"""Truck TransFuser feature / target builders.

Ports the heavy preprocessing from transfuser-truckscenes/dataset/dataset.py
into navsim's `AbstractFeatureBuilder` / `AbstractTargetBuilder` API.

Key deltas vs. vanilla navsim TransfuserFeatureBuilder:

  - Camera: 4-camera input layout (cam_l0=LEFT_FRONT, cam_r0=RIGHT_FRONT,
    cam_l2=LEFT_BACK, cam_r2=RIGHT_BACK). Front pair is directionally
    cropped to a 1.5:1 aspect ratio (left/right cams keep their outer
    field of view); back pair is used uncropped. Final stitched image is
    resized to (camera_height, camera_width) = (256, 1536).
  - Lidar: same histogram pipeline as vanilla. We do NOT raise when the
    point cloud is missing (lidar_pc=None) -- this happens when the
    scene_dict pkl was emitted with `lidar_path=None` (PDMS-eval-only
    flavor; see project_map_free_pdms memory). In that case we return a
    zero tensor so the rest of the forward path still runs. Training
    pkls must be emitted with sensor blobs materialized -- a flag for
    that lives in `build_navsim_logs.py` (not yet implemented).
  - Status: 4D `(vx, vy, ax, ay)` regardless of `use_driving_command`.
    The optional driving_command goes into a separate features key so
    the model can mask vx/vy with status_dropout independently.

Key deltas vs. vanilla navsim TransfuserTargetBuilder:

  - No bev_semantic_map target (no HD map; bev_semantic_weight=0 default).
  - Trailer trajectory target (+ mask) when `use_trailer_head=True`.
    Trailer poses live in `Frame.truckscenes_extras["trailer_pose"]`
    (GLOBAL frame; see `navsim/common/truckscenes/trailer_extras.py`) so
    we convert global -> current-tractor-ego using
    `convert_absolute_to_relative_se2_array`. Chain-break padding
    matches transfuser-truckscenes (pad with previous local pose; first
    invalid stays zero).
  - Agent category filter matches our category mapping
    (`navsim/common/truckscenes/category_mapping.py`): `name.startswith(
    "vehicle.")` rather than vanilla's `name == "vehicle"`.

`BoundingBox2DIndex` is re-exported from `_indices.py` so the heavyweight
imports here (which transitively pull nuplan-devkit via
`navsim.common.dataclasses`) do not leak into `truck_transfuser_model.py`
or `truck_transfuser_loss.py`.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torchvision import transforms

from navsim.agents.truck_transfuser._indices import BoundingBox2DIndex
from navsim.agents.truck_transfuser.truck_transfuser_config import TruckTransfuserConfig
from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


# Re-export so existing callers that imported BoundingBox2DIndex from this
# module continue to work (the canonical home is _indices.py).
__all__ = [
    "BoundingBox2DIndex",
    "TruckTransfuserFeatureBuilder",
    "TruckTransfuserTargetBuilder",
]


# ============================================================ FeatureBuilder


class TruckTransfuserFeatureBuilder(AbstractFeatureBuilder):
    """Builds the feature dict consumed by `TruckTransfuserModel.forward`.

    Produces:
        camera_feature:   (3, camera_height, camera_width) stitched RGB.
        lidar_feature:    (1 or 2, lidar_resolution_height, lidar_resolution_width).
        status_feature:   (4,) [vx, vy, ax, ay].
        driving_command:  (3,) one-hot -- ONLY when use_driving_command=True.
    """

    def __init__(self, config: TruckTransfuserConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "truck_transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {
            "camera_feature": self._get_camera_feature(agent_input),
            "lidar_feature": self._get_lidar_feature(agent_input),
            "status_feature": self._get_status_feature(agent_input),
        }
        if self._config.use_driving_command:
            features["driving_command"] = self._get_driving_command(agent_input)
        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """4-camera stitch port of transfuser-truckscenes:_get_camera_feature.

        Slot mapping (set by `navsim.common.truckscenes.camera_mapping`):
            cam_l0 = CAMERA_LEFT_FRONT   -> keep left FOV  (crop right)
            cam_r0 = CAMERA_RIGHT_FRONT  -> keep right FOV (crop left)
            cam_l2 = CAMERA_LEFT_BACK    -> no crop
            cam_r2 = CAMERA_RIGHT_BACK   -> no crop

        Concatenation order along width: [l0_cropped, r0_cropped, l2, r2].
        """
        cameras = agent_input.cameras[-1]
        plan = [
            (cameras.cam_l0, "left"),
            (cameras.cam_r0, "right"),
            (cameras.cam_l2, None),
            (cameras.cam_r2, None),
        ]
        images: List[npt.NDArray[np.uint8]] = []
        for cam, crop_side in plan:
            img = cam.image
            if img is None:
                # Sensor dropout or empty slot. We do not synthesize a
                # placeholder here -- raise so a misconfigured pkl
                # (missing one of the 4 truck cams) fails loudly during
                # training rather than silently zero-filling.
                raise ValueError(
                    "TruckTransfuserFeatureBuilder: a required camera slot "
                    "(cam_l0/r0/l2/r2) has no image. Check the SensorConfig "
                    "used to build the AgentInput."
                )
            if crop_side is not None:
                img = _crop_to_aspect(img, 1.5, crop_side)
            images.append(img)
        stitched = np.concatenate(images, axis=1)
        resized = cv2.resize(stitched, (self._config.camera_width, self._config.camera_height))
        return transforms.ToTensor()(resized)

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """BEV 2D histogram from the latest history-frame point cloud.

        Returns zeros when `lidar_pc` is None (the PDMS-eval-only pkl
        flavor emits no sensor blobs -- see module docstring). Training
        pkls must have lidar materialized.
        """
        in_ch = 2 if self._config.use_ground_plane else 1
        zero_lidar = torch.zeros(
            (in_ch, self._config.lidar_resolution_height, self._config.lidar_resolution_width),
            dtype=torch.float32,
        )
        lidar = agent_input.lidars[-1]
        if lidar.lidar_pc is None:
            return zero_lidar
        # (6, N) -> (N, 3): x, y, z only.
        pc = lidar.lidar_pc[LidarIndex.POSITION].T
        return self._compute_lidar_histogram(pc)

    def _compute_lidar_histogram(self, pc: np.ndarray) -> torch.Tensor:
        """Port of transfuser-truckscenes/dataset/dataset.py:_compute_lidar_histogram."""
        config = self._config

        def splat_points(point_cloud: np.ndarray) -> np.ndarray:
            xbins = np.linspace(
                config.lidar_min_x,
                config.lidar_max_x,
                int((config.lidar_max_x - config.lidar_min_x) * config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                config.lidar_min_y,
                config.lidar_max_y,
                int((config.lidar_max_y - config.lidar_min_y) * config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
            return hist / config.hist_max_per_pixel

        pc = pc[pc[:, 2] < config.max_height_lidar]
        above = pc[pc[:, 2] > config.lidar_split_height]
        above_features = splat_points(above)
        if config.use_ground_plane:
            below = pc[pc[:, 2] <= config.lidar_split_height]
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return torch.tensor(features)

    def _get_status_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """Status feature = [vx, vy, ax, ay] from the latest ego_status.

        The optional 3D driving_command is emitted separately (see
        compute_features); keeping them split lets the model apply
        status dropout to vx/vy/ax/ay only.
        """
        ego = agent_input.ego_statuses[-1]
        return torch.cat(
            [
                torch.tensor(ego.ego_velocity, dtype=torch.float32),
                torch.tensor(ego.ego_acceleration, dtype=torch.float32),
            ]
        )

    def _get_driving_command(self, agent_input: AgentInput) -> torch.Tensor:
        """One-hot driving command (3D) from the latest ego_status."""
        cmd = agent_input.ego_statuses[-1].driving_command
        return torch.tensor(cmd, dtype=torch.float32)


# ============================================================== TargetBuilder


class TruckTransfuserTargetBuilder(AbstractTargetBuilder):
    """Builds the target dict consumed by `truck_transfuser_loss`.

    Produces:
        trajectory:           (num_poses, 3) local SE(2).
        agent_states:         (num_bounding_boxes, 5) local ego frame.
        agent_labels:         (num_bounding_boxes,) bool (presence).
        trailer_trajectory:   (num_poses, 3) local SE(2) -- only when
                              use_trailer_head=True.
        trailer_mask:         scalar 1.0 / 0.0 -- only when
                              use_trailer_head=True.
    """

    def __init__(self, config: TruckTransfuserConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "truck_transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        targets: Dict[str, torch.Tensor] = {
            "trajectory": self._get_trajectory_target(scene),
        }
        agent_states, agent_labels = self._get_agent_targets(scene)
        targets["agent_states"] = agent_states
        targets["agent_labels"] = agent_labels
        if self._config.use_trailer_head:
            trailer_traj, trailer_mask = self._get_trailer_target(scene)
            targets["trailer_trajectory"] = trailer_traj
            targets["trailer_mask"] = trailer_mask
        return targets

    def _get_trajectory_target(self, scene: Scene) -> torch.Tensor:
        """Future ego trajectory in current-frame ego coordinates.

        `Scene.get_future_trajectory` already returns local SE(2) poses
        for the next `num_trajectory_frames` keyframes; we just convert
        the numpy poses to a tensor.
        """
        traj = scene.get_future_trajectory(num_trajectory_frames=self._config.num_poses)
        return torch.tensor(traj.poses, dtype=torch.float32)

    def _get_agent_targets(self, scene: Scene) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-N nearest in-range vehicle boxes from current-frame annotations.

        Boxes are ALREADY in current-frame ego coordinates (see
        `navsim.common.truckscenes.scene_dict_from_sample._build_annotations`).
        We filter by name (any "vehicle.*" category survives our category
        mapping; ego_trailer is dropped at the adapter layer) and by
        lidar-range membership, then keep the `num_bounding_boxes`
        nearest.
        """
        idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[idx].annotations
        max_agents = self._config.num_bounding_boxes
        config = self._config

        def _xy_in_lidar(x: float, y: float) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                config.lidar_min_y <= y <= config.lidar_max_y
            )

        states_list: List[npt.NDArray[np.float32]] = []
        for box, name in zip(annotations.boxes, annotations.names):
            if not name.startswith("vehicle."):
                continue
            box_x = float(box[BoundingBoxIndex.X])
            box_y = float(box[BoundingBoxIndex.Y])
            if not _xy_in_lidar(box_x, box_y):
                continue
            box_heading = float(box[BoundingBoxIndex.HEADING])
            box_length = float(box[BoundingBoxIndex.LENGTH])
            box_width = float(box[BoundingBoxIndex.WIDTH])
            states_list.append(
                np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
            )

        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)
        if states_list:
            arr = np.stack(states_list, axis=0)
            distances = np.linalg.norm(arr[:, BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]
            arr = arr[argsort]
            agent_states[: len(arr)] = arr
            agent_labels[: len(arr)] = True
        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _get_trailer_target(self, scene: Scene) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trailer future trajectory in current-tractor ego frame + mask.

        Walks future frames; for each, reads `truckscenes_extras["trailer_pose"]`
        (global frame) and converts to local. Chain-break (no trailer in
        that frame or scene ends early) is padded with the previous local
        pose -- first-frame invalid stays zero. Mask is 1 iff the CURRENT
        frame had a trailer annotation; trailer-free scenes return all
        zeros + mask=0.
        """
        # Lazy imports: these touch nuplan and would otherwise pull it in
        # at module load, defeating the truck_transfuser_loss /
        # truck_transfuser_model import chain working in environments
        # without nuplan installed.
        from nuplan.common.actor_state.state_representation import StateSE2
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
            convert_absolute_to_relative_se2_array,
        )

        idx = scene.scene_metadata.num_history_frames - 1
        num_poses = self._config.num_poses
        trajectory = np.zeros((num_poses, 3), dtype=np.float32)

        cur_extras = scene.frames[idx].truckscenes_extras
        if not cur_extras.get("has_trailer", False):
            return torch.tensor(trajectory), torch.tensor(0.0, dtype=torch.float32)

        current_origin = StateSE2(*scene.frames[idx].ego_status.ego_pose)

        for k in range(num_poses):
            future_idx = idx + 1 + k
            future_has = (
                future_idx < len(scene.frames)
                and scene.frames[future_idx].truckscenes_extras.get("has_trailer", False)
            )
            if future_has:
                global_pose = np.asarray(
                    scene.frames[future_idx].truckscenes_extras["trailer_pose"],
                    dtype=np.float64,
                ).reshape(1, 3)
                local_pose = convert_absolute_to_relative_se2_array(current_origin, global_pose)
                trajectory[k] = local_pose[0].astype(np.float32)
            elif k > 0:
                # Chain break or scene end -- pad with previous local pose
                # (matches transfuser-truckscenes:_get_trailer_trajectory_target).
                trajectory[k] = trajectory[k - 1]
            # else: first-frame invalid (no future[0] trailer) stays zero,
            # which is also the transfuser-truckscenes default.

        return torch.tensor(trajectory), torch.tensor(1.0, dtype=torch.float32)


# ================================================================== helpers


def _crop_to_aspect(
    image: npt.NDArray[np.uint8],
    aspect_ratio: float,
    side: str = "center",
) -> npt.NDArray[np.uint8]:
    """Crop `image` to the requested width/height aspect ratio.

    Ported verbatim from transfuser-truckscenes/dataset/dataset.py:611-637.
    `side` controls which edge to keep when cropping width:
      - "left":  drop right side (keep left FOV)
      - "right": drop left side  (keep right FOV)
      - "center": drop both equally
    """
    height, width = image.shape[:2]
    current_aspect = width / float(height)

    if current_aspect > aspect_ratio:
        crop_width = int(round(height * aspect_ratio))
        if side == "left":
            return image[:, :crop_width]
        elif side == "right":
            return image[:, width - crop_width:]
        else:
            left = (width - crop_width) // 2
            return image[:, left: left + crop_width]

    crop_height = int(round(width / aspect_ratio))
    top = (height - crop_height) // 2
    return image[top: top + crop_height, :]
