"""Feature / Target builders for the TruckScenes-adapted DiffusionDrive agent.

The feature side is identical to our TruckTransfuser pipeline -- same
4-cam stitched image, same LiDAR BEV histogram, same (vx, vy, ax, ay)
status, optional driving_command. So we reuse
`TruckTransfuserFeatureBuilder` unchanged.

The target side reuses `TruckTransfuserTargetBuilder` but adds a
zeros-filled `bev_semantic_map` so the upstream
`diffusiondrive.transfuser_loss` (which references that key
unconditionally) does not KeyError. With
`config.bev_semantic_weight=0.0`, the cross-entropy contribution to
total loss is exactly zero. The wasted CE compute is accepted in
exchange for not forking the upstream model / loss.

The trailer-trajectory target (`trailer_trajectory`, `trailer_mask`)
emitted by `TruckTransfuserTargetBuilder` is preserved here for
forward compatibility -- DiffusionDrive's upstream loss does NOT
consume it (no `trailer_weight` field), so it has no effect on
training right now. When we wire a trailer head into DiffusionDrive
later, the target is already in place.
"""
from typing import Dict

import numpy as np
import torch

from navsim.agents.truck_diffusiondrive.truck_diffusiondrive_config import (
    TruckDiffusionDriveConfig,
)
from navsim.agents.truck_transfuser.truck_transfuser_features import (
    TruckTransfuserFeatureBuilder,
    TruckTransfuserTargetBuilder,
)
from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class TruckDiffusionDriveFeatureBuilder(TruckTransfuserFeatureBuilder):
    """Truck-adapted DiffusionDrive features.

    Camera + LiDAR feature path is identical to TruckTransfuser (4-cam
    stitch + BEV histogram, see TruckTransfuserFeatureBuilder). The
    `status_feature` is REPLACED to match a VAD §4.2 shortcut-free
    protocol -- a fair-comparison decision documented in the
    project_paper_deadline / project_map_free_pdms memory:

      Upstream DiffusionDrive: status_feature = concat(
          driving_command (4D nuplan one-hot),
          ego_velocity (2D),
          ego_acceleration (2D),
      ) -> shape (8,), Linear(8, tf_d_model).

      Our truck variant: status_feature = driving_command (3D from
      VAD-style trajectory heading threshold). Ego status omitted to
      avoid the open-loop shortcut where vx*dt approximately equals
      the future trajectory. Status encoder becomes Linear(3,
      tf_d_model) -- see TruckV2TransfuserModel which patches it after
      super().__init__.

    Both TruckTransfuser and TruckDiffusionDrive baselines therefore
    consume the SAME 3D driving_command input under the SAME shortcut
    -free regime; the paper's articulation PDMS comparison is on equal
    input footing.
    """

    def __init__(self, config: TruckDiffusionDriveConfig):
        super().__init__(config)

    def get_unique_name(self) -> str:
        return "truck_diffusiondrive_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # Inherit camera + lidar (4-cam stitch + BEV histogram); replace
        # status_feature with driving_command only (3D, no ego status).
        ego = agent_input.ego_statuses[-1]
        return {
            "camera_feature": self._get_camera_feature(agent_input),
            "lidar_feature": self._get_lidar_feature(agent_input),
            "status_feature": torch.tensor(ego.driving_command, dtype=torch.float32),
        }


class TruckDiffusionDriveTargetBuilder(TruckTransfuserTargetBuilder):
    """Truck targets + a zeros-filled bev_semantic_map placeholder.

    The placeholder satisfies upstream `diffusiondrive.transfuser_loss`'s
    unconditional `targets["bev_semantic_map"]` lookup. Shape matches
    `config.bev_semantic_frame = (bev_pixel_height, bev_pixel_width)` and
    dtype is int64 (CE expects long class indices). All zeros = class 0
    everywhere; weight=0 in the loss kills its contribution.
    """

    def __init__(self, config: TruckDiffusionDriveConfig):
        super().__init__(config)

    def get_unique_name(self) -> str:
        return "truck_diffusiondrive_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        targets = super().compute_targets(scene)
        # Zeros placeholder for the BEV semantic head's CE loss.
        # config.bev_semantic_frame returns (bev_pixel_height, bev_pixel_width).
        h, w = self._config.bev_semantic_frame
        targets["bev_semantic_map"] = torch.tensor(np.zeros((h, w), dtype=np.int64))
        return targets
