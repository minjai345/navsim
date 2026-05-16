"""L2 displacement + collision metrics for truck open-loop planning eval.

Ports the metric logic from `transfuser-truckscenes/evaluate.py` (the
prior code base) into navsim's Scene/AgentInput world. This is the
lightweight "what we can ship today" evaluator -- a stand-in until the
full no-map PDMS scorer in `docs/truck_pdm_scorer_design.md` lands.

What it computes per sample
---------------------------
For each evaluation horizon h ∈ EVAL_HORIZONS (default 1, 2, 3 s):

  * L2 error: ||pred_xy@h - gt_xy@h||_2 in metres (current-ego frame).
  * Collision: 1 if the ego rectangle at pred pose @h intersects any
    surrounding vehicle bbox at that timestep (transformed into the
    current-ego frame). The ego dimensions default to the truck
    tractor (6.9 × 2.5 m).

Aggregated metrics: mean L2 and mean collision rate (%), per horizon
plus average across horizons. Follows the UniAD / VAD / ST-P3
open-loop convention.

What it does NOT do
-------------------
  * No HD map, no traffic light, no progress metric -- those are PDMS
    scorer territory and require navsim's metric_cache machinery.
  * No articulation-aware (off-tracking / swept-volume / hitch
    stability) terms -- those are the paper's Phase-2 contribution
    described in `docs/truck_pdm_scorer_design.md`.

Both follow on once the no-map PDMS scaffolding is in place.
"""
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.common.dataclasses import Scene
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

# Open-loop horizons (seconds). UniAD / VAD / ST-P3 convention.
EVAL_HORIZONS = (1.0, 2.0, 3.0)

# Default sampling interval for our trajectory targets. Matches
# TruckTransfuserConfig.trajectory_sampling_interval = 0.5 s.
DEFAULT_INTERVAL_S = 0.5

# Truck tractor dimensions used for the ego rectangle in collision
# checks. Same defaults as transfuser-truckscenes/evaluate.py:127-128;
# justified by MAN TGX dimensions cited in the TruckScenes paper.
DEFAULT_EGO_LENGTH_M = 6.9
DEFAULT_EGO_WIDTH_M = 2.5

# Vehicle category names we count as "agents" for collision purposes.
# Mirrors transfuser-truckscenes/dataset/dataset.py::_is_vehicle_category
# but uses navsim's mapped names (see category_mapping.py for the
# TruckScenes->navsim mapping).
_VEHICLE_NAMES = frozenset({
    "vehicle.car",
    "vehicle.truck",
    "vehicle.trailer",
    "vehicle.bus",
    "vehicle.construction_vehicle",
    "vehicle.motorcycle",
    "vehicle.bicycle",
})


@dataclass
class _AgentBox2D:
    """A 2D oriented bbox in some reference frame. (x, y) is centre."""

    x: float
    y: float
    heading: float
    length: float
    width: float


def _oriented_box_polygon(x: float, y: float, heading: float,
                          length: float, width: float) -> Polygon:
    """Build a Shapely Polygon for an oriented rectangle (centre x,y).

    Mirrors transfuser-truckscenes/evaluate.py:31-40. Length is along
    the local x-axis (heading direction), width along local y.
    """
    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    hl, hw = length / 2.0, width / 2.0
    corners = np.array(
        [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]],
        dtype=np.float64,
    )
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    corners = corners @ rot.T + np.array([x, y], dtype=np.float64)
    return Polygon(corners)


def _check_collision(ego_pose_xyh: Sequence[float], ego_length: float,
                     ego_width: float, agent_boxes: Iterable[_AgentBox2D]) -> bool:
    """True if ego rectangle at pose (x, y, heading) intersects any agent."""
    ego_poly = _oriented_box_polygon(
        ego_pose_xyh[0], ego_pose_xyh[1], ego_pose_xyh[2],
        ego_length, ego_width,
    )
    for box in agent_boxes:
        if ego_poly.intersects(
            _oriented_box_polygon(box.x, box.y, box.heading, box.length, box.width)
        ):
            return True
    return False


def _get_future_agent_boxes_current_ego_frame(
    scene: Scene, future_step_idx: int,
) -> List[_AgentBox2D]:
    """Future-time agent boxes expressed in the CURRENT ego frame.

    navsim stores each future frame's annotations in THAT frame's ego
    frame; for collision against the predicted ego trajectory (which
    we predict in the *current* ego frame) we need them re-expressed
    in the current-ego coordinate system.

    Pipeline per agent box:
      future-ego coords  -- (rotate by +future_yaw, translate by future_xy) →
      global coords      -- (translate by -current_xy, rotate by -current_yaw) →
      current-ego coords

    The convert_absolute_to_relative_se2_array helper handles the second
    leg; the first leg is just an inverse SE2.

    :param scene: navsim Scene
    :param future_step_idx: 0-based offset into the FUTURE frames; 0 means
        the first frame after the "current" reference frame.
    :return: list of agent boxes (vehicle category) in current ego frame.
    """
    num_history = scene.scene_metadata.num_history_frames
    current_idx = num_history - 1
    future_frame_idx = current_idx + 1 + future_step_idx
    if future_frame_idx >= len(scene.frames):
        return []

    current_pose = scene.frames[current_idx].ego_status.ego_pose
    future_pose = scene.frames[future_frame_idx].ego_status.ego_pose
    future_frame = scene.frames[future_frame_idx]
    boxes = future_frame.annotations.boxes  # (N, 7): [X, Y, Z, L, W, H, HEADING]
    names = future_frame.annotations.names

    if len(boxes) == 0:
        return []

    # ---- Leg 1: future-ego → global. Inverse SE2 around future_pose. ----
    fx, fy, fyaw = float(future_pose[0]), float(future_pose[1]), float(future_pose[2])
    cos_f, sin_f = float(np.cos(fyaw)), float(np.sin(fyaw))
    box_xy_local = np.asarray(boxes[:, :2], dtype=np.float64)
    # Rotate by +future_yaw, then translate by future_xy.
    R_f = np.array([[cos_f, -sin_f], [sin_f, cos_f]], dtype=np.float64)
    box_xy_global = box_xy_local @ R_f.T + np.array([fx, fy], dtype=np.float64)
    box_yaw_global = np.asarray(boxes[:, 6], dtype=np.float64) + fyaw

    # ---- Leg 2: global → current-ego via the existing helper. ----
    se2_global = np.stack(
        [box_xy_global[:, 0], box_xy_global[:, 1], box_yaw_global], axis=-1
    ).astype(np.float64)
    current_se2 = StateSE2(
        float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
    )
    se2_current = convert_absolute_to_relative_se2_array(current_se2, se2_global)

    # ---- Filter to vehicle categories. ----
    out: List[_AgentBox2D] = []
    for i in range(len(names)):
        if names[i] not in _VEHICLE_NAMES:
            continue
        length = float(boxes[i, 3])
        width = float(boxes[i, 4])
        out.append(_AgentBox2D(
            x=float(se2_current[i, 0]),
            y=float(se2_current[i, 1]),
            heading=float(se2_current[i, 2]),
            length=length,
            width=width,
        ))
    return out


def compute_sample_metrics(
    pred_trajectory: npt.NDArray[np.float64],
    gt_trajectory: npt.NDArray[np.float64],
    scene: Scene,
    horizons_s: Sequence[float] = EVAL_HORIZONS,
    interval_s: float = DEFAULT_INTERVAL_S,
    ego_length_m: float = DEFAULT_EGO_LENGTH_M,
    ego_width_m: float = DEFAULT_EGO_WIDTH_M,
) -> Tuple[List[float], List[bool]]:
    """Compute per-horizon L2 + collision for ONE sample.

    :param pred_trajectory: (num_poses, 3) array of [x, y, heading] in
        the current ego frame. Comes from the model.
    :param gt_trajectory: (num_poses, 3) ground truth, same frame.
    :param scene: the navsim Scene (used for future agent boxes).
    :param horizons_s: list of horizons (seconds) to evaluate at.
    :param interval_s: per-step interval of the trajectory (default 0.5 s).
    :return: (l2_per_horizon, collision_per_horizon) -- two parallel
        lists matching horizons_s.
    """
    l2_list: List[float] = []
    col_list: List[bool] = []
    for h in horizons_s:
        # horizon→step index in (num_poses) array. 0.5s interval, 1s
        # horizon → idx 1 (= step 2 = 1s into future). transfuser
        # convention: int(h / interval) - 1. Verified vs original.
        idx = int(round(h / interval_s)) - 1
        if idx < 0 or idx >= pred_trajectory.shape[0]:
            l2_list.append(float("nan"))
            col_list.append(False)
            continue

        dxy = pred_trajectory[idx, :2] - gt_trajectory[idx, :2]
        l2_list.append(float(np.linalg.norm(dxy)))

        agents = _get_future_agent_boxes_current_ego_frame(scene, idx)
        col_list.append(_check_collision(
            pred_trajectory[idx], ego_length_m, ego_width_m, agents
        ))

    return l2_list, col_list


def aggregate_metrics(
    l2_per_sample: Sequence[Sequence[float]],
    col_per_sample: Sequence[Sequence[bool]],
    horizons_s: Sequence[float] = EVAL_HORIZONS,
) -> dict:
    """Stack per-sample metrics into mean L2 / mean collision (%) by horizon.

    Returns flat dict keys like ``"l2/1s": 1.23, "col/3s": 5.4, "l2/avg": ...``.
    """
    l2_arr = np.asarray(l2_per_sample, dtype=np.float64)  # (N, H)
    col_arr = np.asarray(col_per_sample, dtype=np.float64)

    out: dict = {}
    l2_means: List[float] = []
    col_means: List[float] = []
    for j, h in enumerate(horizons_s):
        l2_m = float(np.nanmean(l2_arr[:, j])) if l2_arr.size else float("nan")
        col_m = float(np.mean(col_arr[:, j])) * 100.0 if col_arr.size else float("nan")
        out[f"l2/{int(h)}s"] = l2_m
        out[f"col/{int(h)}s"] = col_m
        l2_means.append(l2_m)
        col_means.append(col_m)
    out["l2/avg"] = float(np.mean(l2_means)) if l2_means else float("nan")
    out["col/avg"] = float(np.mean(col_means)) if col_means else float("nan")
    out["num_samples"] = int(l2_arr.shape[0])
    return out
