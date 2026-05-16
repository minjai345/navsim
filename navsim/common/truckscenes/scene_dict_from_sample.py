"""Single source of truth for `TruckScenes Sample -> navsim scene_dict`.

This module owns the actual conversion. Two consumers wrap it:
  - `runtime_loader.TruckScenesSceneLoader` (Path B, runtime)
  - `build_navsim_logs.main` (Path A, batch pickle dump)

The returned dict matches the schema navsim's `AgentInput.from_scene_dict_list`
and `Scene` factories already consume (see navsim/common/dataclasses.py:148+
for the canonical keys), with two truck-specific additions:

    {
        # ---- navsim-standard keys ----
        "token":                    str,
        "timestamp":                int,        # microseconds (TruckScenes native)
        "ego2global_translation":   (3,)  float64,
        "ego2global_rotation":      (4,)  float64,   # quaternion (w, x, y, z)
        "ego_dynamic_state":        (4,)  float32,   # [vx, vy, ax, ay]
        "driving_command":          (3,)  int / one-hot,
        "cams":                     dict[slot_name -> dict | None],   # 8 keys
        "lidar_path":               None,                              # see note
        "anns":                     dict (boxes, names, velocity_3d,
                                          instance_tokens, track_tokens),
        "roadblock_ids":            [],         # always empty -- no HD map
        "traffic_lights":           [],         # always empty -- no annotation

        # ---- truck-specific additions ----
        "truckscenes_extras":       dict (trailer pose, hitch angle, ...),
    }

Note on `lidar_path`:
  Always None at this layer. We do not materialize merged PCDs to disk
  here. PDMS metric caching does not need sensor blobs (verified against
  navsim/planning/metric_caching/metric_cache_processor.py -- it only
  iterates ego state + annotations). For training / inference where the
  point cloud IS needed, the wrapping loader calls
  `lidar_merge.merge_truckscenes_lidars` separately and attaches the
  result. This split keeps the composer focused on metadata.

The category mapping drops `vehicle.ego_trailer` from `anns` and routes
that information into `truckscenes_extras` instead -- see
`category_mapping.map_truckscenes_category` and `trailer_extras`.
"""
import math
from typing import Any, Dict

import numpy as np
from pyquaternion import Quaternion

from navsim.common.truckscenes.camera_mapping import (
    ALL_NAVSIM_CAM_SLOTS,
    build_cam_dict_for_navsim,
)
from navsim.common.truckscenes.category_mapping import map_truckscenes_category
from navsim.common.truckscenes.trailer_extras import build_trailer_extras

# Default future window for the driving_command derivation. Matches
# transfuser-truckscenes/configs/_base.py:
#   num_poses=8, trajectory_sampling_interval=0.5 -> 4 s horizon.
# Also matches navsim's TrajectorySampling default (time_horizon=4,
# interval_length=0.5).
DEFAULT_NUM_FUTURE_FRAMES = 8

# Heading-mode threshold for driving_command, in degrees, matching
# transfuser-truckscenes/configs/_base.py:
#   driving_command_threshold_heading_deg = 15.0.
DEFAULT_DRIVING_COMMAND_HEADING_DEG = 15.0

# Lateral-mode threshold for driving_command, in metres. Mirrors VAD's
# nuScenes converter (|local_y@4s| > 2.0 -> Left/Right). Catches lane
# changes (small Δyaw, large lateral offset) that heading mode misses.
# See docs/paper_open_decisions.md::D10 for the heading-vs-lateral
# rationale.
DEFAULT_DRIVING_COMMAND_LATERAL_M = 2.0

# Allowed values for driving_command_mode. Adding new modes? Update the
# helper `_driving_command_from_future` below.
DRIVING_COMMAND_MODES = ("heading", "lateral")
DEFAULT_DRIVING_COMMAND_MODE = "heading"


def scene_dict_from_sample(
    ts,
    sample_token: str,
    num_future_frames: int = DEFAULT_NUM_FUTURE_FRAMES,
    driving_command_heading_deg: float = DEFAULT_DRIVING_COMMAND_HEADING_DEG,
    driving_command_lateral_m: float = DEFAULT_DRIVING_COMMAND_LATERAL_M,
    driving_command_mode: str = DEFAULT_DRIVING_COMMAND_MODE,
) -> Dict[str, Any]:
    """Convert a single TruckScenes sample into a navsim-compatible dict.

    :param ts: initialized `TruckScenes` devkit instance.
    :param sample_token: target sample token (navsim's per-frame identity).
    :param num_future_frames: future trajectory length used to derive
        driving_command. The sample must have at least this many `next`
        keyframes available; caller is responsible for filtering.
    :param driving_command_heading_deg: |Δyaw| threshold (degrees) that
        decides STRAIGHT vs LEFT/RIGHT. Used only when
        `driving_command_mode == "heading"`.
    :param driving_command_lateral_m: |local_y| threshold (metres) at the
        final future frame. Used only when
        `driving_command_mode == "lateral"`.
    :param driving_command_mode: one of `DRIVING_COMMAND_MODES`. "heading"
        matches transfuser-truckscenes / nuPlan route-derived semantics.
        "lateral" matches VAD's nuScenes converter and catches lane
        changes.
    :return: dict with the schema documented at module top.
    """
    sample = ts.get("sample", sample_token)
    ref_channel = _get_reference_channel(sample)
    ref_sd = ts.get("sample_data", sample["data"][ref_channel])
    ego_pose = ts.get("ego_pose", ref_sd["ego_pose_token"])

    # Ego pose in GLOBAL frame -- navsim's standard scene_dict expects
    # global translation + quaternion here; local-frame conversion happens
    # in AgentInput.from_scene_dict_list.
    ego2global_translation = np.asarray(ego_pose["translation"], dtype=np.float64)
    ego2global_rotation = np.asarray(ego_pose["rotation"], dtype=np.float64)

    # Ego dynamic state. [vx, vy] from finite differences over ego_pose
    # (chassis CAN vx/vy is unreliable -- 32% of samples are zero-stuck,
    # per transfuser-truckscenes/dataset/dataset.py:289-293); [ax, ay] from
    # ego_motion_chassis IMU.
    vx, vy = _estimate_ego_velocity(ts, sample, ref_channel)
    chassis = ts.getclosest("ego_motion_chassis", ref_sd["timestamp"])
    ax = float(chassis["ax"])
    ay = float(chassis["ay"])
    ego_dynamic_state = np.array([vx, vy, ax, ay], dtype=np.float32)

    # Cameras and annotations.
    cams = build_cam_dict_for_navsim(sample, ts)
    # Annotation conversion needs current ego pose (boxes are emitted in
    # ego frame to match navsim's BoundingBoxIndex convention).
    ego_pos = np.asarray(ego_pose["translation"], dtype=np.float64)
    ego_yaw = _quaternion_to_yaw(ego_pose["rotation"])
    anns = _build_annotations(ts, ref_sd, ego_pos, ego_yaw)
    extras = build_trailer_extras(sample, ts)

    # driving_command derived from the future ego trajectory. Mode picks
    # between heading (|Δyaw@4s|) and lateral (|local_y@4s|); see D10 in
    # docs/paper_open_decisions.md.
    if driving_command_mode not in DRIVING_COMMAND_MODES:
        raise ValueError(
            f"driving_command_mode={driving_command_mode!r} not in "
            f"{DRIVING_COMMAND_MODES}"
        )
    driving_command = _driving_command_from_future(
        ts,
        sample,
        ref_channel,
        num_future_frames=num_future_frames,
        mode=driving_command_mode,
        heading_threshold_deg=driving_command_heading_deg,
        lateral_threshold_m=driving_command_lateral_m,
    )

    # Scene-level identifiers required by `Scene.from_scene_dict_list`.
    # navsim's pkl format treats one pkl as one "log"; we use the
    # TruckScenes scene_token for both `log_name` (pkl filename stem)
    # and `scene_token`. `map_location="no_map"` triggers the
    # forked `Scene._build_map_api` path that returns a NullMap stub
    # (TruckScenes has no HD map).
    scene_token = sample["scene_token"]

    return {
        "token": sample_token,
        "timestamp": int(sample["timestamp"]),
        "log_name": scene_token,
        "scene_token": scene_token,
        "map_location": "no_map",
        "ego2global_translation": ego2global_translation,
        "ego2global_rotation": ego2global_rotation,
        "ego_dynamic_state": ego_dynamic_state,
        "driving_command": driving_command,
        "cams": cams,
        # Always None -- merged PCD is produced on demand by the wrapping
        # loader, not materialized here. See module docstring.
        "lidar_path": None,
        "anns": anns,
        # No HD map / no traffic-light annotation in TruckScenes.
        "roadblock_ids": [],
        "traffic_lights": [],
        "truckscenes_extras": extras,
    }


# -------------------------------------------------------------------- helpers


def _get_reference_channel(sample: dict) -> str:
    """Reference sensor channel for ego_pose lookup. Prefer LIDAR_TOP_FRONT.

    Duplicated from `trailer_extras._get_reference_channel` to keep modules
    independent. If a third copy shows up, lift to a shared `_common.py`.
    """
    if "LIDAR_TOP_FRONT" in sample["data"]:
        return "LIDAR_TOP_FRONT"
    for key in sample["data"]:
        if "LIDAR" in key.upper():
            return key
    return next(iter(sample["data"]))


def _quaternion_to_yaw(rotation_wxyz) -> float:
    """Yaw extracted by rotating the unit-x vector. Matches
    transfuser-truckscenes/dataset/dataset.py:651-655."""
    v = Quaternion(rotation_wxyz).rotate(np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(v[1], v[0]))


def _estimate_ego_velocity(ts, sample, ref_channel, max_time_diff=1.5):
    """Port of transfuser-truckscenes/dataset/dataset.py:_estimate_ego_velocity
    (lines 306-358).

    Returns (vx, vy) in EGO frame. Falls back to (0.0, 0.0) when the local
    neighborhood is degenerate (no prev/next, or time gap too large).
    """
    sd_cur = ts.get("sample_data", sample["data"][ref_channel])
    cur_ep = ts.get("ego_pose", sd_cur["ego_pose_token"])
    cur_yaw = _quaternion_to_yaw(cur_ep["rotation"])

    has_prev = sample.get("prev", "") != ""
    has_next = sample.get("next", "") != ""
    if not has_prev and not has_next:
        return 0.0, 0.0

    # Earlier reference point: prev sample if available, else current.
    if has_prev:
        prev_s = ts.get("sample", sample["prev"])
        prev_sd = ts.get("sample_data", prev_s["data"][ref_channel])
        prev_ep = ts.get("ego_pose", prev_sd["ego_pose_token"])
        first_pos = np.array(prev_ep["translation"][:2])
        first_t = prev_sd["timestamp"]
    else:
        first_pos = np.array(cur_ep["translation"][:2])
        first_t = sd_cur["timestamp"]

    # Later reference point: next sample if available, else current.
    if has_next:
        next_s = ts.get("sample", sample["next"])
        next_sd = ts.get("sample_data", next_s["data"][ref_channel])
        next_ep = ts.get("ego_pose", next_sd["ego_pose_token"])
        last_pos = np.array(next_ep["translation"][:2])
        last_t = next_sd["timestamp"]
    else:
        last_pos = np.array(cur_ep["translation"][:2])
        last_t = sd_cur["timestamp"]

    dt = (last_t - first_t) / 1e6  # microseconds -> seconds
    # Allow 2x the per-sample budget when using centered (prev + next).
    if has_prev and has_next:
        max_time_diff *= 2
    if dt <= 0 or dt > max_time_diff:
        return 0.0, 0.0

    # Global velocity -> rotate into current ego heading.
    v_global = (last_pos - first_pos) / dt
    cos_y, sin_y = math.cos(-cur_yaw), math.sin(-cur_yaw)
    vx = v_global[0] * cos_y - v_global[1] * sin_y
    vy = v_global[0] * sin_y + v_global[1] * cos_y
    return float(vx), float(vy)


def _build_annotations(ts, ref_sd, ego_pos: np.ndarray, ego_yaw: float) -> Dict[str, Any]:
    """Build the per-frame `anns` dict in navsim shape.

    Boxes are stored in this frame's EGO frame, matching navsim's
    convention (see `navsim/common/enums.py:BoundingBoxIndex` and the
    use of `_xy_in_lidar` directly against `box[X], box[Y]` in
    `navsim.agents.transfuser.transfuser_features._compute_agent_targets`).

    Column layout matches `BoundingBoxIndex`:
        [X, Y, Z, LENGTH, WIDTH, HEIGHT, HEADING]

    TruckScenes devkit `box.wlh` orders dimensions as [width, length, height];
    we swap to [length, width, height] to match navsim's expectation.
    Trailer state lives in `truckscenes_extras` -- `vehicle.ego_trailer` is
    dropped from this annotations dict.
    """
    cos_y, sin_y = math.cos(-ego_yaw), math.sin(-ego_yaw)

    boxes_list = []
    names_list = []
    velocity_list = []
    instance_tokens = []
    track_tokens = []

    for box in ts.get_boxes(ref_sd["token"]):
        mapped = map_truckscenes_category(box.name)
        if mapped is None:
            continue
        ann = ts.get("sample_annotation", box.token)

        # Global -> ego frame (this frame's tractor ego_pose).
        gx, gy, gz = float(box.center[0]), float(box.center[1]), float(box.center[2])
        dx, dy = gx - float(ego_pos[0]), gy - float(ego_pos[1])
        ex = dx * cos_y - dy * sin_y
        ey = dx * sin_y + dy * cos_y
        # Z is shared between global and ego (no roll/pitch correction --
        # vanilla navsim does not do it either).
        ez = gz - float(ego_pos[2])

        box_yaw = _quaternion_to_yaw(box.orientation.q)
        local_heading = (box_yaw - ego_yaw + math.pi) % (2 * math.pi) - math.pi

        # devkit wlh = [width, length, height]; navsim wants [length, width, height].
        w = float(box.wlh[0])
        l = float(box.wlh[1])
        h = float(box.wlh[2])

        boxes_list.append([ex, ey, ez, l, w, h, local_heading])
        names_list.append(mapped)
        velocity_list.append(_box_velocity(ts, ann["token"]))
        instance_tokens.append(ann["instance_token"])
        # TruckScenes does not carry a scene-level track id distinct from
        # the instance token. Use instance_token for both -- matches the
        # devkit's effective track identity.
        track_tokens.append(ann["instance_token"])

    # Key names use the `gt_` prefix expected by
    # `Scene._build_annotations` (`gt_boxes`, `gt_names`, `gt_velocity_3d`).
    # `instance_tokens` / `track_tokens` keep their bare names per navsim.
    return {
        "gt_boxes": np.asarray(boxes_list, dtype=np.float32).reshape(-1, 7),
        "gt_names": names_list,
        "gt_velocity_3d": np.asarray(velocity_list, dtype=np.float32).reshape(-1, 3),
        "instance_tokens": instance_tokens,
        "track_tokens": track_tokens,
    }


def _box_velocity(ts, ann_token: str) -> tuple:
    """Wrap devkit `box_velocity` and replace NaN with zero so downstream
    code doesn't have to special-case missing motion. devkit returns
    (vx, vy, vz)."""
    v = ts.box_velocity(ann_token)
    return (
        float(v[0]) if np.isfinite(v[0]) else 0.0,
        float(v[1]) if np.isfinite(v[1]) else 0.0,
        float(v[2]) if np.isfinite(v[2]) else 0.0,
    )


def _driving_command_from_future(
    ts,
    sample,
    ref_channel,
    num_future_frames: int,
    mode: str,
    heading_threshold_deg: float,
    lateral_threshold_m: float,
) -> np.ndarray:
    """Derive a one-hot driving_command from the future ego trajectory.

    Two derivation modes (see docs/paper_open_decisions.md::D10):

      * "heading" -- mirrors transfuser-truckscenes _get_driving_command
        (dataset.py:360-415). Compares |Δyaw@N| to
        `heading_threshold_deg`. Misses lane changes (small Δyaw, large
        lateral offset).
      * "lateral" -- mirrors VAD's nuScenes converter. Compares the
        final-frame local y (lateral offset in the CURRENT ego frame)
        to `lateral_threshold_m`. Catches lane changes; classifies
        gradual curves as LEFT/RIGHT.

    Returns a (3,) one-hot int array in the order [Turn Right, Turn
    Left, Go Straight]. Falls back to STRAIGHT when the future chain
    is shorter than `num_future_frames`.
    """
    sd_cur = ts.get("sample_data", sample["data"][ref_channel])
    ego_cur = ts.get("ego_pose", sd_cur["ego_pose_token"])
    current_yaw = _quaternion_to_yaw(ego_cur["rotation"])
    current_pos = np.asarray(ego_cur["translation"][:2], dtype=np.float64)

    # Walk forward up to N keyframes to find the latest ego pose. The
    # loop tolerates short chains (e.g. last samples in a scene) and
    # records the final valid yaw + global xy.
    next_token = sample.get("next", "")
    last_valid_yaw = None
    last_valid_pos = None
    for _ in range(num_future_frames):
        if not next_token:
            break
        next_sample = ts.get("sample", next_token)
        next_sd = ts.get("sample_data", next_sample["data"][ref_channel])
        next_ego = ts.get("ego_pose", next_sd["ego_pose_token"])
        last_valid_yaw = _quaternion_to_yaw(next_ego["rotation"])
        last_valid_pos = np.asarray(next_ego["translation"][:2], dtype=np.float64)
        next_token = next_sample.get("next", "")

    # One-hot order is fixed across modes for downstream parity:
    # [Turn Right, Turn Left, Go Straight]. "Left" = positive lateral
    # in the right-handed ego frame (same sign convention as Δyaw>0).
    if mode == "heading":
        if last_valid_yaw is None:
            return np.array([0, 0, 1], dtype=np.int64)
        delta = (last_valid_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
        threshold = math.radians(heading_threshold_deg)
        if delta >= threshold:
            return np.array([0, 1, 0], dtype=np.int64)
        if delta <= -threshold:
            return np.array([1, 0, 0], dtype=np.int64)
        return np.array([0, 0, 1], dtype=np.int64)

    if mode == "lateral":
        if last_valid_pos is None:
            return np.array([0, 0, 1], dtype=np.int64)
        # Rotate (last_pos - current_pos) into the CURRENT ego frame so
        # local_y is lateral offset perpendicular to the ego heading at
        # t=0. Same convention as `_estimate_ego_velocity` above.
        dx = float(last_valid_pos[0] - current_pos[0])
        dy = float(last_valid_pos[1] - current_pos[1])
        cos_y, sin_y = math.cos(-current_yaw), math.sin(-current_yaw)
        local_y = dx * sin_y + dy * cos_y
        if local_y >= lateral_threshold_m:
            return np.array([0, 1, 0], dtype=np.int64)
        if local_y <= -lateral_threshold_m:
            return np.array([1, 0, 0], dtype=np.int64)
        return np.array([0, 0, 1], dtype=np.int64)

    # Unreachable -- mode is validated by scene_dict_from_sample.
    raise AssertionError(f"unhandled driving_command_mode={mode!r}")
