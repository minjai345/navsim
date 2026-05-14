"""TruckScenes-specific side-channel data attached to each `Frame`.

navsim's standard `Annotations` slot is for "other agents" only. The truck
trailer is rigidly attached to the ego tractor (via hitch) and is treated
as an extension of ego, not as a foreign agent (see design discussion --
Option C). To keep this distinction explicit in the data flow, trailer
state lives in a separate per-frame `Dict[str, Any]`.

Consumers:
  - Truck FeatureBuilder / TargetBuilder (training, inference): pull
    `trailer_pose`, `hitch_angle`, etc. for the trailer head.
  - Forked articulation PDMS scorer: pull trailer footprint for
    swept-volume / off-tracking metrics.

Default navsim consumers (collision check on `Annotations`, agent detection
target, agent encoder) NEVER look at this dict, so the trailer can never be
silently mistaken for a foreign vehicle.

Schema (per-frame dict):

    {
        "has_trailer":     bool,                     # False -> other keys may be NaN
        "trailer_pose":    (x, y, yaw) float64,      # GLOBAL frame
        "trailer_length":  float,                     # meters
        "trailer_width":   float,                     # meters
        "hitch_angle":     float,                     # tractor_yaw - trailer_yaw, radians
    }

Trailer pose is ALWAYS hitch-corrected here -- we do not preserve the
`use_hitch_corrected_trailer` toggle from transfuser-truckscenes. Rationale:
the toggle existed so v3~v9 baselines stay reproducible against their
training history, but a single, well-defined ground truth is what we want
for paper-stage articulation metrics. If a baseline ablation needs the raw
annotation center, the toggle can be reintroduced later via a config switch
on the truck FeatureBuilder, not by re-emitting different scene_dicts.

Velocity is intentionally not stored. The trailer head paper contribution
will derive trailer velocity either from finite differences over the future
trajectory or from a bicycle model -- both are downstream of this data
layer.
"""
import math
from typing import Any, Dict

import numpy as np

# Hitch geometry constants. Match transfuser-truckscenes/configs/_base.py:
# trailer_hitch_x = 0.3 (rear-axle origin + 0.3 m forward),
# trailer_hitch_y = 0.0. These follow the MAN TGX viewer convention.
HITCH_X: float = 0.3
HITCH_Y: float = 0.0


def _hitch_corrected_trailer_center(
    trailer_length: float,
    trailer_yaw: float,
    tractor_x: float,
    tractor_y: float,
    tractor_yaw: float,
) -> tuple:
    """Return trailer center anchored to the tractor hitch.

    Ported from transfuser-truckscenes/dataset/dataset.py:50-70.
    Hitch geometry constants are baked in; if a different truck model is
    integrated later, lift them to a config.
    """
    cos_t, sin_t = math.cos(tractor_yaw), math.sin(tractor_yaw)
    hitch_global_x = tractor_x + HITCH_X * cos_t - HITCH_Y * sin_t
    hitch_global_y = tractor_y + HITCH_X * sin_t + HITCH_Y * cos_t

    center_x = hitch_global_x - (trailer_length / 2.0) * math.cos(trailer_yaw)
    center_y = hitch_global_y - (trailer_length / 2.0) * math.sin(trailer_yaw)
    return center_x, center_y


def _quaternion_to_yaw(rotation_wxyz) -> float:
    """Yaw extracted by rotating the unit-x vector (matches transfuser-truckscenes
    _quaternion_to_yaw, dataset.py:651-655)."""
    from pyquaternion import Quaternion

    v = Quaternion(rotation_wxyz).rotate(np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(v[1], v[0]))


def build_trailer_extras(sample: dict, ts) -> Dict[str, Any]:
    """Build the per-frame `truckscenes_extras` dict for one sample.

    If the sample carries a `vehicle.ego_trailer` annotation, populate the
    trailer fields with hitch-corrected pose. Otherwise return
    `has_trailer=False` with placeholder values (NaN) so downstream code
    can mask consistently.

    :param sample: TruckScenes `sample` record.
    :param ts: initialized TruckScenes devkit instance.
    """
    # Find the reference LiDAR sample_data so we can read ego_pose. Same
    # convention as transfuser-truckscenes/dataset/dataset.py
    # (_get_reference_channel: prefer LIDAR_TOP_FRONT).
    ref_channel = _get_reference_channel(sample)
    ref_sd = ts.get("sample_data", sample["data"][ref_channel])
    tractor_ep = ts.get("ego_pose", ref_sd["ego_pose_token"])
    tractor_pos = np.asarray(tractor_ep["translation"][:2], dtype=np.float64)
    tractor_yaw = _quaternion_to_yaw(tractor_ep["rotation"])

    nan = float("nan")
    no_trailer: Dict[str, Any] = {
        "has_trailer": False,
        "trailer_pose": np.array([nan, nan, nan], dtype=np.float64),
        "trailer_length": nan,
        "trailer_width": nan,
        "hitch_angle": nan,
    }

    # Find the ego_trailer box in this sample, if any.
    trailer_box = None
    for box in ts.get_boxes(ref_sd["token"]):
        if box.name == "vehicle.ego_trailer":
            trailer_box = box
            break
    if trailer_box is None:
        return no_trailer

    trailer_yaw = _quaternion_to_yaw(trailer_box.orientation.q)
    # box.wlh order is [width, length, height] in TruckScenes
    # (matches dataset.py:498 comment).
    trailer_width = float(trailer_box.wlh[0])
    trailer_length = float(trailer_box.wlh[1])

    corrected_x, corrected_y = _hitch_corrected_trailer_center(
        trailer_length=trailer_length,
        trailer_yaw=trailer_yaw,
        tractor_x=float(tractor_pos[0]),
        tractor_y=float(tractor_pos[1]),
        tractor_yaw=tractor_yaw,
    )

    # hitch_angle: signed difference, normalized to [-pi, pi].
    hitch_angle = (tractor_yaw - trailer_yaw + math.pi) % (2 * math.pi) - math.pi

    return {
        "has_trailer": True,
        "trailer_pose": np.array([corrected_x, corrected_y, trailer_yaw], dtype=np.float64),
        "trailer_length": trailer_length,
        "trailer_width": trailer_width,
        "hitch_angle": float(hitch_angle),
    }


def _get_reference_channel(sample: dict) -> str:
    """Reference sensor channel for ego_pose lookup. Prefer LIDAR_TOP_FRONT.

    Same logic as transfuser-truckscenes/dataset/dataset.py:640-648 but
    inlined here so this module stays self-contained (no cross-package
    dependency back into transfuser-truckscenes).
    """
    if "LIDAR_TOP_FRONT" in sample["data"]:
        return "LIDAR_TOP_FRONT"
    for key in sample["data"]:
        if "LIDAR" in key.upper():
            return key
    # Cameras carry ego_pose tokens too. Fall through to any sensor.
    return next(iter(sample["data"]))
