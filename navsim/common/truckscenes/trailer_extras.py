"""TruckScenes-specific side-channel data attached to each `Frame`.

navsim's standard `Annotations` slot is for "other agents" only. The truck
trailer is rigidly attached to the ego tractor (via hitch) and is treated
as an extension of ego, not as a foreign agent (see design discussion --
Option C). To keep this distinction explicit in the data flow, trailer
state and any other truck-specific per-frame fields live in a separate
`Dict[str, Any]` attached to the frame.

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
        "trailer_pose": (x, y, yaw),                # GLOBAL frame, float
        "trailer_length": float,                     # meters
        "trailer_width": float,                      # meters
        "hitch_angle": float,                        # tractor_yaw - trailer_yaw, radians
        "trailer_velocity": (vx, vy),                # ego frame, m/s
        "has_trailer": bool,                         # False -> all other keys may be NaN
    }

Frames without a trailer (some TruckScenes scenes are tractor-only) set
`has_trailer=False`; downstream loss masking is the consumer's job.
"""
from typing import Any, Dict


def build_trailer_extras(sample: dict, ts) -> Dict[str, Any]:
    """Build the per-frame `truckscenes_extras` dict for one sample.

    Pulls the `vehicle.ego_trailer` annotation if present and, depending on
    the chosen GT mode, either uses its raw center or re-anchors to the
    tractor hitch (current transfuser-truckscenes `_hitch_corrected_trailer_center`).

    TODO:
      - Port `_hitch_corrected_trailer_center` logic from
        transfuser-truckscenes/dataset/dataset.py:50-70.
      - Decide: do we always emit hitch-corrected (cleaner) or mirror the
        current config toggle (`use_hitch_corrected_trailer`)? Recommend
        always-corrected once paper protocol is fixed; toggle complicates
        the side-channel schema.
      - Trailer velocity: centered diff over trailer_pose history vs
        bicycle-model propagation from tractor pose + hitch. Decide once
        kinematic head spec is finalized (see project_baseline_vs_kinematic_head).
      - Handle frames with no trailer annotation (set has_trailer=False,
        rest defaults).
    """
    raise NotImplementedError("trailer_extras.build_trailer_extras: skeleton")
