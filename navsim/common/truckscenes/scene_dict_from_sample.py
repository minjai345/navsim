"""Single source of truth for `TruckScenes Sample -> navsim scene_dict`.

This module owns the actual conversion. Two consumers wrap it:
  - `runtime_loader.TruckScenesSceneLoader` (Path B, runtime)
  - `build_navsim_logs.main` (Path A, batch pickle dump)

The returned dict matches the schema navsim's `AgentInput.from_scene_dict_list`
and `Scene` factories already consume (see navsim/common/dataclasses.py:148+
for the canonical keys), with one truck-specific addition:

    {
        "token":                    str,        # navsim sample token = truckscenes sample_token
        "timestamp":                int,        # microseconds (TruckScenes native)
        "ego2global_translation":   (3,) float64,
        "ego2global_rotation":      (4,) float64,  # quaternion (w, x, y, z)
        "ego_dynamic_state":        (4,) float32,  # [vx, vy, ax, ay]
        "driving_command":          (3,) int / one-hot,
        "cams":                     dict[slot_name -> dict | None],   # 8 keys
        "lidar_path":               str | None,                       # path to merged PCD, or None
        "anns":                     dict (boxes, names, velocity_3d, instance_tokens, track_tokens),
        "roadblock_ids":            [],         # always empty -- no HD map
        "traffic_lights":           [],         # always empty -- no annotation
        "truckscenes_extras":       dict,       # trailer pose, hitch angle, ... (see trailer_extras.py)
    }

Why a dict and not a `Frame`/`Scene` directly?
  - Path A wants to pickle this and feed it back through navsim's stock
    `SceneLoader` / `filter_scenes`, which expects dicts.
  - Path B's loader is free to build `AgentInput` / `Scene` from the same
    dict using navsim's existing classmethods (`AgentInput.from_scene_dict_list`
    etc.), so we get the same downstream surface in both paths.
"""
from typing import Any, Dict


def scene_dict_from_sample(ts, sample_token: str) -> Dict[str, Any]:
    """Convert a single TruckScenes sample into a navsim-compatible dict.

    :param ts: initialized `TruckScenes` devkit instance.
    :param sample_token: target sample token (navsim's per-frame identity).
    :return: dict with the schema documented at module top.

    TODO -- ordered by dependency:
      1. Pull `sample` record, resolve ego_pose via reference LiDAR channel
         (matches transfuser-truckscenes:_get_reference_channel convention).
      2. Build ego_dynamic_state:
           [vx, vy]  <- _estimate_ego_velocity (dataset.py:306-358).
           [ax, ay]  <- ego_motion_chassis IMU (dataset.py:299-300).
      3. Build cams dict via `camera_mapping.build_cam_dict_for_navsim`.
         Fill the 4 unused slots with None.
      4. Build lidar_path: write the merged (6, N) tensor to a per-sample
         file under <output_root>/sensor_blobs/<sample_token>.npy?
         Decision deferred -- maybe in Path A we materialize a file, in
         Path B we hand back the in-memory array directly via a custom
         loader. Need to check if AgentInput.from_scene_dict_list accepts
         either.
      5. Build annotations dict (skip ego_trailer; map categories).
      6. Build truckscenes_extras via `trailer_extras.build_trailer_extras`.
      7. Compute driving_command from this sample's *future* trajectory
         using the heading-threshold logic
         (transfuser-truckscenes/dataset/dataset.py:_get_driving_command).
         CAUTION: this requires future samples to exist (caller must filter
         out samples too close to scene end -- same as
         transfuser-truckscenes/dataset/dataset.py:_collect_valid_samples).
    """
    raise NotImplementedError("scene_dict_from_sample: skeleton")
