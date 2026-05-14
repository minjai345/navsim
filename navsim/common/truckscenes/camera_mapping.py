"""TruckScenes (4 cameras) -> navsim SensorConfig (8 slots) mapping.

navsim's SensorConfig hard-codes 8 camera slots
(cam_f0 / cam_l0,1,2 / cam_r0,1,2 / cam_b0), inherited from the OpenScene
nuPlan layout. TruckScenes only ships 4 cameras; the remaining 4 slots stay
empty (`Camera()` with no image) per navsim's convention for missing sensors.

Slot assignment (Option B from the design discussion -- raw per-camera, not
stitched -- so downstream ablations can drop or recombine views without
re-running the adapter):

    cam_l0 <- CAMERA_LEFT_FRONT
    cam_r0 <- CAMERA_RIGHT_FRONT
    cam_l2 <- CAMERA_LEFT_BACK
    cam_r2 <- CAMERA_RIGHT_BACK

Front-pair stitching (the form fed into TransFuser image encoder today) is
performed inside the truck-specific FeatureBuilder, not here. Keeping raw
4-cam in the scene_dict preserves degrees of freedom for later experiments
(e.g. rear-cam-only or asymmetric crops).
"""
from typing import Dict

# Truckscenes channel name -> navsim cam slot name.
# Channels not in this dict have no navsim equivalent and are dropped.
TRUCKSCENES_TO_NAVSIM_CAM: Dict[str, str] = {
    "CAMERA_LEFT_FRONT": "cam_l0",
    "CAMERA_RIGHT_FRONT": "cam_r0",
    "CAMERA_LEFT_BACK": "cam_l2",
    "CAMERA_RIGHT_BACK": "cam_r2",
}

# Slots that are always empty when loading TruckScenes (no source camera).
# Listed for documentation; consumers can build empty `Camera()` for these.
NAVSIM_UNUSED_CAM_SLOTS = ("cam_f0", "cam_l1", "cam_r1", "cam_b0")


def build_cam_dict_for_navsim(sample: dict, ts) -> dict:
    """Build navsim's `cams` sub-dict for one sample.

    Returns a dict keyed by lower-case navsim slot name (cam_l0, cam_r0, ...)
    holding the per-camera fields navsim's `Cameras.from_camera_dict` expects:
      data_path, sensor2lidar_rotation, sensor2lidar_translation,
      cam_intrinsic, distortion.

    Unused slots map to `None` so the downstream loader instantiates an empty
    `Camera()`.

    TODO:
      - Pull `calibrated_sensor` per camera channel for intrinsics + extrinsics.
      - Express sensor2lidar in the reference-LiDAR frame TruckScenes uses
        (matches existing transfuser-truckscenes lidar merging convention).
      - Confirm distortion model parity with navsim's expectations.
    """
    raise NotImplementedError("camera_mapping.build_cam_dict_for_navsim: skeleton")
