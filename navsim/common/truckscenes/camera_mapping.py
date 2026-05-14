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

Frame convention note: TruckScenes `calibrated_sensor` extrinsics are
expressed in the EGO BODY frame (per docs/schema_truckscenes.md). navsim's
field is named `sensor2lidar_rotation/translation` (inherited from
nuScenes' top-LiDAR-anchored convention). We populate those navsim fields
with the EGO-FRAME extrinsics directly -- the truck-specific FeatureBuilder
knows this convention. Downstream code that genuinely needs sensor-to-
top-LiDAR transforms must compose with the reference LiDAR's extrinsics.
"""
from typing import Any, Dict, Optional

import numpy as np

# Truckscenes channel name -> navsim cam slot name.
# Channels not in this dict have no navsim equivalent and are dropped.
TRUCKSCENES_TO_NAVSIM_CAM: Dict[str, str] = {
    "CAMERA_LEFT_FRONT": "cam_l0",
    "CAMERA_RIGHT_FRONT": "cam_r0",
    "CAMERA_LEFT_BACK": "cam_l2",
    "CAMERA_RIGHT_BACK": "cam_r2",
}

# Slots that are always empty when loading TruckScenes (no source camera).
NAVSIM_UNUSED_CAM_SLOTS = ("cam_f0", "cam_l1", "cam_r1", "cam_b0")

# All 8 navsim slot names (used to enforce uniform dict shape).
ALL_NAVSIM_CAM_SLOTS = tuple(TRUCKSCENES_TO_NAVSIM_CAM.values()) + NAVSIM_UNUSED_CAM_SLOTS


def build_cam_dict_for_navsim(sample: dict, ts) -> Dict[str, Optional[Dict[str, Any]]]:
    """Build navsim's `cams` sub-dict for one sample.

    Returns a dict keyed by lower-case navsim slot name (cam_l0, cam_r0, ...).
    Each value matches the per-camera schema navsim's
    `Cameras.from_camera_dict` expects:

        {
            "data_path":               str (relative to dataroot),
            "sensor2lidar_rotation":   (3, 3) float -- ego-frame extrinsic R,
            "sensor2lidar_translation":(3,)   float -- ego-frame extrinsic t,
            "cam_intrinsic":           (3, 3) float,
            "distortion":              (k,)   float (zeros; TruckScenes images
                                                    are pre-undistorted),
        }

    Unused slots map to `None` so the downstream loader instantiates an
    empty `Camera()`.

    :param sample: TruckScenes `sample` record.
    :param ts: initialized TruckScenes devkit instance.
    """
    from pyquaternion import Quaternion

    cams: Dict[str, Optional[Dict[str, Any]]] = {slot: None for slot in ALL_NAVSIM_CAM_SLOTS}

    for ts_channel, navsim_slot in TRUCKSCENES_TO_NAVSIM_CAM.items():
        if ts_channel not in sample["data"]:
            # Channel missing from this sample (sensor dropout). Leave slot None.
            continue
        sd = ts.get("sample_data", sample["data"][ts_channel])
        cs = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])

        # camera_intrinsic is stored as nested list in JSON; cast to (3, 3) array.
        intrinsic = np.asarray(cs["camera_intrinsic"], dtype=np.float32)
        # TruckScenes images come undistorted and rectified -> zero distortion.
        # Use a length-5 array (k1, k2, p1, p2, k3) to match the nuScenes shape
        # navsim downstream may assume.
        distortion = np.zeros(5, dtype=np.float32)

        cams[navsim_slot] = {
            "data_path": sd["filename"],
            "sensor2lidar_rotation": Quaternion(cs["rotation"]).rotation_matrix.astype(np.float32),
            "sensor2lidar_translation": np.asarray(cs["translation"], dtype=np.float32),
            "cam_intrinsic": intrinsic,
            "distortion": distortion,
        }

    return cams
