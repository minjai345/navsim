"""Merge TruckScenes 6 LiDARs into navsim's (6, N) `lidar_pc` layout.

TruckScenes provides 6 LiDAR channels (TOP_FRONT/TOP_LEFT/TOP_RIGHT/LEFT/
RIGHT/REAR) as individual PCD files; each load gives (4, N) with
[x, y, z, intensity]. navsim's `Lidar.lidar_pc` expects (6, N) with
[x, y, z, intensity, ring, lidar_id] (see navsim/common/enums.py:LidarIndex).

Strategy (decided in design discussion):
  - Transform each LiDAR's points into the EGO BODY frame using its
    `calibrated_sensor` extrinsics (which TruckScenes already expresses
    relative to ego, per docs/schema_truckscenes.md).
  - Synthesize the `lidar_id` channel by tagging points 0..5 by source
    channel. Order is fixed by `TRUCKSCENES_LIDAR_CHANNELS` so the same id
    always means the same physical sensor across the dataset.
  - Synthesize `ring` as 0 (TruckScenes PCDs do not carry per-beam ring
    indices; the current transfuser-truckscenes pipeline does not consume
    this channel either).

Matches the merge logic already proven in
transfuser-truckscenes/dataset/dataset.py:_get_lidar_feature (the (3, N)
slice). We add the intensity column plus the two synthetic channels here.
"""
from typing import List

import numpy as np
import numpy.typing as npt

# Fixed ordering -- `lidar_id` channel indices into this list.
# Same as transfuser-truckscenes LIDAR_CHANNELS to keep parity.
TRUCKSCENES_LIDAR_CHANNELS: List[str] = [
    "LIDAR_TOP_FRONT",
    "LIDAR_TOP_LEFT",
    "LIDAR_TOP_RIGHT",
    "LIDAR_LEFT",
    "LIDAR_RIGHT",
    "LIDAR_REAR",
]


def merge_truckscenes_lidars(sample: dict, ts) -> npt.NDArray[np.float32]:
    """Load all 6 TruckScenes LiDARs, transform to ego frame, return (6, N).

    Layout matches navsim's LidarIndex:
      row 0..2  -> x, y, z (ego frame)
      row 3     -> intensity (raw from PCD)
      row 4     -> ring (stubbed to 0)
      row 5     -> lidar_id (0..5 by channel ordering)

    Missing channels are skipped silently (so a frame with e.g. REAR dropout
    still produces a usable point cloud).

    :param sample: TruckScenes `sample` record.
    :param ts: initialized TruckScenes devkit instance.
    :return: (6, N) float32 array, possibly with N=0 if no LiDARs available.
    """
    from pathlib import Path
    from pyquaternion import Quaternion
    from truckscenes.utils.data_classes import LidarPointCloud

    columns = []
    for lidar_id, channel in enumerate(TRUCKSCENES_LIDAR_CHANNELS):
        if channel not in sample["data"]:
            continue
        sd = ts.get("sample_data", sample["data"][channel])
        cs = ts.get("calibrated_sensor", sd["calibrated_sensor_token"])

        # Load (4, N): [x, y, z, intensity]. Devkit auto-detects PCD format.
        pc = LidarPointCloud.from_file(str(Path(ts.dataroot) / sd["filename"]))
        n = pc.points.shape[1]
        if n == 0:
            continue

        # Sensor frame -> ego body frame.
        # calibrated_sensor stores R, t in EGO frame
        # (truckscenes-devkit docs/schema_truckscenes.md):
        #     p_ego = R @ p_sensor + t
        R = Quaternion(cs["rotation"]).rotation_matrix          # (3, 3)
        t = np.asarray(cs["translation"], dtype=np.float64)     # (3,)
        xyz_ego = (R @ pc.points[:3]).astype(np.float32)        # (3, N)
        xyz_ego += t.astype(np.float32).reshape(3, 1)
        intensity = pc.points[3:4].astype(np.float32)           # (1, N)
        # Synthetic channels.
        ring = np.zeros((1, n), dtype=np.float32)
        lidar_id_row = np.full((1, n), lidar_id, dtype=np.float32)

        # Stack into (6, N) for this lidar; concat across lidars after the loop.
        columns.append(np.vstack([xyz_ego, intensity, ring, lidar_id_row]))

    if not columns:
        return np.zeros((6, 0), dtype=np.float32)
    return np.concatenate(columns, axis=1)
