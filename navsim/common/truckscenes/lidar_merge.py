"""Merge TruckScenes 6 LiDARs into navsim's (6, N) `lidar_pc` layout.

TruckScenes provides 6 LiDAR channels (TOP_FRONT/TOP_LEFT/TOP_RIGHT/LEFT/
RIGHT/REAR) as individual PCD files; each load gives (4, N) with
[x, y, z, intensity]. navsim's `Lidar.lidar_pc` expects (6, N) with
[x, y, z, intensity, ring, lidar_id] (see navsim/common/enums.py:LidarIndex).

Strategy (decided in design discussion):
  - Transform each LiDAR's points into the ego frame using its
    `calibrated_sensor` extrinsics, then concatenate.
  - Synthesize the `lidar_id` channel by tagging points 0..5 by source
    channel. Order is fixed by `TRUCKSCENES_LIDAR_CHANNELS` so the same id
    always means the same physical sensor across the dataset.
  - Synthesize `ring` as 0 (TruckScenes PCDs do not carry per-beam ring
    indices; the current transfuser-truckscenes pipeline does not consume
    this channel either).

This matches the merge logic already proven in
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

    TODO:
      - Port extrinsic transform from transfuser-truckscenes/dataset/dataset.py
        _get_lidar_feature (sensor -> ego via calibrated_sensor).
      - Decide whether to drop intensity (navsim downstream may not consume
        it; current transfuser path drops to (3, N) before histogram).
        Keeping it here preserves the (6, N) contract; truck FeatureBuilder
        can slice as needed.
    """
    raise NotImplementedError("lidar_merge.merge_truckscenes_lidars: skeleton")
