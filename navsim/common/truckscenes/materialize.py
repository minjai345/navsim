"""Sensor blob materialization for `build_navsim_logs --materialize-blobs`.

When the flag is on, `build_navsim_logs` post-processes each emitted
scene_dict so the consumer (`Lidar.from_paths`, `Cameras.from_camera_dict`)
can find sensor data on disk:

  - LiDAR: the 6 TruckScenes channels are merged into one (6, N) float32
    array (via `lidar_merge.merge_truckscenes_lidars`) and written as a
    binary PCD with fields matching navsim's `LidarIndex` (x, y, z,
    intensity, ring, lidar_id). nuplan-devkit's
    `LidarPointCloud.from_buffer(..., "pcd")` ingests this layout
    directly.
  - Camera: not materialized. The original TruckScenes image files are
    used as-is. We just rewrite the scene_dict cam `data_path` from the
    devkit-relative form to an absolute path so the consumer's
    `sensor_blobs_path / data_path` resolves regardless of the
    `sensor_blobs_path` it was launched with (an absolute right-hand
    operand makes `Path` ignore the left-hand path).

Output layout (under `<blob_dir>/`):

    <blob_dir>/lidars/<sample_token>.pcd

After this post-processing each frame dict has:

    "lidar_path": "<blob_dir>/lidars/<sample_token>.pcd"  (absolute)
    "cams":  { slot: { "data_path": "<truckscenes_root>/<orig>", ... }, ... }

Both are absolute, so the consumer can pass any `sensor_blobs_path`
(including the empty path) and still resolve the file.
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt


# PCD field layout matches navsim/common/enums.py:LidarIndex
# (x, y, z, intensity, ring, lidar_id). All float32.
_PCD_FIELDS = ("x", "y", "z", "intensity", "ring", "lidar_id")


def write_pcd_binary(path: Path, points: npt.NDArray[np.float32]) -> None:
    """Write (6, N) float32 array as a binary PCD with 6 float32 fields.

    Format matches nuplan-devkit's `LidarPointCloud.from_buffer(..., "pcd")`
    parser. ASCII header followed by raw little-endian float32 bytes,
    row-major (one point per row, 6 floats per row).
    """
    if points.ndim != 2 or points.shape[0] != 6:
        raise ValueError(f"write_pcd_binary expects (6, N), got {points.shape}")
    n_points = int(points.shape[1])
    # PCD on-disk layout: N rows of 6 float32 fields. numpy (6, N) -> (N, 6)
    # via transpose + ascontiguousarray so .tobytes() is row-major.
    rows = np.ascontiguousarray(points.astype(np.float32).T)

    header = (
        "# .PCD v0.7 - TruckScenes merged 6-lidar point cloud\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(_PCD_FIELDS)}\n"
        "SIZE 4 4 4 4 4 4\n"
        "TYPE F F F F F F\n"
        "COUNT 1 1 1 1 1 1\n"
        f"WIDTH {n_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n_points}\n"
        "DATA binary\n"
    ).encode("ascii")

    # Atomic write so a crashed run never leaves a half-finished pcd that
    # the consumer would happily try to parse.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fp:
        fp.write(header)
        fp.write(rows.tobytes())
    tmp_path.replace(path)


def materialize_scene_dict_blobs(
    scene_dict: Dict[str, Any],
    ts,
    truckscenes_root: Path,
    blob_dir: Path,
) -> Dict[str, Any]:
    """Augment one frame dict in-place with materialized sensor paths.

    Side effects:
      - Writes `<blob_dir>/lidars/<sample_token>.pcd` containing the
        merged 6-LiDAR point cloud.
      - Mutates the frame dict so `lidar_path` and every populated cam
        slot's `data_path` are absolute paths.

    The original TruckScenes image files are referenced directly; no
    file copies happen.
    """
    # Imports inside the function so this module stays cheap to import
    # outside the materialize code path.
    from navsim.common.truckscenes.lidar_merge import merge_truckscenes_lidars

    sample_token = scene_dict["token"]
    sample = ts.get("sample", sample_token)

    # ---- LiDAR: merge + write PCD ----
    lidars_dir = blob_dir / "lidars"
    lidars_dir.mkdir(parents=True, exist_ok=True)
    pcd_path = lidars_dir / f"{sample_token}.pcd"
    points = merge_truckscenes_lidars(sample, ts)
    if points.shape[1] == 0:
        # No usable LiDAR for this sample (extreme dropout). Leave
        # lidar_path None so the consumer's empty-Lidar branch fires.
        scene_dict["lidar_path"] = None
    else:
        write_pcd_binary(pcd_path, points)
        scene_dict["lidar_path"] = str(pcd_path.resolve())

    # ---- Cameras: rewrite relative paths to absolute ----
    cams = scene_dict.get("cams") or {}
    truckscenes_root_abs = truckscenes_root.resolve()
    for slot_name, cam_entry in list(cams.items()):
        if cam_entry is None:
            continue
        data_path = cam_entry.get("data_path")
        if data_path is None:
            continue
        # `data_path` is the devkit-relative path stored in sample_data;
        # join with the dataset root and absolutize so `Path(blob_path) /
        # data_path` returns this absolute path regardless of blob_path.
        cam_entry["data_path"] = str((truckscenes_root_abs / data_path).resolve())

    return scene_dict
