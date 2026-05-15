"""Sensor blob materialization for `build_navsim_logs --materialize-blobs`.

When the flag is on, `build_navsim_logs` post-processes each emitted
scene_dict so the consumer (`Lidar.from_paths`, `Cameras.from_camera_dict`)
can find sensor data on disk:

  - LiDAR: the 6 TruckScenes channels are merged into one (6, N) float32
    array (via `lidar_merge.merge_truckscenes_lidars`) and written as a
    **binary** PCD with fields matching navsim's `LidarIndex` (x, y, z,
    intensity, ring, lidar_id). Critical: nuplan-devkit exposes TWO
    different PCD readers with OPPOSING format requirements:
      * `LidarPointCloud.from_file(path)` -> ASCII only (text iteration)
      * `LidarPointCloud.from_buffer(bytes, "pcd")` -> BINARY only
        (`PointCloud.parse` raises on `DATA ascii`).
    navsim's runtime `Lidar.from_paths` calls `from_buffer`, so we MUST
    emit binary. The standalone `from_file` API is informational only.
  - Camera: not materialized. The original TruckScenes image files are
    used as-is. We just rewrite the scene_dict cam `data_path` from the
    devkit-relative form to an absolute path so the consumer's
    `sensor_blobs_path / data_path` resolves regardless of the
    `sensor_blobs_path` it was launched with (an absolute right-hand
    operand makes `Path` ignore the left-hand path).
  - Optional viz: when --viz is on, a BEV scatter + agent-box overlay
    PNG is saved next to each PCD for quick sanity.

Output layout (under `<blob_dir>/`):

    <blob_dir>/lidars/<sample_token>.pcd
    <blob_dir>/viz/<sample_token>.png        (only when viz=True)

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


# PCD field layout for nuplan-devkit's PointCloud.to_pcd_bin2 (which
# navsim Lidar.from_paths uses). Required field order: x, y, z,
# intensity, ring, lidar_info. The 6th field is `lidar_info` (not
# `lidar_id`!) per nuplan/database/utils/pointclouds/pointcloud.py;
# navsim/common/enums.py:LidarIndex calls the same column `_LIDAR_ID`,
# but the on-disk name is `lidar_info`.
_PCD_FIELDS = ("x", "y", "z", "intensity", "ring", "lidar_info")


def write_pcd_binary(path: Path, points: npt.NDArray[np.float32]) -> None:
    """Write (6, N) float32 array as a binary PCD with 6 float32 fields.

    Format consumable by nuplan-devkit's
    `PointCloud.parse` (called from `LidarPointCloud.from_buffer(...,
    "pcd")`). ASCII header followed by raw little-endian float32 bytes,
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


def write_viz_png(
    path: Path,
    points_6N: npt.NDArray[np.float32],
    anns: Dict[str, Any],
    truckscenes_extras: Dict[str, Any],
) -> None:
    """Render a BEV scatter + agent-box overlay sanity image.

    Plots LiDAR points (xy, ego frame) as low-alpha scatter, overlays
    annotated boxes as polygons, marks the ego at the origin, and adds a
    trailer box if `truckscenes_extras` carries one. Output: PNG.

    This is sanity-grade -- not a publication figure. Visual evidence
    that LiDAR + annotations + trailer extras all landed in coherent
    coordinates after the adapter pipeline.
    """
    # Lazy matplotlib import -- materialize.py is also used in non-viz
    # paths and matplotlib startup is a few hundred ms.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(8, 8))

    # Subsample to avoid plotting hundreds of thousands of points.
    if points_6N.shape[1] > 0:
        step = max(1, points_6N.shape[1] // 50000)
        xy = points_6N[:2, ::step].T
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="gray", alpha=0.4, linewidths=0)

    # Box overlays (ego frame, [X, Y, Z, LENGTH, WIDTH, HEIGHT, HEADING]).
    boxes = anns.get("boxes", np.zeros((0, 7), dtype=np.float32))
    names = anns.get("names", [])
    polygons: List[Polygon] = []
    for i, box in enumerate(boxes):
        x, y, _z, length, width, _h, heading = (float(v) for v in box)
        # Local box corners (length along x, width along y).
        hl, hw = length / 2.0, width / 2.0
        corners = np.array(
            [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32
        )
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        rotated = corners @ rot.T
        rotated += np.array([x, y])
        polygons.append(Polygon(rotated, closed=True))
    if polygons:
        pc = PatchCollection(polygons, facecolor="none", edgecolor="tab:blue", linewidth=1.2)
        ax.add_collection(pc)

    # Ego marker.
    ax.plot(0, 0, marker="^", color="tab:red", markersize=10, label="ego (tractor)")

    # Trailer (if present) -- drawn in tractor-ego frame.
    extras = truckscenes_extras or {}
    if extras.get("has_trailer", False):
        # trailer_pose is GLOBAL frame; we render it relative to the same
        # tractor ego pose used by the annotations. For viz simplicity we
        # don't re-project -- if anns are in this frame, trailer should be
        # too once we convert. For now just print the hitch_angle in the
        # title (visual placement deferred to a dedicated viz pass).
        ax.set_title(
            f"sample={path.stem}  hitch_angle={extras['hitch_angle']:.2f} rad",
            fontsize=10,
        )
    else:
        ax.set_title(f"sample={path.stem}  (no trailer)", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlim(-40, 60)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("x (m, ego forward)")
    ax.set_ylabel("y (m, ego left)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def materialize_scene_dict_blobs(
    scene_dict: Dict[str, Any],
    ts,
    truckscenes_root: Path,
    blob_dir: Path,
    viz: bool = False,
) -> Dict[str, Any]:
    """Augment one frame dict in-place with materialized sensor paths.

    Side effects:
      - Writes `<blob_dir>/lidars/<sample_token>.pcd` containing the
        merged 6-LiDAR point cloud (ASCII PCD).
      - Optionally writes `<blob_dir>/viz/<sample_token>.png` when
        `viz=True` (BEV scatter + annotation boxes overlay).
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

    # ---- LiDAR: merge + write binary PCD ----
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

    # ---- Optional viz: BEV scatter + agent boxes ----
    if viz:
        viz_path = blob_dir / "viz" / f"{sample_token}.png"
        write_viz_png(
            viz_path,
            points_6N=points,
            anns=scene_dict.get("anns", {}),
            truckscenes_extras=scene_dict.get("truckscenes_extras", {}),
        )

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
