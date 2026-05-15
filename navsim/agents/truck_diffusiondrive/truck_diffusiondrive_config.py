"""Config for the TruckScenes-adapted DiffusionDrive agent.

Subclasses `navsim.agents.diffusiondrive.transfuser_config.TransfuserConfig`
(verbatim upstream from hustvl/DiffusionDrive) and overrides the fields
that matter for TruckScenes:

  - Camera: 4-cam stitched layout produces a wider image
    (`camera_width=1536`, anchors 48 horizontally) instead of the
    nuScenes 3-cam-stitched 1024.
  - LiDAR: v9 truck baseline range (-32 / +48 forward, 320 vertical
    pixels, 10 anchors) + ground plane enabled.
  - BEV semantic head: STAYS BUILT (use_bev_semantic=True) so the
    upstream V2TransfuserModel forward path can emit
    `predictions["bev_semantic_map"]` -- the unmodified
    `diffusiondrive.transfuser_loss` references this key
    unconditionally. We set `bev_semantic_weight=0.0` so its loss
    contribution is zero. The TargetBuilder emits a zeros placeholder
    of the right shape. Wasted compute on the head + CE call accepted
    in exchange for not forking the upstream model / loss.
  - `plan_anchor_path` -- TruckScenes-trained k-means anchor file
    (produced by `checks/compute_truck_kmeans_anchor.py`). Upstream's
    hardcoded path was a private workstation path; ours must be set
    explicitly in the Hydra yaml.
  - `bkb_path` -- left as empty default; upstream's image-encoder load
    first tries `timm.create_model(pretrained=True)` (HF cache) and
    only falls back to `bkb_path` on failure. With HF cache primed on
    B200, the fallback path is never hit.
"""
from dataclasses import dataclass

from navsim.agents.diffusiondrive.transfuser_config import (
    TransfuserConfig as DiffusionDriveTransfuserConfig,
)


@dataclass
class TruckDiffusionDriveConfig(DiffusionDriveTransfuserConfig):
    """Truck variant of DiffusionDrive's config."""

    # ===== Camera (4-cam truck layout) =====
    camera_width: int = 1536
    img_horz_anchors: int = 1536 // 32  # 48

    # ===== LiDAR (v9 baseline parity: forward-wider BEV + ground plane) =====
    lidar_min_x: float = -32
    lidar_max_x: float = 48
    lidar_resolution_height: int = 320
    lidar_vert_anchors: int = 10
    use_ground_plane: bool = True

    # ===== BEV semantic (no HD map -> weight 0, head still built) =====
    # See module docstring for why we keep `use_bev_semantic=True` here
    # (upstream loss references the key unconditionally).
    bev_semantic_weight: float = 0.0

    # ===== Anchor file =====
    # TruckScenes-specific k-means centers (shape (k, num_poses, 2)).
    # Override in Hydra yaml to point at the materialized anchor .npy.
    plan_anchor_path: str = ""

    # ===== Backbone weights =====
    # Empty -> upstream backbone load uses HF cache via
    # `timm.create_model(pretrained=True)` first (works on B200).
    bkb_path: str = ""

    # ===== Truck builder-compat fields =====
    # TruckTransfuser{Feature,Target}Builder accesses several flat
    # fields that don't exist on upstream DiffusionDrive's
    # TransfuserConfig. Mirror them here so the shared builder code
    # works against this config.
    # `num_poses` mirrors trajectory_sampling.num_poses (=8 default).
    num_poses: int = 8
    # `use_trailer_head` matches v9 truck-only baseline so the
    # DiffusionDrive comparison stays apples-to-apples with v9. Flip
    # to True only for an explicit trailer-head ablation.
    use_trailer_head: bool = False
