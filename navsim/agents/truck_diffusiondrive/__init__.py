"""DiffusionDrive agent specialized for MAN TruckScenes.

Mirrors `navsim/agents/diffusiondrive/` (upstream from hustvl/DiffusionDrive)
but adapts the camera input layout (3-cam stitch -> truck 4-cam),
disables BEV semantic loss (TruckScenes has no HD map), and points to a
TruckScenes-trained k-means anchor file (the upstream `plan_anchor_path`
is computed from nuScenes/navsim trajectories which do not match truck
trajectory statistics).

Most of DiffusionDrive's code (model, backbone, loss, diffusion modules)
is reused as-is from the upstream agent package -- the truck variant is
a thin wrapper that overrides:

  - `TruckDiffusionDriveFeatureBuilder` (4-cam stitch via
    TruckTransfuserFeatureBuilder.image-stitching logic)
  - `TruckDiffusionDriveConfig` (truck camera_width / lidar range / no
    bev_semantic / truck plan_anchor_path)
  - `TruckDiffusionDriveAgent` (truck SensorConfig + 4-cam slots only)

This skeleton is filled in across multiple sessions. See task #27 and
[[project_paper_deadline]] for the integration plan.
"""
