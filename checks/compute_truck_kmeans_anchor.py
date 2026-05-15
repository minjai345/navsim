"""K-means anchor computation for TruckDiffusionDrive.

DiffusionDrive's diffusion process samples trajectories from a set of
fixed anchors (k-means cluster centers of training trajectories). The
upstream `plan_anchor_path` is a hardcoded .npy computed from
nuScenes/navsim trajectories (shape (20, 8, 2) = 20 clusters × 8 future
poses × xy). For TruckScenes those statistics are wrong: trucks have
different speed envelopes, turning radii, and rare-event distributions.

This script computes a TruckScenes-specific anchor file:

  - Walks every scene in v1.2-trainval (or a specified split)
  - For every sample that has `num_future_poses` next-keyframes, extracts
    the future ego trajectory in the current ego's local frame (just
    like our scene_dict_from_sample / Scene.get_future_trajectory).
  - Runs sklearn KMeans with k clusters.
  - Saves cluster centers as (k, num_future_poses, 2) float32 .npy.

Output: <output>/truck_kmeans_anchor_<k>_<num_poses>.npy

Usage (on B200 navsim env):

    PYTHONPATH=. python checks/compute_truck_kmeans_anchor.py \\
        --truckscenes-root /home/.../man-truckscenes \\
        --version v1.2-trainval \\
        --split train \\
        --num-clusters 20 \\
        --num-future-poses 8 \\
        --output data/truck_kmeans_anchor_20_8.npy
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _quaternion_to_yaw(rotation_wxyz):
    """Yaw via unit-x rotation. Same formula as scene_dict_from_sample."""
    from pyquaternion import Quaternion

    v = Quaternion(rotation_wxyz).rotate(np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(v[1], v[0]))


def _future_xy_local(ts, sample_token: str, num_future_poses: int):
    """Return (num_future_poses, 2) future xy in current ego local frame,
    or None if the chain is too short.
    """
    sample = ts.get("sample", sample_token)
    # Reference channel for ego_pose lookup -- prefer LIDAR_TOP_FRONT.
    ref = "LIDAR_TOP_FRONT" if "LIDAR_TOP_FRONT" in sample["data"] else next(iter(sample["data"]))
    sd_cur = ts.get("sample_data", sample["data"][ref])
    ep_cur = ts.get("ego_pose", sd_cur["ego_pose_token"])
    cur_pos = np.asarray(ep_cur["translation"][:2], dtype=np.float64)
    cur_yaw = _quaternion_to_yaw(ep_cur["rotation"])
    cos_y, sin_y = math.cos(-cur_yaw), math.sin(-cur_yaw)

    out = np.zeros((num_future_poses, 2), dtype=np.float32)
    next_token = sample.get("next", "")
    for k in range(num_future_poses):
        if not next_token:
            return None
        ns = ts.get("sample", next_token)
        nsd = ts.get("sample_data", ns["data"][ref])
        nep = ts.get("ego_pose", nsd["ego_pose_token"])
        fut_pos = np.asarray(nep["translation"][:2], dtype=np.float64)
        dx, dy = fut_pos[0] - cur_pos[0], fut_pos[1] - cur_pos[1]
        out[k, 0] = dx * cos_y - dy * sin_y
        out[k, 1] = dx * sin_y + dy * cos_y
        next_token = ns.get("next", "")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truckscenes-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.2-trainval")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"])
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument("--num-future-poses", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap on sample count (sanity testing).")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    from sklearn.cluster import KMeans
    from truckscenes import TruckScenes
    from truckscenes.utils.splits import create_splits_scenes

    ts = TruckScenes(version=args.version, dataroot=str(args.truckscenes_root), verbose=False)
    splits = create_splits_scenes()

    # Resolve scene tokens for the chosen split (TruckScenes naming maps
    # scene["name"] -> split entry directly).
    name_to_token = {s["name"]: s["token"] for s in ts.scene}
    if args.split == "all":
        scene_tokens = [s["token"] for s in ts.scene]
    else:
        scene_tokens = [name_to_token[n] for n in splits[args.split] if n in name_to_token]
    print(f"resolved {len(scene_tokens)} scene(s) for split={args.split}")

    # Walk every sample in those scenes, collect future xy.
    trajectories = []
    for i, scene_token in enumerate(scene_tokens, 1):
        scene = ts.get("scene", scene_token)
        token = scene["first_sample_token"]
        while token:
            traj = _future_xy_local(ts, token, args.num_future_poses)
            if traj is not None:
                trajectories.append(traj)
                if args.max_samples is not None and len(trajectories) >= args.max_samples:
                    break
            sample = ts.get("sample", token)
            token = sample.get("next", "") or ""
        if args.max_samples is not None and len(trajectories) >= args.max_samples:
            break
        if i % 50 == 0:
            print(f"  [{i}/{len(scene_tokens)}] scenes processed, {len(trajectories)} trajectories so far")

    arr = np.stack(trajectories, axis=0)
    print(f"collected {arr.shape[0]} trajectories (shape {arr.shape})")

    # Flatten (N, num_poses*2) for KMeans, reshape centers back.
    flat = arr.reshape(arr.shape[0], -1)
    print(f"running KMeans(n_clusters={args.num_clusters})")
    km = KMeans(n_clusters=args.num_clusters, random_state=args.random_state, n_init=10)
    km.fit(flat)
    centers = km.cluster_centers_.reshape(args.num_clusters, args.num_future_poses, 2).astype(np.float32)
    print(f"anchor shape: {centers.shape}, inertia: {km.inertia_:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, centers)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
