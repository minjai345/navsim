"""Path A batch script: materialize TruckScenes as navsim per-log pkls.

navsim's `filter_scenes` (navsim/common/dataloader.py:16) expects each pkl
file at `<data_path>/<log_name>.pkl` to contain a list of frame dicts in
chronological order. A "scene" (in navsim terminology) is then a sliding
window over that list, governed by `SceneFilter.num_frames` and
`SceneFilter.frame_interval`.

Naming clash (intentional, documented for the reader):
  - TruckScenes "scene" ~~ navsim "log"   (a contiguous recording).
  - navsim "scene"      ~~ a windowed view inside one log.

We therefore emit ONE pkl per TruckScenes scene token, named
`<scene_token>.pkl`. Each pkl holds the full chronological list of frame
dicts produced by `scene_dict_from_sample` for every sample in that scene.

Downstream consumer expectations:
  - `SceneFilter.has_route=True` cannot be used: TruckScenes has no HD
    map, so every emitted frame dict has `roadblock_ids=[]`. Callers must
    set `has_route=False` (or filter scenes some other way) or every
    sample is dropped.
  - PDMS metric caching does NOT need sensor blobs (verified against
    metric_cache_processor.py -- iterates ego state + annotations only).
    Run WITHOUT --materialize-blobs for that use case; the emitted pkls
    have `lidar_path=None` and cam `data_path` values relative to the
    TruckScenes root (consumer must pass `sensor_blobs_path=$TRUCKSCENES`).
  - Lightning-trainer training needs sensor blobs. Run WITH
    --materialize-blobs: per-sample merged LiDAR is written as a binary
    PCD under `<blob_dir>/lidars/<sample>.pcd` (matches nuplan-devkit's
    `LidarPointCloud.from_buffer("pcd")`), and cam paths are rewritten
    to absolute. Both become absolute, so the consumer can pass any
    `sensor_blobs_path` (Path operator drops it when the right operand
    is absolute).

Usage (PDMS / metadata only):
    python -m navsim.common.truckscenes.build_navsim_logs \\
        --truckscenes-root $TRUCKSCENES \\
        --version v1.2-trainval \\
        --output-dir /path/to/navsim_truck_logs/ \\
        [--scene-tokens TOKEN1 TOKEN2 ...] \\
        [--max-scenes N] [--num-workers N]

Usage (training-ready, with sensor blobs):
    python -m navsim.common.truckscenes.build_navsim_logs \\
        --truckscenes-root $TRUCKSCENES \\
        --version v1.2-trainval \\
        --output-dir /path/to/navsim_truck_logs/ \\
        --materialize-blobs \\
        --blob-dir /path/to/sensor_blobs/ \\
        [--max-scenes N] [--num-workers N]
"""
import argparse
import multiprocessing as mp
import pickle
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple


def _iter_scene_sample_tokens(ts, scene_token: str) -> List[str]:
    """Walk `first_sample_token` -> `next` chain for one scene, in order."""
    scene = ts.get("scene", scene_token)
    tokens: List[str] = []
    cur = scene["first_sample_token"]
    while cur:
        tokens.append(cur)
        sample = ts.get("sample", cur)
        cur = sample.get("next", "") or ""
    return tokens


def _build_one_log(
    scene_token: str,
    truckscenes_root: Path,
    version: str,
    output_dir: Path,
    materialize_blobs: bool,
    blob_dir: Optional[Path],
    viz: bool = False,
) -> Tuple[str, Optional[str]]:
    """Worker entry: build and pickle one log. Returns (scene_token, error).

    A separate `TruckScenes` instance is constructed per worker so each
    process has its own device IO context (the devkit is not designed for
    cross-process sharing of a single instance).

    When `materialize_blobs=True`, per-sample sensor blobs are written
    under `blob_dir` and the scene_dict paths are rewritten to absolute
    via `materialize.materialize_scene_dict_blobs`.
    """
    try:
        # Imports inside the worker so the parent process need not load
        # truckscenes-devkit unless the user actually runs the script.
        from truckscenes import TruckScenes

        from navsim.common.truckscenes.scene_dict_from_sample import (
            scene_dict_from_sample,
        )
        if materialize_blobs:
            from navsim.common.truckscenes.materialize import (
                materialize_scene_dict_blobs,
            )

        ts = TruckScenes(version=version, dataroot=str(truckscenes_root), verbose=False)

        sample_tokens = _iter_scene_sample_tokens(ts, scene_token)
        scene_dict_list = []
        for sample_token in sample_tokens:
            sd = scene_dict_from_sample(ts, sample_token)
            if materialize_blobs:
                materialize_scene_dict_blobs(
                    sd,
                    ts=ts,
                    truckscenes_root=truckscenes_root,
                    blob_dir=blob_dir,
                    viz=viz,
                )
            scene_dict_list.append(sd)

        output_path = output_dir / f"{scene_token}.pkl"
        # Atomic write: pickle to a tmp file then rename so a partial write
        # never leaves a half-finished pkl that downstream loading silently
        # accepts.
        tmp_path = output_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as fp:
            pickle.dump(scene_dict_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(output_path)
        return scene_token, None
    except Exception:
        return scene_token, traceback.format_exc()


def _resolve_scene_tokens(
    truckscenes_root: Path,
    version: str,
    explicit: Optional[List[str]],
    max_scenes: Optional[int],
) -> List[str]:
    """Determine which scene tokens to process.

    If `explicit` is given, use it verbatim (after dedup). Otherwise
    iterate every scene in the chosen version. `max_scenes` truncates.
    """
    from truckscenes import TruckScenes

    ts = TruckScenes(version=version, dataroot=str(truckscenes_root), verbose=False)
    if explicit:
        # Preserve user-given order; dedup defensively.
        seen = set()
        tokens = []
        for t in explicit:
            if t in seen:
                continue
            seen.add(t)
            tokens.append(t)
    else:
        tokens = [scene["token"] for scene in ts.scene]
    if max_scenes is not None:
        tokens = tokens[:max_scenes]
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truckscenes-root",
        type=Path,
        required=True,
        help="Path to the TruckScenes dataset root (== $TRUCKSCENES).",
    )
    parser.add_argument(
        "--version",
        default="v1.1-trainval",
        help="TruckScenes version tag, passed to the devkit constructor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-log pkl files into. Created if missing.",
    )
    parser.add_argument(
        "--scene-tokens",
        nargs="*",
        default=None,
        help="Optional explicit scene token allowlist. If omitted, every "
             "scene in the chosen version is processed.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap on the number of scenes processed (useful for "
             "smoke testing on a few scenes first).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes. Each instantiates its own "
             "TruckScenes devkit handle.",
    )
    parser.add_argument(
        "--materialize-blobs",
        action="store_true",
        help="Write merged LiDAR PCDs (under <blob-dir>/lidars/<sample>.pcd) "
             "and rewrite scene_dict paths to absolute. Required for "
             "Lightning-trainer training; not needed for PDMS-eval-only.",
    )
    parser.add_argument(
        "--blob-dir",
        type=Path,
        default=None,
        help="Directory to write materialized sensor blobs into. Defaults "
             "to <output-dir>/sensor_blobs/ when --materialize-blobs is set.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Also write a BEV scatter + agent-box overlay PNG per sample "
             "under <blob-dir>/viz/. Requires --materialize-blobs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.materialize_blobs:
        if args.blob_dir is None:
            args.blob_dir = args.output_dir / "sensor_blobs"
        args.blob_dir.mkdir(parents=True, exist_ok=True)

    scene_tokens = _resolve_scene_tokens(
        truckscenes_root=args.truckscenes_root,
        version=args.version,
        explicit=args.scene_tokens,
        max_scenes=args.max_scenes,
    )
    print(f"build_navsim_logs: {len(scene_tokens)} scene(s) to process "
          f"-> {args.output_dir}", flush=True)

    if args.viz and not args.materialize_blobs:
        parser.error("--viz requires --materialize-blobs")

    worker = partial(
        _build_one_log,
        truckscenes_root=args.truckscenes_root,
        version=args.version,
        output_dir=args.output_dir,
        materialize_blobs=args.materialize_blobs,
        blob_dir=args.blob_dir,
        viz=args.viz,
    )

    failures: List[Tuple[str, str]] = []
    if args.num_workers <= 1:
        # Sequential -- simpler tracebacks, easier to debug single-sample
        # issues during initial integration.
        for i, scene_token in enumerate(scene_tokens, 1):
            token, err = worker(scene_token)
            status = "ok" if err is None else "FAIL"
            print(f"[{i}/{len(scene_tokens)}] {token} {status}", flush=True)
            if err is not None:
                failures.append((token, err))
    else:
        # Use spawn to avoid sharing a parent-process TruckScenes handle.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            for i, (token, err) in enumerate(
                pool.imap_unordered(worker, scene_tokens), 1
            ):
                status = "ok" if err is None else "FAIL"
                print(f"[{i}/{len(scene_tokens)}] {token} {status}", flush=True)
                if err is not None:
                    failures.append((token, err))

    if failures:
        print(f"\n{len(failures)} scene(s) failed:", file=sys.stderr)
        for token, err in failures:
            print(f"--- {token} ---", file=sys.stderr)
            print(err, file=sys.stderr)
        sys.exit(1)
    print(f"\nbuild_navsim_logs: done. {len(scene_tokens)} pkl(s) written.")


if __name__ == "__main__":
    main()
