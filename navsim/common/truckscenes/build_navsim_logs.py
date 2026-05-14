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
  - PDMS metric caching does not consume sensor blobs (verified against
    metric_cache_processor.py -- iterates ego state + annotations only).
    This script therefore does NOT materialize merged LiDAR PCDs or
    camera image copies. The lidar_path in every dict is None and cams
    refer to TruckScenes' original image paths under $TRUCKSCENES.

Usage:
    python -m navsim.common.truckscenes.build_navsim_logs \\
        --truckscenes-root $TRUCKSCENES \\
        --version v1.1-trainval \\
        --output-dir /path/to/navsim_truck_logs/ \\
        [--scene-tokens TOKEN1 TOKEN2 ...] \\
        [--max-scenes N] \\
        [--num-workers N]
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
) -> Tuple[str, Optional[str]]:
    """Worker entry: build and pickle one log. Returns (scene_token, error).

    A separate `TruckScenes` instance is constructed per worker so each
    process has its own device IO context (the devkit is not designed for
    cross-process sharing of a single instance).
    """
    try:
        # Imports inside the worker so the parent process need not load
        # truckscenes-devkit unless the user actually runs the script.
        from truckscenes import TruckScenes

        from navsim.common.truckscenes.scene_dict_from_sample import (
            scene_dict_from_sample,
        )

        ts = TruckScenes(version=version, dataroot=str(truckscenes_root), verbose=False)

        sample_tokens = _iter_scene_sample_tokens(ts, scene_token)
        scene_dict_list = []
        for sample_token in sample_tokens:
            scene_dict_list.append(scene_dict_from_sample(ts, sample_token))

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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_tokens = _resolve_scene_tokens(
        truckscenes_root=args.truckscenes_root,
        version=args.version,
        explicit=args.scene_tokens,
        max_scenes=args.max_scenes,
    )
    print(f"build_navsim_logs: {len(scene_tokens)} scene(s) to process "
          f"-> {args.output_dir}", flush=True)

    worker = partial(
        _build_one_log,
        truckscenes_root=args.truckscenes_root,
        version=args.version,
        output_dir=args.output_dir,
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
