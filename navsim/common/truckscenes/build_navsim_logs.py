"""Path A (batch script) -- materialize TruckScenes as navsim pkl logs.

Iterates a TruckScenes split, calls `scene_dict_from_sample()` for every
sample, groups results by scene_token, and dumps `<output_dir>/<log_name>.pkl`
files in the layout navsim's stock `SceneLoader` / `MetricCacheProcessor`
already understand. This is the one-shot conversion required to run the
forked map-free PDMS evaluation pipeline on TruckScenes.

Once these pkls exist, downstream navsim code does not need to know about
truckscenes-devkit -- it consumes the pkl logs like any nuPlan/OpenScene
dataset.

Usage (planned):
    python -m navsim.common.truckscenes.build_navsim_logs \\
        --truckscenes-root $TRUCKSCENES \\
        --version v1.1-trainval \\
        --split train \\
        --output-dir /path/to/output/navsim_logs/ \\
        [--num-workers N]
"""
import argparse
from pathlib import Path


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
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Subset to convert. 'all' iterates every scene in the version.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-log pkl files into.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallelism for sample-level conversion. Per-scene grouping "
             "happens after, so workers are independent.",
    )
    args = parser.parse_args()

    # TODO -- ordered:
    #   1. Initialize TruckScenes(dataroot=args.truckscenes_root, version=args.version).
    #   2. Resolve split -> list of scene tokens (TruckScenes has explicit
    #      train/val/test splits in v1.1-trainval; load `splits.json`).
    #   3. For each scene, enumerate valid sample_tokens (same future-window
    #      validity check as transfuser-truckscenes:_collect_valid_samples).
    #   4. For each valid sample_token, call
    #      `scene_dict_from_sample(ts, sample_token)`.
    #   5. Group resulting dicts by scene_token (or log_token if TruckScenes
    #      groups multiple scenes per log -- to confirm).
    #   6. Write `<output_dir>/<log_name>.pkl` with the list of dicts, in
    #      the exact shape `filter_scenes` (navsim/common/dataloader.py:16)
    #      expects.
    #   7. Optional: emit a `sensor_blobs/` tree if we decide to materialize
    #      merged LiDAR PCDs and stitched images here. Decision pending --
    #      MetricCacheProcessor only iterates ego state + annotations and
    #      does NOT load sensor blobs, so this can probably be skipped for
    #      the PDMS-eval-only use case.
    raise NotImplementedError("build_navsim_logs.main: skeleton")


if __name__ == "__main__":
    main()
