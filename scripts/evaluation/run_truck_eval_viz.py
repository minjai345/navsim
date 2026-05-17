"""BEV visualization of predicted vs ground-truth trajectories.

Thin wrapper over navsim's existing viz module
(`navsim/visualization/plots.py::plot_bev_with_agent`). For a trained
truck agent we score the val split, pick worst / best / random samples
by avg L2, render each picked sample as a navsim-style BEV PNG, and
dump a `picks.json` for traceability.

Layer config: `["annotations", "lidar"]` -- map layer is excluded
because TruckScenes has no HD map (Scene.map_api is a NullMap stub
that would crash navsim's map-querying code path).

Usage:

    PYTHONPATH=. python scripts/evaluation/run_truck_eval_viz.py \\
        --config-name default_training_truck \\
        train_test_split=truckscenes_mini \\
        navsim_log_path=/NHNHOME/.../navsim_truck_logs \\
        sensor_blobs_path=/NHNHOME/.../navsim_truck_blobs \\
        +checkpoint=/tmp/ckpt_beta_heading.ckpt \\
        +viz.num_per_category=3 \\
        wandb.enable=false
"""
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

# Side-effect import: extends navsim's tracked_object_types dict so
# `add_annotations_to_bev_ax` accepts our hierarchical names. Must run
# before any navsim.visualization import that touches that dict.
import navsim.visualization.truck_viz_shim  # noqa: F401

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.evaluation.truck_l2_collision import DEFAULT_INTERVAL_S, EVAL_HORIZONS
import navsim.visualization.config as viz_config
from navsim.visualization.plots import plot_bev_with_agent

logger = logging.getLogger(__name__)

CONFIG_PATH = "../../navsim/planning/script/config/training"
CONFIG_NAME = "default_training_truck"

CMD_LABELS = {0: "Right", 1: "Left", 2: "Straight"}


def _load_checkpoint_into_agent(agent: AbstractAgent, ckpt_path: Path) -> None:
    """Strip Lightning's `agent.` prefix and load into a fresh navsim Agent."""
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = payload.get("state_dict", payload)
    stripped = {}
    for k, v in sd.items():
        if k.startswith("agent."):
            stripped[k[len("agent."):]] = v
        else:
            stripped[k] = v
    missing, unexpected = agent.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        logger.warning(
            "Load mismatch: %d missing, %d unexpected", len(missing), len(unexpected)
        )


def _l2_per_horizon(pred: np.ndarray, gt: np.ndarray) -> List[float]:
    """L2 at each EVAL_HORIZON step. Same indexing as compute_sample_metrics."""
    out = []
    for h in EVAL_HORIZONS:
        idx = int(round(h / DEFAULT_INTERVAL_S)) - 1
        if idx < 0 or idx >= pred.shape[0]:
            out.append(float("nan"))
            continue
        out.append(float(np.linalg.norm(pred[idx, :2] - gt[idx, :2])))
    return out


def _pick_samples(
    summaries: List[Tuple[str, float, int, float]],
    num_per_category: int,
    min_gt_displacement_m: float = 2.0,
    seed: int = 0,
) -> Dict[str, List[Tuple[str, float, int, float]]]:
    """Pick `num_per_category` samples each of worst / best / random.

    Filters out near-stationary scenes (GT max displacement
    < `min_gt_displacement_m`) so "best" L2≈0 doesn't trivially win on
    parked-truck samples where pred-and-GT both predict no motion.

    `summaries` entries are (token, avg_l2, cmd_class, gt_max_disp).
    Returned tuples preserve the same shape for downstream consumers.
    """
    active = [s for s in summaries if s[3] >= min_gt_displacement_m]
    if not active:
        # Should never happen on a real val split, but guard anyway.
        active = summaries

    sorted_by_l2 = sorted(active, key=lambda x: x[1], reverse=True)
    worst = sorted_by_l2[:num_per_category]
    best = sorted_by_l2[-num_per_category:][::-1]
    middle = sorted_by_l2[
        num_per_category: max(num_per_category, len(sorted_by_l2) - num_per_category)
    ]
    rng = random.Random(seed)
    rand = rng.sample(middle, min(num_per_category, len(middle)))
    return {"worst": worst, "best": best, "random": rand}


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise ValueError(
            "Pass `+checkpoint=/path/to/last.ckpt` so the script knows what "
            "weights to load."
        )
    ckpt_path = Path(cfg.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    num_per_cat = int(cfg.get("viz", {}).get("num_per_category", 3))
    output_dir = Path(cfg.output_dir)
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # TruckScenes has no HD map -- drop the "map" layer from the shared
    # navsim BEV config so `add_map_to_bev_ax` is never invoked on NullMap.
    # Add "lidar" so the point cloud shows up alongside annotations.
    viz_config.BEV_PLOT_CONFIG["layers"] = ["annotations", "lidar"]

    # Widen BEV extent: navsim default 64x64 m is too narrow for a 4-s
    # truck trajectory at highway speed (~100 m forward). figure_margin
    # = (margin_x_forward, margin_y_lateral) -- doubled by configure_bev_ax
    # before applied as ±half. (160, 100) gives ±80 m forward × ±50 m lat.
    viz_config.BEV_PLOT_CONFIG["figure_margin"] = (160, 100)

    # Push LiDAR to the background. Defaults (alpha=0.5, size=0.1,
    # zorder=3) drown annotations + trajectories. The shim's monkey-
    # patched renderer already uses uniform light grey, so we just bump
    # alpha/size/zorder via the shared config.
    viz_config.LIDAR_CONFIG.update(alpha=0.20, size=0.05, zorder=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent: AbstractAgent = instantiate(cfg.agent)
    _load_checkpoint_into_agent(agent, ckpt_path)
    agent.to(device).eval()

    # val scene loader (75-scene split).
    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    val_scene_filter.log_names = list(cfg.val_logs)
    val_scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    val_tokens = val_scene_loader.tokens
    # Optional `+viz.score_subset=N` -- on CPU the full 2170-sample
    # scoring pass takes ~1.5 h. When N is set we deterministically
    # sample N tokens (random with fixed seed) and rank within that
    # subset; the pick distribution is still meaningful for a preview.
    score_subset = cfg.get("viz", {}).get("score_subset", None)
    if score_subset is not None and score_subset > 0 and score_subset < len(val_tokens):
        rng = random.Random(0)
        val_tokens = rng.sample(list(val_tokens), int(score_subset))
        logger.info("Subsetting to %d random val samples for fast scoring",
                    len(val_tokens))
    logger.info("Scoring %d val samples to pick %d×3 viz samples",
                len(val_tokens), num_per_cat)

    # ---- pass 1: score all val samples to rank by avg L2 ----
    summaries: List[Tuple[str, float, int, float]] = []
    for token in tqdm(val_tokens, desc="scoring"):
        scene = val_scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        gt_trajectory = scene.get_future_trajectory().poses

        features = {}
        for b in agent.get_feature_builders():
            features.update(b.compute_features(agent_input))
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
        with torch.no_grad():
            predictions = agent.forward(features)
        pred_trajectory = predictions["trajectory"].squeeze(0).detach().cpu().numpy()

        l2_list = _l2_per_horizon(pred_trajectory, gt_trajectory)
        avg_l2 = float(np.nanmean(l2_list))
        cmd = int(np.argmax(agent_input.ego_statuses[-1].driving_command))
        # max GT displacement from origin -- used to filter near-
        # stationary samples (parked/stopped trucks where L2≈0 is trivial).
        gt_max_disp = float(np.max(np.linalg.norm(gt_trajectory[:, :2], axis=-1)))
        summaries.append((token, avg_l2, cmd, gt_max_disp))

    picks = _pick_samples(summaries, num_per_cat)

    # ---- pass 2: render picked samples via navsim plot_bev_with_agent ----
    # Re-set the agent's compute_trajectory device to CPU temporarily would
    # break our GPU acceleration; instead let plot_bev_with_agent call
    # compute_trajectory which uses the agent's current device.
    count = 0
    for category, items in picks.items():
        cat_dir = viz_dir / category
        cat_dir.mkdir(exist_ok=True)
        for token, avg_l2, cmd, gt_disp in items:
            scene = val_scene_loader.get_scene_from_token(token)

            fig, ax = plot_bev_with_agent(scene, agent)
            ax.set_title(
                f"[{category}] cmd={CMD_LABELS.get(cmd, '?')}  "
                f"avg L2={avg_l2:.2f} m  gt_disp={gt_disp:.1f} m\n{token}",
                fontsize=9,
            )
            fig.tight_layout()
            out_png = cat_dir / f"{token}.png"
            fig.savefig(out_png, dpi=120)
            plt.close(fig)
            count += 1

    # picks.json for traceability.
    (viz_dir / "picks.json").write_text(json.dumps({
        category: [
            {"token": t, "avg_l2": l, "cmd": c, "gt_disp_m": d}
            for t, l, c, d in items
        ]
        for category, items in picks.items()
    }, indent=2))

    # ---- optional GIFs ----
    # For each picked sample also render a short GIF that scrubs a
    # "current-time" marker along the GT and predicted trajectories
    # against the same BEV background. Useful for spotting where /
    # when the prediction diverges from the human driver.
    if bool(cfg.get("viz", {}).get("emit_gif", True)):
        _emit_gifs(picks, val_scene_loader, agent, viz_dir, device)


def _emit_gifs(picks, val_scene_loader, agent, viz_dir, device) -> None:
    """Render one trajectory-scrub GIF per picked sample.

    Each GIF: BEV background (annotations + lidar) drawn once, with a
    large marker that walks step-by-step along the GT (green) and
    predicted (red) trajectories. Saved next to the corresponding PNG
    as `<token>.gif`. Uses Pillow directly since navsim's
    frame_plot_to_gif expects a per-frame plot callable and we'd
    re-render the whole BEV every frame -- much slower.
    """
    import io
    from PIL import Image

    for category, items in picks.items():
        cat_dir = viz_dir / category
        for token, avg_l2, cmd, gt_disp in items:
            scene = val_scene_loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()
            gt_trajectory = scene.get_future_trajectory().poses
            features = {}
            for b in agent.get_feature_builders():
                features.update(b.compute_features(agent_input))
            features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
            with torch.no_grad():
                predictions = agent.forward(features)
            pred_trajectory = (
                predictions["trajectory"].squeeze(0).detach().cpu().numpy()
            )

            fig, ax = plot_bev_with_agent(scene, agent)
            ax.set_title(
                f"[{category}] cmd={CMD_LABELS.get(cmd, '?')}  "
                f"avg L2={avg_l2:.2f} m  gt_disp={gt_disp:.1f} m\n{token}",
                fontsize=9,
            )

            # The BEV ax uses (y, x) -> (x_screen, y_screen) per
            # add_trajectory_to_bev_ax. Replicate that mapping for the
            # animated marker.
            gt_xy = np.concatenate([np.zeros((1, 2)), gt_trajectory[:, :2]], axis=0)
            pr_xy = np.concatenate([np.zeros((1, 2)), pred_trajectory[:, :2]], axis=0)

            # Persistent scatter handles -- updated in place each frame.
            gt_marker = ax.scatter([], [], s=120, c="#2ca02c", edgecolors="black",
                                   linewidths=1.2, zorder=6)
            pr_marker = ax.scatter([], [], s=120, c="#d62728", edgecolors="black",
                                   linewidths=1.2, zorder=6)

            num_steps = min(gt_xy.shape[0], pr_xy.shape[0])
            frames: List[Image.Image] = []
            for k in range(num_steps):
                # NOTE: matplotlib BEV plots y on x-screen and x on y-screen
                # (see add_trajectory_to_bev_ax).
                gt_marker.set_offsets([[gt_xy[k, 1], gt_xy[k, 0]]])
                pr_marker.set_offsets([[pr_xy[k, 1], pr_xy[k, 0]]])
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110)
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()

            plt.close(fig)
            gif_path = cat_dir / f"{token}.gif"
            frames[0].save(
                gif_path, save_all=True, append_images=frames[1:],
                duration=400, loop=0,
            )

    print()
    print(f"Wrote {count} BEV PNG(s) under {viz_dir}/")
    print("  (worst/, best/, random/ + picks.json)")


if __name__ == "__main__":
    main()
