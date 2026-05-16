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
    summaries: List[Tuple[str, float, int]],
    num_per_category: int,
    seed: int = 0,
) -> Dict[str, List[Tuple[str, float, int]]]:
    """Pick `num_per_category` samples each of worst / best / random.

    `summaries` is a list of (token, avg_l2, cmd_class).
    """
    sorted_by_l2 = sorted(summaries, key=lambda x: x[1], reverse=True)
    worst = sorted_by_l2[:num_per_category]
    best = sorted_by_l2[-num_per_category:][::-1]
    middle = sorted_by_l2[num_per_category: max(num_per_category, len(sorted_by_l2) - num_per_category)]
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
    logger.info("Scoring %d val samples to pick %d×3 viz samples",
                len(val_tokens), num_per_cat)

    # ---- pass 1: score all val samples to rank by avg L2 ----
    summaries: List[Tuple[str, float, int]] = []
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
        summaries.append((token, avg_l2, cmd))

    picks = _pick_samples(summaries, num_per_cat)

    # ---- pass 2: render picked samples via navsim plot_bev_with_agent ----
    # Re-set the agent's compute_trajectory device to CPU temporarily would
    # break our GPU acceleration; instead let plot_bev_with_agent call
    # compute_trajectory which uses the agent's current device.
    count = 0
    for category, items in picks.items():
        cat_dir = viz_dir / category
        cat_dir.mkdir(exist_ok=True)
        for token, avg_l2, cmd in items:
            scene = val_scene_loader.get_scene_from_token(token)

            fig, ax = plot_bev_with_agent(scene, agent)
            ax.set_title(
                f"[{category}] cmd={CMD_LABELS.get(cmd, '?')}  "
                f"avg L2={avg_l2:.2f} m\n{token}",
                fontsize=9,
            )
            fig.tight_layout()
            out_png = cat_dir / f"{token}.png"
            fig.savefig(out_png, dpi=120)
            plt.close(fig)
            count += 1

    # picks.json for traceability.
    (viz_dir / "picks.json").write_text(json.dumps({
        category: [{"token": t, "avg_l2": l, "cmd": c} for t, l, c in items]
        for category, items in picks.items()
    }, indent=2))

    print()
    print(f"Wrote {count} BEV PNG(s) under {viz_dir}/")
    print("  (worst/, best/, random/ + picks.json)")


if __name__ == "__main__":
    main()
