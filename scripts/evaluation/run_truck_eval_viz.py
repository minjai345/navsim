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
from navsim.visualization.bev import (
    add_configured_bev_on_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.camera import add_camera_ax
from navsim.visualization.plots import (
    configure_ax,
    configure_bev_ax,
    plot_bev_with_agent,
)

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


def _render_bev_panel(
    ax,
    scene,
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
) -> None:
    """Fill one matplotlib axis with the BEV (annotations + lidar) + GT/Pred
    trajectories. Same content as plot_bev_with_agent but writes to a given
    ax instead of creating its own figure. Lets us drop the BEV into a
    composite grid alongside the camera panels.
    """
    from navsim.visualization.config import TRAJECTORY_CONFIG
    from navsim.common.dataclasses import Trajectory

    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, Trajectory(gt_trajectory), TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, Trajectory(pred_trajectory), TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)


# Map our 4-cam TruckScenes layout to a 2x2 grid: top row = front, bottom = back.
# Slot names are navsim attribute names on Cameras (cam_l0=LEFT_FRONT etc).
_CAM_GRID = [
    ("cam_l0", "LEFT_FRONT"),
    ("cam_r0", "RIGHT_FRONT"),
    ("cam_l2", "LEFT_BACK"),
    ("cam_r2", "RIGHT_BACK"),
]


def _render_composite_frame(
    fig,
    grid_spec,
    scene,
    frame_idx_for_cams: int,
    gt_trajectory_t0: np.ndarray,
    pred_trajectory_t0: np.ndarray,
    marker_idx: int,
    stats_lines: List[str],
) -> None:
    """Draw a 2x3 composite figure: 4 cameras + BEV(t=0) + stats text.

    Layout:
        ┌──────────────┬──────────────┬──────────────┐
        │  LEFT_FRONT  │              │  RIGHT_FRONT │
        ├──────────────┤     BEV      ├──────────────┤
        │  LEFT_BACK   │              │  RIGHT_BACK  │
        └──────────────┴──────────────┴──────────────┘
                                  +
                                stats

    Cameras come from `scene.frames[frame_idx_for_cams]` so they change over
    time across GIF frames. BEV is drawn at t=0 with the GT/Pred trajectory
    overlay + a large marker at `marker_idx` along each trajectory.
    """
    fig.clear()
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.18],
                          hspace=0.04, wspace=0.04)

    # ---- camera panels ----
    cam_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 2]),
    ]
    frame = scene.frames[frame_idx_for_cams]
    for ax, (cam_attr, label) in zip(cam_axes, _CAM_GRID):
        camera = getattr(frame.cameras, cam_attr, None)
        if camera is not None and getattr(camera, "image", None) is not None:
            add_camera_ax(ax, camera)
        else:
            ax.set_facecolor("#222")
            ax.text(0.5, 0.5, "no image", color="white",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- BEV panel ----
    bev_ax = fig.add_subplot(gs[0:2, 1])
    _render_bev_panel(bev_ax, scene, gt_trajectory_t0, pred_trajectory_t0)

    # Trajectory progress markers (BEV axes convention: y_world plotted on
    # x-screen, x_world on y-screen, x-axis inverted -- mirrors
    # add_trajectory_to_bev_ax).
    gt_xy = np.concatenate(
        [np.zeros((1, 2)), gt_trajectory_t0[:, :2]], axis=0
    )
    pr_xy = np.concatenate(
        [np.zeros((1, 2)), pred_trajectory_t0[:, :2]], axis=0
    )
    k = max(0, min(marker_idx, gt_xy.shape[0] - 1, pr_xy.shape[0] - 1))
    bev_ax.scatter([gt_xy[k, 1]], [gt_xy[k, 0]],
                   s=120, c="#2ca02c", edgecolors="black",
                   linewidths=1.2, zorder=6)
    bev_ax.scatter([pr_xy[k, 1]], [pr_xy[k, 0]],
                   s=120, c="#d62728", edgecolors="black",
                   linewidths=1.2, zorder=6)

    # ---- stats banner (bottom row spanning all 3 cols) ----
    stats_ax = fig.add_subplot(gs[2, :])
    stats_ax.axis("off")
    stats_ax.text(
        0.5, 0.5, " | ".join(stats_lines),
        ha="center", va="center", transform=stats_ax.transAxes,
        fontsize=9, family="monospace",
    )


def _emit_composite_gif(
    scene,
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    out_path: Path,
    title_prefix: str,
    duration_ms: int = 400,
) -> None:
    """Render a 2x3 composite GIF.

    Each GIF frame `k` (k = 0..num_future_frames):
      * cameras come from `scene.frames[num_history-1 + k]`
        -> the 4 camera views actually advance through time
      * BEV(t=0) stays fixed (LiDAR + annotations at the starting frame)
      * GT and predicted markers scrub along their trajectories at step k

    Cameras updating means you see the road change as the truck moves;
    BEV stays fixed so the two trajectories remain visible end-to-end
    (paper-style trajectory comparison).
    """
    import io
    from PIL import Image

    num_history = scene.scene_metadata.num_history_frames
    num_future = scene.scene_metadata.num_future_frames
    # GIF spans the current frame + num_future_frames forward poses, so
    # camera frames and trajectory marker step together.
    n_steps = min(num_future + 1, gt_trajectory.shape[0] + 1, pred_trajectory.shape[0] + 1)

    fig = plt.figure(figsize=(14, 7))
    frames = []
    for k in range(n_steps):
        frame_idx_for_cams = min(num_history - 1 + k, len(scene.frames) - 1)
        stats = [
            f"{title_prefix}",
            f"t=+{k * DEFAULT_INTERVAL_S:.1f}s",
            f"frame={frame_idx_for_cams}",
        ]
        _render_composite_frame(
            fig=fig,
            grid_spec=None,
            scene=scene,
            frame_idx_for_cams=frame_idx_for_cams,
            gt_trajectory_t0=gt_trajectory,
            pred_trajectory_t0=pred_trajectory,
            marker_idx=k,
            stats_lines=stats,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
    plt.close(fig)

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0,
    )


def _curvy_picks_from_json(
    curvy_json_path: Path,
    top_k: int,
    val_token_set: set,
) -> List[Tuple[str, float]]:
    """Load the existing curvy ranking and return the top-K val tokens.

    `scene_curvy_split.json` (from transfuser-truckscenes/tools/find_curvy_scenes.py)
    is a list of {token, curvature, ...} already sorted descending by curvature.
    """
    payload = json.loads(curvy_json_path.read_text())
    scenes = payload.get("scenes", [])
    picks = []
    for s in scenes:
        if s["token"] in val_token_set:
            picks.append((s["token"], float(s.get("curvature", 0.0))))
            if len(picks) >= top_k:
                break
    return picks


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

    # ---- pass 2: render picked samples via navsim BEV helpers ----
    # We cannot use `plot_bev_with_agent` directly because its internal
    # `agent.compute_trajectory(...)` keeps features on CPU and would
    # crash when the agent lives on GPU (CPU input vs GPU weights).
    # Instead, compute the predicted trajectory manually with explicit
    # device move and pass to `_render_bev_panel`.
    count = 0
    for category, items in picks.items():
        cat_dir = viz_dir / category
        cat_dir.mkdir(exist_ok=True)
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

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            _render_bev_panel(ax, scene, gt_trajectory, pred_trajectory)
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

    # ---- optional curvy-scene composite mode ----
    # `+viz.curvy_top_k=N +viz.curvy_json=/path/to/scene_curvy_split.json`
    # picks the top-N most-curvy val scenes and renders a 2x3 composite
    # (4 cams + BEV + stats) PNG plus a GIF where the cameras advance
    # through the future frames while the BEV stays fixed at t=0.
    curvy_top_k = int(cfg.get("viz", {}).get("curvy_top_k", 0) or 0)
    curvy_json = cfg.get("viz", {}).get("curvy_json", None)
    if curvy_top_k > 0 and curvy_json:
        curvy_dir = viz_dir / "curvy_composite"
        curvy_dir.mkdir(exist_ok=True)
        val_token_set = set(val_scene_loader.tokens)
        curvy = _curvy_picks_from_json(
            Path(curvy_json), curvy_top_k, val_token_set
        )
        logger.info("Curvy composite mode: %d scenes selected", len(curvy))
        curvy_records = []
        for token, curvature in curvy:
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
            cmd = int(np.argmax(agent_input.ego_statuses[-1].driving_command))
            avg_l2 = float(np.nanmean(_l2_per_horizon(pred_trajectory, gt_trajectory)))

            title_prefix = (
                f"cmd={CMD_LABELS.get(cmd, '?')}  "
                f"curvature={curvature:.2f}  avg_L2={avg_l2:.2f} m  "
                f"{token[:12]}…"
            )
            # static PNG: t=0 frame
            fig = plt.figure(figsize=(14, 7))
            _render_composite_frame(
                fig=fig, grid_spec=None,
                scene=scene,
                frame_idx_for_cams=scene.scene_metadata.num_history_frames - 1,
                gt_trajectory_t0=gt_trajectory,
                pred_trajectory_t0=pred_trajectory,
                marker_idx=0,
                stats_lines=[title_prefix, "t=+0.0s", "frame=current"],
            )
            png_path = curvy_dir / f"{token}.png"
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            # GIF: cameras advance per future frame, BEV stays at t=0
            if bool(cfg.get("viz", {}).get("emit_gif", True)):
                _emit_composite_gif(
                    scene=scene,
                    gt_trajectory=gt_trajectory,
                    pred_trajectory=pred_trajectory,
                    out_path=curvy_dir / f"{token}.gif",
                    title_prefix=title_prefix,
                )
            curvy_records.append({
                "token": token, "curvature": curvature, "cmd": cmd,
                "avg_l2": avg_l2,
            })

        (curvy_dir / "picks.json").write_text(
            json.dumps(curvy_records, indent=2, ensure_ascii=False)
        )


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

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            _render_bev_panel(ax, scene, gt_trajectory, pred_trajectory)
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
