"""Open-loop L2 + collision evaluator for trained Truck agents.

Run with the SAME --config-name / agent= overrides you used at training
time so the agent class + config matches the checkpoint. Pass the
Lightning checkpoint with `+checkpoint=/path/to/last.ckpt`.

Examples (β heading, training cache reused for feature builders):

    PYTHONPATH=. python scripts/evaluation/run_truck_l2_collision_eval.py \\
        --config-name default_training_truck \\
        train_test_split=truckscenes_mini \\
        navsim_log_path=/NHNHOME/.../navsim_truck_logs \\
        sensor_blobs_path=/NHNHOME/.../navsim_truck_blobs \\
        +checkpoint=/NHNHOME/.../truck_navsim_exp/<exp_name>/.../lightning_logs/version_0/checkpoints/last.ckpt

Output:
  * stdout: pretty table of L2 / collision per horizon.
  * `<output_dir>/l2_collision_metrics.json` -- same dict, machine-readable.
  * optional wandb run (if cfg.wandb.enable) with the same metrics
    posted to the run summary.

Scope: this is the "what we can ship today" evaluator. It mirrors the
transfuser-truckscenes evaluate.py protocol (UniAD / VAD / ST-P3
open-loop L2 + collision) and stands in until the full no-map PDMS
scorer described in docs/truck_pdm_scorer_design.md lands.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.evaluation.truck_l2_collision import (
    EVAL_HORIZONS,
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    DEFAULT_INTERVAL_S,
    aggregate_metrics,
    compute_sample_metrics,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "../../navsim/planning/script/config/training"
CONFIG_NAME = "default_training_truck"


def _load_checkpoint_into_agent(agent: AbstractAgent, ckpt_path: Path) -> None:
    """Load a Lightning-saved checkpoint into a fresh navsim Agent.

    Lightning wraps the agent under the attribute name `agent` inside
    AgentLightningModule, so state_dict keys look like
    `agent._transfuser_model.image_encoder...`. We strip the leading
    `agent.` and load into the bare Agent. Strict=False catches the
    rare case where Lightning serialises extra training-only buffers
    (e.g. EMA / optimizer-aux state in the same dict).
    """
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = payload.get("state_dict", payload)
    stripped = {}
    for k, v in sd.items():
        if k.startswith("agent."):
            stripped[k[len("agent."):]] = v
        else:
            stripped[k] = v
    missing, unexpected = agent.load_state_dict(stripped, strict=False)
    if missing:
        logger.warning("Checkpoint missing %d keys (showing up to 5): %s",
                       len(missing), missing[:5])
    if unexpected:
        logger.warning("Checkpoint had %d unexpected keys (showing up to 5): %s",
                       len(unexpected), unexpected[:5])


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg.get("checkpoint"):
        raise ValueError(
            "Pass `+checkpoint=/path/to/last.ckpt` so the eval knows what "
            "weights to load."
        )
    ckpt_path = Path(cfg.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Building agent + loading checkpoint %s on %s", ckpt_path, device)

    agent: AbstractAgent = instantiate(cfg.agent)
    _load_checkpoint_into_agent(agent, ckpt_path)
    agent.to(device).eval()

    # ---- scene loader: VAL split only (we evaluate on the 75-scene split) ----
    logger.info("Building val SceneLoader")
    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    # Restrict to the official val log list (defined in
    # truckscenes_train_val_log_split.yaml -> cfg.val_logs).
    val_scene_filter.log_names = list(cfg.val_logs)

    val_scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    val_tokens = val_scene_loader.tokens
    logger.info("Evaluating on %d val samples", len(val_tokens))

    # Optional wandb logging (mirrors run_training.py's gate).
    wandb_run = None
    if cfg.get("wandb") is not None and cfg.wandb.get("enable", False):
        import wandb as _wandb

        wandb_run = _wandb.init(
            project=cfg.wandb.get("project", "truck_navsim"),
            entity=cfg.wandb.get("entity", None),
            name=f"eval__{cfg.get('experiment_name', ckpt_path.parent.parent.name)}",
            tags=list(cfg.wandb.get("tags", []) or []) + ["eval:l2-collision"],
            reinit=True,
        )

    # ---- main eval loop ----
    l2_per_sample: List[List[float]] = []
    col_per_sample: List[List[bool]] = []
    # also retain (token, cmd_class) parallel arrays so the per-sample
    # JSON can drive viz / per-class breakdowns later without re-running.
    sample_tokens: List[str] = []
    cmd_classes: List[int] = []

    for token in tqdm(val_tokens, desc="L2+col eval"):
        scene = val_scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        gt_trajectory = scene.get_future_trajectory().poses  # (num_poses, 3)

        # forward pass on GPU. compute_trajectory builds features on
        # the cpu side then forwards on whatever device the agent lives
        # on. We move feature tensors to device manually to avoid an
        # AbstractAgent edit.
        features: Dict[str, torch.Tensor] = {}
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
        with torch.no_grad():
            predictions = agent.forward(features)
        pred_trajectory = predictions["trajectory"].squeeze(0).detach().cpu().numpy()

        l2_list, col_list = compute_sample_metrics(
            pred_trajectory=pred_trajectory,
            gt_trajectory=gt_trajectory,
            scene=scene,
            horizons_s=EVAL_HORIZONS,
            interval_s=DEFAULT_INTERVAL_S,
            ego_length_m=DEFAULT_EGO_LENGTH_M,
            ego_width_m=DEFAULT_EGO_WIDTH_M,
        )
        l2_per_sample.append(l2_list)
        col_per_sample.append(col_list)
        sample_tokens.append(token)
        cmd_classes.append(int(np.argmax(agent_input.ego_statuses[-1].driving_command)))

    metrics = aggregate_metrics(l2_per_sample, col_per_sample, EVAL_HORIZONS)

    # ---- output ----
    out_json = output_dir / "l2_collision_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))

    # Per-sample dump for downstream consumers (viz pickers, ablation
    # breakdowns by cmd class, statistical tests). Same horizon order as
    # EVAL_HORIZONS.
    per_sample_payload = {
        "horizons_s": list(EVAL_HORIZONS),
        "samples": [
            {
                "token": tok,
                "cmd": cmd,
                "l2_per_horizon": l2_per_sample[i],
                "collision_per_horizon": [bool(x) for x in col_per_sample[i]],
                "avg_l2": float(np.nanmean(l2_per_sample[i])),
            }
            for i, (tok, cmd) in enumerate(zip(sample_tokens, cmd_classes))
        ],
    }
    out_per_sample = output_dir / "l2_collision_per_sample.json"
    out_per_sample.write_text(json.dumps(per_sample_payload, indent=2))

    print()
    print("=== L2 + Collision evaluation ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"samples:    {metrics['num_samples']}")
    print()
    print(f"{'horizon':<10}{'L2 (m)':>12}{'Collision (%)':>18}")
    print("-" * 40)
    for h in EVAL_HORIZONS:
        h_key = f"{int(h)}s"
        print(f"{h_key:<10}{metrics[f'l2/{h_key}']:>12.3f}{metrics[f'col/{h_key}']:>18.2f}")
    print("-" * 40)
    print(f"{'avg':<10}{metrics['l2/avg']:>12.3f}{metrics['col/avg']:>18.2f}")
    print()
    print(f"JSON written to:        {out_json}")
    print(f"Per-sample JSON:        {out_per_sample}")

    if wandb_run is not None:
        wandb_run.summary.update(metrics)
        wandb_run.finish()


if __name__ == "__main__":
    main()
