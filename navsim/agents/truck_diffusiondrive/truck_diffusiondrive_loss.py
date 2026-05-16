"""Loss for the TruckScenes-adapted DiffusionDrive agent.

Minimal fork of upstream `diffusiondrive.transfuser_loss`. Only one
change: the BEV semantic CE branch is gated on
`config.bev_semantic_weight > 0 AND "bev_semantic_map" in targets`,
matching the TruckTransfuser convention. Everything else (Hungarian
agent matching, trajectory L1 fallback, diffusion loss read-through,
loss_dict layout) is byte-for-byte identical to upstream so we keep the
same logging structure for paper experiments.

Why fork: TruckScenes has no HD map -- bev_semantic_weight=0 here -- so
the upstream unconditional `targets["bev_semantic_map"]` lookup would
crash (KeyError) or produce shape-mismatched CE (if the target shape
mis-aligns with model output, which happens when our config widens
forward BEV but the inherited `bev_pixel_height` stays at the parent's
default; see project_paper_deadline / commit history).
"""
from typing import Dict

import torch

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import _agent_loss
from torch.nn import functional as F


def truck_transfuser_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: TransfuserConfig,
):
    """Truck-adapted DiffusionDrive loss.

    See module docstring for the single semantic-branch gating change vs.
    upstream `diffusiondrive.transfuser_loss`.
    """
    if "trajectory_loss" in predictions:
        trajectory_loss = predictions["trajectory_loss"]
    else:
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)

    # ── Truck-only divergence: only compute the BEV semantic CE when the
    # head is active. Without this gate, the upstream code crashes on
    # TruckScenes (no HD map -> no target / shape mismatch with the
    # unconfigured semantic head).
    if config.bev_semantic_weight > 0 and "bev_semantic_map" in targets:
        bev_semantic_loss = F.cross_entropy(
            predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
        )
    else:
        bev_semantic_loss = torch.zeros((), device=trajectory_loss.device)

    if "diffusion_loss" in predictions:
        diffusion_loss = predictions["diffusion_loss"]
    else:
        diffusion_loss = 0

    loss = (
        config.trajectory_weight * trajectory_loss
        + config.diff_loss_weight * diffusion_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
    )
    loss_dict = {
        "loss": loss,
        "trajectory_loss": config.trajectory_weight * trajectory_loss,
        "diffusion_loss": config.diff_loss_weight * diffusion_loss,
        "agent_class_loss": config.agent_class_weight * agent_class_loss,
        "agent_box_loss": config.agent_box_weight * agent_box_loss,
        "bev_semantic_loss": config.bev_semantic_weight * bev_semantic_loss,
    }
    if "trajectory_loss_dict" in predictions:
        trajectory_loss_dict = predictions["trajectory_loss_dict"]
        loss_dict.update(trajectory_loss_dict)
    return loss_dict
