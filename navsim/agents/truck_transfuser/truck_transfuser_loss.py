"""Truck TransFuser loss.

Ported from transfuser-truckscenes/model/loss.py. Differences vs. navsim's
vanilla transfuser_loss:

  - Returns `(loss, components)` instead of a bare scalar. Lightning
    logging and W&B traces consume `components` to split out per-term
    curves. Vanilla navsim's loss returns the scalar only; we keep the
    richer interface so existing transfuser-truckscenes training logs
    stay drop-in.
  - Trailer trajectory term: gated on key presence (no key in
    predictions/targets -> term silently skipped), masked by
    `targets["trailer_mask"]` (1 for samples with an ego_trailer
    annotation, 0 otherwise). Denominator clamped to 1 so a trailer-free
    batch does not nan-divide.
  - BEV semantic term: gated on `bev_semantic_weight > 0` AND key
    presence, since TruckScenes does not emit a BEV semantic target
    (use_bev_semantic=False default). Vanilla navsim always computes it.

The Hungarian-matched agent detection loss is unchanged from the vanilla
implementation; we keep the same helpers (_get_ce_cost, _get_l1_cost,
_get_src_permutation_idx).
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from navsim.agents.truck_transfuser.truck_transfuser_config import TruckTransfuserConfig
from navsim.agents.truck_transfuser.truck_transfuser_features import BoundingBox2DIndex


def truck_transfuser_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: TruckTransfuserConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Total loss (weighted sum) + per-term breakdown.

    Returns:
        loss:       scalar tensor used for backward.
        components: dict of un-weighted per-term losses for logging.
                    Always present: truck_l1, agent_cls, agent_box.
                    Conditionally present: trailer_l1 (when targets carry
                    trailer trajectory + mask), bev_semantic (when the
                    BEV head is enabled and target exists).
    """
    truck_l1 = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_cls, agent_box = _agent_loss(targets, predictions, config)

    components: Dict[str, torch.Tensor] = {
        "truck_l1": truck_l1,
        "agent_cls": agent_cls,
        "agent_box": agent_box,
    }

    loss = (
        config.trajectory_weight * truck_l1
        + config.agent_class_weight * agent_cls
        + config.agent_box_weight * agent_box
    )

    # Trailer trajectory term.
    # - trailer present (mask=1): supervised; trailer absent (mask=0):
    #   contribution is zero in the numerator.
    # - denominator = number of trailer-present samples in the batch,
    #   clamped to 1 to avoid div-by-zero on trailer-free batches.
    if (
        config.trailer_weight > 0
        and "trailer_trajectory" in predictions
        and "trailer_trajectory" in targets
        and "trailer_mask" in targets
    ):
        pred_t = predictions["trailer_trajectory"]   # (B, num_poses, 3)
        gt_t = targets["trailer_trajectory"]         # (B, num_poses, 3)
        mask = targets["trailer_mask"]               # (B,)

        per_sample = F.l1_loss(pred_t, gt_t, reduction="none").mean(dim=(-1, -2))
        denom = mask.sum().clamp(min=1.0)
        trailer_l1 = (per_sample * mask).sum() / denom
        components["trailer_l1"] = trailer_l1
        loss = loss + config.trailer_weight * trailer_l1

    # BEV semantic term (off by default for TruckScenes: no HD map).
    if config.bev_semantic_weight > 0 and "bev_semantic_map" in targets:
        bev_loss = F.cross_entropy(
            predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
        )
        components["bev_semantic"] = bev_loss
        loss = loss + config.bev_semantic_weight * bev_loss

    return loss, components


def _agent_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config: TruckTransfuserConfig,
):
    """Hungarian-matched bounding box loss (identical to vanilla TransFuser)."""
    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        # Restrict matching to a forward cone (latent TF setting).
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )
        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    # Guard the divisor; if no GT in the entire batch, treat as 1 so the
    # loss stays a meaningful zero (rather than NaN).
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable BCE cost matrix for Hungarian matching."""
    gt_valid_expanded = gt_valid[:, :, None].detach().float()
    pred_logits_expanded = pred_logits[:, None, :].detach()

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val))
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term
    ce_cost = ce_cost.permute(0, 2, 1)
    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """L1 (x, y) cost matrix for Hungarian matching."""
    gt_states_expanded = gt_states[:, :, None, :2].detach()
    pred_states_expanded = pred_states[:, None, :, :2].detach()
    l1_cost = gt_valid[..., None].float() * (
        gt_states_expanded - pred_states_expanded
    ).abs().sum(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """Flatten Hungarian-matched (batch, src) index pairs into a single index tensor."""
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
