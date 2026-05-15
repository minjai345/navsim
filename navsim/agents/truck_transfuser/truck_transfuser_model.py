"""Truck TransFuser model.

Ported from transfuser-truckscenes/model/model.py. Reuses navsim's
existing `TransfuserBackbone` (the vanilla and truck variants share an
identical backbone class -- only config values differ, all of which are
duck-typable). Truck-specific deltas vs. navsim's vanilla TransfuserModel:

  - Status feature is composed at forward time from optional channels
    (4D ego_status + 3D one-hot driving_command), gated by
    `use_ego_status` / `use_driving_command`. Status dropout is applied
    inside forward (training only) to discourage the vx*dt shortcut.
  - keyval embedding length is computed from
    `lidar_vert_anchors * lidar_horz_anchors`, not the hard-coded 8x8 in
    the vanilla model -- truck configs (e.g. wider BEV range) change the
    grid.
  - Optional trailer trajectory head emits a separate "trailer_trajectory"
    key when `use_trailer_head=True`. Trailer-free samples are handled by
    a mask in the loss, not by a per-sample skip here.
  - BEV semantic head is built unconditionally for compatibility but only
    emits an output when `bev_semantic_weight > 0` (default 0 for
    TruckScenes -- no HD map).
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

# Reuse navsim's existing backbone -- truck variant only differs in config
# values, and TransfuserBackbone consumes its config by field name (duck
# typing), so passing a TruckTransfuserConfig works without inheritance.
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.truck_transfuser._indices import BoundingBox2DIndex
from navsim.agents.truck_transfuser.truck_transfuser_config import TruckTransfuserConfig
from navsim.common.enums import StateSE2Index


class TruckTransfuserModel(nn.Module):
    """TransFuser variant for MAN TruckScenes.

    Query layout: [truck_trajectory(1), (optional) trailer_trajectory(1),
    agent_detection(N)]. The trailer query slot is added only when
    `use_trailer_head=True` (strict truck-only ablation drops it
    entirely; setting `trailer_weight=0` alone keeps the capacity).
    """

    def __init__(self, config: TruckTransfuserConfig):
        super().__init__()

        # Query splits, in order.
        self._query_splits = [1]  # truck trajectory query
        if config.use_trailer_head:
            self._query_splits.append(1)  # trailer trajectory query
        self._query_splits.append(config.num_bounding_boxes)  # agent queries

        self._config = config
        self._backbone = TransfuserBackbone(config)

        # keyval = (lidar BEV tokens after /32 stem) + 1 status token.
        # lidar BEV tokens = lidar_vert_anchors * lidar_horz_anchors. v3/v4
        # use 8x8 = 64; range-widened configs (e.g. x in [-32, 48]) produce
        # different grids and the embedding length must follow.
        n_keyval = config.lidar_vert_anchors * config.lidar_horz_anchors + 1
        self._keyval_embedding = nn.Embedding(n_keyval, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)

        # Status encoder input is concatenated from optional channels:
        #   - ego status (vx, vy, ax, ay) = 4 dims (toggle: use_ego_status)
        #   - driving_command one-hot      = 3 dims (toggle: use_driving_command)
        # VAD §4.2 drops ego status to avoid shortcut learning. At least
        # one must be on so the status token has content.
        status_dim = (4 if config.use_ego_status else 0) + (
            3 if config.use_driving_command else 0
        )
        assert status_dim > 0, (
            "Either use_ego_status or use_driving_command must be True; "
            "otherwise the status token has no content."
        )
        self._status_encoding = nn.Linear(status_dim, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        # Trailer trajectory head -- same architecture as the truck head
        # but a separate parameter copy so the two trajectories are
        # learned independently. Skipping the build when
        # use_trailer_head=False genuinely reduces model capacity (strict
        # truck-only ablation).
        if config.use_trailer_head:
            self._trailer_trajectory_head = TrajectoryHead(
                num_poses=config.num_poses,
                d_ffn=config.tf_d_ffn,
                d_model=config.tf_d_model,
            )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]

        # Build the status encoder input from optional channels. The
        # status_dropout below only masks the ego-status channels --
        # driving_command is a navigation signal (not a leakage target)
        # and must always pass through unmasked.
        ego_status: torch.Tensor = features["status_feature"]  # (B, 4)
        batch_size = ego_status.shape[0]

        # Status dropout -- training only, sample-level zero-out of ego
        # status. Forces image/lidar branches to carry signal (vx*dt is a
        # trivial trajectory mapping otherwise). Masked zeros also appear
        # in training distribution so inference behaves consistently.
        if (
            self._config.use_ego_status
            and self.training
            and self._config.status_dropout_p > 0.0
        ):
            keep_mask = (
                torch.rand(batch_size, 1, device=ego_status.device)
                > self._config.status_dropout_p
            ).to(ego_status.dtype)
            ego_status = ego_status * keep_mask

        parts = []
        if self._config.use_ego_status:
            parts.append(ego_status)
        if self._config.use_driving_command:
            parts.append(features["driving_command"])
        status_feature = torch.cat(parts, dim=-1)

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        # Query split order: [truck, (optional) trailer, agents].
        splits = query_out.split(self._query_splits, dim=1)
        if self._config.use_trailer_head:
            trajectory_query, trailer_query, agents_query = splits
        else:
            trajectory_query, agents_query = splits
            trailer_query = None

        output: Dict[str, torch.Tensor] = {}
        # bev_semantic_map is consumed by the loss only when
        # bev_semantic_weight > 0. Skip the head call when disabled to
        # avoid the unused tensor (and the small wasted forward cost).
        if self._config.bev_semantic_weight > 0:
            output["bev_semantic_map"] = self._bev_semantic_head(bev_feature_upscale)

        # Truck trajectory under the "trajectory" key (navsim convention).
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)

        # Trailer trajectory under "trailer_trajectory" -- only present
        # when the head is built. Downstream loss/eval guards on key
        # existence so absence is silently skipped.
        if self._config.use_trailer_head:
            trailer_pred = self._trailer_trajectory_head(trailer_query)
            output["trailer_trajectory"] = trailer_pred["trajectory"]

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head -- column layout = BoundingBox2DIndex."""

    def __init__(self, num_agents: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )
        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )
        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)
        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    """SE(2) trajectory regression head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}
