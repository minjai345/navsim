"""Truck-adapted V2TransfuserModel for DiffusionDrive.

The upstream V2TransfuserModel hardcodes
`self._status_encoding = nn.Linear(4 + 2 + 2, tf_d_model)`, expecting
status_feature to concatenate (driving_command 4D, ego_velocity 2D,
ego_acceleration 2D). Our truck variant uses the VAD §4.2 shortcut-free
protocol -- ego status is omitted and driving_command is 3D
(TruckScenes adapter convention). To plug in cleanly, this subclass
re-creates `self._status_encoding` as `Linear(3, tf_d_model)` right
after the parent constructor runs. The original 8D weights are
discarded (a few hundred parameters, negligible).

Every other component (TransfuserBackbone, diffusion decoder,
plan_anchor handling, agent head, bev_semantic head) is inherited
unchanged from the upstream V2TransfuserModel.

Decision rationale: see project_paper_deadline / project_map_free_pdms
memory. Both TruckTransfuser and TruckDiffusionDrive baselines must
share the SAME 3D driving_command input under the SAME shortcut-free
regime so the paper's articulation PDMS comparison is on equal
footing.
"""
from typing import Dict

import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from navsim.agents.truck_diffusiondrive.truck_diffusiondrive_config import (
    TruckDiffusionDriveConfig,
)


class TruckV2TransfuserModel(V2TransfuserModel):
    """V2TransfuserModel patched for the truck setup.

    Two layers re-created after `super().__init__`:

      1. `_status_encoding`: upstream is `Linear(4 + 2 + 2, d_model)`
         for nuPlan's (driving_command 4D, vel 2D, accel 2D) layout.
         Truck protocol uses driving_command (3D) only -- the VAD §4.2
         shortcut-free regime, see truck_diffusiondrive_features.py.
         Replaced with `Linear(3, d_model)`.
      2. `_keyval_embedding`: upstream hardcodes `Embedding(8**2 + 1, ...)`
         for vanilla 8x8 BEV grid + 1 status token. v9 truck baseline
         widens the forward BEV to (-32, +48), so the post-/32-stem BEV
         is 10x8 = 80 tokens + 1 status = 81. Replaced to compute the
         grid size from config (matches TruckTransfuserModel's pattern).
    """

    def __init__(self, config: TruckDiffusionDriveConfig):
        super().__init__(config)
        self._status_encoding = nn.Linear(3, config.tf_d_model)
        n_keyval = config.lidar_vert_anchors * config.lidar_horz_anchors + 1
        self._keyval_embedding = nn.Embedding(n_keyval, config.tf_d_model)
