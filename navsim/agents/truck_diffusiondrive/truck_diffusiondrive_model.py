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
    """V2TransfuserModel with a 3D-only status encoder."""

    def __init__(self, config: TruckDiffusionDriveConfig):
        super().__init__(config)
        # Replace status_encoding (Linear(8, d) -> Linear(3, d)) for the
        # truck VAD-style protocol: driving_command only (3D), no ego
        # status. status_feature shape from TruckDiffusionDriveFeatureBuilder
        # is (3,) per sample.
        self._status_encoding = nn.Linear(3, config.tf_d_model)
