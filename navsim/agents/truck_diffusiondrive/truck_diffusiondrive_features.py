"""Feature / Target builders for the TruckScenes-adapted DiffusionDrive agent.

The feature side is identical to our TruckTransfuser pipeline -- same
4-cam stitched image, same LiDAR BEV histogram, same (vx, vy, ax, ay)
status, optional driving_command. So we reuse
`TruckTransfuserFeatureBuilder` unchanged.

The target side reuses `TruckTransfuserTargetBuilder` but adds a
zeros-filled `bev_semantic_map` so the upstream
`diffusiondrive.transfuser_loss` (which references that key
unconditionally) does not KeyError. With
`config.bev_semantic_weight=0.0`, the cross-entropy contribution to
total loss is exactly zero. The wasted CE compute is accepted in
exchange for not forking the upstream model / loss.

The trailer-trajectory target (`trailer_trajectory`, `trailer_mask`)
emitted by `TruckTransfuserTargetBuilder` is preserved here for
forward compatibility -- DiffusionDrive's upstream loss does NOT
consume it (no `trailer_weight` field), so it has no effect on
training right now. When we wire a trailer head into DiffusionDrive
later, the target is already in place.
"""
from typing import Dict

import numpy as np
import torch

from navsim.agents.truck_diffusiondrive.truck_diffusiondrive_config import (
    TruckDiffusionDriveConfig,
)
from navsim.agents.truck_transfuser.truck_transfuser_features import (
    TruckTransfuserFeatureBuilder,
    TruckTransfuserTargetBuilder,
)
from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class TruckDiffusionDriveFeatureBuilder(TruckTransfuserFeatureBuilder):
    """Identical features to the TruckTransfuser FeatureBuilder.

    Defined as a subclass for naming consistency only; downstream code
    (e.g. cache `get_unique_name` collisions) benefits from a distinct
    name when both agents are used in the same training pipeline.
    """

    def __init__(self, config: TruckDiffusionDriveConfig):
        super().__init__(config)

    def get_unique_name(self) -> str:
        return "truck_diffusiondrive_feature"


class TruckDiffusionDriveTargetBuilder(TruckTransfuserTargetBuilder):
    """Truck targets + a zeros-filled bev_semantic_map placeholder.

    The placeholder satisfies upstream `diffusiondrive.transfuser_loss`'s
    unconditional `targets["bev_semantic_map"]` lookup. Shape matches
    `config.bev_semantic_frame = (bev_pixel_height, bev_pixel_width)` and
    dtype is int64 (CE expects long class indices). All zeros = class 0
    everywhere; weight=0 in the loss kills its contribution.
    """

    def __init__(self, config: TruckDiffusionDriveConfig):
        super().__init__(config)

    def get_unique_name(self) -> str:
        return "truck_diffusiondrive_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        targets = super().compute_targets(scene)
        # Zeros placeholder for the BEV semantic head's CE loss.
        # config.bev_semantic_frame returns (bev_pixel_height, bev_pixel_width).
        h, w = self._config.bev_semantic_frame
        targets["bev_semantic_map"] = torch.tensor(np.zeros((h, w), dtype=np.int64))
        return targets
