"""TruckScenes-adapted DiffusionDrive agent.

Subclasses upstream `diffusiondrive.transfuser_agent.TransfuserAgent` and
overrides two pieces:

  1. `get_feature_builders` / `get_target_builders` -- swap in our
     TruckDiffusionDrive builders (4-cam stitch instead of vanilla 3-cam,
     plus zeros bev_semantic_map placeholder).
  2. `get_sensor_config` -- only the 4 cam slots TruckScenes actually
     populates (cam_l0/r0/l2/r2). Vanilla `build_all_sensors([3])`
     causes `Cameras.from_camera_dict` to subscript None on empty
     slots; same problem we hit in Phase C iter 2 with TruckTransfuser.

Everything else (model = `V2TransfuserModel`, loss =
`diffusiondrive.transfuser_loss`, optimizer = AdamW + CosLR via
upstream `WarmupCosLR`, forward signature `(features, targets=None)`,
diffusion process, plan_anchor handling, ...) is reused from upstream
without modification. The upstream `bev_semantic_map` head still emits
its output; `bev_semantic_weight=0` in our config kills its loss
contribution.
"""
from typing import List

from navsim.agents.diffusiondrive.transfuser_agent import (
    TransfuserAgent as DiffusionDriveTransfuserAgent,
)
from navsim.agents.truck_diffusiondrive.truck_diffusiondrive_config import (
    TruckDiffusionDriveConfig,
)
from navsim.agents.truck_diffusiondrive.truck_diffusiondrive_features import (
    TruckDiffusionDriveFeatureBuilder,
    TruckDiffusionDriveTargetBuilder,
)
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class TruckDiffusionDriveAgent(DiffusionDriveTransfuserAgent):
    """DiffusionDrive agent specialized for MAN TruckScenes."""

    def name(self) -> str:
        return self.__class__.__name__

    def get_sensor_config(self) -> SensorConfig:
        """Enable only the 4 cam slots TruckScenes populates + lidar.

        Same rationale as TruckTransfuserAgent.get_sensor_config:
        `Cameras.from_camera_dict` subscripts `camera_dict[name]`
        unconditionally when the slot is in `sensor_names`, so empty
        TruckScenes slots (None entries) crash with NoneType is not
        subscriptable. Listing only the 4 used slots avoids that path.

        `[3]` assumes 4 history frames (navsim default).
        """
        return SensorConfig(
            cam_f0=False,
            cam_l0=[3],
            cam_l1=False,
            cam_l2=[3],
            cam_r0=[3],
            cam_r1=False,
            cam_r2=[3],
            cam_b0=False,
            lidar_pc=[3],
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [TruckDiffusionDriveFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TruckDiffusionDriveTargetBuilder(config=self._config)]
