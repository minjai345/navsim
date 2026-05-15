"""Truck TransFuser agent (AbstractAgent wrapper).

Thin wrapper that plugs `TruckTransfuserModel` + `truck_transfuser_loss` +
`TruckTransfuser{Feature,Target}Builder` into navsim's `AbstractAgent`
interface. Mirrors `navsim/agents/transfuser/transfuser_agent.py`
exactly except for the truck-specific deltas:

  - Loss returns `(scalar, components)` -- we forward only the scalar to
    Lightning here. Component metrics (truck_l1, agent_cls, agent_box,
    trailer_l1, bev_semantic) need a callback or a training_step
    override to surface in W&B. Deferred (no callback yet).
  - `get_training_callbacks()` returns `[]`. The vanilla TransfuserCallback
    is heavily tied to BEV-semantic visualization (no use under our
    no-HD-map setting) and does not know about the trailer trajectory
    head. A truck-specific callback (4-cam stitched + LiDAR BEV +
    truck/trailer trajectory + agent boxes overlay) is a follow-up task.
  - `get_sensor_config()` loads ALL 8 navsim cam slots at the latest
    history frame; only cam_l0/r0/l2/r2 actually carry images for
    TruckScenes (the other 4 slots stay empty `Camera()`). This is the
    same `[3]` convention vanilla uses -- index 3 = last of 4 history
    frames.
  - Optimizer hard-coded to Adam with `lr` to match the vanilla
    interface. Experiment configs that need AdamW + weight_decay +
    warmup (e.g. transfuser-truckscenes v6_lr_schedule onward) can
    override via Lightning trainer wiring once that lands; we do not
    plumb optimizer choice through TruckTransfuserConfig because navsim
    treats training hyperparams as Lightning-side concerns.
"""
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.truck_transfuser.truck_transfuser_config import TruckTransfuserConfig
from navsim.agents.truck_transfuser.truck_transfuser_features import (
    TruckTransfuserFeatureBuilder,
    TruckTransfuserTargetBuilder,
)
from navsim.agents.truck_transfuser.truck_transfuser_loss import truck_transfuser_loss
from navsim.agents.truck_transfuser.truck_transfuser_model import TruckTransfuserModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class TruckTransfuserAgent(AbstractAgent):
    """TransFuser agent specialized for MAN TruckScenes."""

    def __init__(
        self,
        config: TruckTransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        :param config: TruckTransfuser model config.
        :param lr: Adam learning rate.
        :param checkpoint_path: optional .ckpt path to restore from.
        """
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._truck_transfuser_model = TruckTransfuserModel(config)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """Restore weights from `checkpoint_path` if it was supplied.

        Lightning checkpoint state dicts prefix keys with `agent.`; we
        strip that so the inner model accepts them. CPU fallback matches
        vanilla TransfuserAgent.
        """
        if self._checkpoint_path is None:
            return
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Enable only the 4 cam slots TruckScenes populates + lidar.

        navsim's `Cameras.from_camera_dict` tries to subscript
        `camera_dict[name]["data_path"]` whenever the slot is in the
        enabled list. Empty TruckScenes slots are emitted as None
        entries, so enabling ALL eight slots (the vanilla
        `build_all_sensors([3])` pattern) raises
        `TypeError: 'NoneType' object is not subscriptable` on the
        first None slot. Enable only the 4 cams the truck adapter
        actually populates -- the remaining 4 are passed through as
        empty `Camera()` instances via the else branch in
        from_camera_dict.

        `[3]` assumes 4 history frames (navsim default). When we expose
        num_history_frames on the config, derive the iteration list
        from it.
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
        return [TruckTransfuserFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TruckTransfuserTargetBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._truck_transfuser_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns scalar loss only.

        `truck_transfuser_loss` actually returns `(scalar, components)`;
        we drop components here because `AbstractAgent.compute_loss`'s
        return type is a single tensor. A callback / training_step
        override is the natural place to surface per-term curves.
        """
        loss, _components = truck_transfuser_loss(targets, predictions, self._config)
        return loss

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._truck_transfuser_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Empty until the truck-specific viz callback lands.

        See module docstring -- vanilla TransfuserCallback is BEV-semantic
        heavy and does not know about the trailer trajectory head.
        """
        return []
