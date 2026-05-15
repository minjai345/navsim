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
  - Optimizer / LR schedule are configurable via the agent's `__init__`
    so a v9-equivalent run can be expressed in the Hydra yaml. Matches
    the transfuser-truckscenes `_setup_optimizer_and_scheduler` shape
    exactly (AdamW or Adam + optional LinearLR warmup + CosineAnnealing,
    SequentialLR-combined). Hyperparam parity is required for the
    direction-(ii) conformance test (#18) -- comparing navsim-form
    loss curves against the prior v9_cmd_no_status_seed0 baseline is
    only meaningful if optimizer choice / weight_decay / warmup match.
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
        optimizer_type: str = "adam",
        weight_decay: float = 0.0,
        lr_warmup_epochs: int = 0,
        max_epochs: int = 100,
        checkpoint_path: Optional[str] = None,
    ):
        """
        :param config: TruckTransfuser model config.
        :param lr: initial learning rate.
        :param optimizer_type: "adam" (default, vanilla parity) or "adamw"
            (v9_cmd_no_status protocol). AdamW uses `weight_decay`; Adam
            also accepts it (0 default = identical to plain Adam).
        :param weight_decay: passed to (Adam)W. v9 protocol = 0.01.
        :param lr_warmup_epochs: epochs of linear warmup (start_factor=0.01).
            0 = no warmup, CosineAnnealing starts from epoch 0. v9 = 2.
        :param max_epochs: total training epochs (= CosineAnnealing T_max
            after subtracting warmup). MUST match the Lightning
            trainer's `max_epochs` -- caller responsibility.
        :param checkpoint_path: optional .ckpt path to restore from.
        """
        super().__init__()
        self._config = config
        self._lr = lr
        self._optimizer_type = optimizer_type
        self._weight_decay = weight_decay
        self._lr_warmup_epochs = lr_warmup_epochs
        self._max_epochs = max_epochs
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
        """Return Lightning-style {optimizer, lr_scheduler} dict.

        Mirrors transfuser-truckscenes/train.py:190-214 exactly so a v9
        config (`optimizer_type="adamw", weight_decay=0.01,
        lr_warmup_epochs=2, max_epochs=48`) reproduces the same schedule
        the prior baselines were trained with. With defaults
        (optimizer_type="adam", weight_decay=0.0, lr_warmup_epochs=0)
        the behavior matches vanilla navsim's TransfuserAgent.

        Schedule:
          - LinearLR warmup (start_factor=0.01) for lr_warmup_epochs.
          - CosineAnnealingLR for the remainder (T_max = max_epochs
            - lr_warmup_epochs).
          - SequentialLR combines the two at the warmup milestone.
          - When lr_warmup_epochs == 0, cosine annealing alone (T_max =
            max_epochs).
          - Scheduler steps per EPOCH, not per training batch -- matches
            transfuser-truckscenes/train.py and avoids needing a
            steps-per-epoch estimate at agent construction time.
        """
        params = self._truck_transfuser_model.parameters()
        if self._optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self._lr, weight_decay=self._weight_decay)
        elif self._optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(params, lr=self._lr, weight_decay=self._weight_decay)
        else:
            raise ValueError(
                f"Unknown optimizer_type {self._optimizer_type!r}; expected 'adam' or 'adamw'."
            )

        if self._lr_warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=self._lr_warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._max_epochs - self._lr_warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self._lr_warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._max_epochs
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Empty until the truck-specific viz callback lands.

        See module docstring -- vanilla TransfuserCallback is BEV-semantic
        heavy and does not know about the trailer trajectory head.
        """
        return []
