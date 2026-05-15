"""Pytorch Lightning wrapper around `AbstractAgent`.

This file is a *minimal* fork of vanilla navsim's AgentLightningModule.
Two changes vs. upstream so DiffusionDrive (and the truck DiffusionDrive
fork) can plug in without requiring its own wrapper:

  1. `forward(features, targets)` -- targets are now passed alongside
     features. Diffusion-based agents need the target trajectory at
     train time to construct the noisy input for denoising. Agents that
     do not need targets at forward time MUST accept `targets=None` as a
     keyword argument and ignore it (vanilla navsim TransFuser /
     TruckTransfuser do this).
  2. `compute_loss` may return either a scalar tensor (vanilla
     TransFuser / TruckTransfuser) OR a dict (DiffusionDrive --
     individual component losses for separate logging). The dict path
     logs each entry separately and uses dict["loss"] as the
     backprop scalar.

Everything else (training_step / validation_step / configure_optimizers)
is unchanged from upstream.
"""
import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple, Union

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        # Always pass targets so diffusion-based agents work; agents that
        # do not need targets at forward time must accept and ignore it.
        prediction = self.agent.forward(features, targets=targets)
        loss = self.agent.compute_loss(features, targets, prediction)

        if isinstance(loss, dict):
            # DiffusionDrive convention: per-component losses logged
            # separately, dict["loss"] used for backprop.
            for name, value in loss.items():
                if value is None:
                    continue
                self.log(
                    f"{logging_prefix}/{name}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=len(batch[0]),
                )
            return loss["loss"]

        # Vanilla scalar return (TransFuser / TruckTransfuser).
        self.log(
            f"{logging_prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
