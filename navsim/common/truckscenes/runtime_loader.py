"""Path B (runtime adapter) -- build navsim Scene/AgentInput on the fly.

Wraps a `TruckScenes` devkit instance and exposes a `tokens` + `__getitem__`
surface mirroring navsim's `SceneLoader` (navsim/common/dataloader.py:110)
so navsim's training / inference dataloaders can consume TruckScenes
without ever touching the pkl log format.

For PDMS metric-caching, use the Path A batch script (`build_navsim_logs.py`)
to materialize pkl logs that navsim's stock SceneLoader + MetricCacheProcessor
can read. This loader is the training-side equivalent only.
"""
from pathlib import Path
from typing import List, Optional

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig

from navsim.common.truckscenes.scene_dict_from_sample import scene_dict_from_sample


class TruckScenesSceneLoader:
    """Runtime-only adapter: builds navsim dataclasses from a `TruckScenes`
    devkit instance, with no on-disk pkl intermediate.

    Mirrors the public surface of `navsim.common.dataloader.SceneLoader`
    (the parts the training pipeline actually consumes: `tokens`,
    `get_agent_input_from_token`, `get_scene_from_token`). We do not
    subclass the upstream SceneLoader because its `__init__` is built
    around pkl log loading; we want a sibling that satisfies the same duck
    type.

    Intentionally NOT supported here:
      - synthetic scenes (no second-stage data in TruckScenes)
      - metric cache loading (use Path A + MetricCacheLoader)
    """

    def __init__(
        self,
        ts,                                    # initialized TruckScenes devkit
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        num_history_frames: int = 4,
        num_future_frames: int = 8,
    ):
        """
        :param ts: TruckScenes devkit instance.
        :param scene_filter: forwarded for parity with SceneLoader API; only
            `tokens` / `log_names` fields are honored here.
        :param sensor_config: which cameras / lidar to actually load. We
            still produce empty `Camera()` placeholders for the 4 unused
            slots so the Cameras dataclass shape stays uniform.
        :param num_history_frames: history window length, used when
            assembling AgentInput.
        :param num_future_frames: future window length, used when
            assembling Scene's full frame list (for target builders).
        """
        self._ts = ts
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        self._num_history_frames = num_history_frames
        self._num_future_frames = num_future_frames

        # TODO: collect valid sample tokens (those with enough past + future
        # samples in the same scene). Same logic as
        # transfuser-truckscenes/dataset/dataset.py:_collect_valid_samples
        # but parameterized by both history and future windows.
        self._tokens: List[str] = []

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """Build an `AgentInput` (history-only, runtime-facing) for a token.

        TODO:
          - Walk back `num_history_frames - 1` samples from `token` via
            sample['prev'], collect history scene_dicts.
          - Hand the list to `AgentInput.from_scene_dict_list` (or our own
            assembly, since the upstream classmethod assumes sensor blobs
            live on disk under `sensor_blobs_path`). For Path B we want to
            assemble Cameras / Lidar / EgoStatus directly from the
            scene_dict so no disk write is needed -- consider adding a new
            `AgentInput.from_runtime_dict_list` classmethod, or building
            here directly.
        """
        raise NotImplementedError("TruckScenesSceneLoader.get_agent_input_from_token: skeleton")

    def get_scene_from_token(self, token: str) -> Scene:
        """Build a full `Scene` (history + future + annotations) for a token.

        Used by training (TargetBuilder) and any non-PDMS evaluation that
        runs against the runtime loader.

        TODO:
          - Assemble history + future frame lists.
          - Build per-frame `Frame` dataclasses including `truckscenes_extras`.
            navsim's current `Frame` dataclass does not have this field --
            decide: (a) add it via minimal fork of dataclasses.py, or (b)
            stash extras in a side dict keyed by frame token and have the
            truck FeatureBuilder look it up there. (a) is cleaner; (b)
            avoids fork. Recommend (a) once Path A is also being written.
          - Plug `NullMap()` for `Scene.map_api`.
        """
        raise NotImplementedError("TruckScenesSceneLoader.get_scene_from_token: skeleton")
