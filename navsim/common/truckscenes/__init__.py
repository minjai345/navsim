"""TruckScenes adapter for navsim.

Bridges MAN TruckScenes (devkit-loaded) into navsim's scene_dict / Scene
log format via a single conversion core:

    scene_dict_from_sample()
        |
        +-- build_navsim_logs  (Path A: batch pickle dump)
                                Output pkls are consumed by navsim's stock
                                SceneLoader + MetricCacheProcessor + (in
                                the long run) the truck TransFuser agent
                                training pipeline.

The earlier design considered a runtime adapter (Path B) that bypassed pkl
materialization, but it was removed (commit deleting runtime_loader.py):
direction (ii) puts the canonical truck training stack in
`navsim/agents/truck_transfuser/`, which consumes pkls the same way every
other navsim agent does. A separate runtime data path would just
duplicate the surface.

Companion modules:
  - `null_map`            -- AbstractMap stub for HD-map-less Scene construction.
  - `camera_mapping`      -- 4 TruckScenes cameras -> navsim 8-slot SensorConfig.
  - `lidar_merge`         -- 6 LiDARs -> (6, N) [x, y, z, intensity, ring, lidar_id].
  - `trailer_extras`      -- per-frame truckscenes_extras side-channel
                             (hitch-corrected trailer pose, etc.).
  - `category_mapping`    -- TruckScenes vocabulary -> navsim categories.
  - `scene_dict_from_sample` -- composes the above into one frame dict.
  - `build_navsim_logs`   -- batch script: walk a TruckScenes split and
                             emit per-log pkls in navsim's log shape.

Design notes are in this package's docstrings; longer rationale lives in
the `project_map_free_pdms` / `project_dual_repo_structure` /
`project_b200_training` memory entries.
"""
