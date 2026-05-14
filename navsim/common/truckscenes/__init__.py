"""TruckScenes adapter for navsim.

Bridges MAN TruckScenes (devkit-loaded) into navsim's Scene / AgentInput / pkl
log format. Two entry points share a single conversion core:

    scene_dict_from_sample()         <- single source of truth
        |
        +-- TruckScenesSceneLoader   (runtime, Path B: training / inference)
        |
        +-- build_navsim_logs        (batch, Path A: PDMS metric-cache pipeline)

Design notes are in this package's docstrings; longer rationale lives in the
`project_map_free_pdms` and `project_dual_repo_structure` memory entries.
"""
