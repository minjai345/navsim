"""Import-time monkey-patch so navsim's viz code accepts TruckScenes names.

navsim's `tracked_object_types` dict (in
`navsim.planning.scenario_builder.navsim_scenario_utils`) maps bare
category names ("vehicle", "pedestrian", ...) to nuPlan's
`TrackedObjectType` enum. Our TruckScenes adapter emits hierarchical
names ("vehicle.car", "vehicle.truck", "human.pedestrian", ...) which
mirror the nuScenes vocabulary -- that's the right form for our own
training pipeline, but it crashes navsim's `add_annotations_to_bev_ax`
(and any other code that does `tracked_object_types[name]`) with a
KeyError.

Importing this module extends the dict with the TruckScenes hierarchy
in-place. Idempotent -- safe to import multiple times.

This is a viz-only band-aid; if/when navsim upstream gains a registry-
style category resolver we should drop this shim.
"""
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.planning.scenario_builder.navsim_scenario_utils import (
    tracked_object_types,
)

def _patch_add_lidar_to_bev_ax() -> None:
    """Replace navsim.visualization.bev.add_lidar_to_bev_ax with a simpler
    uniform-grey background renderer.

    The shipped version (1) computes a viridis colour-map from
    distance/intensity which dominates the figure, and (2) has a typo
    bug for `color_element="none"` (`np.uin8` instead of `np.uint8`)
    that AttributeErrors on call. Both are noisy for our paper figures
    where annotations + trajectories should be foreground. We swap in a
    quiet light-grey scatter with low alpha and zorder=1 (i.e. behind
    annotation boxes + trajectories).
    """
    from navsim.common.enums import LidarIndex
    from navsim.visualization import bev as _bev_mod
    from navsim.visualization.config import LIDAR_CONFIG

    def add_lidar_to_bev_ax_quiet(ax, lidar):
        pc = lidar.lidar_pc
        if pc is None or pc.shape[1] == 0:
            return ax
        # navsim's `filter_lidar_pc` clips by LIDAR_CONFIG x/y/z limits.
        # We bypass it -- the figure xlim/ylim from configure_bev_ax
        # already crop the visible area.
        ax.scatter(
            pc[LidarIndex.Y],
            pc[LidarIndex.X],
            c="#888888",
            alpha=LIDAR_CONFIG.get("alpha", 0.20),
            s=LIDAR_CONFIG.get("size", 0.05),
            zorder=LIDAR_CONFIG.get("zorder", 1),
            linewidths=0,
        )
        return ax

    _bev_mod.add_lidar_to_bev_ax = add_lidar_to_bev_ax_quiet


_patch_add_lidar_to_bev_ax()

_TRUCK_EXTRA = {
    # Vehicles -- all map to VEHICLE for viz colouring; for true class
    # separation, downstream code should consult the original string.
    "vehicle.car": TrackedObjectType.VEHICLE,
    "vehicle.truck": TrackedObjectType.VEHICLE,
    "vehicle.bus": TrackedObjectType.VEHICLE,
    "vehicle.trailer": TrackedObjectType.VEHICLE,
    "vehicle.motorcycle": TrackedObjectType.VEHICLE,
    "vehicle.bicycle": TrackedObjectType.BICYCLE,
    "vehicle.other": TrackedObjectType.GENERIC_OBJECT,
    # Pedestrians (collapsed from hierarchical sub-categories in
    # category_mapping.py).
    "human.pedestrian": TrackedObjectType.PEDESTRIAN,
    # Static / movable objects.
    "movable_object.barrier": TrackedObjectType.BARRIER,
    "movable_object.trafficcone": TrackedObjectType.TRAFFIC_CONE,
    "static_object.traffic_sign": TrackedObjectType.GENERIC_OBJECT,
    # Animals -- no first-class TrackedObjectType, route to GENERIC.
    "animal": TrackedObjectType.GENERIC_OBJECT,
}
tracked_object_types.update(_TRUCK_EXTRA)
