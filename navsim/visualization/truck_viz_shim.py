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
