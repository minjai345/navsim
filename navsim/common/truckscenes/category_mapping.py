"""TruckScenes annotation category -> navsim/nuScenes-style category mapping.

TruckScenes annotates objects with a hierarchical string vocabulary
(`vehicle.car`, `vehicle.bus.bendy`, `vehicle.bus.rigid`, `vehicle.truck`,
`vehicle.trailer`, `vehicle.ego_trailer`, `human.pedestrian.adult`, ...).

The authoritative enumeration is in the official devkit at
truckscenes-devkit/src/truckscenes/eval/detection/utils.py:14-45
(`category_to_detection_name`). We mirror that mapping here, with two
adjustments for the navsim adapter:

  - `vehicle.ego_trailer` is dropped from `Annotations` entirely -- it
    lives in `truckscenes_extras` instead (see `trailer_extras.py`).
  - The mapping target uses navsim/nuScenes-style hierarchical names
    (`vehicle.car`, `human.pedestrian`, ...) rather than the flat
    detection-name vocabulary (`car`, `pedestrian`, ...). This keeps
    downstream navsim filters (which were written against the
    hierarchical form) working.
"""
from typing import Optional

# Source of truth: TruckScenes devkit
# `truckscenes.eval.detection.utils.category_to_detection_name`.
# We collapse detection-class subcategories back into navsim's hierarchical
# names so existing navsim category filters continue to work.
TRUCKSCENES_TO_NAVSIM_CATEGORY = {
    # Vehicles
    "vehicle.bicycle": "vehicle.bicycle",
    "vehicle.bus.bendy": "vehicle.bus",
    "vehicle.bus.rigid": "vehicle.bus",
    "vehicle.car": "vehicle.car",
    "vehicle.motorcycle": "vehicle.motorcycle",
    "vehicle.construction": "vehicle.other",
    "vehicle.other": "vehicle.other",
    "vehicle.trailer": "vehicle.trailer",
    "vehicle.truck": "vehicle.truck",
    # Humans
    "human.pedestrian.adult": "human.pedestrian",
    "human.pedestrian.child": "human.pedestrian",
    "human.pedestrian.construction_worker": "human.pedestrian",
    "human.pedestrian.police_officer": "human.pedestrian",
    # Static / movable objects
    "movable_object.barrier": "movable_object.barrier",
    "movable_object.trafficcone": "movable_object.trafficcone",
    "static_object.traffic_sign": "static_object.traffic_sign",
    # Animals
    "animal": "animal",
    # vehicle.ego_trailer is intentionally NOT here; it routes to
    # truckscenes_extras (see `map_truckscenes_category` for the drop logic).
}


def map_truckscenes_category(name: str) -> Optional[str]:
    """Map a TruckScenes category to navsim's category vocabulary.

    Returns `None` if the category should be dropped from `Annotations`
    (currently: `vehicle.ego_trailer`, and any category not enumerated
    above -- the caller is responsible for filtering these out before
    constructing `Annotations`).
    """
    if name == "vehicle.ego_trailer":
        # Routed to truckscenes_extras, not the regular agent stream.
        return None
    return TRUCKSCENES_TO_NAVSIM_CATEGORY.get(name)
