"""TruckScenes annotation category -> navsim/nuScenes-style category mapping.

TruckScenes annotates objects with a hierarchical string vocabulary
(`vehicle.car`, `vehicle.bus.bendy`, `vehicle.bus.rigid`, `vehicle.truck`,
`vehicle.trailer`, `vehicle.ego_trailer`, `human.pedestrian.*`, ...).

navsim's downstream consumers (especially the TransFuser detection target
builder and PDMS collision scorer) were written against nuScenes / nuPlan
category strings. To avoid forking every consumer's category-filter logic
we collapse the TruckScenes vocabulary to navsim's expected names here.

Two special cases:
  - `vehicle.ego_trailer` is NEVER passed through to `Annotations` -- it
    goes into `truckscenes_extras` (see `trailer_extras.py`). Filtered out
    before this mapping runs.
  - `vehicle.trailer` (foreign trailers attached to other trucks) IS a
    regular "other agent" and stays in annotations.
"""
from typing import Optional


# TODO: populate after enumerating actual TruckScenes category strings.
# Skeleton entries below cover the obvious cases; cross-check against
# truckscenes-devkit `category.json` and the categories the current
# transfuser-truckscenes pipeline already handles
# (dataset.py:_is_vehicle_category).
TRUCKSCENES_TO_NAVSIM_CATEGORY = {
    # Vehicles
    "vehicle.car": "vehicle.car",
    "vehicle.truck": "vehicle.truck",
    "vehicle.bus.rigid": "vehicle.bus",
    "vehicle.bus.bendy": "vehicle.bus",
    "vehicle.trailer": "vehicle.trailer",
    "vehicle.motorcycle": "vehicle.motorcycle",
    "vehicle.bicycle": "vehicle.bicycle",
    # Humans
    "human.pedestrian": "human.pedestrian",
    # TODO: enumerate full set
}


def map_truckscenes_category(name: str) -> Optional[str]:
    """Map a TruckScenes category to navsim's category vocabulary.

    Returns `None` if the category should be dropped (e.g. ego_trailer,
    static objects we don't track). Caller is responsible for filtering
    these out before constructing `Annotations`.
    """
    if name == "vehicle.ego_trailer":
        return None
    return TRUCKSCENES_TO_NAVSIM_CATEGORY.get(name)
