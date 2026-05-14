"""Minimal `AbstractMap` stub so navsim's `Scene` can be constructed without
HD-map data.

navsim's `Scene.map_api: AbstractMap` field is non-Optional (see
navsim/common/dataclasses.py:333-339). TruckScenes has no HD map (no lane
graph, no drivable polygons -- see project_map_free_pdms memory). To keep
the upstream `Scene` dataclass unmodified we plug in a `NullMap` that
satisfies the type contract.

Safety: this object is only ever instantiated for TruckScenes scenes. Any
PDMS code path that *would* hit it has been forked elsewhere
(navsim/planning/metric_caching/truck_*.py and the truck PDMS scorer fork)
so the AbstractMap methods below should never be invoked in practice. They
raise `NotImplementedError` defensively rather than returning empty data,
to make it loud if a map-using code path slips through.
"""
from nuplan.common.maps.abstract_map import AbstractMap


class NullMap(AbstractMap):
    """AbstractMap stub for HD-map-less datasets (TruckScenes).

    All map queries raise. If you hit one of these, either the call site
    needs to be guarded (`isinstance(map_api, NullMap)`) or the code path
    needs to be moved to a TruckScenes-specific fork that does not consume
    the map.
    """

    @property
    def map_name(self) -> str:
        return "no_map"

    # TODO: enumerate the AbstractMap abstract methods that need to be
    # implemented to make this class instantiable. nuplan's AbstractMap has
    # ~15 abstract methods (get_proximal_map_objects, get_map_object, etc.).
    # The simplest path is to implement each as
    #     raise NotImplementedError(f"{type(self).__name__}.{<name>}: TruckScenes has no HD map")
    # Concrete enumeration deferred to first integration smoke test, where
    # Python will list every still-abstract method on instantiation.
