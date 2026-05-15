"""HD-map-less stub plugged into `Scene.map_api` for TruckScenes scenes.

`Scene.map_api: AbstractMap` is non-Optional (see
navsim/common/dataclasses.py). TruckScenes has no HD map (see
project_map_free_pdms memory), so the truck adapter routes the
`map_location="no_map"` case through the forked `Scene._build_map_api`
to return this stub.

Design choice: `NullMap` does NOT inherit from `AbstractMap`.

  Why: `AbstractMap` has ~15 abstract methods. Subclassing forces every
  method to be overridden just to instantiate (ABCMeta enforcement).
  Python's dataclass / type-hint system is *not* enforced at runtime, so
  setting `Scene.map_api = NullMap()` works as long as no code path
  actually queries the map. For TruckScenes that path is guaranteed
  silent: the truck FeatureBuilder / TargetBuilder never touch
  `scene.map_api`, and the forked truck PDMS scorer (when written) will
  skip every metric that needs an `AbstractMap`.

If a code path that *was* expected to skip the map ever does try to
query NullMap, AttributeError fires loudly -- which is the desired
fail-fast behavior.
"""


class NullMap:
    """No-HD-map placeholder for TruckScenes scenes."""

    map_name = "no_map"

    def __repr__(self) -> str:  # pragma: no cover -- trivial
        return "NullMap()"
