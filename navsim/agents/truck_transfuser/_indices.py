"""Column-layout enums for truck TransFuser tensors.

Kept in a separate private module so importing it does NOT pull in
`navsim.common.dataclasses` (which transitively imports nuplan-devkit).
This lets `truck_transfuser_model.py` and `truck_transfuser_loss.py`
stay nuplan-free for unit-level smoke tests in environments that ship
only torch + timm + scipy (see project_b200_training memory).

`truck_transfuser_features.py` re-exports `BoundingBox2DIndex` from here
for the heavier FeatureBuilder / TargetBuilder code path, which already
needs nuplan via AgentInput / Scene.
"""
from enum import IntEnum


class BoundingBox2DIndex(IntEnum):
    """Column layout for agent bounding box tensors (x, y, heading, length, width)."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
