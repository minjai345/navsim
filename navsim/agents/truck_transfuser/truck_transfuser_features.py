"""Truck TransFuser feature / target builders (skeleton).

Full FeatureBuilder + TargetBuilder implementations land in a follow-up
commit (project task #15). This module currently only defines the
agent-local enums that `truck_transfuser_model.py` and
`truck_transfuser_loss.py` need to import.

The `BoundingBox2DIndex` definition mirrors
navsim/agents/transfuser/transfuser_features.py exactly so downstream
visualization and metric code that consumes either agent's outputs sees
the same column layout. We keep a local copy (rather than importing from
the vanilla transfuser agent) to avoid cross-agent coupling.
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
