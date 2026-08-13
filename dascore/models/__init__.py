"""
DASCore's model layer: base classes and the annotated types they are built from.

Anything which inherits from [DascoreBaseModel](`dascore.models.base.DascoreBaseModel`)
can appear in a DASCore document. Models used only to shuttle values inside one
module are plain pydantic models instead.
"""

from __future__ import annotations

from dascore.models.base import (
    DascoreBaseModel,
    InventoryModel,
    TimeRangedModel,
    sensible_model_equals,
    sensible_model_hash,
    values_equal,
)
from dascore.models.types import (
    ArrayLike,
    CommaSeparatedStr,
    DateTime64,
    DTypeLike,
    FiniteFloat,
    FrozenDictType,
    OptionalFiniteFloat,
    PositiveFiniteFloat,
    PositiveInt,
    TimeDelta64,
    UnitQuantity,
    UTF8Str,
    frozen_dict_serializer,
    frozen_dict_validator,
)

__all__ = [
    "ArrayLike",
    "CommaSeparatedStr",
    "DTypeLike",
    "DascoreBaseModel",
    "DateTime64",
    "FiniteFloat",
    "FrozenDictType",
    "InventoryModel",
    "OptionalFiniteFloat",
    "PositiveFiniteFloat",
    "PositiveInt",
    "TimeDelta64",
    "TimeRangedModel",
    "UTF8Str",
    "UnitQuantity",
    "frozen_dict_serializer",
    "frozen_dict_validator",
    "sensible_model_equals",
    "sensible_model_hash",
    "values_equal",
]
