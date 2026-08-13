"""
Deprecated home of DASCore's models; import from [dascore.models](`dascore.models`).

Everything here is re-exported from its new home so out-of-tree readers which
import from this path keep working.
"""

from __future__ import annotations

from pydantic import BaseModel

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
    "BaseModel",
    "CommaSeparatedStr",
    "DTypeLike",
    "DascoreBaseModel",
    "DateTime64",
    "FiniteFloat",
    "FrozenDictType",
    "InventoryModel",
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
