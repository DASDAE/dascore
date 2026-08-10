"""Utilities for models."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    model_validator,
)
from typing_extensions import Self

from dascore.compat import array, is_array_like
from dascore.exceptions import InvalidInventoryError
from dascore.units import Quantity, get_quantity, get_quantity_str
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import _all_null, all_close, to_str, unbyte
from dascore.utils.time import to_datetime64, to_timedelta64

# --- A list of custom types with appropriate serialization/deserialization
# these can just be use with pydantic type-hints.

frozen_dict_validator = PlainValidator(lambda x: FrozenDict(x))
frozen_dict_serializer = PlainSerializer(lambda x: dict(x))

# A datetime64
DateTime64 = Annotated[
    np.datetime64,
    PlainValidator(to_datetime64),
    PlainSerializer(to_str, when_used="json"),  # getting undefined name
]

TimeDelta64 = Annotated[
    np.timedelta64,
    PlainValidator(to_timedelta64),
    PlainSerializer(to_str, when_used="json"),  # getting undefined name
]

# The validator may preserve non-numpy array-likes (see compat.array), but
# ndarray is deliberately the single static face of array values; a structural
# protocol is not worth the complexity it spreads through every signature.
ArrayLike = Annotated[
    np.ndarray,
    PlainValidator(array),
]

DTypeLike = Annotated[
    str,
    PlainValidator(np.dtype),
]

UnitQuantity = Annotated[
    Quantity | str | None,
    PlainValidator(get_quantity),
    PlainSerializer(get_quantity_str),
]

CommaSeparatedStr = Annotated[
    str, PlainValidator(lambda x: x if isinstance(x, str) else ",".join(x))
]

FrozenDictType = Annotated[
    FrozenDict,
    frozen_dict_validator,
    frozen_dict_serializer,
]

UTF8Str = Annotated[str, PlainValidator(unbyte)]

# A positive (> 0) integer.
PositiveInt = Annotated[int, Field(gt=0)]

# A positive (> 0), finite (no nan/inf) float.
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def sensible_model_equals(self: BaseModel | Mapping, other: object) -> bool:
    """Custom equality to not compare private attrs and handle numpy arrays."""
    d1 = self.model_dump() if isinstance(self, BaseModel) else self
    if isinstance(other, BaseModel):
        d2 = other.model_dump()
    elif isinstance(other, Mapping):
        d2 = other
    else:  # nothing else can carry the same fields
        return NotImplemented
    if not set(d1) == set(d2):  # different keys, not equal
        return False
    for name in set(x for x in d1 if not x.startswith("_")):
        # skip any private attributes.
        if not values_equal(d1[name], d2[name]):
            return False
    return True


def values_equal(val1, val2) -> bool:
    """Recursively compare dumped values; nulls are equal only to nulls."""
    if is_array_like(val1) or is_array_like(val2):
        arr1, arr2 = np.asarray(val1), np.asarray(val2)
        if arr1.shape != arr2.shape:
            return False
        if not np.array_equal(pd.isnull(arr1), pd.isnull(arr2)):
            return False
        return bool(all_close(arr1, arr2))
    if isinstance(val1, Mapping) and isinstance(val2, Mapping):
        if set(val1) != set(val2):
            return False
        return all(values_equal(val1[key], val2[key]) for key in val1)
    if isinstance(val1, list | tuple) and isinstance(val2, list | tuple):
        if len(val1) != len(val2):
            return False
        return all(values_equal(v1, v2) for v1, v2 in zip(val1, val2))
    return bool(val1 == val2 or (_all_null(val1) and _all_null(val2)))


class DascoreBaseModel(BaseModel):
    """A base model with sensible configurations."""

    _cache = {}

    model_config = ConfigDict(
        extra="ignore",  # TODO: change to raise, then let subclass overwrite
        validate_assignment=True,
        ignored_types=(cached_property,),
        frozen=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def new(self, **kwargs) -> Self:
        """Create new instance with some attributed updated."""
        out = self.model_dump(exclude_unset=True)
        out.update(kwargs)
        return self.__class__(**out)

    @classmethod
    def get_summary_df(cls):
        """Get dataframe of attributes and descriptions for display."""
        fields = cls.model_fields
        names_desc = {
            i: v.description
            for i, v in fields.items()
            if getattr(v, "description", False)
        }
        out = pd.Series(names_desc).to_frame(name="description")
        out.index.name = "attribute"
        return out

    __eq__ = sensible_model_equals


class InventoryModel(DascoreBaseModel):
    """
    Base class for immutable DASDAE inventory objects.

    Every inventory object carries two uniform attachment points for
    information the model does not otherwise represent: ``description``
    (free prose for humans, matching StationXML's Description element)
    and ``extra_fields`` (typed key-values, e.g. for round-tripping
    unmodeled metadata from external formats).
    """

    description: str = Field(default="", description="Free-text description.")
    extra_fields: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Extra metadata not represented by standardized fields.",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def new(self, **kwargs) -> Self:
        """
        Create a new instance with some attributes updated.

        Dumps all fields (not just the set ones) so union discriminators and
        validator-normalized fields survive reconstruction.
        """
        out = self.model_dump()
        out.update(kwargs)
        return self.__class__(**out)


class TimeRangedModel(InventoryModel):
    """Base class for inventory objects with time-validity epochs.

    Validity intervals are half-open, ``[start_time, end_time)``; an unset
    (NaT) end time means the epoch is ongoing. All times are UTC.
    """

    start_time: DateTime64 = Field(
        default=np.datetime64("NaT", "ns"),
        description="Start time for which this metadata item is valid (UTC).",
    )
    end_time: DateTime64 = Field(
        default=np.datetime64("NaT", "ns"),
        description=(
            "End time for which this metadata item is valid (UTC); NaT while ongoing."
        ),
    )

    @model_validator(mode="after")
    def _check_time_order(self):
        """A set end time must follow the start time."""
        start, end = self.start_time, self.end_time
        if not pd.isnull(start) and not pd.isnull(end) and end <= start:
            msg = f"end_time {end} must be after start_time {start}."
            raise InvalidInventoryError(msg)
        return self

    def is_effective_at(self, time) -> bool:
        """Return True if this epoch is valid at the supplied time (half-open)."""
        time = to_datetime64(time)
        if pd.isnull(time):
            return True
        start = self.start_time
        end = self.end_time
        after_start = pd.isnull(start) or start <= time
        before_end = pd.isnull(end) or time < end
        return bool(after_start and before_end)

    def overlaps(self, other: TimeRangedModel) -> bool:
        """
        Return True if two half-open validity intervals overlap.

        Unset (NaT) starts are unbounded past; unset ends are ongoing.
        """
        s1, e1, s2, e2 = (
            self.start_time,
            self.end_time,
            other.start_time,
            other.end_time,
        )
        first_starts_before = pd.isnull(e2) or pd.isnull(s1) or s1 < e2
        second_starts_before = pd.isnull(e1) or pd.isnull(s2) or s2 < e1
        return bool(first_starts_before and second_starts_before)
