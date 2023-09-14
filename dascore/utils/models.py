"""Utilities for models."""
from __future__ import annotations

from functools import cached_property
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator
from typing_extensions import Self

from dascore.compat import array
from dascore.units import Quantity, get_quantity, get_quantity_str
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    all_close,
    to_str,
    unbyte,
)
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


def sensible_model_equals(self, other):
    """Custom equality to not compare private attrs and handle numpy arrays."""
    try:
        d1, d2 = self.model_dump(), other.model_dump()
    except (TypeError, ValueError, AttributeError):
        return False
    if not set(d1) == set(d2):  # different keys, not equal
        return False
    for name, val1 in d1.items():
        # skip any private attributes.
        if name.startswith("_"):
            continue
        val2 = d2[name]
        if isinstance(val1, np.ndarray):
            if not all_close(val1, val2):
                return False
        else:
            if not val1 == val2:
                return False
    return True


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
        out = dict(self)
        for item, value in kwargs.items():
            out[item] = value
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
