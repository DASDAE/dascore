"""Utilities for models."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator
from typing_extensions import Self

from dascore.compat import array, is_array
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


def sensible_model_equals(
    self: BaseModel | Mapping, other: BaseModel | Mapping
) -> bool:
    """Custom equality to not compare private attrs and handle numpy arrays."""
    d1 = self.model_dump() if hasattr(self, "model_dump") else self
    d2 = other.model_dump() if hasattr(other, "model_dump") else other
    if not set(d1) == set(d2):  # different keys, not equal
        return False
    for name in set(x for x in d1 if not x.startswith("_")):
        # skip any private attributes.
        val1, val2 = d1[name], d2[name]
        if is_array(val1):
            if not all_close(val1, val2):
                return False
        else:
            if val1 != val2 and not (_all_null(val1) and _all_null(val2)):
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
