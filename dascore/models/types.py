"""Annotated types with DASCore's validation and serialization attached."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, TypeVar

import numpy as np
from pydantic import (
    AfterValidator,
    Field,
    PlainSerializer,
    PlainValidator,
)

from dascore.compat import array
from dascore.units import Quantity, get_quantity, get_quantity_str
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import to_str, unbyte
from dascore.utils.time import to_datetime64, to_timedelta64

# --- A list of custom types with appropriate serialization/deserialization
# these can just be use with pydantic type-hints.

# Freezes without validating contents. Use FrozenDictType below when the
# declared value types must still be enforced.
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


def _to_unit_quantity(value):
    """Read units, refusing a quantity that carries many magnitudes."""
    out = get_quantity(value)
    try:
        # Passing a sequence makes pint build an array magnitude, which is
        # writable through the frozen model holding it. Asking whether it can
        # be hashed is the direct question; no real unit spelling fails it.
        hash(out)
    except TypeError:
        msg = f"Units must name a single unit, got {value!r}."
        raise ValueError(msg) from None
    return out


UnitQuantity = Annotated[
    Quantity | str | None,
    PlainValidator(_to_unit_quantity),
    PlainSerializer(get_quantity_str),
]

CommaSeparatedStr = Annotated[
    str, PlainValidator(lambda x: x if isinstance(x, str) else ",".join(x))
]

K = TypeVar("K")
V = TypeVar("V")

# Mapping, not dict: the runtime value is a FrozenDict, so a dict annotation
# would let type checkers pass writes that raise. AfterValidator, not
# PlainValidator: a plain validator would replace the declared value-type check.
FrozenDictType = Annotated[
    Mapping[K, V],
    AfterValidator(lambda x: FrozenDict(x)),
    PlainSerializer(dict),
]

UTF8Str = Annotated[str, PlainValidator(unbyte)]

# A positive (> 0) integer.
PositiveInt = Annotated[int, Field(gt=0)]

# A positive (> 0), finite (no nan/inf) float.
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
