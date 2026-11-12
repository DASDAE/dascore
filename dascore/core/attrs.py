"""Pydantic schemas used by DASCore."""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Mapping
from contextlib import suppress
from typing import Annotated, Any, Self, cast

import numpy as np
import pandas as pd
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    PlainValidator,
    model_validator,
)

from dascore.constants import (
    DataCategory,
    DataType,
    max_lens,
)
from dascore.models import DascoreBaseModel, UnitQuantity
from dascore.units import Quantity, Unit
from dascore.utils.misc import (
    to_str,
    unbyte,
    validate_acquisition_key,
)
from dascore.utils.time import numpy_time

str_validator = PlainValidator(to_str)
_ARRAYS = frozenset({np.ndarray, list, tuple})
# What an extra attr holds, with None and a scalar quantity; see the
# PatchAttrs note. bool is an int, and np.timedelta64 an np.integer.
_HELD = (str, int, float, np.bool_, np.integer, np.floating, np.datetime64, Unit)
_INT64 = range(-(2**63), 2**64)
_PLAIN = frozenset({str, float, bool, type(None), np.float64, np.datetime64})


class _ArrayAttrError(ValueError):
    """An attr holds several values, which belong on a coordinate."""


def _held_value(name: str, value: Any) -> Any:
    """Return an extra attr's value as a type an attr holds, or raise."""
    if type(value) in _ARRAYS:
        # As objects, so a ragged list counts and a python scalar stays one.
        value = np.asarray(value, None if type(value) is np.ndarray else object)
        if value.size != 1:
            raise _ArrayAttrError("an array belongs on a coordinate.")
        value = value.reshape(-1)[0]  # how HDF5 and netCDF spell a scalar
    if getattr(value, "tzinfo", None) is not None:
        msg = f"The attr {name!r} holds a time-zone-aware time; convert it to UTC."
        raise ValueError(msg)
    if isinstance(value, datetime.date | datetime.timedelta):  # pd.NaT included
        value = numpy_time(value)
    elif isinstance(value, bytes | np.str_):  # HDF5 refuses an np.str_
        with suppress(UnicodeDecodeError):
            value = str(unbyte(value))
    elif value is pd.NA:
        value = None
    held = isinstance(value, _HELD) and (type(value) is not int or value in _INT64)
    if held or value is None:
        return value
    if isinstance(value, Quantity) and np.isrealobj(value.m) and not np.ndim(value.m):
        return value
    msg = (
        f"The attr {name!r} holds a {type(value).__qualname__!r}; an attr holds "
        "a string, real number, bool, None, numpy time, or scalar quantity or "
        "unit (see the PatchAttrs note)."
    )
    raise ValueError(msg)


def _hold_extras(attrs: Mapping, fields, skipped: type[ValueError]) -> dict:
    """Return attrs with each extra one held, warning for each skipped."""
    out = dict(attrs)
    # A value of a plain type is held as it is; the common case, so fast.
    odd = [k for k, v in out.items() if type(v) not in _PLAIN and k not in fields]
    for name in odd:
        try:
            out[name] = _held_value(name, out[name])
        except skipped as error:
            warnings.warn(f"Skipped attr {name!r}: {error}", UserWarning, stacklevel=3)
            del out[name]
    return out


def drop_unheld_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return attrs without the extra ones `PatchAttrs` refuses, warning for each."""
    return _hold_extras(attrs, PatchAttrs.model_fields, ValueError)


# Each 0.1.x coordinate attr suffix, and how its coordinate now gives it.
_COORD_FIELD_SUFFIXES = {
    "_min": "min()",
    "_max": "max()",
    "_step": "step",
    "_units": "units",
}


def _coordinate_field_hint(name: str) -> str:
    """
    A hint for an attr name which 0.1.x derived from the coordinates.

    `time_min`, `distance_step` and the like, and `coords` and `dims`, were
    attrs before 1.0 and now live on the patch; an empty string otherwise.
    """
    if name in ("coords", "dims"):
        return f"Coordinate metadata moved off the attrs: use `patch.{name}`."
    for suffix, call in _COORD_FIELD_SUFFIXES.items():
        if name.endswith(suffix) and (dim := name[: -len(suffix)]):
            return (
                f"Coordinate metadata moved off the attrs: use "
                f"`patch.get_coord({dim!r}).{call}`."
            )
    return ""


class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    `PatchAttrs` stores non-structural metadata. Nested coordinate payloads in
    `coords` are rejected, while flat coord-like keys are treated like any
    other extra attrs. `dims` is ignored during normalization.

    The default attributes are:
    ```{python}
    #| echo: false

    import dascore as dc
    from IPython.display import Markdown

    df_str = (
        dc.PatchAttrs.get_summary_df()
        .reset_index()
        .to_markdown(index=False, stralign="center")
    )
    Markdown(df_str)
    ```
    """

    model_config = ConfigDict(
        title="Patch Summary",
        extra="allow",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    data_type: Annotated[DataType, str_validator] = Field(
        description="Describes the quantity being measured.", default=""
    )
    data_category: Annotated[DataCategory, str_validator] = Field(
        description="Describes the type of data.",
        default="",
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="The units of the data measurements"
    )
    acquisition_key: Annotated[str, AfterValidator(validate_acquisition_key)] = Field(
        default="",
        max_length=max_lens["acquisition_key"],
        description=(
            "Inventory identity of the data source, spelled "
            "network.fiber_array.location.acquisition."
        ),
    )
    tag: str = Field(
        default="", max_length=max_lens["tag"], description="A custom string field."
    )
    history: str | tuple[str, ...] = Field(
        default_factory=tuple,
        description="A list of processing performed on the patch.",
    )
    origin_id: str = Field(
        default="",
        description=(
            "Identifies which stored data this came from. It survives every "
            "operation, and changes only when data from more than one "
            "origin is combined."
        ),
    )
    data_id: str = Field(
        default="",
        description=(
            "Identifies which array this is: its origin's id until an "
            "operation runs, then an id derived from the inputs' ids and "
            "the operation, so equal ids mean the same route from the same data."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def reject_coordinate_attributes(cls, data: Any) -> Any:
        """Reject nested coord payloads and ignore structural dims input."""
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        if "coords" in data and not isinstance(data["coords"], str):
            msg = "PatchAttrs no longer accepts coordinate metadata. Received: coords."
            raise ValueError(msg)
        data.pop("dims", None)
        # What the two ids were called before they were renamed.
        for old, new in (("patch_id", "origin_id"), ("processing_id", "data_id")):
            if value := data.pop(old, None):
                data.setdefault(new, value)
        return _hold_extras(data, cls.model_fields, _ArrayAttrError)

    def __getitem__(self, item):
        return getattr(self, item)

    def __getattr__(self, item: str):
        # pydantic's __getattr__ serves extras; a miss on a name 0.1.x
        # derived from the coordinates gets a pointer to where it went.
        try:
            # Looked up dynamically: pydantic defines it outside type checking.
            return getattr(super(), "__getattr__")(item)
        except AttributeError as exc:
            if hint := _coordinate_field_hint(item):
                raise AttributeError(f"{exc} {hint}") from None
            raise

    def __len__(self):
        return len(self.model_dump())

    def get(self, item, default=None):
        """dict-like get method."""
        try:
            return self[item]
        except (AttributeError, KeyError):
            return default

    def items(self):
        """Yield (attribute, values) just like dict.items()."""
        yield from self.model_dump().items()

    @classmethod
    def from_dict(
        cls,
        attr_map: Mapping | PatchAttrs | None,
    ) -> Self:
        """
        Get a new instance of the PatchAttrs.

        Parameters
        ----------
        attr_map
            Anything convertible to a dict that contains attr info. `dims`
            entries are ignored during normalization.
        """
        if isinstance(attr_map, cls):
            return attr_map
        if attr_map is None:
            out = {}
        elif callable(model_dump := getattr(attr_map, "model_dump", None)):
            out = model_dump()
        else:
            out = attr_map
        if isinstance(out, Mapping):
            out = dict(out)
            out.pop("dims", None)
        # Anything else may still be unpackable -- a pandas Series, say --
        # and the constructor has always been what rejects the rest.
        return cls(**cast("Mapping[str, Any]", out))

    def update(self, **kwargs) -> Self:
        """Update an attribute in the model, return new model."""
        out = self.model_dump(exclude_unset=True)
        out.update(kwargs)
        return self.from_dict(out)

    def drop(self, *args):
        """Drop specific keys if they exist."""
        contents = dict(self)
        ok_to_keep = set(contents) - set(args)
        out = {i: v for i, v in contents.items() if i in ok_to_keep}
        return self.__class__(**out)

    def drop_private(self) -> Self:
        """Drop all private attributes."""
        contents = dict(self)
        out = {i: v for i, v in contents.items() if not i.startswith("_")}
        return self.__class__(**out)

    def flat_dump(self, exclude=None) -> dict:
        """Dump attrs to a flat dict."""
        return self.model_dump(exclude=exclude)
