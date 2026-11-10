"""Pydantic schemas used by DASCore."""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Mapping
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
# The types an extra attr may hold, besides None and a scalar quantity
# checked apart; see the PatchAttrs note.
_SIMPLE = (
    str,
    bool,
    int,
    float,
    np.bool_,
    np.number,
    np.datetime64,
    np.timedelta64,
    Unit,
)
_INT_RANGE = (np.iinfo(np.int64).min, np.iinfo(np.uint64).max)
_SKIPPED = object()


def _simple_attr_value(name: str, value: Any) -> Any:
    """
    Return an extra attr's value as one of the types an attr holds.

    Python and pandas times become numpy times of the same precision, text
    and decodable bytes become `str`, and `pd.NA` becomes None; any other
    type is refused.
    """
    if isinstance(value, complex | np.complexfloating):
        pass  # a number, but no format stores it
    elif isinstance(value, int) and not isinstance(value, bool):
        if not _INT_RANGE[0] <= value <= _INT_RANGE[1]:
            msg = f"The attr {name!r} holds {value}, which no 64-bit int holds."
            raise ValueError(msg)
        return value
    elif value is None or isinstance(value, _SIMPLE):
        # A numpy string is a str, but HDF5 refuses it as an attr.
        return str(value) if isinstance(value, np.str_) else value
    elif value is pd.NA:  # how a table spells unknown
        return None
    elif isinstance(value, datetime.datetime | datetime.date):  # pd.NaT included
        if getattr(value, "tzinfo", None) is not None:
            msg = (
                f"The attr {name!r} holds a time-zone-aware time, {value}. "
                "Convert it to UTC and drop the zone first."
            )
            raise ValueError(msg)
        return numpy_time(value)
    elif isinstance(value, datetime.timedelta):  # pd.Timedelta included
        return numpy_time(value)
    elif isinstance(value, bytes | np.bytes_):  # how binary files spell text
        try:
            return unbyte(value)
        except UnicodeDecodeError:
            pass
    elif isinstance(value, Quantity):
        magnitude = value.magnitude
        if np.ndim(magnitude) == 0 and not np.iscomplexobj(magnitude):
            return value
    msg = (
        f"The attr {name!r} holds a {type(value).__module__}."
        f"{type(value).__qualname__}, but an attr holds a string, number, "
        "bool, None, time, timedelta, or a scalar quantity or unit. Store "
        "an array on a coordinate, and structured metadata as fields of a "
        "PatchAttrs subclass."
    )
    raise ValueError(msg)


def _extra_attr_value(name: str, value: Any) -> Any:
    """
    Return an extra attr's value, or `_SKIPPED` for an array of several
    values, which belongs on a coordinate; refuse any other type.
    """
    if type(value) in _ARRAYS:
        # As objects, so a ragged list counts rather than raising and a
        # python scalar stays one.
        value = np.asarray(value, None if type(value) is np.ndarray else object)
        if value.size != 1:
            msg = f"Skipped attr {name!r}: an array belongs on a coordinate."
            warnings.warn(msg, UserWarning, stacklevel=3)
            return _SKIPPED
        value = value.reshape(-1)[0]  # how HDF5 and netCDF spell a scalar
    return _simple_attr_value(name, value)


def drop_unheld_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return attrs without the extra attrs `PatchAttrs` refuses, warning for each.

    For a reader which copies every attr a file holds, so that one attr of
    a type no patch holds is skipped rather than failing the read.
    """
    out = {}
    for name, value in attrs.items():
        if name not in PatchAttrs.model_fields:
            try:
                value = _extra_attr_value(name, value)
            except ValueError as error:
                msg = f"Skipped attr {name!r}: {error}"
                warnings.warn(msg, UserWarning, stacklevel=2)
                continue
            if value is _SKIPPED:
                continue
        out[name] = value
    return out


class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    `PatchAttrs` stores non-structural metadata. Nested coordinate payloads in
    `coords` are rejected, while flat coord-like keys are treated like any
    other extra attrs. `dims` is ignored during normalization.

    An extra attr holds one string, real number, bool, None, numpy time or
    timedelta, or scalar quantity or unit. Python and pandas times become
    numpy times and bytes become text; a dict, array or other object is
    refused. See the PatchAttrs note for why attrs hold no arrays or maps.

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
        for name in [k for k in data if k not in cls.model_fields]:
            value = _extra_attr_value(name, data.pop(name))
            if value is not _SKIPPED:
                data[name] = value
        return data

    def __getitem__(self, item):
        return getattr(self, item)

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
