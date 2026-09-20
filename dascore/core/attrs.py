"""Pydantic schemas used by DASCore."""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Mapping, Sequence, Set
from typing import Annotated, Any, NoReturn, Self, cast

import numpy as np
import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
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
from dascore.utils.misc import (
    to_str,
    validate_acquisition_key,
)

str_validator = PlainValidator(to_str)

# What an attr value may be. str and bytes come first because they are
# also Sequences, which the refused tuple below covers.
_SCALAR_TYPES = (
    str,
    bytes,
    bool,
    int,
    float,
    complex,
    np.generic,
    datetime.datetime,
    datetime.date,
    datetime.timedelta,
    type(None),
)

# What it may not be: anything holding more than one value.
_COLLECTION_TYPES = (
    BaseModel,
    Mapping,
    Sequence,
    Set,
    pd.Series,
    pd.DataFrame,
    pd.Index,
)


def _scalar_attr(name: str, value: Any) -> Any:
    """
    Return the scalar an attr value is, raising for anything with a shape.

    A 0-d array is the scalar it wraps; anything else with a shape, and
    any collection, belongs on the patch as a coordinate.
    """
    if isinstance(value, _SCALAR_TYPES):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value[()]
        _raise_not_scalar(name, value)
    if isinstance(value, _COLLECTION_TYPES) or np.ndim(value) != 0:
        _raise_not_scalar(name, value)
    return value


def _raise_not_scalar(name: str, value: Any) -> NoReturn:
    """Say that an attr holds no arrays, and where an array goes instead."""
    msg = (
        f"Attrs hold scalars, so {name!r} cannot be a "
        f"{type(value).__name__}. An array belongs on the patch as a "
        f"coordinate: patch.update_coords({name}=(dims, array)), or "
        f"patch.update_coords({name}=(None, array)) for one which rides "
        "no dimension."
    )
    raise ValueError(msg)


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
        """Reject coord payloads and non-scalars; ignore structural dims."""
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
        # Declared fields are whatever their annotations allow; the rest
        # are scalars, so that an array is a coordinate and nothing else.
        # Value first: this runs on every patch, and almost every value
        # is already a scalar.
        for name, value in data.items():
            if isinstance(value, _SCALAR_TYPES) or name in cls.model_fields:
                continue
            data[name] = _scalar_attr(name, value)
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


def drop_non_scalar_attrs(
    attrs: Mapping[str, Any], attr_class: type[PatchAttrs] = PatchAttrs
) -> dict[str, Any]:
    """
    Return stored attrs without the values a patch attr cannot hold.

    A file written before attrs were required to be scalars may carry
    arrays or collections in its attr namespace. Dropping them, with one
    warning naming the lot, keeps such a file readable. Declared fields
    are left alone, as they are during validation.

    Parameters
    ----------
    attrs
        The attr names and values a file stored.
    attr_class
        The class the values are destined for, which decides which names
        are declared fields.
    """
    fields = attr_class.model_fields
    out: dict[str, Any] = {}
    dropped = []
    for name, value in attrs.items():
        if name in fields:
            out[name] = value
            continue
        try:
            out[name] = _scalar_attr(name, value)
        except ValueError:
            dropped.append(name)
    if dropped:
        msg = (
            f"Dropping stored attrs which are not scalars: {sorted(dropped)}. "
            "Attrs hold scalars; such values belong on the patch as "
            "coordinates."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
    return out
