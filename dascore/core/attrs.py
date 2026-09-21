"""Pydantic schemas used by DASCore."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence, Set
from functools import cache
from typing import Annotated, Any, Self, cast

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    model_validator,
)

from dascore.constants import (
    WARN_LEVELS,
    DataCategory,
    DataType,
    max_lens,
)
from dascore.exceptions import PatchAttributeError
from dascore.models import DascoreBaseModel, UnitQuantity
from dascore.utils.misc import (
    to_str,
    unbyte,
    validate_acquisition_key,
    validate_warn_level,
    warn_or_raise,
)

str_validator = PlainValidator(to_str)

# The exact types nearly every attr has; anything else gets a closer look.
_PLAIN = frozenset(
    {str, int, float, bool, type(None), np.str_, np.float64, np.float32}
    | {np.int64, np.int32, np.bool_, np.datetime64, np.timedelta64}
)

# What holds more than one value whatever its length. A record scalar has
# fields, and an iterator is spent by reading it.
_MANY = (BaseModel, Mapping, Sequence, Set, Iterator, np.void)

# What goes in place of an attr holding more than one value.
_NOT_SCALAR = object()

# What no attr is called, because they say how a patch is built.
_STRUCTURAL = frozenset({"dims", "coords"})


def _scalar_attr(value: Any) -> Any:
    """
    Return the one value an attr holds, or `_NOT_SCALAR` for more.

    Anything holding exactly one value is that value: a 0-d or length-1
    array is how HDF5 and netCDF spell a scalar.
    """
    if type(value) in _PLAIN or isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return unbyte(value)
    if isinstance(value, np.ndarray):
        return _scalar_attr(value.reshape(-1)[0]) if value.size == 1 else _NOT_SCALAR
    if isinstance(value, list | tuple):
        return _scalar_attr(value[0]) if len(value) == 1 else _NOT_SCALAR
    # A quantity says how much it holds through its magnitude. Asked before
    # the abstract classes below, which are slow to rule out.
    held = getattr(value, "magnitude", None)
    if held is not None:
        one = held.size == 1 if isinstance(held, np.ndarray) else True
        return value if one else _NOT_SCALAR
    # Another library's array says so through its shape.
    many = isinstance(value, _MANY) or getattr(value, "ndim", 0) > 0
    return _NOT_SCALAR if many else value


@cache
def _declared(attr_class: type[PatchAttrs]) -> frozenset[str]:
    """What an attrs class names itself; `model_fields` builds a dict."""
    return frozenset(attr_class.model_fields) | _STRUCTURAL


def _scalar_pass(data: dict, on_non_scalar: WARN_LEVELS, declared) -> None:
    """Reduce each extra in place to the one value it holds."""
    skipped = []
    for name, value in data.items():
        if type(value) in _PLAIN:
            continue
        # A declared field is left to its annotation, but for the one-value
        # array a file spells a scalar with.
        if name in declared and not (type(value) is np.ndarray and value.size == 1):
            continue
        got = _scalar_attr(value)
        if got is not _NOT_SCALAR:
            data[name] = got
        elif name not in declared:
            skipped.append(name)
    if not skipped:
        return
    kinds = ", ".join(f"{x!r} ({type(data[x]).__name__})" for x in skipped)
    msg = (
        f"Attrs hold scalars, so these hold more than one value: {kinds}. "
        "An array belongs on the patch as a coordinate: "
        f"patch.update_coords({skipped[0]}=(dims, array)), or "
        f"patch.update_coords({skipped[0]}=(None, array)) for one which "
        "rides no dimension."
    )
    warn_or_raise(msg, exception=PatchAttributeError, behavior=on_non_scalar)
    for name in skipped:
        data.pop(name)


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
        # Warning rather than refusing is what keeps every reader, plugins
        # included, able to read a file which stored one.
        # Checked in C first: only a value of another type is looked at.
        if not _PLAIN.issuperset(map(type, data.values())):
            _scalar_pass(data, "warn", _declared(cls))
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
        on_non_scalar: WARN_LEVELS = "warn",
    ) -> Self:
        """
        Get a new instance of the PatchAttrs.

        Parameters
        ----------
        attr_map
            Anything convertible to a dict that contains attr info. `dims`
            entries are ignored during normalization.
        on_non_scalar
            What to do with an extra attr holding more than one value:
            "warn" (the default) skips it and says so, "raise" refuses it
            naming the coordinate it should be, and "ignore" skips it
            silently. A value holding exactly one thing becomes that thing
            in every mode.

        Examples
        --------
        >>> import numpy as np
        >>> import dascore as dc
        >>>
        >>> foreign = {"project": np.array(["survey"]), "epsg_code": [4326]}
        >>> attrs = dc.PatchAttrs.from_dict(foreign, on_non_scalar="ignore")
        >>> assert attrs.project == "survey" and attrs.epsg_code == 4326
        """
        validate_warn_level(on_non_scalar, "on_non_scalar")
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
            # Said here so the constructor's own default never sees one.
            if on_non_scalar != "warn":
                out = scalar_attrs(out, on_non_scalar, cls)
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


def scalar_attrs(
    attrs: Mapping[str, Any],
    on_non_scalar: WARN_LEVELS = "warn",
    attr_class: type[PatchAttrs] = PatchAttrs,
) -> dict[str, Any]:
    """
    Return stored attrs as the scalars a patch attr may hold.

    A value holding exactly one thing — a 0-d array, a length-1 array or
    sequence, which is how HDF5 and netCDF spell a scalar — becomes that
    thing, bytes read as text. Anything holding more belongs on the patch
    as a coordinate, and is handled per `on_non_scalar`. Declared fields
    of `attr_class` are whatever their annotations allow, `history`
    included, and `dims` and `coords` are structural rather than attrs.

    Parameters
    ----------
    attrs
        The attr names and values, as a file or another library wrote them.
    on_non_scalar
        "warn" (the default) to skip such a value and say so once, naming
        the lot; "raise" to refuse it; "ignore" to skip it silently.
    attr_class
        The class the values are destined for, which decides which names
        are declared fields.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.attrs import scalar_attrs
    >>>
    >>> stored = {"project": np.array(["survey"]), "gauge": np.array([1.0, 2.0])}
    >>> assert scalar_attrs(stored, "ignore") == {"project": "survey"}
    """
    validate_warn_level(on_non_scalar, "on_non_scalar")
    out = dict(attrs)
    _scalar_pass(out, on_non_scalar, _declared(attr_class))
    return out
