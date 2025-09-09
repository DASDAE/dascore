"""Pydantic schemas used by DASCore."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, PlainValidator, model_validator
from typing_extensions import Self

import dascore as dc
from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    max_lens,
)
from dascore.core.coords import BaseCoord, CoordSummary
from dascore.utils.attrs import separate_coord_info
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    to_str,
)
from dascore.utils.models import (
    CommaSeparatedStr,
    DascoreBaseModel,
    UnitQuantity,
    frozen_dict_serializer,
    frozen_dict_validator,
)

str_validator = PlainValidator(to_str)
_coord_summary_suffixes = set(CoordSummary.model_fields)
_coord_required = {"min", "max"}


def _get_coords_dict(data_dict):
    """
    Add coords dict to data dict, pop out any coordinate attributes.

    For example, if time_min, time_step are in data_dict, these will be
    grouped into the coords sub dict under "time".
    """

    def _get_dims(data_dict):
        """Try to get dim tuple."""
        dims = None
        if "dims" in data_dict:
            dims = data_dict["dims"]
        elif hasattr(coord := data_dict.get("coords"), "dims"):
            dims = coord.dims
        if isinstance(dims, str):
            dims = tuple(dims.split(","))
        return dims

    dims = _get_dims(data_dict)
    coord_info, new_attrs = separate_coord_info(
        data_dict, dims, required=("min", "max")
    )
    new_attrs["coords"] = {i: dc.core.CoordSummary(**v) for i, v in coord_info.items()}
    return new_attrs


class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

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

    data_type: Annotated[Literal[VALID_DATA_TYPES], str_validator] = Field(
        description="Describes the quantity being measured.", default=""
    )
    data_category: Annotated[Literal[VALID_DATA_CATEGORIES], str_validator] = Field(
        description="Describes the type of data.",
        default="",
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="The units of the data measurements"
    )
    instrument_id: str = Field(
        description="A unique id for the instrument which generated the data.",
        default="",
        max_length=max_lens["instrument_id"],
    )
    acquisition_id: str = Field(
        description="A unique identifier linking this data to an experiment.",
        default="",
        max_length=max_lens["experiment_id"],
    )
    tag: str = Field(
        default="", max_length=max_lens["tag"], description="A custom string field."
    )
    station: str = Field(
        default="", max_length=max_lens["station"], description="A station code."
    )
    network: str = Field(
        default="", max_length=max_lens["network"], description="A network code."
    )
    history: str | tuple[str, ...] = Field(
        default_factory=tuple,
        description="A list of processing performed on the patch.",
    )

    dims: CommaSeparatedStr = Field(
        default="",
        max_length=max_lens["dims"],
        description="A tuple of comma-separated dimensions names.",
    )

    coords: Annotated[
        FrozenDict[str, CoordSummary],
        frozen_dict_validator,
        frozen_dict_serializer,
    ] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse_coord_attributes(cls, data: Any) -> Any:
        """Parse the coordinate attributes into coord dict."""
        if isinstance(data, dict):
            data = _get_coords_dict(data)
            # add dims as coords if dims is not included.
            if "dims" not in data:
                data["dims"] = tuple(data["coords"])
        return data

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(self.model_dump())

    def __getattr__(self, item):
        """Enables dynamic attributes such as time_min, time_max, etc."""
        split = item.split("_")
        # this only works on names like time_max, distance_step, etc.
        if not len(split) == 2:
            return super().__getattr__(item)
        first, second = split
        if first == "d":
            first, second = second, "step"
            msg = f"'{item}' is deprecated; use '{first}_{second}' instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        if first not in self.coords:
            return super().__getattr__(item)
        coord_sum = self.coords[first]
        return getattr(coord_sum, second)

    def get(self, item, default=None):
        """dict-like get method."""
        try:
            return self[item]
        except (AttributeError, KeyError):
            return default

    def items(self):
        """Yield (attribute, values) just like dict.items()."""
        yield from self.model_dump().items()

    def coords_from_dims(self) -> Mapping[str, BaseCoord]:
        """Return coordinates from dimensions assuming evenly sampled."""
        out = {}
        for dim in self.dim_tuple:
            out[dim] = self.coords[dim].to_coord()
        return out

    @classmethod
    def from_dict(
        cls,
        attr_map: Mapping | PatchAttrs,
    ) -> Self:
        """
        Get a new instance of the PatchAttrs.

        Optionally, give preference to data contained in a
        [`CoordManager`](`dascore.core.coordmanager.CoordManager`).

        Parameters
        ----------
        attr_map
            Anything convertible to a dict that contains the attr info.
        """
        if isinstance(attr_map, cls):
            return attr_map
        out = {} if attr_map is None else attr_map
        return cls(**out)

    @property
    def dim_tuple(self):
        """Return a tuple of dimensions. The dims attr is a string."""
        dim_str = self.dims
        if not dim_str:
            return tuple()
        return tuple(self.dims.split(","))

    def rename_dimension(self, **kwargs):
        """Rename one or more dimensions if in kwargs. Return new PatchAttrs."""
        if not (dims := set(kwargs) & set(self.dim_tuple)):
            return self
        new = self.model_dump(exclude_defaults=True)
        coords = new.get("coords", {})
        new_dims = list(self.dim_tuple)
        for old_name, new_name in {x: kwargs[x] for x in dims}.items():
            new_dims[new_dims.index(old_name)] = new_name
            coords[new_name] = coords.pop(old_name, None)
        new["dims"] = tuple(new_dims)
        return self.__class__(**new)

    def update(self, **kwargs) -> Self:
        """Update an attribute in the model, return new model."""
        passed_in_coords = kwargs.get("coords", True)
        out = self.model_dump(exclude_unset=True)
        # If coord manager passed in, base new coordinates on that.
        new_coords = dict(out.get("coords", {}))
        if isinstance(passed_in_coords, dc.CoordManager):
            new_coords = passed_in_coords.model_dump(exclude_unset=True)

        coord_info, attr_info = separate_coord_info(kwargs, dims=self.dim_tuple)
        out.update(attr_info)
        # Iterate the coordinate information and update.
        for name, coord_dict in coord_info.items():
            if name not in new_coords:
                new_coords[name] = coord_dict
            else:
                new_coords[name].update(coord_dict)
        out["coords"] = new_coords
        # If falsy coords passed in, clear the coordinates.
        if not passed_in_coords:
            out["coords"] = {}
        return self.__class__(**out)

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

    def flat_dump(self, dim_tuple=False, exclude=None) -> dict:
        """
        Flatten the coordinates and dump to dict.

        Parameters
        ----------
        dim_tuple
            If True, return dimensional tuple instead of range. EG, the
            output will have {time: (min, max)} rather than
            {time_min: ..., time_max: ...,}. This is useful because it can
            be passed to read, scan, select, etc.
        exclude
            keys to exclude.
        """
        out = self.model_dump(exclude=exclude)
        for coord_name, coord in out.pop("coords").items():
            names = list(coord)
            if dim_tuple:
                names = sorted(set(names) - {"min", "max"})
                out[coord_name] = (coord["min"], coord["max"])
            for name in names:
                out[f"{coord_name}_{name}"] = coord[name]
            # ensure step has right type if nullish
            step_name = f"{coord_name}_step"
            step, start = out[step_name], coord["min"]
            if step is None:
                is_time = isinstance(start, np.datetime64 | np.timedelta64)
                if is_time:
                    out[step_name] = np.timedelta64("NaT")
                elif isinstance(start, float | np.floating):
                    out[step_name] = np.nan
        return out
