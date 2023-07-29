"""Pydantic schemas used by DASCore."""
from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, Any

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

import dascore as dc
from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    basic_summary_attrs,
    max_lens,
)
from dascore.core.coords import BaseCoord, CoordRange, CoordSummary
from dascore.core.coordmanager import CoordManager
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import to_str
from dascore.utils.mapping import FrozenDict
from dascore.utils.models import (
    DascoreBaseModel,
    PlainValidator,
    UnitQuantity,
    CommaSeparatedStr,
    frozen_dict_validator,
    frozen_dict_serializer,
    # FrozenDictType,
)

# from dascore.utils.mapping import FrozenDict

str_validator = PlainValidator(to_str)
_coord_summary_suffixes = set(CoordSummary.model_fields)


def _get_coords_dict(data_dict, fields):
    """
    Add coords dict to data dict, pop out any coordinate attributes.

    For example, if time_min, time_step are in data_dict, these will be
    grouped into the coords sub dict under "time".
    """

    def _get_coord_dict(data_dict):
        """Get a dict for stuffing outputs in."""
        # first handle coords.
        coords = data_dict.get("coords", {})
        if isinstance(coords, CoordManager):
            data_dict["dims"] = coords.dims
            coords = coords.to_summary_dict()
        coords = dict(coords)
        return coords

    def _ensure_all_coord_summaries(coords):
        """Enssure each coordinate entry is a CoordSummary."""
        # then iterate and convert each coordinate
        for i, v in coords.items():
            if isinstance(v, BaseCoord):
                coords[i] = v.to_summary()
            # converts normal dicts to coord summary
            elif not isinstance(v, CoordSummary):
                coords[i] = CoordSummary(**v)
        return coords

    def _get_coords_from_attrs(data_dict, fields, coords):
        """Use fields in attrs to get coord dict."""
        # finally, pull out any attributes on the top level
        extra_fields = set(data_dict) - set(fields)
        new_coords = defaultdict(dict)
        for extra_field in extra_fields:
            if len(split := extra_field.split("_")) != 2:
                continue
            first, second = split
            # handle d_{coord_name}, it should now be {coord_name}_step
            if first == "d":
                first, second = second, "step"
                msg = f"{extra_field} is deprecated, use {first}_{second}"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            # if coords were passed we don't want attrs to overwrite them.
            if second in _coord_summary_suffixes:
                # we need to pop out the key either way, but only use
                # it if we haven't already defined the coord summary
                val = data_dict.pop(extra_field)
                if first not in coords:
                    new_coords[first][second] = val
        # now convert new values to Summaries
        for i, v in new_coords.items():
            new_coords[i] = CoordSummary(**v)
        return new_coords

    coords = _get_coord_dict(data_dict)
    coords_with_summaries = _ensure_all_coord_summaries(coords)
    new_coords = _get_coords_from_attrs(data_dict, fields, coords_with_summaries)
    # update coords and return
    coords.update(new_coords)
    data_dict["coords"] = coords
    return data_dict


@compose_docstring(basic_params=basic_summary_attrs)
class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    Parameter
    ---------
    {basic_params}

    Notes
    -----
    These attributes go into the HDF5 index used by dascore. Therefore,
    when they are changed the index version needs to be incremented so
    previous indices are invalidated.
    """

    model_config = ConfigDict(
        title="Patch Summary",
        extra="allow",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    data_type: Annotated[Literal[VALID_DATA_TYPES], str_validator] = ""
    data_category: Annotated[Literal[VALID_DATA_CATEGORIES], str_validator] = ""
    data_units: UnitQuantity | None = None

    instrument_id: str = Field("", max_length=max_lens["instrument_id"])
    cable_id: str = Field("", max_length=max_lens["cable_id"])

    tag: str = Field("", max_length=max_lens["tag"])
    station: str = Field("", max_length=max_lens["station"])
    network: str = Field("", max_length=max_lens["network"])
    history: str | Sequence[str] = Field(default_factory=list)

    dims: CommaSeparatedStr = Field("", max_length=max_lens["dims"])

    coords: Annotated[
        FrozenDict[str, CoordSummary],
        frozen_dict_validator,
        frozen_dict_serializer,
    ] = {}

    @model_validator(mode="before")  # noqa
    @classmethod
    def parse_coord_attributes(cls, data: Any) -> Any:
        """
        Parse the coordinate attributes into coord dict.
        """
        if isinstance(data, dict):
            data = _get_coords_dict(data, cls.model_fields)
        return data

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(self.model_dump())

    def __getattr__(self, item):
        """
        This enables dynamic attributes such as time_min, time_max, etc.
        """
        split = item.split("_")
        # this only works on names like time_max, distance_step, etc.
        if not len(split) == 2:
            return super().__getattr__(item)
        first, second = split
        if first == "d":
            first, second = second, "step"
            msg = f"{item} is depreciated, use {first}_{second} instead."
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

    @classmethod
    def get_defaults(cls):
        """return a dict of default values"""
        new = cls()
        return new.model_dump()

    def coords_from_dims(self) -> Mapping[str, BaseCoord]:
        """Return coordinates from dimensions assuming evenly sampled."""
        out = {}
        for dim in self.dim_tuple:
            start, stop = self[f"{dim}_min"], self[f"{dim}_max"]
            step = self[f"d_{dim}"]
            out[dim] = CoordRange(start=start, stop=stop + step, step=step)
        return out

    @classmethod
    def from_dict(
        cls,
        attr_map: Mapping | PatchAttrs,
        coord_manager: dc.core.coordmanager.CoordManager | None = None,
    ) -> Self:
        """
        Get a new instance of the PatchAttrs.

        Optionally, give preference to data contained in a
        [`CoordManager`](`dascore.core.coordmanager.CoordManager`).

        Parameters
        ----------
        attr_map
            Anything convertible to a dict that contains the attr info.
        coord_manager
            A coordinate manager to fill in/overwrite attributes.

        """
        if isinstance(attr_map, cls) and coord_manager is None:
            return attr_map
        out = {} if attr_map is None else dict(attr_map)
        if coord_manager is None:
            return cls(**out)
        out["dims"] = ",".join(coord_manager.dims)
        out["coords"] = coord_manager.to_summary_dict()
        return cls(**out)

    @property
    def dim_tuple(self):
        """Return a tuple of dimensions. The dims attr is a string."""
        return tuple(self.dims.split(","))

    def rename_dimension(self, **kwargs):
        """
        Rename one or more dimensions if in kwargs. Return new PatchAttrs.
        """
        if not (dims := set(kwargs) & set(self.dim_tuple)):
            return self
        new = dict(self)
        new_dims = list(self.dim_tuple)
        for old_name, new_name in {x: kwargs[x] for x in dims}.items():
            new_dims[new_dims.index(old_name)] = new_name
        new["dims"] = tuple(new_dims)
        return self.__class__(**new)

    def update(self, **kwargs) -> Self:
        """Update an attribute in the model, return new model."""
        out = dict(self)
        out.update(kwargs)
        return self.__class__(**out)

    def drop_private(self) -> Self:
        """Drop all private attributes."""
        contents = dict(self)
        out = {i: v for i, v in contents.items() if not i.startswith("_")}
        return self.__class__(**out)

    def flat_dump(self) -> dict:
        """
        Flatten the coordinates and dump to dict
        """
        out = self.model_dump()
        for coord_name, coord in out.pop("coords").items():
            for name, val in coord.items():
                out[f"{coord_name}_{name}"] = val
        return out
