"""Pydantic schemas used by DASCore."""
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator
from typing_extensions import Self

import dascore as dc
from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    basic_summary_attrs,
    max_lens,
)
from dascore.core.coords import BaseCoord, CoordRange
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import to_str
from dascore.utils.models import (
    DascoreBaseModel,
    DateTime64,
    PlainValidator,
    TimeDelta64,
    UnitQuantity,
)

str_validator = PlainValidator(to_str)


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

    data_type: Annotated[Literal[VALID_DATA_TYPES], str_validator] = ""
    data_category: Annotated[Literal[VALID_DATA_CATEGORIES], str_validator] = ""
    data_units: UnitQuantity | None = None
    time_min: DateTime64 = np.datetime64("NaT")
    time_max: DateTime64 = np.datetime64("NaT")
    d_time: TimeDelta64 = np.timedelta64("NaT")
    time_units: UnitQuantity | None = None
    distance_min: float = np.NaN
    distance_max: float = np.NaN
    d_distance: float = np.NaN
    distance_units: UnitQuantity | None = None
    instrument_id: str = Field("", max_length=max_lens["instrument_id"])
    cable_id: str = Field("", max_length=max_lens["cable_id"])
    dims: str = Field("", max_length=max_lens["dims"])
    tag: str = Field("", max_length=max_lens["tag"])
    station: str = Field("", max_length=max_lens["station"])
    network: str = Field("", max_length=max_lens["network"])
    history: str | Sequence[str] = Field(default_factory=list)

    model_config = ConfigDict(
        title="Patch Summary",
        extra="allow",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("dims", mode="before")
    @classmethod
    def _flatten_dims(cls, value):
        """Some dims are passed as a tuple; we just want str"""
        if not isinstance(value, str):
            value = ",".join(value)
        return value

    # In order to maintain backward compatibility, these dunders make the
    # class also behave like a dict.

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

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
        attr_map,
        coord_manager: Optional["dc.core.coordmanager.CoordManager"] = None,
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
        for name in coord_manager.dims:
            coord = coord_manager.coord_map[name]
            out[f"{name}_min"], out[f"{name}_max"] = coord.min(), coord.max()
            out[f"d_{name}"] = np.NaN if pd.isnull(coord.step) else coord.step
            out[f"{name}_units"] = coord.units
        out["dims"] = coord_manager.dims
        return cls(**out)

    @property
    def dim_tuple(self):
        """Return a tuple of dimensions. The dims attr is a string."""
        return tuple(self.dims.split(","))

    @property
    def time_step(self):
        """Return d_time. Start transition to time_step."""
        return self.d_time

    @property
    def distance_step(self):
        """Return d_time. Start transition to distance_step."""
        return self.d_distance

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


class PatchFileSummary(PatchAttrs):
    """
    The expected minimum attributes for a Patch/spool file.
    """

    file_version: str = ""
    file_format: str = ""
    path: str | Path = ""

    @classmethod
    def get_index_columns(cls) -> tuple[str, ...]:
        """Return the column names which should be used for indexing."""
        excluders = (
            "data_units",
            "time_units",
            "distance_units",
            "history",
        )
        fields = set(cls.model_fields) - set(excluders)
        return tuple(sorted(fields))
