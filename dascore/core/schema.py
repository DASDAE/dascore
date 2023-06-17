"""Pydantic schemas used by DASCore."""
from pathlib import Path
from typing import Literal, Mapping, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, validator

from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    basic_summary_attrs,
    max_lens,
)
from dascore.utils.docs import compose_docstring
from dascore.utils.time import to_datetime64, to_timedelta64


class SimpleValidator:
    """
    A custom class for getting simple validation behavior in pydantic.

    Subclass, then define function to be used as validator. func
    """

    @classmethod
    def func(cls, value):
        """A method to overwrite with custom validation."""
        return value

    @classmethod
    def __get_validators__(cls):
        """Hook used by pydantic."""
        yield cls.validate

    @classmethod
    def validate(cls, validator):
        """Simply call func."""
        return cls.func(validator)


class DateTime64(np.datetime64, SimpleValidator):
    """DateTime64 validator"""

    func = to_datetime64


class TimeDelta64(np.timedelta64, SimpleValidator):
    """TimeDelta64 validator"""

    func = to_timedelta64


@compose_docstring(basic_params=basic_summary_attrs)
class PatchAttrs(BaseModel):
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

    See also [PatchAttrs](`dascore.core.schema.PatchAttrs`).
    """

    data_type: Literal[VALID_DATA_TYPES] = ""
    data_category: Literal[VALID_DATA_CATEGORIES] = ""
    data_units: str = ""
    time_min: DateTime64 = np.datetime64("NaT")
    time_max: DateTime64 = np.datetime64("NaT")
    d_time: TimeDelta64 = np.timedelta64("NaT")
    time_units: str = ""
    distance_min: float = np.NaN
    distance_max: float = np.NaN
    d_distance: float = np.NaN
    distance_units: str = ""
    instrument_id: str = Field("", max_length=max_lens["instrument_id"])
    cable_id: str = Field("", max_length=max_lens["cable_id"])
    dims: str = Field("", max_length=max_lens["dims"])
    tag: str = Field("", max_length=max_lens["tag"])
    station: str = Field("", max_length=max_lens["station"])
    network: str = Field("", max_length=max_lens["network"])
    history: Union[str, Sequence[str]] = Field(default_factory=list)

    # In order to maintain backward compatibility, these dunders make the
    # class also behave like a dict.

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(self.dict())

    def get(self, item, default=None):
        """dict-like get method."""
        try:
            return self[item]
        except (AttributeError, KeyError):
            return default

    def items(self):
        """Yield (attribute, values) just like dict.items()."""
        for item, value in self.dict().items():
            yield item, value

    @classmethod
    def get_defaults(cls):
        """return a dict of default values"""
        new = cls()
        return new.dict()

    def coords_from_dims(self) -> Mapping[str, np.ndarray]:
        """Return coordinates from dimensions assuming evenly sampled."""
        out = {}
        for dim in self.dim_tuple:
            # TODO replace this with simple coords
            start, stop = self[f"{dim}_min"], self[f"{dim}_max"]
            step = self[f"d_{dim}"]
            ar = np.arange(start, stop + step, step)
            # due to float imprecision the last value can be slightly larger
            # than stop, just trim
            if ar[-1] > stop:
                ar = ar[:-1]
            out[dim] = ar
        return out

    class Config:
        """Configuration for Patch Summary"""

        title = "Patch Summary"
        extra = "allow"
        allow_mutation = False
        json_encoders = {
            np.datetime64: lambda x: str(x),
            np.timedelta64: lambda x: str(x),
        }

    @property
    def dim_tuple(self):
        """Return a tuple of dimensions. The dims attr is a string."""
        return tuple(self.dims.split(","))

    @validator("dims", pre=True)
    def _flatten_dims(cls, value):
        """Some dims are passed as a tuple; we just want str"""
        if not isinstance(value, str):
            value = ",".join(value)
        return value


class PatchFileSummary(PatchAttrs):
    """
    The expected minimum attributes for a Patch/spool file.
    """

    # These attributes are excluded from the HDF index.
    _excluded_index = ("data_units", "time_units", "distance_units", "history")

    file_version: str = ""
    file_format: str = ""
    path: Union[str, Path] = ""

    @classmethod
    def get_index_columns(cls) -> tuple[str, ...]:
        """Return the column names which should be used for indexing."""
        fields = set(cls.__fields__) - set(cls._excluded_index)
        return tuple(sorted(fields))
