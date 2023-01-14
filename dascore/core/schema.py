"""Pydantic schemas used by DASCore."""
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from pydantic import BaseModel, Field

from dascore.constants import basic_summary_attrs, max_lens
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


class DateTime64(SimpleValidator):
    """Datetime64 validator"""

    func = to_datetime64


class TimeDelta64(SimpleValidator):
    """Datetime64 validator"""

    func = to_timedelta64


@compose_docstring(basic_params=basic_summary_attrs)
class PatchSummary(BaseModel):
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

    data_type: str = ""
    category: str = ""
    time_min: DateTime64 = np.datetime64("NaT")
    time_max: DateTime64 = np.datetime64("NaT")
    d_time: TimeDelta64 = np.timedelta64("NaT")
    distance_min: float = np.NaN
    distance_max: float = np.NaN
    d_distance: float = np.NaN
    instrument_id: str = Field("", max_length=max_lens["instrument_id"])
    cable_id: str = Field("", max_length=max_lens["cable_id"])
    dims: Union[tuple[str, ...], str] = tuple()
    tag: str = Field("", max_length=max_lens["tag"])
    station: str = Field("", max_length=max_lens["station"])
    network: str = Field("", max_length=max_lens["network"])

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

    class Config:
        """Configuration for Patch Summary"""

        title = "Patch Summary"
        extra = "allow"
        allow_mutation = False
        json_encoders = {
            np.datetime64: lambda x: str(x),
            np.timedelta64: lambda x: str(x),
        }


class PatchSummaryWithHistory(PatchSummary):
    """Patch summary which includes history."""

    history: Union[str, Sequence[str]] = ""


class PatchFileSummary(PatchSummary):
    """
    The expected minimum attributes for a Patch/spool file.
    """

    file_version: str = ""
    file_format: str = ""
    path: Union[str, Path] = ""


@compose_docstring(basic_params=basic_summary_attrs)
class PatchAttrs(PatchSummary):
    """
    The schema for the metadata attached to a patch.

    Attributes
    ----------
    {basic_summary_attrs}
    data_units
        The units of data (e.g., m/s)
    time_units
        Units of time axis (if time coord is not datetime64)
    distance_units
        Units of distance axis (default is m)
    history
        A list keeping track of processing occurring on patch.

    Notes
    -----
    `PatchAttrs` is a superset of [PatchSummary](`dascore.core.schema.PatchSummary`).
    `PatchAttrs` is the actual object attached to Patches, whereas `PatchSummary` is
    the information required to index and filter a patch.
    """

    data_units = ""
    time_units: str = "s"
    distance_units: str = "m"
    history: list[str] = []
