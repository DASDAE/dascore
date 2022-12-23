"""Pydantic schemas used by DASCore."""
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from pydantic import BaseModel, Field

from dascore.constants import max_lens
from dascore.utils.time import to_datetime64, to_timedelta64


class SimpleValidator:
    """
    A custom class for getting simple validation behavior in pydantic.

    Subclass, then define function to be used as validator. func
    """

    @classmethod
    def func(cls, x):
        """A method to overwrite with custom validation."""
        return x

    @classmethod
    def __get_validators__(cls):
        """Hook used by pydantic."""
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """Simply call func."""
        return cls.func(v)


class DateTime64(SimpleValidator):
    """Datetime64 validator"""

    func = to_datetime64


class TimeDelta64(SimpleValidator):
    """Datetime64 validator"""

    func = to_timedelta64


class PatchSummary(BaseModel):
    """
    The expected attributes for a Patch.


    Parameter
    ---------
    d_time
        The temporal sample spacing. If the patch is not evenly sampled
        this should be set to `np.timedelta64('NaT')`
    time_min
        The time represented by the first sample in the patch.
    time_max
        The time represented by the last sample in the patch.
    time_units
        The units of time axis. Not needed when type is `np.datetime64`.
    d_distance
        The spatial sampling rate, set to NaN if the patch is not evenly sampled
        in space.
    distance_min
        The along-fiber distance of the first channel in the patch.
    distance_max
        The along-fiber distance of the last channel in the patch.
    distance_units
        The units of distance, defaults to m.
    data_units
        units
    category
        The category
    network
        The network code an ascii-compatible string up to 2 characters.
    station
        The station code an ascii-compatible string up to 5 characters
    instrument_id
        The identifier of the instrument.
    dims
        A tuple of dimension names in the same order as the data dimensions.
    tag


    """

    class Config:
        """Configuration for Patch Summary"""

        json_encoders = {
            np.datetime64: lambda x: str(x),
            np.timedelta64: lambda x: str(x),
        }

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
    dims: tuple[str, ...] | str = tuple()
    tag: str = Field("", max_length=max_lens["tag"])
    station: str = Field("", max_length=max_lens["station"])
    network: str = Field("", max_length=max_lens["network"])

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


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


class PatchAttrs(PatchSummary):
    """
    The schema for the metadata attached to a patch.

    Attributes
    ----------
    d_time
        The temporal sample spacing. If the patch is not evenly sampled
        this should be set to `np.timedelta64('NaT')`
    time_min
        The time represented by the first sample in the patch.
    time_max
        The time represented by the last sample in the patch.
    time_units
        The units of time axis. Not needed when type is `np.datetime64`.
    d_distance
        The spatial sampling rate, set to NaN if the patch is not evenly sampled
        in space.
    distance_min
        The along-fiber distance of the first channel in the patch.
    distance_max
        The along-fiber distance of the last channel in the patch.
    distance_units
        The units of distance, defaults to m.
    data_type
        d
    data_units
        units
    category
        The category
    network
        The network code an ascii-compatible string up to 2 characters.
    station
        The station code an ascii-compatible string up to 5 characters
    instrument_id

    history: list[str] = []
    dims: tuple = ()
    tag: str = ""



    Notes
    -----
    PatchAttrs behaves like a dictionary for backwards compatibility reasons.

    """

    data_units = ""
    time_units: str = "s"
    distance_units: str = "m"
    history: list[str] = []
