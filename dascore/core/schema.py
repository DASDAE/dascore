"""
Pydantic schemas.
"""
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from pydantic import BaseModel

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
        """Hook used by pydantic"""
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
    """The expected attributes for a Patch."""

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
    instrument_id: str = ""
    cable_id: str = ""
    dims: str = tuple()
    tag: str = ""
    station: str = ""
    network: str = ""


class PatchSummaryWithHistory(PatchSummary):
    """Patch summary which includes history."""

    history: Union[str, Sequence[str]] = ""


class PatchFileSummary(PatchSummary):
    """The expected minimum attributes for a Patch/spool file."""

    file_version: str = ""
    file_format: str = ""
    path: Union[str, Path] = ""
