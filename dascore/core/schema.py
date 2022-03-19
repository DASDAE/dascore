"""
Pydantic schemas.
"""

import numpy as np
from pydantic import BaseModel

from dascore.utils.time import to_datetime64, to_timedelta64


class SimpleValidator:
    """
    A custom class for getting simply validation behavior in pydantic.

    Simply subclass, then define function to be used as validator.
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
        cls.func(v)


class DateTime64(SimpleValidator):
    """Datetime64 validator"""

    func = to_datetime64


class TimeDelta64(SimpleValidator):
    """Datetime64 validator"""

    func = to_timedelta64


class PatchSummary(BaseModel):
    """The expected minimum attributes for a Patch attrs."""

    data_type: str = ""
    category: str = ""
    time_min: DateTime64 = np.datetime64("NaT")
    time_max: DateTime64 = np.datetime64("NaT")
    d_time: TimeDelta64 = np.timedelta64("NaT")
    distance_min: float = np.NaN
    distance_max: float = np.NaN
    d_distance: float = np.NaN
    instrument_id: str = ""
    dims: str = tuple()
    tag: str = ""
