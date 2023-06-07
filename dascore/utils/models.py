"""
Utilities for models.
"""
from functools import cached_property

import numpy as np
import numpy.typing as npt
import pint
from pydantic import BaseModel
from typing_extensions import Self

from dascore.compat import array
from dascore.utils.time import to_datetime64, to_timedelta64


class DascoreBaseModel(BaseModel):
    """A base model with sensible configurations."""

    class Config:
        """Configuration for models."""

        extra = "ignore"
        validate_assignment = True  # validators run on assignment
        keep_untouched = (cached_property,)
        frozen = True
        validate_all = True

    def new(self, **kwargs) -> Self:
        """Create new instance with some attributed updated."""
        out = dict(self)
        for item, value in kwargs.items():
            out[item] = value
        return self.__class__(**out)


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


class ArrayLike(np.ndarray, SimpleValidator):
    """An array like validator."""

    @classmethod
    def func(cls, object):
        """Ensure an object is array-like."""
        assert isinstance(object, np.ndarray)
        return array(object)


class DTypeLike(SimpleValidator):
    """A data-type like validator."""

    @classmethod
    def func(cls, object):
        """Assert an object is datatype-like"""
        assert isinstance(object, npt.DTypeLike)
        return object


class Unit(pint.Unit, SimpleValidator):
    """A pint unit which can be used by pydantic."""

    func = pint.Unit


class UnitStr(str, SimpleValidator):
    """A string which can be converted to a Unit (e.g, m)."""

    @classmethod
    def func(cls, value):
        """validate that a string can be converted to a unit."""
        if value:
            Unit(value)
            # don't leave as a unit; pydantic doesnt know how to serialize it.
            value = str(value)
        return value
