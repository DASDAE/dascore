"""
Custom dascore exceptions.
"""


class DASCoreError(Exception):
    """Base class for dascore errors."""


class InvalidFileFormatter(ValueError, DASCoreError):
    """Raised when an invalid file formatter is defined or used."""


class UnknownFiberFormat(IOError, DASCoreError):
    """Raised when the format of an alleged fiber file is not recognized."""


class ParameterError(ValueError, DASCoreError):
    """Raised when something is wrong with an input parameter."""


class PatchError(DASCoreError):
    """Parent class for more specific Patch Errors."""


class PatchCoordError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's coordinates."""


class PatchDimError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's dimension."""


class PatchAttributeError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's attributes."""


class TimeError(ValueError, DASCoreError):
    """Raised when something is wrong with a time value"""


class InvalidTimeRange(TimeError):
    """Raised when an invalid time range is encountered."""


class FilterValueError(ValueError, DASCoreError):
    """Raise when something goes wrong with filtering or filter inputs."""


class UnsupportedKeyword(TypeError, DASCoreError):
    """Raised when dascore encounters an unexpected keyword."""
