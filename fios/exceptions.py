"""
Custom fios exceptions.
"""


class FiosError(Exception):
    """Base class for FIOS errors."""


class UnknownFiberFormat(IOError, FiosError):
    """Raised when the format of an elledged fiber file is not recognized."""


class PatchError(FiosError):
    """Parent class for more specific Patch Errors."""


class PatchCoordError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's coordinates."""


class PatchDimError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's dimension."""


class PatchAttributeError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's attributes."""


class TimeError(ValueError, FiosError):
    """Raised when something is wrong with a time value"""


class InvalidTimeRange(TimeError):
    """Raised when an invalid time range is encountered."""


class FilterValueError(ValueError, FiosError):
    """Raise when something goes wrong with filtering or filter inputs."""
