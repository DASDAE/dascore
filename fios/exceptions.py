"""
Custom fios exceptions.
"""


class UnknownFiberFormat(IOError):
    """Raised when the format of an elledged fiber file is not recognized."""


class PatchCoordError(ValueError):
    """Raised when something is wrong with a Patch's coordinates."""


class PatchDimError(ValueError):
    """Raised when something is wrong with a Patch's dimension."""


class PatchAttributeError(ValueError):
    """Raised when something is wrong with a Patch's attributes."""


class InvalidTimeRange(ValueError):
    """Raised when an invalid time range is encountered."""
