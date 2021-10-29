"""
Custom fios exceptions.
"""


class UnknownFiberFormat(IOError):
    """Raised when the format of an elledged fiber file is not recognized."""


class IncompatibleCoords(ValueError):
    """Raised when coordinates are not compatible with the data."""


class MissingDimensions(ValueError):
    """Raised when trying to filter an trace on non-existent dimensions."""


class InvalidTimeRange(ValueError):
    """Raised when an invalid time range is encountered."""
