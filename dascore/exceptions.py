"""Custom dascore exceptions."""
from __future__ import annotations


class DASCoreError(Exception):
    """Base class for dascore errors."""


class InvalidFiberIO(ValueError, DASCoreError):
    """Raised when an invalid Fiber IO is defined or used."""


class InvalidFiberFile(IOError, DASCoreError):
    """Raised when a fiber operation is called on an invalid file."""


class UnknownFiberFormat(IOError, DASCoreError):
    """Raised when the format of an alleged fiber file is not recognized."""


class UnknownExample(DASCoreError):
    """Raised when an unregistered example is requested."""


class ParameterError(ValueError, DASCoreError):
    """Raised when something is wrong with an input parameter."""


class PatchError(DASCoreError):
    """Parent class for more specific Patch Errors."""


class IncompatiblePatchError(PatchError):
    """Raised when an operator cannot be performed on a patch."""


class CoordError(ValueError, PatchError):
    """Raised when something is wrong with a Coordinate."""


class CoordMergeError(CoordError):
    """Raised when something is wrong with requested merge operation."""


class CoordSortError(CoordError):
    """Raised when coordinates cannot be sorted."""


class CoordDataError(CoordError):
    """Raised when the data shape doesn't match the coordinates."""


class ChunkError(DASCoreError):
    """Raised when chunking goes awry."""


class PatchDimError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's dimension."""


class PatchAttributeError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's attributes."""


class PatchConversionError(ValueError, PatchError):
    """Raised when a patch conversion to a different format fails."""


class TimeError(ValueError, DASCoreError):
    """Raised when something is wrong with a time value."""


class InvalidTimeRange(TimeError):
    """Raised when an invalid time range is encountered."""


class FilterValueError(ValueError, DASCoreError):
    """Raise when something goes wrong with filtering or filter inputs."""


class UnsupportedKeyword(TypeError, DASCoreError):
    """Raised when dascore encounters an unexpected keyword."""


class InvalidFileHandler(TypeError, DASCoreError):
    """Raised when a writable file handler is requested from a read handle."""


class InvalidIndexVersionError(ValueError, DASCoreError):
    """Raised when a version mismatch occurs in index."""


class MissingOptionalDependency(ImportError, DASCoreError):
    """Raised when an optional package needed for some functionality is missing."""


class InvalidSpoolError(ValueError, DASCoreError):
    """Raised when something is wrong with a spool."""


class UnitError(ValueError, DASCoreError):
    """Raised when an issue is encountered with unit handling."""


class AttributeMergeError(ValueError, DASCoreError):
    """Raised when something is wrong with combining attributes."""
