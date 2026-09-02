"""Custom dascore exceptions."""

from __future__ import annotations


class DASCoreError(Exception):
    """Base class for dascore errors."""


class DependencyError(DASCoreError):
    """Raised when functionality depends on an unavailable compatibility stack."""


class InvalidFiberIOError(ValueError, DASCoreError):
    """Raised when an invalid Fiber IO is defined or used."""


class InvalidFiberFileError(IOError, DASCoreError):
    """Raised when a fiber operation is called on an invalid file."""


class UnknownFiberFormatError(IOError, DASCoreError):
    """Raised when the format of an alleged fiber file is not recognized."""


class UnknownExampleError(DASCoreError):
    """Raised when an unregistered example is requested."""


class ParameterError(ValueError, DASCoreError):
    """Raised when something is wrong with an input parameter."""


class InvalidSpoolQueryError(ParameterError):
    """Raised when a spool query references unknown names or bad values."""


class PatchError(DASCoreError):
    """Parent class for more specific Patch Errors."""


class IncompatiblePatchError(PatchError):
    """Raised when an operator cannot be performed on a patch."""


class UnresolvedPatchError(PatchError):
    """
    Raised when an inventory does not describe a patch.

    The patch names no inventory entry, or names one the inventory does
    not resolve to exactly one of. A patch the inventory describes *twice*
    (one straddling an epoch boundary) is a different condition and raises
    a plain `PatchError`: it needs subdividing, not a missing-data policy.
    """


class MissingPatchError(IndexError, PatchError):
    """
    Raised when no patch can be produced for a spool entry.

    This typically happens when a patch is trimmed to nothing by
    a coordinate selection (see #583). Subclasses IndexError for
    backwards compatibility.
    """


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


class PatchCoordinateError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's coordinates."""


class PatchBroadcastError(ValueError, PatchError):
    """Raised when patch cant be broadcast to a specified shape."""


class PatchAttributeError(ValueError, PatchError):
    """Raised when something is wrong with a Patch's attributes."""


class PatchConversionError(ValueError, PatchError):
    """Raised when a patch conversion to a different format fails."""


class TimeError(ValueError, DASCoreError):
    """Raised when something is wrong with a time value."""


class InvalidTimeRangeError(TimeError):
    """Raised when an invalid time range is encountered."""


class TimeOverflowError(OverflowError, TimeError):
    """
    Raised when a time cannot be represented in nanoseconds.

    Also an OverflowError, which is what numpy raised for the same values
    on the paths where it did not wrap, so existing handlers keep working.
    """


class FilterValueError(ValueError, DASCoreError):
    """Raise when something goes wrong with filtering or filter inputs."""


class UnsupportedKeywordError(TypeError, DASCoreError):
    """Raised when dascore encounters an unexpected keyword."""


class InvalidFileHandlerError(TypeError, DASCoreError):
    """Raised when a writable file handler is requested from a read handle."""


class InvalidIndexError(ValueError, DASCoreError):
    """Raised when a persisted index is invalid or incompatible."""


class InvalidIndexVersionError(InvalidIndexError):
    """Raised when a version mismatch occurs in index."""


class MissingOptionalDependencyError(ImportError, DependencyError):
    """
    Raised when an optional package needed for some functionality is missing.

    The install_name attribute, when set, gives the name of the package to
    install (eg protobuf) which may differ from the import name
    (eg google.protobuf). It defaults on the class so subclasses which don't
    call this init still have it.
    """

    install_name: str | None = None

    def __init__(self, *args, install_name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.install_name = install_name


class DASVaderCompatibilityError(InvalidFiberFileError, DependencyError):
    """Raised when a legacy DASVader file needs an older HDF5 compatibility stack."""


class InvalidSpoolError(ValueError, DASCoreError):
    """Raised when something is wrong with a spool."""


class UnitError(ValueError, DASCoreError):
    """Raised when an issue is encountered with unit handling."""


class AttributeMergeError(ValueError, DASCoreError):
    """Raised when something is wrong with combining attributes."""


class DASCorePluginError(AttributeError, DASCoreError):
    """Raised when something is wrong with plugins."""


class RemoteCacheError(IOError, DASCoreError):
    """Raised when DASCore cannot satisfy remote cache requirements."""


class InvalidInventoryError(ValueError, DASCoreError):
    """Raised when inventory metadata violates the DASDAE inventory model."""


class InvalidAnnotationError(ValueError, DASCoreError):
    """Raised when stored annotations violate the DASCore annotation model."""


class InvalidModelTagError(ValueError, DASCoreError):
    """Raised when a serialized document names its model class illegally."""
