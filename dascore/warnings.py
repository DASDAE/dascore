"""Custom dascore warnings."""

from __future__ import annotations


class DASCoreWarning(UserWarning):
    """Base class for dascore warnings."""


class NumpyFallbackWarning(DASCoreWarning):
    """
    Raised when a patch function converts non-numpy data to numpy.

    Patch functions which have no implementation for the array backend of
    the input patch fall back to their numpy implementation. The data are
    converted to numpy, the function is applied, then the output data are
    converted back to the original backend.
    """
