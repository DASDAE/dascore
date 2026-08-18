"""Custom dascore warnings."""

from __future__ import annotations


class DASCoreWarning(UserWarning):
    """Base class for dascore warnings."""


class NumpyFallbackWarning(DASCoreWarning):
    """
    Issued when an operation converts non-numpy data to numpy.

    Some ufuncs, all numpy functions, and reductions of dtypes the standard
    excludes are applied by numpy: the data are converted, the operation
    applied, and the output converted back to the original backend.
    """
