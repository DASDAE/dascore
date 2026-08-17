"""Custom dascore warnings."""

from __future__ import annotations


class DASCoreWarning(UserWarning):
    """Base class for dascore warnings."""


class NumpyFallbackWarning(DASCoreWarning):
    """
    Raised when an operation converts non-numpy data to numpy.

    Ufuncs, numpy functions and reductions which the array API standard
    cannot express are applied by numpy: the data are converted to numpy,
    the operation is applied, then the output data are converted back to
    the original backend.
    """
