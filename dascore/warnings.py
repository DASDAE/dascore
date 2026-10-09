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


class SnapWarning(DASCoreWarning):
    """
    Issued when a reader snaps stored labels which sit off its grid.

    A reader puts the labels `snap` names on a grid, and warns when a label
    sits further from it than `dascore.config.max_shift` steps: the file may
    hold a gap, or its clock may run apart from its sampling. Pass
    `snap=False` to read the labels as stored.
    """
