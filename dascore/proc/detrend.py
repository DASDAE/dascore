"""Module for detrending."""

from __future__ import annotations

from typing import Literal

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.utils.imports import lazy_import

scipy_detrend = lazy_import("scipy.signal", "detrend")


class Detrend(PatchProcessor):
    """
    Remove a constant or linear trend from the data along one dimension.

    Parameters
    ----------
    patch
        The patch to detrend.
    dim
        Name of the dimension to detrend along.
    type
        The trend to subtract: "linear" (the default) removes a least-squares
        line, "constant" removes the mean.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> # Subtract the best-fitting line from each channel.
    >>> out = pa.detrend("time")
    >>> # Or only the mean.
    >>> demeaned = pa.detrend("time", type="constant")
    """

    dim: str
    type: Literal["linear", "constant"] = "linear"

    def get_metadata(self, meta):
        """Return the axis to detrend along."""
        if self.dim not in meta.dims:
            msg = f"dim '{self.dim}' is not in patch dimensions {meta.dims}"
            raise ParameterError(msg)
        return meta, {"axis": meta.get_axis(self.dim)}

    def numpy_kernel(self, data, *, axis):
        """Return the data with the trend along the axis removed."""
        return scipy_detrend(data, axis=axis, type=self.type)
