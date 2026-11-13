"""Module for detrending."""

from __future__ import annotations

from typing import Literal

import numpy as np

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.utils.imports import lazy_import
from dascore.utils.misc import broadcast_for_index, suppress_warnings

scipy_detrend = lazy_import("scipy.signal", "detrend")


def _nan_detrend(data, axis, type):
    """Subtract a trend fit to the non-NaN samples of each trace."""

    def total(values):
        # Accumulate in at least float64 so narrow floats do not overflow.
        dtype = np.result_type(values.dtype, np.float64)
        return values.sum(axis=axis, keepdims=True, dtype=dtype)

    valid = ~np.isnan(data)
    index = broadcast_for_index(data.ndim, axis, slice(None), None)
    x = np.arange(data.shape[axis])[index]
    y = np.where(valid, data, 0)
    # Traces with too few samples for the fit divide by zero and become NaN.
    with suppress_warnings(RuntimeWarning):
        trend = total(y) / total(valid)
        if type == "linear":
            x_mean = total(x * valid) / total(valid)
            dx = np.where(valid, x - x_mean, 0)
            trend = trend + total(dx * y) / total(dx * dx) * (x - x_mean)
    return data - trend.astype(data.dtype)


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

    Notes
    -----
    NaN samples are treated as missing: the trend is fit to the other
    samples of each trace and NaN samples stay NaN; a trace with too few
    samples for the fit comes back all NaN.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> # Subtract the best-fitting line from each channel.
    >>> out = pa.detrend("time")
    >>> # Or only the mean.
    >>> demeaned = pa.detrend("time", type="constant")
    """

    __version__ = "1.1"
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
        if data.dtype.kind in "fc" and np.isnan(data).any():
            return _nan_detrend(data, axis, self.type)
        return scipy_detrend(data, axis=axis, type=self.type)
