"""Module for detrending."""

from __future__ import annotations

from typing import Literal

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.utils.imports import lazy_import

scipy_detrend = lazy_import("scipy.signal", "detrend")


class Detrend(PatchProcessor):
    """
    Perform detrending along a given dimension (distance or time) of a patch.

    Parameters
    ----------
    patch
        The patch to detrend.
    dim
        The dimension ("distance" or "time") along where detrending is applied.
    type
        Specifies least-squares fit type for detrend,
        with "linear" (default) or "constant" as options.

    Returns
    -------
    The Patch instance after applying the detrend function.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>> out = pa.detrend("time") # detrend along the time dimension
    """

    dim: str
    type: Literal["linear", "constant"] = "linear"

    def plan(self, patch, out):
        """Return the axis to detrend along."""
        if self.dim not in patch.dims:
            msg = f"dim '{self.dim}' is not in patch dimensions {patch.dims}"
            raise ParameterError(msg)
        return {"axis": patch.get_axis(self.dim)}

    def numpy_kernel(self, data, *, axis):
        """Return the data with the trend along the axis removed."""
        return scipy_detrend(data, axis=axis, type=self.type)


detrend = Detrend.patch_function
