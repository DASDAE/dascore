"""
Patch function for 'short-term average' to 'long-term average' ratio transform
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import ConfigDict

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError, UnitError
from dascore.proc.rolling import _PandasPatchRoller, _rolling_mean, rolling
from dascore.units import get_quantity, get_quantity_str
from dascore.utils.array import _is_offset_unit, _quantity
from dascore.utils.misc import check_filter_kwargs


class Stalta(PatchProcessor):
    """
    Compute the short-term / long-term average (STA/LTA) ratio along a patch dimension.

    Parameters
    ----------
    samples
        If True, values specified by kwargs are in samples not coordinate units.
    **kwargs
        Used to pass one dimension name and the short/long-term window lengths.
        For example `time=(0.1, 0.5)` uses windows of 0.1 and 0.5 seconds along
        the time axis.

    Returns
    -------
    PatchType
        A new patch containing the STA/LTA ratio.

    Notes
    -----
    A good first guess is to choose the long-term window 5x the length of the
    short-term window.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.example_event_2()
    >>>
    >>> s = p.envelope(dim="time").stalta(time=(0.002, 0.01))
    >>> s.viz.waterfall(  # doctest: +SKIP
    ...     cmap="RdGy_r", scale=[0, 2], scale_type="absolute"
    ... );
    """

    samples: Any = False

    model_config = ConfigDict(extra="allow")
    _positional_fields = ()
    data_type = "stalta"

    def get_metadata(self, meta):
        """Return the ratio's metadata, and each window and how it is rolled."""
        dim, (sta, lta) = check_filter_kwargs(self.model_extra or {})
        if lta <= sta:
            msg = f"The long-term window must exceed the short-term window, got {lta}."
            raise ParameterError(msg)
        # Let `rolling` (metadata only) pick the window sizes and engine, so the
        # result matches `patch.rolling(...).mean()`.
        sta_roll, lta_roll = (
            rolling(meta, samples=self.samples, **{dim: x}) for x in (sta, lta)
        )
        coords = sta_roll.get_coords()
        attrs, scale = meta.attrs, None
        # Units as dividing two patches in these units gives: the quotient's
        # units, with any scale (e.g. '10 m/s') applied to both means.
        if (units := get_quantity(attrs.data_units)) is not None:
            if _is_offset_unit(units):
                msg = f"{np.divide} is not defined for the offset units {units}."
                raise UnitError(msg)
            quotient = _quantity(1.0, units) / _quantity(1.0, units)
            attrs = attrs.update(data_units=get_quantity_str(quotient.units))
            scale = None if units.magnitude == 1 else units.magnitude
        plan = {
            "axis": sta_roll.axis,
            "sizes": (sta_roll.window, lta_roll.window),
            "pandas": isinstance(sta_roll, _PandasPatchRoller),
            "scale": scale,
        }
        return meta.new(coords=coords, attrs=attrs, dtype=np.float64), plan

    def numpy_kernel(self, data, *, axis, sizes, pandas, scale):
        """Return the short-term rolling mean over the long-term one."""
        sta, lta = (
            _rolling_mean(data, window=x, step=1, axis=axis, pandas=pandas)
            for x in sizes
        )
        if scale is not None:
            sta, lta = sta * scale, lta * scale
        return sta / lta
