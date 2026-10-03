"""
Patch function for 'Frequency-Band Energy' transform
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import ConfigDict

from dascore.core.processor import PatchProcessor
from dascore.exceptions import UnitError
from dascore.proc.filter import PassFilter
from dascore.proc.rolling import _PandasPatchRoller, _rolling_mean, rolling
from dascore.units import get_quantity
from dascore.utils.array import _is_offset_unit
from dascore.utils.misc import check_filter_kwargs
from dascore.utils.patch import get_dim_sampling_rate


class Fbe(PatchProcessor):
    """
    Compute the rolling Frequency Band Energy in a window.
    This is the Root-Mean-Squared (RMS) of the Energy in a Frequency Band (the FBE),
    and commonly called 'the waterfall plot' in DAS-processing.
    This implementation is a wrapper to DASCore functionality:
        1) Apply a 'pass_filter' to the patch
        2) Apply rolling-function of a window along a coordinate
        3) Calculate RMS value


    Parameters
    ----------
    window
        window length in which to calculate energy (in units of the sampling rate)
    step
        time-step for rolling window. Defaults to original sampling rate, but this
        can be used for downsampling of the resulting patch.
        See also [rolling](`dascore.Patch.rolling`)
    db
        Return patch data in decibel [dB] instead of original units.
        Decibel is calculated as 20 * log10( sqrt(mean(x^2)) ).
    **kwargs
        Used to specify the dimension and associated frequency, wavelength, or
        equivalent limits. For example time=(1, 100) applies a time-dimension bandpass
        of 1-100 Hz. See [pass_filter](`dascore.Patch.pass_filter`) for more details.

    Returns
    -------
    PatchType
        A new patch containing FBE-RMS traces.

    Example
    --------
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.example_event_2()
    >>>
    >>> fbe_patch = p.fbe(time=(100,200), window = 0.002)
    >>> ax = fbe_patch.viz.waterfall(cmap = 'Spectral_r')
    >>> _ = ax.set_title('FBE along time-axis')


    Or along the distance-axis:
    >>> import dascore as dc
    >>>
    >>> p = dc.examples.get_example_patch('example_event_2')
    >>>
    >>> fbe_patch = p.fbe(distance=(.01,.05), window = 5)
    >>> ax = fbe_patch.viz.waterfall(cmap = 'Spectral_r')
    >>> _ = ax.set_title('FBE along distance-axis')
    """

    window: Any
    step: Any = None
    db: Any = True

    model_config = ConfigDict(extra="allow")

    def get_metadata(self, meta):
        """Return the energy's metadata, the filter and the rolling windows."""
        extras = self.model_extra or {}
        dim = check_filter_kwargs(extras)[0]
        plan = PassFilter(**extras).get_metadata(meta)[1]
        # Squaring the filtered data, as the energy does, refuses offset units.
        units = get_quantity(meta.attrs.data_units)
        if units is not None and _is_offset_unit(units):
            msg = (
                f"{np.power} is not defined for the offset units {units}; "
                "convert to an absolute unit (kelvin) first."
            )
            raise UnitError(msg)
        step = 1 / get_dim_sampling_rate(meta, dim) if self.step is None else self.step
        # Let `rolling` (metadata only) pick the window size and engine, so the
        # result matches `patch.rolling(...).mean()`.
        roller = rolling(meta, **{dim: self.window, "step": step})
        attrs = {"data_type": "frequency_band_energy"}
        if self.db:
            attrs["data_units"] = "dB"
        out = meta.new(coords=roller.get_coords(), dtype=np.float64)
        return out.new(attrs=attrs), plan | {
            "window": roller.window,
            "step": roller.step,
            "pandas": isinstance(roller, _PandasPatchRoller),
        }

    def numpy_kernel(self, data, *, axis, sos, window, step, pandas):
        """Return the root-mean-square of the filtered data in each window."""
        filtered = PassFilter().numpy_kernel(data, axis=axis, sos=sos)
        mean = _rolling_mean(
            filtered**2, window=window, step=step, axis=axis, pandas=pandas
        )
        energy = mean**0.5
        return np.log10(energy) * 20 if self.db else energy
