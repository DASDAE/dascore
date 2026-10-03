"""
Patch function for 'Frequency-Band Energy' transform
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import ConfigDict

from dascore.core.processor import PatchProcessor
from dascore.proc.filter import PassFilter
from dascore.proc.rolling import (
    _PandasPatchRoller,
    _rolling_numpy,
    _rolling_pandas,
    rolling,
)
from dascore.units import get_filter_units
from dascore.utils.misc import check_filter_kwargs, check_filter_range
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
        dim, (arg1, arg2) = check_filter_kwargs(extras)
        coord_units = meta.coords.coord_map[dim].units
        filt_min, filt_max = get_filter_units(arg1, arg2, to_unit=coord_units, dim=dim)
        sample_rate = get_dim_sampling_rate(meta, dim)
        nyquist = 0.5 * sample_rate
        low = None if filt_min is None else filt_min / nyquist
        high = None if filt_max is None else filt_max / nyquist
        check_filter_range(nyquist, low, high, filt_min, filt_max)
        step = 1 / sample_rate if self.step is None else self.step
        plan = PassFilter(**extras).get_metadata(meta)[1]
        # `rolling` reads only the metadata, and is how the windows were found.
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
        filtered = PassFilter.kernel_for("numpy")(
            PassFilter(), data, axis=axis, sos=sos
        )
        roll = _rolling_pandas if pandas else _rolling_numpy
        mean = roll(
            filtered**2,
            "mean" if pandas else np.mean,
            window=window,
            step=step,
            axis=axis,
            center=False,
            args=(),
            kwargs={},
        )
        energy = mean**0.5
        return np.log10(energy) * 20 if self.db else energy
