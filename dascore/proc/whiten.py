"""Spectral whitening."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.fft as sft
from pydantic import ConfigDict
from scipy.ndimage import uniform_filter1d

from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.proc.basic import Angle
from dascore.proc.taper import TaperRange, _scale_by
from dascore.transform.fourier import Dft, Idft, _dft_kernel, _fft_input, _is_complex
from dascore.utils.array_api import _result_dtype, array_namespace, asarray_like
from dascore.utils.transformatter import FourierTransformatter


def _get_dim_freq_range_from_kwargs(patch, kwargs):
    """Get the dimension and frequency range."""
    dim_set = set(patch.dims)
    # Handles the default case when no kwargs passed.
    if not kwargs:
        expected = {"time", "ft_time"} & dim_set
        if not expected:
            msg = "No dim name provided in kwargs and patch has no time dimension."
            raise ParameterError(msg)
        dim = "time"
        freq_range = None
    # A single kwarg was passed.
    elif len(kwargs) == 1:
        dim, freq_range = next(iter(kwargs.items()))
        fft_dim = FourierTransformatter().rename_dims(dim)[0]
        if dim not in dim_set and fft_dim not in dim_set:
            msg = f"passed dim of {dim} to whiten but it is not in patch dimensions."
            raise ParameterError(msg)
    else:  # Something when wrong.
        msg = (
            "Whiten kwargs must specify a single patch dimension. "
            "Double check the supported input parameters as these may have "
            f"changed. You passed {kwargs}."
        )
        raise ParameterError(msg)

    return dim, freq_range


def _get_amp_envelope(data, axis, window_len, water_level):
    """Get a smoothed amplitude envelope."""
    amp = np.abs(data)
    # Uniform filter is *much* faster than convolve
    uni = uniform_filter1d(amp, window_len, axis=axis, mode="wrap")
    if water_level is not None:
        # Enforce water level to avoid instability in dividing small numbers
        uni[uni < water_level * uni.max()] = water_level * uni.max()
    smoothed_amp = amp / uni
    return smoothed_amp


def _check_smooth(fft_coord, smooth_size, water_level):
    """Check the smooth size."""
    if water_level is not None:
        if not isinstance(water_level, float) or water_level < 0 or water_level > 1:
            msg = "water_level must be a float between 0 and 1."
            raise ParameterError(msg)
    if smooth_size <= 0:
        msg = "Frequency smoothing size must be positive"
        raise ParameterError(msg)
    if smooth_size >= fft_coord.max():
        msg = "Frequency smoothing size is larger than Nyquist"
        raise ParameterError(msg)


def _check_freq_range(fft_coord, freq_range):
    """Check the frequency range."""
    # Note: the freq_range can be a len 2 or 4 sequence
    frange = np.asarray(freq_range)
    diffs = frange[1:] - frange[:-1]
    min_size = fft_coord.step * 2

    if np.any(diffs < min_size):
        msg = "Frequency range is too narrow"
        raise ParameterError(msg)


class Whiten(PatchProcessor):
    """
    Spectral whitening of a signal.

    The whitened signal is returned in the same domain (eq frequency or
    time domain) as the input signal. See also the
    [Whiten Processing Section](`dascore/docs/tutorial/processing.qmd`#whiten).

    Parameters
    ----------
    smooth_size
        Size in transformed domain units (eg Hz) or samples of moving average
        window, used to compute the spectrum before whitening.
        If None, don't smooth signal which results in a uniform amplitude.
        units.
    water_level
        If used, float between 0 and 1 to stabilize frequencies with near
        zero amplitude. Does nothing if smooth_size is None.
        Values between 0.01 and 0.05 usually work well.
    **kwargs
        Used to specify the dimension range in transformed units (e.g, Hz)
        of the smoothing. Can either be a sequence of two values or
        four values to specify a taper range. Simply uses
        [Patch.taper_range](`dascore.Patch.taper_range`) under the hood.
        If kwargs are provided, try to smooth `time` or `ft_time` coords.

    Notes
    -----
    1) The FFT result is divided by the smoothed spectrum before inverting
       back to time-domain signal. The phase is not changed.

    2) Amplitude is NOT preserved

    Example
    -------
    >>> import dascore as dc
    >>>
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Whiten along time dimension
    >>> white_patch = patch.whiten(time=None)
    >>>
    >>> # Band limited whitening
    >>> white_patch = patch.whiten(time=(20, 40))
    >>>
    >>> # Band limited with taper ends
    >>> white_patch = patch.whiten(time=(10, 20, 40, 60))
    >>>
    >>> # Whitening along distance with amplitude smoothing (0.1/m))
    >>> white_patch = patch.whiten(smooth_size=0.1, distance=None)
    """

    __version__ = "1.1"
    smooth_size: Any = None
    water_level: Any = None

    model_config = ConfigDict(extra="allow")

    def get_metadata(self, meta):
        """Return the whitened metadata, and the transforms, smoothing and taper."""
        smooth_size, water_level = self.smooth_size, self.water_level
        dim, freq_range = _get_dim_freq_range_from_kwargs(meta, self.model_extra or {})
        fft_dim = FourierTransformatter().rename_dims(dim)[0]
        # Frequency-domain metadata; unchanged if the input is already transformed.
        dft = Dft(dim=dim, real=not _is_complex(meta.dtype))
        out, dft_plan = dft.get_metadata(meta)
        transform = out is not meta
        fft_coord = out.get_coord(fft_dim)
        plan = {"transform": transform, "axis": out.get_axis(fft_dim)}
        plan |= {"window": None, "env": None, "dft": dft_plan, "idft": None}
        # The smoothing window in samples; None leaves unit amplitudes.
        if smooth_size is not None:
            _check_smooth(fft_coord, smooth_size, water_level)
            count = fft_coord.get_sample_count(smooth_size, enforce_lt_coord=True)
            plan["window"] = int(count)
        # Apply band-limited taper to remove some frequencies.
        if freq_range:
            _check_freq_range(fft_coord, freq_range)
            taper = TaperRange(**{fft_dim: freq_range})
            plan["env"] = taper.get_metadata(out)[1]["env"]
        # Convert back to time domain if input was in time-domain.
        if transform:
            out, plan["idft"] = Idft().get_metadata(out)
        else:  # Unit phases of the data, at their precision.
            out = out.new(dtype=_result_dtype(meta.dtype, np.complex64))
        return out, plan

    def numpy_kernel(self, data, *, transform, axis, window, env, dft, idft):
        """Return the data with flattened amplitudes and their phases kept."""
        if transform:
            data = _dft_kernel(data, np, sft, cast=False, **dft)
        if window is None:
            amp = np.ones_like(data)
        else:
            amp = _get_amp_envelope(data, axis, window, self.water_level)
        # New amplitudes, old phases.
        data = amp * np.exp(1j * np.angle(data))
        if env is not None:
            data = _scale_by(data, env)
        if transform:
            data = Idft().numpy_kernel(data, **idft)
        return data

    def kernel(self, data, *, transform, axis, window, env, dft, idft):
        """As `numpy_kernel`, on the data's own backend."""
        xp = array_namespace(data)
        data = _dft_kernel(data, xp, xp.fft, cast=True, **dft) if transform else data
        data = _fft_input(data, xp, True)
        phases = xp.exp(1j * xp.astype(Angle().kernel(data), data.dtype))
        if window is not None:
            amp = xp.abs(data)
            smooth = _wrapped_mean(amp, xp, window, axis)
            if self.water_level is not None:
                floor = self.water_level * xp.max(smooth)
                smooth = xp.where(smooth < floor, floor, smooth)
            phases = phases * (amp / smooth)
        if env is not None:
            phases = _scale_by(phases, env)
        return Idft().kernel(phases, **idft) if transform else phases


def _wrapped_mean(data, xp, window, axis):
    """
    Return `uniform_filter1d(mode="wrap")` of real data, as a circular
    convolution, which any backend runs natively.
    """
    # A chunked (dask) array transforms along one chunk only.
    if hasattr(data, "rechunk"):
        data = data.rechunk({axis: -1})
    size = data.shape[axis]
    box = np.zeros(size)
    box[window // 2 - np.arange(window)] = 1 / window
    spectrum = xp.fft.rfft(data, axis=axis)
    shape = [1] * data.ndim
    shape[axis] = -1
    box = xp.reshape(asarray_like(np.fft.rfft(box), spectrum), tuple(shape))
    box = xp.astype(box, spectrum.dtype)
    return xp.fft.irfft(spectrum * box, n=size, axis=axis)
