"""Module for calculating cross-correlation over time or distance."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.fft as sft
from pydantic import ConfigDict

import dascore as dc
from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.proc.basic import Pad
from dascore.transform.fourier import (
    Dft,
    Idft,
    _dft_kernel,
    _is_complex,
    _operand,
)
from dascore.units import get_quantity
from dascore.utils.array_api import _result_dtype, array_namespace
from dascore.utils.patch import get_dim_axis_value
from dascore.utils.time import to_float


class CorrelateShift(PatchProcessor):
    """
    Apply a shift to the patch data to undo correlation in frequency domain.

    Also adds the appropriate coordinate prefixed with "lag" and has a datatype
    of float.

    Parameters
    ----------
    patch
        The input patch
    dim
        The dimension name that was correlated in the freq. domain.
    undo_weighting
        If True, also undo the weighting artifact caused by DASCore's dft
        weighting. This is done by simply dividing by the coordinate step,
        and the data units by the coordinate's units.
        See [dft note](`dascore/docs/notes/dft_notes.qmd`) for more details.

    Notes
    -----
    A product of transforms is a correlation circular over the transformed
    length, but `idft` trims a padded transform back to the original length,
    which drops the most negative lags. Transform with `pad=False`, or pad
    first with `patch.pad(time="correlate")`, as below.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Example 1
    >>> # An auto-correlation of the example patch
    >>> dft = patch.dft("time", real=True, pad=False)
    >>> dft_sq = dft * dft.conj()
    >>> idft = dft_sq.idft()
    >>> auto_patch = idft.correlate_shift(dim="time")
    """

    __version__ = "1.2"
    dim: Any
    undo_weighting: Any = True

    data_type = "correlation"

    def get_metadata(self, meta):
        """Return metadata with the lag coordinate, and the axis and step."""
        dim = self.dim
        coord = meta.get_coord(dim, require_evenly_sampled=True)
        axis = meta.get_axis(dim)
        step = coord.step
        new_start = -np.ceil((len(coord) - 1) / 2) * step
        new_end = np.ceil((len(coord) - 1) / 2) * step
        _new_coord = dc.get_coord(
            start=new_start, stop=new_end, step=step, units=coord.units
        )
        new_coord = _new_coord.change_length(len(coord))
        assert len(new_coord) == len(coord)
        # Each lag mixes every sample, so nothing riding them carries over.
        cm = meta.coords.drop_associated(dim, "Correlating along")
        new_cm = cm._update_grid(dim, **{dim: new_coord}).rename_coord(
            **{dim: f"lag_{dim}"}
        )
        out = meta.new(coords=new_cm)
        if not self.undo_weighting:
            return out, {"axis": axis, "step": None}
        units = get_quantity(meta.attrs.data_units)
        if units is not None and coord.units is not None:
            # dividing by the step divides the units by the coordinate's too
            out = out.update_attrs(data_units=units / get_quantity(coord.units))
        weight = _operand(to_float(step), meta.dtype)
        dtype = _result_dtype(meta.dtype, weight)
        return out.new(dtype=dtype), {"axis": axis, "step": weight}

    def kernel(self, data, *, axis, step):
        """Return the data shifted so zero lag is central, divided by the step."""
        return _shift_lags(data, axis=axis, step=step)


def _shift_lags(data, *, axis, step):
    """Return the data shifted so zero lag is central, divided by the step."""
    xp = array_namespace(data)
    # fftshift is this roll, but some backends shift only floats.
    data = xp.roll(data, data.shape[axis] // 2, axis=axis)
    if step is None:
        return data
    if not xp.isdtype(data.dtype, ("real floating", "complex floating")):
        # numpy promotes integers to float64 here; some backends refuse to.
        data = xp.astype(data, xp.float64)
    # Divide by the step as a 0-d array: like a numpy scalar, and unlike
    # a python float, it sets the result's dtype, and every backend
    # accepts it.
    return data / (xp.asarray(step) if isinstance(step, np.generic) else step)


class Correlate(PatchProcessor):
    """
    Correlate source row/columns in a patch with all other row/columns.

    The correlation runs in the frequency domain, transforming the target
    dimension when needed. For an already transformed patch, apply
    [`Patch.correlate_shift`](`dascore.Patch.correlate_shift`) after
    the inverse transform. The output gains one new, last source
    dimension; [`Patch.squeeze`](`dascore.Patch.squeeze`) removes it for a
    single source.

    Parameters
    ----------
    samples
        Interpret source selectors as sample indices rather than coordinate
        values.
    lag_dim
        The dimension to correlate along. A 2D patch defaults to the
        dimension not named in ``kwargs``; a patch with more dimensions
        must name it, and every other dimension is correlated separately.
    normalize
        If True, divide each correlation by ``sqrt(E_trace * E_source)``,
        where ``E`` is the whole-trace energy (sum of squared magnitudes)
        along ``lag_dim``, so ``|cc| <= 1`` and the result is dimensionless.
        The mean is not removed, and the energies do not change with lag:
        for traces with their mean removed, this equals the Pearson
        coefficient only at zero lag. A trace with no energy gives zeros.
    **kwargs
        Source dimension mapped to one or more source values or indices.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.units import m
    >>> patch = dc.get_example_patch(
    ...     "sin_wav",
    ...     sample_rate=100,
    ...     frequency=range(10, 20),
    ...     duration=5,
    ...     channel_count=10,
    ... ).taper(time=0.05).set_units(distance="m")
    >>>
    >>> # Correlate every channel with the 10 m channel.
    >>> cc_patch = patch.correlate(distance=10 * m).squeeze()
    >>>
    >>> # Keep -2 through 2 seconds of lag.
    >>> cc_patch = (
    ...     patch.correlate(distance=10 * m)
    ...     .select(lag_time=(-2, 2))
    ... )
    >>>
    >>> # Normalized correlation, keeping -0.5 through 0.5 seconds of lag.
    >>> cc_patch = patch.correlate(distance=10 * m, normalize=True).select(
    ...     lag_time=(-0.5, 0.5)
    ... )
    >>>
    >>> # Select a source by sample index after decimation.
    >>> cc_patch = (
    ...     patch.decimate(distance=2, filter_type=None)
    ...     .correlate(distance=1, samples=True)
    ... )
    >>>
    >>> cc_patch = patch.correlate(time=100, samples=True)
    >>>
    >>> # Correlate several sources in a frequency-domain pipeline.
    >>> padded_patch = patch.pad(time="correlate")
    >>> dft_patch = padded_patch.dft("time", real=True)
    >>> cc_patch = dft_patch.correlate(distance=[1, 3, 7], samples=True)
    >>> cc_out = cc_patch.idft().correlate_shift("time")
    >>>
    >>> # Correlate along time separately for each source of a 3D patch.
    >>> import numpy as np
    >>> patch_3d = dc.Patch(
    ...     data=np.random.rand(3, 2, 400),
    ...     coords={"source": [0, 1, 2], "distance": [0.0, 1.0], "time": range(400)},
    ...     dims=("source", "distance", "time"),
    ... )
    >>> cc_patch = patch_3d.correlate(distance=0, samples=True, lag_dim="time")

    Notes
    -----
    The result's data units are the square of the patch's (for example
    (m/s)**2 for velocity), as each value is a sum of products of the data.
    Single-precision data are correlated in single precision.

    Correlation runs along ``lag_dim``. That dimension
    becomes a lag dimension prefixed with ``lag_``; for example, selecting a
    ``distance`` source transforms ``time`` into ``lag_time``.
    """

    __version__ = "1.3"
    samples: Any = False
    lag_dim: Any = None
    normalize: bool = False

    model_config = ConfigDict(extra="allow")
    data_type = "correlation"

    def get_metadata(self, meta):
        """Return the correlation's metadata, and each step's plan."""
        extras = self.model_extra or {}
        if "lag" in extras:
            msg = (
                "The 'lag' parameter was removed. Select on the lag coordinate instead."
            )
            raise TypeError(msg)
        if len(meta.dims) < 2:
            msg = "must be at least a 2D patch."
            raise ParameterError(msg)
        dim, source_axis, source = get_dim_axis_value(meta, kwargs=extras)[0]
        # Get the dim along which the fft should be calculated.
        others = [x for x in meta.dims if x != dim]
        fft_dim = others[0] if self.lag_dim is None else self.lag_dim
        if fft_dim not in others or (self.lag_dim is None and len(others) > 1):
            msg = f"lag_dim must be one of {others}, not {self.lag_dim!r}."
            raise ParameterError(msg)
        # Determine if the input patch has already been transformed.
        transform = not fft_dim.startswith("ft_")
        plan: dict[str, Any] = {"transform": transform, "source_axis": source_axis}
        if transform:  # Standard dft workflow for correlation
            meta, pad = Pad(**{fft_dim: "correlate"}).get_metadata(meta)
            real = None if _is_complex(meta.dtype) else fft_dim
            meta, dft = Dft(dim=fft_dim, real=real).get_metadata(meta)
            plan |= pad | {"dft": dft}
        # Get the sources.
        coord = meta.get_coord(dim)
        source = coord.values if source is None else source
        index = coord.get_next_index(source, samples=self.samples)
        plan["index"] = np.atleast_1d(index)
        if self.samples:  # the sources are positions, which have no units
            new_coord = dc.get_coord(data=np.atleast_1d(source))
        else:  # values, as the dimension states them
            values = coord._get_compatible_value(np.atleast_1d(source))
            new_coord = dc.get_coord(data=values, units=coord.units)
        dim_name = f"source_{dim}"
        out = meta.new(coords=meta.coords.update(**{dim_name: (dim_name, new_coord)}))
        lag_dim = fft_dim.removeprefix("ft_")
        if (unpadded := f"_{lag_dim}_unpadded") in out.coords.coord_map:
            # The product is a correlation circular over the padded length, so
            # idft must keep all of it; trimming would drop the negative lags.
            coords = out.coords.update(**{unpadded: (None, out.get_coord(lag_dim))})
            out = out.new(coords=coords)
        if (units := get_quantity(meta.attrs.data_units)) is not None:
            # A product of two spectra carries their units twice. Normalized,
            # the data units cancel, leaving the transform's for idft to undo.
            ft_units = get_quantity(meta.get_coord(f"ft_{lag_dim}").units)
            norm = 1 / ft_units**2 if ft_units is not None else units / units
            out = out.update_attrs(data_units=norm if self.normalize else units**2)
        if self.normalize:
            plan["energy"] = Idft(dim=fft_dim).get_metadata(out)[1]
        # Undo fft if this function did one, shift, and update coord.
        if transform:
            out, idft = Idft(dim=fft_dim).get_metadata(out)
            out, shift = CorrelateShift(dim=fft_dim).get_metadata(out)
            plan |= {"idft": idft, "shift": shift}
        return out, plan

    def numpy_kernel(self, data, *, transform, source_axis, index, **plan):
        """Return each row or column correlated with the sources."""
        if transform:
            data = np.pad(data, plan["pad_width"])
            data = _dft_kernel(data, np, sft, cast=False, **plan["dft"])
        if (energy_plan := plan.get("energy")) is not None:
            # Scale each trace to unit energy before the product, so the
            # result does not depend on amplitude. A trace with no energy
            # stays zero and one with NaN energy stays NaN (data * 0).
            norm = np.sqrt(_energy(data, **energy_plan)).astype(data.real.dtype)
            data = np.divide(data, norm, out=data * 0, where=norm > 0)
        # The sources, along a new last axis so they broadcast with the data.
        selector: list[Any] = [slice(None)] * data.ndim + [None]
        selector[source_axis] = index
        source = np.swapaxes(data[tuple(selector)], source_axis, -1)
        data = data[..., None] * np.conj(source)
        if transform:
            data = Idft().numpy_kernel(data, **plan["idft"])
            data = _shift_lags(data, **plan["shift"])
        return data


def _energy(data, *, axes, real, step, sizes, **_):
    """Return the time-domain energy of a spectrum along its axis (Parseval)."""
    axis = axes[0]
    n = data.shape[axis] if sizes is None else sizes[0]
    power = np.abs(data).astype(np.float64) ** 2
    total = power.sum(axis=axis, keepdims=True)
    if real:  # one-sided bins, but for zero and Nyquist, stand for two
        edges = [0, n // 2] if n % 2 == 0 else [0]
        total = 2 * total - np.take(power, edges, axis=axis).sum(axis, keepdims=True)
    return total / (n * float(step) ** 2)
