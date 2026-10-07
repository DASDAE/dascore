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
        cm = meta.coords
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
    Correlate source row/columns in a 2D patch with all other row/columns.

    The correlation runs in the frequency domain, transforming the target
    dimension when needed. For an already transformed patch, apply
    [`Patch.correlate_shift`](`dascore.Patch.correlate_shift`) after
    the inverse transform. The 2D input becomes 3D, with one new source
    dimension; [`Patch.squeeze`](`dascore.Patch.squeeze`) removes it for a
    single source.

    Parameters
    ----------
    samples
        Interpret source selectors as sample indices rather than coordinate
        values.
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

    Notes
    -----
    The result's data units are the square of the patch's (for example
    (m/s)**2 for velocity), as each value is a sum of products of the data.
    Single-precision data are correlated in single precision.

    Correlation runs along the dimension not named in ``kwargs``. That dimension
    becomes a lag dimension prefixed with ``lag_``; for example, selecting a
    ``distance`` source transforms ``time`` into ``lag_time``.
    """

    __version__ = "1.3"
    samples: Any = False

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
        if len(meta.dims) != 2:
            msg = "must be a 2D patch."
            raise ParameterError(msg)
        dim, source_axis, source = get_dim_axis_value(meta, kwargs=extras)[0]
        # Get the axis and coord over which fft should be calculated.
        fft_axis = next(iter(set(range(len(meta.dims))) - {source_axis}))
        fft_dim = meta.dims[fft_axis]
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
        source = getattr(source, "magnitude", source)  # strips units
        new_coord = dc.get_coord(data=np.atleast_1d(source))
        dim_name = f"source_{dim}"
        out = meta.new(coords=meta.coords.update(**{dim_name: (dim_name, new_coord)}))
        lag_dim = fft_dim.removeprefix("ft_")
        if (unpadded := f"_{lag_dim}_unpadded") in out.coords.coord_map:
            # The product is a correlation circular over the padded length, so
            # idft must keep all of it; trimming would drop the negative lags.
            coords = out.coords.update(**{unpadded: (None, out.get_coord(lag_dim))})
            out = out.new(coords=coords)
        if (units := get_quantity(meta.attrs.data_units)) is not None:
            # a product of two spectra carries their units twice
            out = out.update_attrs(data_units=units**2)
        # Undo fft if this function did one, shift, and update coord.
        if transform:
            out, idft = Idft().get_metadata(out)
            out, shift = CorrelateShift(dim=fft_dim).get_metadata(out)
            plan |= {"idft": idft, "shift": shift}
        return out, plan

    def numpy_kernel(self, data, *, transform, source_axis, index, **plan):
        """Return each row or column correlated with the sources."""
        if transform:
            data = np.pad(data, plan["pad_width"])
            data = _dft_kernel(data, np, sft, cast=False, **plan["dft"])
        # The sources, along a third axis so they broadcast with the data.
        selector: list[Any] = [slice(None), slice(None), None]
        selector[source_axis] = index
        source = np.swapaxes(data[tuple(selector)], source_axis, -1)
        data = data[..., None] * np.conj(source)
        if transform:
            data = Idft().numpy_kernel(data, **plan["idft"])
            data = _shift_lags(data, **plan["shift"])
        return data
