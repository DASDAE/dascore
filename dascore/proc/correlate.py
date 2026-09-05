"""Module for calculating cross-correlation over time or distance."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import (
    get_dim_axis_value,
    patch_function,
)
from dascore.utils.time import to_float


def _get_source_fft(patch, dim, source, source_axis, samples):
    """
    Get an array of coordinate sources.

    This function will place the new sources in a third dimension so
    they broadcast with the original fft matrix.
    """
    # Extract an array containing just the sources
    coord_source = patch.get_coord(dim)
    index_source = coord_source.get_next_index(source, samples=samples)
    selector = [slice(None), slice(None), None]
    selector[source_axis] = np.atleast_1d(index_source)
    source = patch.data[tuple(selector)]
    # Now transpose source so source dim is list. Essentially we just
    # need to swap the source axis with the last axis.
    out = np.swapaxes(source, source_axis, -1)
    return out


@patch_function(data_type="correlation")
def correlate_shift(
    patch: PatchType, dim: str, undo_weighting: bool = True
) -> PatchType:
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
        weighting. This is done by simply dividing by the coordinate step.
        See [dft note](`docs/notes/dft_notes.qmd`) for more details.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Example 1
    >>> # An auto-correlation of the example patch
    >>> dft = patch.dft("time", real=True)
    >>> dft_sq = dft * dft.conj()
    >>> idft = dft_sq.idft()
    >>> auto_patch = idft.correlate_shift(dim="time")
    """
    coord = patch.get_coord(dim, require_evenly_sampled=True)
    axis = patch.get_axis(dim)
    data = np.fft.fftshift(patch.data, axes=axis)
    if undo_weighting:
        data = data / to_float(coord.step)
    step = coord.step
    new_start = -np.ceil((len(coord) - 1) / 2) * step
    new_end = np.ceil((len(coord) - 1) / 2) * step
    _new_coord = dc.get_coord(
        start=new_start, stop=new_end, step=step, units=coord.units
    )
    new_coord = _new_coord.change_length(len(coord))
    assert len(new_coord) == len(coord)
    cm = patch.coords
    new_cm = cm.update(**{dim: new_coord}).rename_coord(**{dim: f"lag_{dim}"})
    out = patch.update(data=data, coords=new_cm)
    return out


@patch_function(data_type="correlation")
def correlate(
    patch: PatchType,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    """
    Correlate source row/columns in a 2D patch with all other row/columns.

    The correlation runs in the frequency domain, transforming the target
    dimension when needed. For an already transformed patch, apply
    [`Patch.correlate_shift`](`dascore.proc.correlate.correlate_shift`) after
    the inverse transform. The 2D input becomes 3D, with one new source
    dimension; [`Patch.squeeze`](`dascore.Patch.squeeze`) removes it for a
    single source.

    Parameters
    ----------
    patch
        Two-dimensional patch in the original or frequency domain.
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
    Correlation runs along the dimension not named in ``kwargs``. That dimension
    becomes a lag dimension prefixed with ``lag_``; for example, selecting a
    ``distance`` source transforms ``time`` into ``lag_time``.
    """
    if "lag" in kwargs:
        msg = "The 'lag' parameter was removed. Select on the lag coordinate instead."
        raise TypeError(msg)
    if len(patch.dims) != 2:
        msg = "must be a 2D patch."
        raise ParameterError(msg)
    dim, source_axis, source = get_dim_axis_value(patch, kwargs=kwargs)[0]
    # Get the axis and coord over which fft should be calculated.
    fft_axis = next(iter(set(range(len(patch.dims))) - {source_axis}))
    fft_dim = patch.dims[fft_axis]
    # Determine if the input patch has already been transformed.
    input_dft = fft_dim.startswith("ft_")
    is_real = not np.issubdtype(patch.data.dtype, np.complexfloating)
    if not input_dft:  # Standard dft workflow for correlation
        # Note: we use .func here to avoid getting these added to the history.
        padded = patch.pad.func(patch, **{fft_dim: "correlate"})
        patch = padded.dft.func(padded, fft_dim, real=fft_dim if is_real else None)
    # Get the sources.
    source = patch.get_coord(dim).values if source is None else source
    source_fft = _get_source_fft(patch, dim, source, source_axis, samples)
    # Need to insert new axis so the arrays broadcast correctly.
    fft_patch_array = patch.data[..., None]
    fft_prod = fft_patch_array * np.conj(source_fft)
    # Create frequency domain patch with results
    source = getattr(source, "magnitude", source)  # strips units
    new_coord = dc.get_coord(data=np.atleast_1d(source))
    dim_name = f"source_{dim}"
    cm = patch.coords.update(**{dim_name: (dim_name, new_coord)})
    out = patch.update(data=fft_prod, coords=cm)
    # Undo fft if this function did one, shift, and update coord.
    if not input_dft:
        idft = out.idft.func(out)
        out = idft.correlate_shift.func(idft, fft_dim)
    return out
