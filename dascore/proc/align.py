"""
Functions to align patches based on some criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dascore.compat import ndarray
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import iterate
from dascore.utils.patch import patch_function


@dataclass(frozen=True)  # frozen=True → makes it immutable (like namedtuple)
class _ShiftInfo:
    """Metadata about the shift based on mode."""

    output_length: float
    mode: str
    start_shift: int
    end_shift: int
    new_shape: tuple[int, ...]
    source_slice: tuple[ndarray, ndarray]
    dest_slice: tuple[ndarray, ndarray]


def _get_source_indices(shifts, start_shift, end_shift, n_samples):
    """Get the indices in the source array."""
    min_start = -shifts + start_shift
    start = np.where(min_start < 0, 0, min_start)
    max_end = n_samples - shifts + end_shift
    end = np.where(max_end > n_samples, n_samples, max_end)
    return (start, end)


def _get_dest_indices(shifts, start_shift, n_samples, output_size):
    """Get indices in the output array."""
    start = np.maximum(0, shifts - start_shift)
    max_ends = start + n_samples
    end = np.where(max_ends > output_size, output_size, max_ends)
    return (start, end)


def _calculate_shift_info(
    mode,
    shifts,
    dim,
    dim_axis,
    patch_shape,
    reverse,
):
    """
    Calculate metadata about the shift.
    """
    # Note: shifts are either all >= 0 or <=0
    shifts = np.asarray(shifts, dtype=np.int64)
    min_shift, max_shift = shifts.min(), shifts.max()
    n_samples = len(dim)

    if mode == "full":
        # Shift value to left/right most index in reference to stationary trace.
        start_shift = min_shift
        end_shift = max_shift
    elif mode == "same":
        # Keep original size (and dimension)
        start_shift = 0
        end_shift = 0
    else:
        assert mode == "valid"
        # Find overlapping region
        # for forward, can only shift start time, reverse only endtime.
        start_shift = max_shift if max_shift > 0 else 0
        end_shift = min_shift if min_shift < 0 else 0

    output_size = n_samples + end_shift - start_shift
    # get slices for start/stop
    source_inds = _get_source_indices(shifts, start_shift, end_shift, n_samples)
    dest_inds = _get_dest_indices(shifts, start_shift, n_samples, output_size)
    # determine shape
    shape = list(patch_shape)
    shape[dim_axis] = output_size
    # Package up and return.
    out = _ShiftInfo(
        mode=mode,
        output_length=output_size,
        start_shift=start_shift,
        end_shift=end_shift,
        new_shape=tuple(shape),
        source_slice=source_inds,
        dest_slice=dest_inds,
    )
    return out


def _create_slice_indexer(start, stop, ndim, idx, coord_axes, dim_axis):
    """Create a slice indexer to go from start/stop idx to coord axes."""
    out = [slice(None)] * ndim
    out[dim_axis] = slice(start, stop)
    for i, ax in zip(idx, coord_axes):
        out[ax] = i
    return tuple(out)


def _apply_shifts_to_data(data, meta, dim_axis, coord_axes, fill_value):
    """
    Apply different shifts to data along specified axis.
    """
    out = np.full(meta.new_shape, fill_value)
    coord_axes = tuple(iterate(coord_axes))
    in_start, in_stop = meta.source_slice
    out_start, out_stop = meta.dest_slice
    for idx in np.ndindex(in_start.shape):
        source_ind = _create_slice_indexer(
            in_start[idx], in_stop[idx], out.ndim, idx, coord_axes, dim_axis
        )
        dest_ind = _create_slice_indexer(
            out_start[idx], out_stop[idx], out.ndim, idx, coord_axes, dim_axis
        )
        out[dest_ind] = data[source_ind]
    return out


def _validate_alignment_inputs(patch, kwargs):
    """Ensure the inputs are usable."""
    # Must have exactly 1 key/value which must both be strings.
    if len(kwargs) != 1:
        msg = "align_to_coords requires exactly one keyword argument"
        raise ParameterError(msg)
    (dim_name, coord_name) = dict(kwargs).popitem()
    if not isinstance(coord_name, str):
        msg = "align_to_coords requires keyword name and value to be strings."
        raise ParameterError(msg)
    # key must be a dimension and value a non-dimensional coord name.
    if dim_name not in patch.dims:
        msg = (
            f"align_to_coords requires the keyword to be a dimension, but "
            f"{dim_name} is not one of the patch dimensions: {patch.dims}"
        )
        raise ParameterError(msg)
    if coord_name not in patch.coords.coord_map or coord_name in patch.dims:
        msg = (
            f"align_to_coords requires the value to be the name of a non-dimensional "
            f"coordinate but '{coord_name}' is either not a coordinate or also "
            f"a dimension. Patch dimensions are: {patch.dims} and "
            f"patch coords: {patch.coords}"
        )
        raise ParameterError(msg)
    # Value must not depend on selected dim.
    value_dims = patch.coords.dim_map[coord_name]
    if dim_name in value_dims:
        msg = (
            f"align_to_coords requires the align coord not depend on the selected"
            f" dimension but {dim_name} does."
        )
        raise ParameterError(msg)
    return dim_name, coord_name


def _get_aligned_coords(patch, dim_name, meta):
    """
    Get the aligned coordinate manager.
    """
    coord = patch.get_coord(dim_name)
    start_shift, end_shift = meta.start_shift, meta.end_shift
    new_coord = coord.update(
        min=coord.min() + coord.step * start_shift,
    ).change_length(len(coord) - start_shift + end_shift)
    return patch.coords.update(**{dim_name: new_coord})


def _get_shift_indices(coord, dim, reverse, samples, relative):
    """
    Get the indices  for shifting.

    Positive values indicate a shift to the right, negative to left.
    """
    # First get naive indices, then subtract min.
    inds = dim.get_next_index(coord.values, samples=samples, relative=relative)
    out = inds - np.min(inds)
    # Reverse index. This way the min value still is at 0 (reference)
    if reverse:
        out *= -1
    return out


@patch_function()
def align_to_coord(
    patch: PatchType,
    mode: Literal["full", "valid", "same"] = "same",
    relative: bool = False,
    samples: bool = False,
    reverse: bool = False,
    fill_value: float = np.nan,
    **kwargs,
) -> PatchType:
    """
    Align patches based on values in a non-dimension coordinate.

    Parameters
    ----------
    mode : str
        Determines the output shape of the patch. Options are:
        "full" - Regardless of shift, all original data are preserved.
            This can result in patches with many fill values along the
            aligned dimension.
        "same" - The patch will retain its shape, however, only one trace
            (and traces that weren't shifted) will remain complete. Parts
            of shifted traces will be discarded.
        "valid" - The output patch will likely be smaller than the input
            patch, as only completely overlapped valid data are kept. This
            means no fill_values will occur in the patch.
    relative
        If True, the values in the alignment coord are relative to alignment
        dimension, else they are absolute.
    samples
        If True, the values in the alignment coord indicate samples rather
        than values in the shift dimension's units.
    reverse
        If True, multiply the alignment coordinate values by -1 to reverse
        a previous alignment operation.
    fill_value
        The value to insert in areas lacking data.

    **kwargs
        Used to specify the dimension which should shift and the coordinate
        to shift it to. The shift coordinate should not depend on the
        specified dimension.

    Returns
    -------
    A patch with aligned coordinates.

    Notes
    -----
    To understand the mode argument, consider a 2D patch with two traces,
    an alignment dimension length of 10, and a relative shift of 5 samples.
    For mode=="full" the output looks like this:
        -----aaaaaaaaaa
        bbbbbbbbbb-----
    For mode=="same" the patch will look like this:
        aaaaaaaaaa
        bbbbb-----
    For mode=="valid":
        aaaaa
        bbbbb
    where a and b represent values in first and second trace, respectively,
    and the - represents the fill values.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Create a suitable dimension for alignment.
    >>> # This will shift each channel 1 time sample relative to previous
    >>> time = patch.get_array("time")
    >>> distance = patch.get_array("distance")
    >>> ref_times = time[np.arange(len(distance))]
    >>> patch_coord = patch.update_coords(shift_time=("distance", ref_times))
    >>>
    >>> # Example 1: Apply the alignment, filling values with NaN
    >>> out = patch_coord.align_to_coord(time="shift_time", mode="full")
    >>>
    >>> # Example 2: Round-trip alignment with reverse parameter
    >>> shifts = np.array([0, 5, 10, 15, 20])  # shifts in samples
    >>> patch_shift = patch.update_coords(my_shifts=("distance", shifts))
    >>> # Forward alignment
    >>> aligned = patch_shift.align_to_coord(
    ...     time="my_shifts", samples=True, mode="full"
    ... )
    >>> # Reverse to get back original (after dropping NaN padding)
    >>> reversed_patch = aligned.align_to_coord(
    ...     time="my_shifts", samples=True, mode="full", reverse=True
    ... )
    >>> original = reversed_patch.dropna("time")
    >>> assert original.equals(patch_shift)
    >>>
    >>> # Example 3: Use 'valid' mode to extract only overlapping region
    >>> valid_aligned = patch_shift.align_to_coord(
    ...     time="my_shifts", samples=True, mode="valid"
    ... )
    >>> # Result contains no NaN values, only the overlapping data
    >>> assert not np.isnan(valid_aligned).any()
    """
    # Validate inputs and get coordinates.
    dim_name, coord_name = _validate_alignment_inputs(patch, kwargs)
    # We only require evenly sampled dim when we might need to expand it.
    must_be_even = mode == "full"
    dim = patch.get_coord(dim_name, require_evenly_sampled=must_be_even)
    coord = patch.get_coord(coord_name)

    # Get axes of shift and other dims.
    dim_axis = patch.dims.index(dim_name)
    coord_dims = patch.coords.dim_map[coord_name]
    coord_axes = tuple(patch.dims.index(x) for x in coord_dims)

    # Get the metadata about shift and the indices for shifting.
    inds = _get_shift_indices(coord, dim, reverse, samples, relative)
    meta = _calculate_shift_info(mode, inds, dim, dim_axis, patch.shape, reverse)

    # Apply shifts to data
    shifted_data = _apply_shifts_to_data(
        patch.data, meta, dim_axis, coord_axes, fill_value
    )
    assert shifted_data.ndim == patch.data.ndim, "dimensionality changed"

    # Get new coordinates.
    new_coords = _get_aligned_coords(patch, dim_name, meta)
    return patch.new(data=shifted_data, coords=new_coords)
