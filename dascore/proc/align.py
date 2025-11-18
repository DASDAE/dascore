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


def _get_dest_indices(shifts, start_shift, end_shift, n_samples, output_size):
    """Get indices in the output array."""
    start = np.maximum(0, shifts - start_shift)
    # Clamp source indices to valid range [0, n_samples]
    min_start = -shifts + start_shift
    source_start = np.clip(min_start, 0, n_samples)
    max_end = n_samples - shifts + end_shift
    source_end = np.clip(max_end, 0, n_samples)
    # Compute source length, ensuring non-negative
    source_length = np.maximum(source_end - source_start, 0)
    # Destination end bounded to [0, output_size]
    end = np.clip(start + source_length, 0, output_size)
    return (start, end)


def _calculate_shift_info(
    mode,
    shifts,
    dim,
    dim_axis,
    patch_shape,
):
    """
    Calculate metadata about the shift.
    """
    # Validate mode parameter
    valid_modes = ("full", "same", "valid")
    if mode not in valid_modes:
        msg = f"mode must be one of {valid_modes} but got '{mode}'"
        raise ParameterError(msg)

    # Convert shifts to array
    shifts = np.asarray(shifts, dtype=np.int64)
    n_samples = len(dim)
    min_shift, max_shift = shifts.min(), shifts.max()

    if mode == "full":
        # Shift value to left/right most index in reference to stationary trace.
        start_shift = min_shift
        end_shift = max_shift
    elif mode == "same":
        # Keep original size (and dimension)
        start_shift = 0
        end_shift = 0
    else:
        # mode == "valid"
        # Find overlapping region
        start_shift = max_shift if max_shift > 0 else 0
        end_shift = min_shift if min_shift < 0 else 0

    output_size = n_samples + end_shift - start_shift
    # get slices for start/stop
    source_inds = _get_source_indices(shifts, start_shift, end_shift, n_samples)
    dest_inds = _get_dest_indices(
        shifts, start_shift, end_shift, n_samples, output_size
    )
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
    for i, ax in zip(idx, coord_axes, strict=True):
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
        msg = "align_to_coord requires exactly one keyword argument"
        raise ParameterError(msg)
    (dim_name, coord_name) = dict(kwargs).popitem()
    if not isinstance(coord_name, str):
        msg = "align_to_coord requires keyword name and value to be strings."
        raise ParameterError(msg)
    # key must be a dimension and value a non-dimensional coord name.
    if dim_name not in patch.dims:
        msg = (
            f"align_to_coord requires the keyword to be a dimension, but "
            f"{dim_name} is not one of the patch dimensions: {patch.dims}"
        )
        raise ParameterError(msg)
    if coord_name not in patch.coords.coord_map or coord_name in patch.dims:
        msg = (
            f"align_to_coord requires the value to be the name of a non-dimensional "
            f"coordinate but '{coord_name}' is either not a coordinate or also "
            f"a dimension. Patch dimensions are: {patch.dims} and "
            f"patch coords: {patch.coords}"
        )
        raise ParameterError(msg)
    # Value must not depend on selected dim.
    value_dims = patch.coords.dim_map[coord_name]
    if dim_name in value_dims:
        msg = (
            f"align_to_coord requires the align coord not depend on the selected"
            f" dimension but {dim_name} does."
        )
        raise ParameterError(msg)
    return dim_name, coord_name


def _get_aligned_coords(patch, dim_name, meta, shift_coord, samples, relative):
    """
    Get the aligned coordinate manager.
    """
    coord = patch.get_coord(dim_name)
    start_shift, end_shift = meta.start_shift, meta.end_shift

    # Calculate new minimum based on mode
    if samples or relative:
        # Samples/Relative mode: use the calculated start_shift from metadata
        # This preserves the original behavior where shifts are relative offsets
        new_min = coord.min() + coord.step * start_shift
    else:
        # Absolute mode: shift coordinate values specify exact absolute positions
        # The new minimum is the minimum value from the shift coordinate
        new_min = np.min(shift_coord.values)

    new_coord = coord.update(
        min=new_min,
    ).change_length(len(coord) - start_shift + end_shift)
    return patch.coords.update(**{dim_name: new_coord})


def _get_shift_indices(coord, dim, reverse, samples, relative):
    """
    Get the indices  for shifting.

    Positive values indicate a shift to the right, negative to left.
    """
    # First get naive indice
    # In absolute mode (not samples, not relative), allow out of bounds
    # since shift values define new absolute positions
    allow_oob = not (samples or relative)
    inds = dim.get_next_index(
        coord.values, samples=samples, relative=relative, allow_out_of_bounds=allow_oob
    )
    # Use the point with the least shift as a reference.
    ref = np.argmin(np.abs(inds))
    out = inds - inds[ref]
    # Reverse index if needed. This way the reference stays the same.
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
        If True, the values in the alignment coordinate specify offsets from
        the original start position (in the dimension's units). If False and
        samples is False (absolute mode), the values specify the exact absolute
        positions where each trace should start.
    samples
        If True, the values in the alignment coordinate specify offsets in
        sample counts from the original start position. If False and relative
        is False (absolute mode), the values specify exact absolute positions
        in the dimension's units.
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

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch()
    >>> # Select a smaller subset for examples
    >>> patch = patch.select(distance=(0, 50))
    >>>
    >>> # Example 1: Absolute mode - align traces to specific times
    >>> # Each trace will start at the specified absolute time value
    >>> time = patch.get_coord("time")
    >>> distance = patch.get_array("distance")
    >>> # Specify exact start times for each trace (as datetime64)
    >>> dt = np.timedelta64(1, 'ms')  # 1 millisecond
    >>> start_times = time.min() + np.arange(len(distance)) * dt
    >>> patch_abs = patch.update_coords(abs_times=("distance", start_times))
    >>> # Traces will now start at specified absolute times
    >>> aligned_abs = patch_abs.align_to_coord(time="abs_times", mode="full")
    >>>
    >>> # Example 2: Round-trip alignment with reverse parameter
    >>> shifts = np.arange(len(distance))  # shifts in samples
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
    >>> assert not np.isnan(valid_aligned.data).any()

    Notes
    -----
    **Alignment Meaning

    The `relative` and `samples` parameters control how shift values are interpreted:

    - **Absolute mode** (relative=False, samples=False): Shift coordinate values
      specify the exact absolute position where each trace should start. For example,
      if the shift coordinate contains [2.0, 2.5, 3.0], traces will start at times
      2.0, 2.5, and 3.0 respectively.

    - **Relative mode** (relative=True): Shift coordinate values specify offsets
      from the original start position in the dimension's units. For example, if
      the original start is 0.0 and shifts are [0.0, 0.5, 1.0], traces will start
      at 0.0, 0.5, and 1.0.

    - **Samples mode** (samples=True): Shift coordinate values specify offsets in
      sample counts. For example, shifts of [0, 1, 2] with step=0.1 will offset
      traces by 0, 0.1, and 0.2 in the dimension's units.

    **Mode Parameter:**

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
    meta = _calculate_shift_info(mode, inds, dim, dim_axis, patch.shape)

    # Apply shifts to data
    shifted_data = _apply_shifts_to_data(
        patch.data, meta, dim_axis, coord_axes, fill_value
    )
    assert shifted_data.ndim == patch.data.ndim, "dimensionality changed"

    # Get new coordinates.
    new_coords = _get_aligned_coords(patch, dim_name, meta, coord, samples, relative)
    return patch.new(data=shifted_data, coords=new_coords)
