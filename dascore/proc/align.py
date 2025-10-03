"""
Functions to align patches based on some criterion.
"""

from typing import Literal

import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import patch_function


def _calculate_mode_parameters(mode, shifts, n_samples):
    """
    Calculate output size and offsets based on alignment mode.

    Parameters
    ----------
    mode
        One of "full", "valid", or "same".
    shifts
        Array of shift amounts.
    n_samples
        Original number of samples along the dimension.

    Returns
    -------
    dict with keys:
        output_size: Size of output dimension
        adjusted_shifts: Shifts adjusted for the output coordinate frame
        start_offset_samples: Offset in samples from original start
    """
    shifts = np.asarray(shifts, dtype=np.int64)
    min_shift, max_shift = shifts.min(), shifts.max()
    if mode == "full":
        # Expand dimension to fit all shifts
        output_size = n_samples + max_shift - min_shift
        adjusted_shifts = shifts - min_shift
        start_offset_samples = -min_shift
    elif mode == "same":
        # Keep original size
        output_size = n_samples
        adjusted_shifts = shifts
        start_offset_samples = 0
    else:  # mode == "valid"
        # Find overlapping region
        output_size = n_samples - (max_shift - min_shift)
        adjusted_shifts = shifts - min_shift
        start_offset_samples = max_shift - min_shift
    return {
        "output_size": output_size,
        "adjusted_shifts": adjusted_shifts,
        "start_offset_samples": start_offset_samples,
    }


def _apply_shifts_to_data(data, shifts, dim_axis, coord_axes, mode, fill_value):
    """
    Apply different shifts to data along specified axis.

    Parameters
    ----------
    data
        The input data array.
    shifts
        Array of shift amounts with dimensions matching coord_axes.
    dim_axis
        The axis along which to apply shifts.
    coord_axes
        The axes along which shift amounts vary.
    mode
        One of "full", "valid", or "same".
    fill_value
        Value to use for positions without data.
    """
    coord_axes = tuple(coord_axes) if not isinstance(coord_axes, tuple) else coord_axes
    # Calculate output parameters
    n_samples = data.shape[dim_axis]
    mode_params = _calculate_mode_parameters(mode, shifts, n_samples)
    output_size = mode_params["output_size"]
    adjusted_shifts = mode_params["adjusted_shifts"]
    start_pad = mode_params["start_offset_samples"]
    # Create output array
    out_shape = list(data.shape)
    out_shape[dim_axis] = output_size
    output = np.full(out_shape, fill_value, dtype=data.dtype)
    # Iterate over all shift positions
    for idx in np.ndindex(shifts.shape):
        shift = int(adjusted_shifts[idx])
        # Build slice for this position
        src_slice = [slice(None)] * data.ndim
        dst_slice = [slice(None)] * data.ndim
        for ax, i in zip(coord_axes, idx):
            src_slice[ax] = i
            dst_slice[ax] = i
        if mode == "valid":
            # All traces extract same window
            src_slice[dim_axis] = slice(start_pad, start_pad + output_size)
            dst_slice[dim_axis] = slice(None)
        else:
            # Calculate source and destination ranges
            src_start = 0
            src_end = n_samples
            dst_start = shift + start_pad
            dst_end = dst_start + n_samples
            # Clip to valid ranges
            if dst_start < 0:
                src_start -= dst_start
                dst_start = 0
            if dst_end > output_size:
                src_end -= dst_end - output_size
                dst_end = output_size
            src_slice[dim_axis] = slice(src_start, src_end)
            dst_slice[dim_axis] = slice(dst_start, dst_end)
        # Copy data
        output[tuple(dst_slice)] = data[tuple(src_slice)]
    return output


def _validate_alignment_inputs(patch, kwargs):
    """Ensure the inputs are usable."""
    # Must have exactly 1 key/value which must both be strings.
    if len(kwargs) != 1:
        msg = "align_to_coords requires exactly one keyword argument"
        raise ParameterError(msg)
    (dim_name, coord_name) = dict(kwargs).popitem()
    if not (isinstance(dim_name, str) and dim_name is not None):
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


def _get_aligned_coords(patch, dim_name, shifted_data, mode, inds, dim_axis):
    """
    Get the aligned coordinate manager.

    Parameters
    ----------
    patch
        The original patch.
    dim_name
        Name of the dimension that was shifted.
    shifted_data
        The shifted data array.
    mode
        The alignment mode (full/valid/same).
    inds
        The shift indices.
    dim_axis
        The axis index of the shifted dimension.
    """
    # Get the original coordinate for the shifted dimension
    dim_coord = patch.get_coord(dim_name)
    new_size = shifted_data.shape[dim_axis]
    # Calculate how the coordinate needs to be adjusted based on mode
    mode_params = _calculate_mode_parameters(mode, inds, len(dim_coord))
    offset_samples = mode_params["start_offset_samples"]
    # Update coordinate start and length
    if offset_samples == 0:
        new_coord = dim_coord.change_length(new_size)
    else:
        new_start = dim_coord.start + offset_samples * dim_coord.step
        new_coord = dim_coord.update(start=new_start).change_length(new_size)
    # Add reference coordinate with original dim values
    ref_coord_name = f"reference_{dim_name}"
    coords_dict = {dim_name: new_coord, ref_coord_name: (None, dim_coord.values)}
    return patch.coords.update(**coords_dict)


@patch_function()
def align_to_coord(
    patch: PatchType,
    mode: Literal["full", "valid", "same"] = "same",
    relative: bool = False,
    samples: bool = False,
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
        that values in the shift dimension's units.
    fill_value
        The value to insert in areas lacking data.

    **kwargs
        Used to specify the dimension which should shift and the coordinate
        to shift it to. The shift coordinate should not depend on the
        specified dimension.

    Returns
    -------
    A patch with new coordinate reference_{dim}

    Notes
    -----
    To understand the mode argument, consider a 2D patch with two traces,
    an alignment dimension length of 10, and a relative shift of 5 sample.
    For mode=="full" the output looks like this:
        -----aaaaaaaaaa
        bbbbbbbbbb-----
    For mode=="same" the patch will look like this:
        aaaaaaaaaa
        bbbbb-----
    for mode=="valid":
        aaaaa
        bbbbb
    where a and b represent values in first and second trace, respectively,
    and the - represents the fill values.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Create a suitable dimension for alignment.
    >>> # This will shift each channel 1 time sample relative to previous
    >>> time = patch.get_array("time")
    >>> distance = patch.get_array("distance")
    >>> ref_times = time[np.arange(len(distance))]
    >>> patch_coord = patch.update_coords(shift_time=("distance", ref_times))
    >>>
    >>> # Apply the alignment
    >>> out = patch_coord.align_to_coord(time="shift_time", mode="full")
    """
    dim_name, coord_name = _validate_alignment_inputs(patch, kwargs)
    dim = patch.get_coord(dim_name, require_evenly_sampled=True)
    coord = patch.get_coord(coord_name)
    # Get alignment indices.
    inds = dim.get_next_index(coord.values, samples=samples, relative=relative)
    # Get axes for shifting
    dim_axis = patch.dims.index(dim_name)
    coord_dims = patch.coords.dim_map[coord_name]
    coord_axes = tuple(patch.dims.index(x) for x in coord_dims)
    # Apply shifts to data
    shifted_data = _apply_shifts_to_data(
        patch.data, inds, dim_axis, coord_axes, mode, fill_value
    )
    assert shifted_data.ndim == patch.data.ndim, "dimensionality changed"
    # Get new coord manager
    new_coords = _get_aligned_coords(
        patch, dim_name, shifted_data, mode, inds, dim_axis
    )
    return patch.new(data=shifted_data, coords=new_coords)
