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


def _apply_shifts_to_data(data, shifts, dim_axis, coord_axis, mode, fill_value):
    """
    Apply different shifts to data along specified axis using vectorization.

    Parameters
    ----------
    data
        The input data array.
    shifts
        Array of shift amounts, one per position along coord_axis.
    dim_axis
        The axis along which to apply shifts.
    coord_axis
        The axis along which shift amounts vary.
    mode
        One of "full", "valid", or "same".
    fill_value
        Value to use for positions without data.
    """
    # Move axes so dim_axis is last and coord_axis is first for easier indexing
    data = np.moveaxis(data, [coord_axis, dim_axis], [0, -1])
    n_traces, n_samples = data.shape[0], data.shape[-1]
    # Determine output size and padding based on mode
    mode_params = _calculate_mode_parameters(mode, shifts, n_samples)
    output_size = mode_params["output_size"]
    adjusted_shifts = mode_params["adjusted_shifts"]
    start_pad = mode_params["start_offset_samples"]
    # Create output array filled with fill_value
    out_shape = list(data.shape)
    out_shape[-1] = output_size
    output = np.full(out_shape, fill_value, dtype=data.dtype)
    # Use advanced indexing to assign shifted data
    trace_indices = np.arange(n_traces)[:, None]
    if mode == "valid":
        # For valid mode, extract overlapping region from source WITHOUT shifting
        # All traces read the same window: the overlapping region
        sample_indices = np.arange(output_size)
        # For valid, we take the window from max_shift to max_shift + output_size
        # This is the region that overlaps for all traces after alignment
        # Broadcast to all traces
        source_indices = np.broadcast_to(
            sample_indices + start_pad, (n_traces, output_size)
        )
        dest_indices = np.broadcast_to(sample_indices, (n_traces, output_size))
        # All indices should be valid by construction
        valid_mask = np.ones((n_traces, output_size), dtype=bool)
    else:
        # For full/same modes, shift data into destination positions
        sample_indices = np.arange(n_samples)
        dest_indices = sample_indices + adjusted_shifts[:, None] + start_pad
        source_indices = sample_indices
        # Mask valid destination indices
        valid_mask = (dest_indices >= 0) & (dest_indices < output_size)
    # Expand indices to handle arbitrary number of middle dimensions
    for _ in range(data.ndim - 2):
        trace_indices = trace_indices[..., None]
        valid_mask = valid_mask[..., None]
        source_indices = source_indices[..., None]
        dest_indices = dest_indices[..., None]
    # Use advanced indexing - need to flatten and rebuild
    flat_valid = valid_mask.ravel()
    flat_traces = np.broadcast_to(trace_indices, valid_mask.shape).ravel()[flat_valid]
    flat_source = np.broadcast_to(source_indices, valid_mask.shape).ravel()[flat_valid]
    flat_dest = np.broadcast_to(dest_indices, valid_mask.shape).ravel()[flat_valid]
    # Build full index tuple for all dimensions
    source_idx = [flat_traces, ..., flat_source]
    dest_idx = [flat_traces, ..., flat_dest]
    output[tuple(dest_idx)] = data[tuple(source_idx)]
    # Move axes back to original positions
    output = np.moveaxis(output, [0, -1], [coord_axis, dim_axis])
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
        patch.data, inds, dim_axis, coord_axes[0], mode, fill_value
    )
    assert shifted_data.ndim == patch.data.ndim, "dimensionality changed"
    # Get new coord manager
    new_coords = _get_aligned_coords(
        patch, dim_name, shifted_data, mode, inds, dim_axis
    )
    return patch.new(data=shifted_data, coords=new_coords)
