"""Processing for muting (zeroing) patch data in specified regions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import Quantity
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import patch_function
from dascore.utils.signal import WINDOW_FUNCTIONS, _get_window_function
from dascore.utils.time import to_float


parameter_docstring = """
Parameters
----------
patch
    The patch instance.
mode
    - "union": mute data inside the specified region [default]
    - "complement": mute data outside the specified region
taper
    Taper width at mute boundaries. Can be:
    - None: sharp mute (no taper)
    - float (0.0-1.0): fraction of dimension range (e.g., 0.05 = 5%)
      The number of samples in the mute is held constant across dimensions
      by calculating the samples the fraction represents for each dimension
      and using the minimum sample count. This helps avoid distorted mutes
      based on dimensions with vastly different lengths.
    - Quantity with units: absolute value (e.g., 0.02*dc.units.s)
      Note: If multiple dimensions specified, must use dict.
    - dict: {dim: taper_value} for dimension-specific taper
      Values can be floats (fractions) or Quantities (absolute)
    - Callable: custom function that receives and modifies envelope array.
window_type
    Window function for tapering (only used for non-callable taper).
    Options:
        {taper_type}.
relative
    If True (default), values are relative to coordinate edges.
    Positive values are offsets from start, negative from end.
samples
    If True, values specified in samples rather than coordinate values.
**kwargs
    Dimension specifications as (boundary_1, boundary_2) pairs.
    Each boundary can be:
    - Scalar: constant value (defines plane perpendicular to dimension)
    - Array/list: coordinates defining line/curve/surface
    - None or ...: edge of coordinate (min or max)

"""



@patch_function()
@compose_docstring(taper_type=sorted(WINDOW_FUNCTIONS), params=parameter_docstring)
def mute_envelope(
    patch: PatchType,
    *,
    mode: Literal["union", "complement"] = "union",
    taper: float | dict | None = None,
    window_type: str = "hann",
    relative: bool = True,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    """
    Calculate a patch envelope which applies mute defined by parameters.

    The resulting patch simply needs to be multiplied by the original
    patch to achieve muting. This allows for more fine-grained control
    of the

    """


@patch_function()
@compose_docstring(
    taper_type=sorted(WINDOW_FUNCTIONS),
    params=parameter_docstring,
)
def mute(
    patch: PatchType,
    *,
    mode: Literal["union", "complement"] = "union",
    taper: float | dict | None = None,
    window_type: str = "hann",
    relative: bool = True,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    """
    Mute (zero out) patch data in specified regions.

    Each dimension accepts two boundary specifications that define
    the mute region. Boundaries can be scalar values (for planes/blocks)
    or arrays of values (for lines/curves/surfaces).

    {params}

    Examples
    --------
    >>> import dascore as dc
    >>> from scipy.ndimage import gaussian_filter
    >>>
    >>> patch = dc.get_example_patch("ricker_moveout")
    >>>
    >>>
    >>> # Mute first 0.5s (relative to start by default)
    >>> muted = patch.mute(time=(0, 0.5))
    >>>
    >>> # Mute everything except middle section
    >>> kept = patch.mute(time=(0.2, -0.2), mode="complement")
    >>>
    >>> # Taper with absolute units
    >>> muted = patch.mute(
    ...     time=(0.2, 0.8),
    ...     taper={'time': 0.02 * dc.units.s},
    ... )
    >>>
    >>> # Custom taper function
    >>> def custom_taper(envelope):
    ...     # Apply gaussian filter to envelope to smooth sharp edges.
    ...     return gaussian_filter(envelope, sigma=2)
    >>>
    >>> muted = patch.mute(time=(0.2, 0.8), taper=custom_taper)
    >>>
    >>> # --- LINEAR VELOCITY MUTES ---
    >>>
    >>> # Classic first break mute: mute early arrivals
    >>> # Line from (t=0, d=0) to (t=0.3, d=300) defines velocity=1000 m/s
    >>> muted = patch.mute(
    ...     time=(0, [0, 0.3]),
    ...     distance=(0, [0, 300]),
    ...     taper=0.02,
    ...     relative=False,
    ... )
    >>>
    >>> # Mute late arrivals: from velocity line to end
    >>> muted = patch.mute(
    ...     time=([0, 0.3], None),
    ...     distance=([0, 300], None),
    ...     relative=False,
    ... )
    >>>
    >>> # Mute wedge between two velocity lines
    >>> muted = patch.mute(
    ...     time=([0, 0.375], [0, 0.25]),
    ...     distance=([0, 300], [0, 300]),
    ...     relative=False,
    ... )
    >>>
    >>> # Curved mute using multiple control points
    >>> muted = patch.mute(
    ...     time=(0, [0, 0.1, 0.25, 0.35]),
    ...     distance=(0, [0, 50, 200, 300]),
    ...     relative=False,
    ... )
    >>>
    >>> # --- MIXED GEOMETRY ---
    >>>
    >>> # Linear mute in time-distance, block in other dimension
    >>> patch_3d = dc.get_example_patch("nd_patch", dim_count=3)
    >>> muted = patch_3d.mute(
    ...     dim_1=(0, [0, 5]),
    ...     dim_2=(0, [0, 5]),
    ...     dim_3=(2, 8),
    ... )

    Notes
    -----
    - For linear/planar mutes, all specified dimensions must have the
      same number of points to define the geometry.
    - Taper is applied perpendicular to the mute boundary.
    - Block mutes can be combined with planar mutes across
      different dimensions.
    - By default, relative=True means values are offsets from coordinate
      edges. Use relative=False for absolute coordinate values.

    See Also
    --------
    [`Patch.select`](`dascore.Patch.select`)
    [`Patch.taper_range`](`dascore.Patch.taper_range`)
    """
    # Validate we have dimension specifications
    if not kwargs:
        msg = "At least one dimension must be specified for muting"
        raise ParameterError(msg)

    # Parse boundaries and categorize them
    scalar_bounds, array_bounds = _parse_mute_boundaries(patch, kwargs)

    # Validate array boundaries have matching lengths
    if array_bounds:
        _validate_array_boundaries(array_bounds)

    # Handle coordinate conversions based on mode
    if samples:
        # Samples mode: values are already indices
        # If relative, treat as relative indices (offset from start/end)
        # If not relative, treat as absolute indices
        if relative:
            # Convert relative sample indices to absolute
            scalar_bounds = _convert_relative_samples(patch, scalar_bounds)
            array_bounds = _convert_relative_samples_arrays(patch, array_bounds)
        # else: already absolute sample indices, use directly
    else:
        # Coordinate mode: values are coordinate values
        if relative:
            # Convert relative coordinates to absolute
            scalar_bounds = _convert_scalar_relative(patch, scalar_bounds)
            array_bounds = _convert_array_relative(patch, array_bounds)
        # Now convert coordinate values to sample indices
        scalar_bounds = _convert_to_samples(patch, scalar_bounds)
        array_bounds = _convert_arrays_to_samples(patch, array_bounds)

    # Build mute envelope
    envelope = np.ones(patch.shape, dtype=float)

    # Apply block mutes (scalar boundaries)
    if scalar_bounds:
        block_envelope = _get_block_mute_envelope(
            patch, scalar_bounds, taper, window_type
        )
        envelope *= block_envelope

    # Apply geometric mutes (array boundaries) - Phase 2
    if array_bounds:
        geom_envelope = _get_geometric_mute_envelope(
            patch, array_bounds, taper, window_type
        )
        envelope *= geom_envelope

    # Apply custom taper function if provided
    if callable(taper):
        envelope = taper(envelope)
        if not isinstance(envelope, np.ndarray):
            msg = (
                "Custom taper function must return a numpy array. "
                f"Got {type(envelope)}"
            )
            raise ParameterError(msg)
        if envelope.shape != patch.shape:
            msg = (
                f"Custom taper function must return array with shape {patch.shape}. "
                f"Got shape {envelope.shape}"
            )
            raise ParameterError(msg)

    # Invert envelope for complement mode
    if mode == "complement":
        envelope = 1.0 - envelope

    # Apply envelope to data
    return patch.new(data=patch.data * envelope)


def _parse_mute_boundaries(patch, kwargs):
    """
    Parse kwargs into boundary specifications.

    Returns
    -------
    scalar_bounds : dict
        {dim: (min, max)} for dimensions with scalar boundaries
    array_bounds : dict
        {dim: (array1, array2)} for dimensions with array boundaries
    """
    scalar_bounds = {}
    array_bounds = {}

    for dim, value in kwargs.items():
        # Validate dimension exists
        if dim not in patch.dims:
            valid_dims = sorted(patch.dims)
            msg = f"Dimension '{dim}' not found. Valid dimensions: {valid_dims}"
            raise ParameterError(msg)

        # Must be a tuple of length 2
        if not isinstance(value, tuple) or len(value) != 2:
            msg = (
                f"Each dimension must specify 2 boundaries as a tuple. "
                f"Got {value} for dimension '{dim}'"
            )
            raise ParameterError(msg)

        bound1, bound2 = value

        # Check if boundaries are scalars or arrays
        is_scalar_1 = _is_scalar_or_none(bound1)
        is_scalar_2 = _is_scalar_or_none(bound2)

        if is_scalar_1 and is_scalar_2:
            scalar_bounds[dim] = (bound1, bound2)
        else:
            # At least one is an array
            # Only convert to array if it's actually array-like (not scalar)
            if is_scalar_1:
                # bound1 is scalar, bound2 is array - this is an error for now
                msg = (
                    f"Cannot mix scalar and array boundaries for dimension '{dim}'. "
                    f"Both boundaries must be scalars or both must be arrays with "
                    f"matching lengths."
                )
                raise ParameterError(msg)
            arr1 = None if bound1 is None else np.atleast_1d(bound1)
            arr2 = None if bound2 is None else np.atleast_1d(bound2)
            array_bounds[dim] = (arr1, arr2)

    return scalar_bounds, array_bounds


def _is_scalar_or_none(value):
    """Check if value is scalar or None (not an array/list)."""
    if value is None or value is ...:
        return True
    return np.isscalar(value)


def _validate_array_boundaries(array_bounds):
    """Validate that all array boundaries have matching lengths."""
    lengths = {}
    for dim, (arr1, arr2) in array_bounds.items():
        len1 = len(arr1) if arr1 is not None else None
        len2 = len(arr2) if arr2 is not None else None
        if len1 is not None:
            lengths[f"{dim}_bound1"] = len1
        if len2 is not None:
            lengths[f"{dim}_bound2"] = len2

    # Check all non-None lengths match
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        msg = (
            "All array boundaries must have the same number of points. "
            f"Got varying lengths: {lengths}"
        )
        raise ParameterError(msg)


def _convert_scalar_relative(patch, scalar_bounds):
    """Convert relative scalar values to absolute coordinate values."""
    result = {}
    for dim, (val1, val2) in scalar_bounds.items():
        coord = patch.get_coord(dim)
        coord_min, coord_max = coord.min(), coord.max()

        # Convert each boundary
        abs_val1 = _relative_to_absolute_scalar(val1, coord_min, coord_max)
        abs_val2 = _relative_to_absolute_scalar(val2, coord_min, coord_max)
        result[dim] = (abs_val1, abs_val2)

    return result


def _convert_array_relative(patch, array_bounds):
    """Convert relative array values to absolute coordinate values."""
    result = {}
    for dim, (arr1, arr2) in array_bounds.items():
        coord = patch.get_coord(dim)
        coord_min, coord_max = coord.min(), coord.max()

        # Convert each boundary array
        abs_arr1 = _relative_to_absolute_array(arr1, coord_min, coord_max)
        abs_arr2 = _relative_to_absolute_array(arr2, coord_min, coord_max)
        result[dim] = (abs_arr1, abs_arr2)

    return result


def _relative_to_absolute_scalar(value, coord_min, coord_max):
    """Convert single relative value to absolute."""
    if value is None or value is ...:
        return value

    # Handle datetime/timedelta types
    if isinstance(coord_min, np.datetime64):
        # Convert value to timedelta if it's a number
        if not isinstance(value, (np.timedelta64, np.datetime64)):
            value = np.timedelta64(int(value * 1e9), 'ns')
        if value >= np.timedelta64(0):
            return coord_min + value
        else:
            return coord_max + value
    else:
        # Numeric types
        if value >= 0:
            return coord_min + value
        else:
            return coord_max + value


def _relative_to_absolute_array(arr, coord_min, coord_max):
    """Convert array of relative values to absolute."""
    if arr is None:
        return None

    # Handle datetime/timedelta types
    if isinstance(coord_min, np.datetime64):
        # Convert values to timedeltas if they're numbers
        if not isinstance(arr[0], (np.timedelta64, np.datetime64)):
            arr = np.array([np.timedelta64(int(v * 1e9), 'ns') for v in arr])
        result = np.empty_like(arr, dtype=coord_min.dtype)
        positive_mask = arr >= np.timedelta64(0)
        result[positive_mask] = coord_min + arr[positive_mask]
        result[~positive_mask] = coord_max + arr[~positive_mask]
        return result
    else:
        result = np.empty_like(arr, dtype=float)
        positive_mask = arr >= 0
        result[positive_mask] = coord_min + arr[positive_mask]
        result[~positive_mask] = coord_max + arr[~positive_mask]
        return result


def _convert_relative_samples(patch, scalar_bounds):
    """Convert relative sample indices to absolute indices."""
    result = {}
    for dim, (val1, val2) in scalar_bounds.items():
        coord = patch.get_coord(dim)
        length = len(coord)

        # Convert each boundary
        abs_val1 = _relative_sample_to_absolute(val1, length)
        abs_val2 = _relative_sample_to_absolute(val2, length)
        result[dim] = (abs_val1, abs_val2)

    return result


def _convert_relative_samples_arrays(patch, array_bounds):
    """Convert relative sample indices in arrays to absolute indices."""
    result = {}
    for dim, (arr1, arr2) in array_bounds.items():
        coord = patch.get_coord(dim)
        length = len(coord)

        # Convert each boundary array
        abs_arr1 = _relative_samples_array_to_absolute(arr1, length)
        abs_arr2 = _relative_samples_array_to_absolute(arr2, length)
        result[dim] = (abs_arr1, abs_arr2)

    return result


def _relative_sample_to_absolute(value, length):
    """Convert relative sample index to absolute."""
    if value is None or value is ...:
        return value
    if value >= 0:
        return int(value)
    else:
        return int(length + value)


def _relative_samples_array_to_absolute(arr, length):
    """Convert array of relative sample indices to absolute."""
    if arr is None:
        return None

    result = np.empty_like(arr, dtype=int)
    for i, val in enumerate(arr):
        if val >= 0:
            result[i] = int(val)
        else:
            result[i] = int(length + val)
    return result


def _convert_to_samples(patch, scalar_bounds):
    """Convert scalar boundaries from coordinate values to sample indices."""
    result = {}
    for dim, (val1, val2) in scalar_bounds.items():
        coord = patch.get_coord(dim)
        # Convert to indices
        idx1 = _value_to_index(coord, val1) if val1 is not None else None
        idx2 = _value_to_index(coord, val2) if val2 is not None else None
        result[dim] = (idx1, idx2)
    return result


def _convert_arrays_to_samples(patch, array_bounds):
    """Convert array boundaries from coordinate values to sample indices."""
    result = {}
    for dim, (arr1, arr2) in array_bounds.items():
        coord = patch.get_coord(dim)
        idx_arr1 = _array_to_indices(coord, arr1) if arr1 is not None else None
        idx_arr2 = _array_to_indices(coord, arr2) if arr2 is not None else None
        result[dim] = (idx_arr1, idx_arr2)
    return result


def _value_to_index(coord, value):
    """Convert coordinate value to index."""
    if value is None or value is ...:
        return None
    return coord.get_next_index(value)


def _array_to_indices(coord, arr):
    """Convert array of coordinate values to indices."""
    if arr is None:
        return None
    return np.array([coord.get_next_index(val) for val in arr])


def _parse_taper_value(taper_val, coord, dim_name):
    """
    Parse a taper value into absolute coordinate units.

    Parameters
    ----------
    taper_val
        Can be float (fraction), Quantity (absolute), or None
    coord
        The coordinate object for this dimension
    dim_name
        Name of the dimension (for error messages)

    Returns
    -------
    float or timedelta
        Taper width in coordinate units, or None if taper_val is None
    """
    if taper_val is None:
        return None

    # Check if it's a Quantity with units
    if isinstance(taper_val, Quantity):
        # For time quantities, keep as timedelta; for others convert to float
        coord_range = coord.max() - coord.min()
        if isinstance(coord_range, np.timedelta64):
            # Convert Quantity to timedelta64
            # First convert to seconds, then to nanoseconds for timedelta64
            seconds = taper_val.to('s').magnitude
            return np.timedelta64(int(seconds * 1e9), 'ns')
        else:
            # Convert to float (handles distance, etc.)
            return float(taper_val.magnitude)

    # Otherwise it's a float - treat as fraction of dimension range
    if not isinstance(taper_val, (int, float)):
        msg = (
            f"Taper value for dimension '{dim_name}' must be a float "
            f"(fraction) or Quantity (with units). Got {type(taper_val)}"
        )
        raise ParameterError(msg)

    # Get dimension range and multiply by fraction
    coord_range = coord.max() - coord.min()
    if isinstance(coord_range, np.timedelta64):
        # For datetime coords, multiply timedelta by fraction
        return coord_range * taper_val
    else:
        return taper_val * coord_range


def _validate_taper_dict(taper, scalar_bounds):
    """
    Validate that taper dict has valid dimension names.

    Parameters
    ----------
    taper : dict
        Dictionary of dimension names to taper values
    scalar_bounds : dict
        Dictionary of scalar boundaries being used
    """
    if not isinstance(taper, dict):
        return

    for dim in taper:
        if dim not in scalar_bounds:
            valid_dims = sorted(scalar_bounds.keys())
            msg = (
                f"Taper dimension '{dim}' not found in mute dimensions. "
                f"Valid dimensions: {valid_dims}"
            )
            raise ParameterError(msg)


def _get_block_mute_envelope(patch, scalar_bounds, taper, window_type):
    """
    Create envelope for rectangular block mutes.

    Similar to taper_range but creates a mute envelope instead.
    """
    # Validate taper if it's a dict
    if isinstance(taper, dict):
        _validate_taper_dict(taper, scalar_bounds)

    # Check if taper is a Quantity with multiple dimensions
    if isinstance(taper, Quantity) and len(scalar_bounds) > 1:
        msg = (
            "Cannot use Quantity (with units) for taper when multiple dimensions "
            "are specified. Use a dict like: taper={'time': 0.02*dc.units.s, "
            "'distance': 10*dc.units.m}"
        )
        raise ParameterError(msg)

    envelope = np.ones(patch.shape, dtype=float)

    for dim, (val1, val2) in scalar_bounds.items():
        coord = patch.get_coord(dim)
        axis = patch.get_axis(dim)

        # Values are already sample indices at this point
        # Handle None values (edges)
        if val1 is None or val1 is ...:
            val1 = 0
        if val2 is None or val2 is ...:
            val2 = len(coord) - 1

        # Create slice object from indices
        slice_obj = slice(int(val1), int(val2) + 1)

        # Create 1D envelope for this dimension
        dim_envelope = np.ones(len(coord))
        dim_envelope[slice_obj] = 0  # Zero out mute region

        # Apply taper if specified and not callable
        if taper is not None and not callable(taper):
            # Get taper value for this dimension
            if isinstance(taper, dict):
                taper_val = taper.get(dim)
            else:
                taper_val = taper

            if taper_val is not None:
                # Parse taper value (handles fraction vs Quantity)
                taper_width = _parse_taper_value(taper_val, coord, dim)
                dim_envelope = _apply_taper_1d(
                    dim_envelope, slice_obj, taper_width, coord, window_type
                )

        # Broadcast to full shape and multiply
        indexer = broadcast_for_index(
            patch.ndim, axis, value=slice(None), fill=None
        )
        envelope *= dim_envelope[indexer]

    return envelope


def _apply_taper_1d(envelope, slice_obj, taper_width, coord, window_type):
    """Apply taper to 1D envelope at boundaries."""
    func = _get_window_function(window_type)

    # Get indices of mute region
    start_idx = slice_obj.start if slice_obj.start is not None else 0
    stop_idx = slice_obj.stop if slice_obj.stop is not None else len(coord)

    # Convert taper width to samples
    if hasattr(coord, 'step') and coord.step is not None:
        # Handle both timedelta and numeric step/taper_width
        if isinstance(taper_width, np.timedelta64):
            # Both are timedeltas - can divide directly
            taper_samples = int(taper_width / coord.step)
        elif isinstance(coord.step, np.timedelta64):
            # taper_width is numeric, step is timedelta - shouldn't happen
            # Convert timedelta step to float for division
            step_float = to_float(coord.step)
            taper_samples = int(taper_width / step_float)
        else:
            # Both numeric
            taper_samples = int(taper_width / coord.step)
    else:
        # For non-evenly sampled coords, estimate
        coord_range = coord.max() - coord.min()
        if isinstance(taper_width, np.timedelta64) and isinstance(coord_range, np.timedelta64):
            # Both timedeltas
            taper_samples = int(len(coord) * (taper_width / coord_range))
        elif isinstance(taper_width, np.timedelta64):
            # Convert timedelta to float
            taper_float = to_float(taper_width)
            range_float = to_float(coord_range)
            taper_samples = int(len(coord) * taper_float / range_float)
        elif isinstance(coord_range, np.timedelta64):
            # Convert coord_range to float
            range_float = to_float(coord_range)
            taper_samples = int(len(coord) * taper_width / range_float)
        else:
            # Both numeric
            taper_samples = int(len(coord) * taper_width / coord_range)

    taper_samples = max(1, min(taper_samples, (stop_idx - start_idx) // 2))

    # Apply taper at start of mute region
    if taper_samples > 0 and start_idx > 0:
        window = func(2 * taper_samples)[:taper_samples]
        taper_start = max(0, start_idx - taper_samples)
        envelope[taper_start:start_idx] *= window

    # Apply taper at end of mute region
    if taper_samples > 0 and stop_idx < len(coord):
        window = func(2 * taper_samples)[taper_samples:]
        taper_end = min(len(coord), stop_idx + taper_samples)
        envelope[stop_idx:taper_end] *= window[:taper_end - stop_idx]

    return envelope


def _get_geometric_mute_envelope(patch, array_bounds, taper, window_type):
    """
    Create envelope for geometric (line/plane/hyperplane) mutes.

    This is Phase 2 - to be implemented for linear/planar mutes.
    For now, raise an informative error.
    """
    msg = (
        "Geometric mutes (using arrays to define lines/planes) are not yet "
        "implemented. Use scalar boundaries for block mutes, or stay tuned "
        "for the next release!"
    )
    raise NotImplementedError(msg)
