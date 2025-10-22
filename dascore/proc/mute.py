"""Processing for muting (zeroing) patch data in specified regions."""

from __future__ import annotations

from collections.abc import Mapping, Sized
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import (
    get_2d_line_intersection,
)
from dascore.utils.models import DascoreBaseModel
from dascore.utils.patch import get_dim_axis_value, patch_function


class _MuteGeometry(DascoreBaseModel):
    """
    Parent class for Mute Geometry.
    """

    dims: tuple[str, ...]
    axes: tuple[int, ...]
    relative: bool = True

    @classmethod
    def from_params(cls, vals, dims, axes, patch, relative):
        """Initialize Mute Geometry from input parameters."""

    def _mask_array(
        self,
        array: NDArray,
    ):
        """Apply the mask on the array for envelope calculation."""

    def _apply_smoothing(self, array, smooth, patch):
        """Apply smoothing to the array."""
        sigma, axes = self._get_smooth_sigma(smooth, patch)
        return gaussian_filter(array, sigma=sigma, axes=axes)

    def _get_smooth_sigma(self, smooth, patch):
        """Get sigma values, in samples, for the gaussian kernel."""

        def _broadcast_smooth_to_dims(dims, smooth):
            """Broadcast smooth samples to dims length."""
            # First, we need to get the smooth parameters to line up with the dims.
            # For a single value, just broadcast to dim length.
            if not isinstance(smooth, Mapping):
                vals = [smooth] * len(dims)
                axes = self.axes
            else:
                # Otherwise, the smooth dict must be a subset of the dimensions.
                if not set(smooth).issubset(set(dims)):
                    msg = (
                        f"If a smooth dictionary is used in Mute, it must be a "
                        f"subset of the dims in kwargs. Kwarg dims are {dims} and"
                        f"smooth keys are {list(smooth)}."
                    )
                    raise ParameterError(msg)
                vals = [smooth[dim] for dim in dims if dim in smooth]
                axes = [self.dims.index(x) for x in smooth]

            return vals, axes

        def _convert_to_samples(smooth, dims, patch):
            """Convert the smooth parameter to number of samples."""
            out = []
            for dim, val in zip(dims, smooth):
                coord = patch.get_coord(dim)
                if val is None:
                    out.append(0)
                elif isinstance(val, int | np.integer):
                    out.append(val)
                elif isinstance(val, float | np.floating):
                    if not 0 <= val <= 1:
                        msg = (
                            f"Mute's smooth parameter for {dim} must be between 0 "
                            f"and 1 when using a floating point value."
                        )
                        raise ParameterError(msg)
                    out.append(int(np.round(len(coord) * val)))
                else:  # should capture quantities
                    out.append(coord.get_sample_count(val))
            return out

        smooth_by_dims, axes = _broadcast_smooth_to_dims(self.dims, smooth)
        smooth_ints = _convert_to_samples(smooth_by_dims, self.dims, patch)
        # Now convert to input format for scipy's gaussian filter.
        return smooth_ints, axes


class _MuteGeometry1D(_MuteGeometry):
    """
    Private container to manage 1D Mute Geometry.
    """

    lims: tuple

    @classmethod
    def from_params(cls, vals, dims, axes, patch, relative):
        """Initialize Mute Geometry from input parameters."""
        vals = np.array(vals)
        if vals.size != 2:  # Handle single dimension Mute.
            msg = "Mute requires two boundaries when using a single dimension."
            raise ParameterError(msg)
        lims = (vals[0][0], vals[0][1])
        return cls(dims=dims, axes=axes, relative=relative, lims=lims)

    def _apply_mask(self, array: NDArray, patch: dc.Patch, fill_value) -> NDArray:
        coord = patch.get_coord(self.dims[0])
        _, c_index = coord.select(self.lims, relative=self.relative)
        index = [slice(None)] * array.ndim
        index[self.axes[0]] = c_index
        array[tuple(index)] = fill_value
        return array


class _MuteGeometry2D(_MuteGeometry):
    """
    Private container to help manage the geometry of the slope filter.

    Generally this is initialized with the `from_point_list` class method.

    Parameters
    ----------
    origins
        The origin for each point. If lines are not parallel, this is the
        shared origin. If they are parallel the first value of each point
        is used.
    line1
        A numpy array of the un-normalized first line.
    line2
        A numpy array of the un-normalized second line.
    norm
        The corresponding coordinate range for each dimension (as float)
    line1_norm
        The first line divided by the norm.
    line2_norm
        The second line divided by the norm.
    parallel
        If True, the lines are parallel to each other.

    The mute region is selected by finding the cross product between each
    normalized (scaled) line
    """

    origins: tuple[NDArray[np.floating], NDArray[np.floating]]
    norm: NDArray[np.floating]
    line1_norm: NDArray[np.floating]
    line2_norm: NDArray[np.floating]
    parallel: bool

    # For determining if any points are degenerate (norms close to 0)
    # or parallel (dot products close to 1)
    _tolerance: ClassVar[float] = 1e-9

    @classmethod
    def _get_line_params(cls, points, patch, dims, fill_ind):
        """Get the parameters from a list of points."""
        tol = cls._tolerance
        # Get the origin (where two lines intersect), (nan, nan) if parallel.
        coord_norm = np.array(
            [dc.to_float(patch.get_coord(x).coord_range()) for x in dims]
        )
        origin = get_2d_line_intersection(*points)
        # Get vectors with various stages of normalization.
        # We need to first normalize in coord space and then by l2 norm.
        l1 = points[1] - points[0]
        l2 = points[3] - points[2]
        v1, v2 = l1 / coord_norm, l2 / coord_norm
        norm_v1, norm_v2 = norm(v1), norm(v2)
        with np.errstate(divide="ignore", invalid="ignore"):
            v1_norm, v2_norm = v1 / norm_v1, v2 / norm_v2
        # Then get array for easier manipulation
        v_norms = np.stack([v1_norm, v2_norm], axis=0)

        # Check if any points are degenerate.
        if (norm_v1 < tol) or (norm_v2 < tol):
            msg = f"A line provided to mute ({v1} or {v2}) is degenerate!"
            raise ParameterError(msg)
        # Determine if vectors are parallel and point in same direction
        dot = np.dot(v_norms[0], v_norms[1])
        parallel = bool((1 - abs(dot)) < tol)
        same_direction = bool(dot >= 0)
        if not same_direction:
            if parallel or fill_ind != -1:
                preferred = fill_ind if fill_ind != -1 else 0
                # Just reverse order of line
                v_norms[preferred] *= -1
            else:
                msg = "Non-parallel mute vectors must point in the same direction."
                raise ParameterError(msg)

        # Get the origin tuple (origin for l1, origin for l2)
        if parallel:
            origin_tuple = (points[0], points[2])
        else:
            origin_tuple = (origin, origin)
        out = dict(
            origins=origin_tuple,
            norm=coord_norm,
            line1_norm=v_norms[0],
            line2_norm=v_norms[1],
            parallel=parallel,
        )
        return out

    @staticmethod
    def _get_filled_value_list(patch, coords, value_list, relative):
        """Create an array of points, swapping out implicit values."""

        def _get_coord_float_values(coord, vals, relative):
            """Get the coordinate float values relative to start of coords."""
            out = dc.to_float(np.array(vals))
            if not relative:
                out = out - dc.to_float(coord.min())
            return out

        # Output is dim by column and point by row.
        out = np.full((4, 2), fill_value=np.nan, dtype=np.float64)
        # Indicates an implicit value was used.
        ifill_index = -1

        for ind, (coord, row) in enumerate(zip(coords, value_list)):
            coord = patch.get_coord(patch.dims[ind])
            # We iterate each pair because it might be an implicit value.
            for pair_ind, pair in enumerate(row):
                is_sized = isinstance(pair, Sized)
                # The range is clearly defined.
                if is_sized:
                    array_inds = (slice(pair_ind * 2, pair_ind * 2 + 2), ind)
                    out[array_inds] = _get_coord_float_values(coord, pair, relative)
                    continue
                # This pair value is a place holder (None, ...)
                elif not is_sized and (pair is None or pair is Ellipsis):
                    continue
                # This is an implicit value; here is where things get crazy.
                # We handle this by creating a new set of points to sub into
                # the output array. This new set shares a first point with the
                # other line, as well as the point on the implicit dimension.
                # The value for the non-implicit dimension is held fixed.
                else:
                    ifill_index = pair_ind
                    other_col_ind = 0 if ind == 1 else 1
                    fixed_vals = _get_coord_float_values(coord, [pair], relative)
                    out[pair_ind * 2 : pair_ind * 2 + 2, ind] = fixed_vals
                    out[pair_ind * 2 : pair_ind * 2 + 2, other_col_ind] = np.array(
                        [0, 1]
                    )
        assert not np.any(np.isnan(out))
        return out, ifill_index

    @classmethod
    def from_params(cls, vals, dims, axes, patch, relative=True):
        """
        Return values which are two lines in absolute coordinate space,
        (expressed as floats)
        """
        coords = [patch.get_coord(x) for x in dims]
        # Iterate over each row (consists of (start, stop))
        points, fill_ind = cls._get_filled_value_list(patch, coords, vals, relative)
        # Then put the 4 points together. This gives us a len 4 array with rows
        # as points from first line, then points from second.
        line_params = cls._get_line_params(points, patch, dims, fill_ind)
        kwargs = dict(dims=dims, axes=axes, relative=relative) | line_params
        return cls(**kwargs)

    def _get_normalized_array_coord(self, array, patch):
        """
        Get an array that matches the dimensionality of envelope but of its
        normalized, relative coordinates.
        """

        def _nan_normalize(array):
            """Normalize along last axis, handle norm close to 0."""
            array_norm = norm(array, axis=-1, keepdims=True)
            array_norm[array_norm == 0] = -np.finfo(np.float64).min
            return array / array_norm

        out = [[], []]
        for dim_num, dim in enumerate(self.dims):
            ax = patch.get_axis(dim)
            coord = patch.get_coord(dim)
            # Get the coordinate values normalized to coord range.
            coord_vals = dc.to_float(coord.values)
            coord_range = dc.to_float(coord.coord_range())
            if self.relative:
                coord_vals -= dc.to_float(coord.min())
            # Iterate over each origin.
            for onum, origin in enumerate(self.origins):
                # We need to transform coordinate values to values between 0 and 1.
                # with the same dimensionality as array.
                norm_vals = (coord_vals - origin[dim_num]) / coord_range
                bcast_inds = [None] * array.ndim
                bcast_inds[ax] = slice(None)
                norms = norm_vals[tuple(bcast_inds)]
                # Next, we set those values on an array with the same shape as array.
                inds = [slice(None)] * array.ndim
                inds[ax] = slice(None, len(coord))
                carray = np.empty_like(array)
                carray[tuple(inds)] = norms
                out[onum].append(carray)
        # Return new axis as -1 so it will broadcast with lines.
        out_1 = [np.stack(x, axis=-1) for x in out]
        out_norm = [_nan_normalize(x) for x in out_1]
        return out_norm

    def _cross_product_ok(self, coord_array_1, coord_array_2):
        """
        Determine if each point is in the region based on the cross product
        requirement.
        """
        # Get padded arrays with 0 z values for cross product.
        # Create padding for all dimensions: (0,0) for all but last, (0,1) for last
        array_widths = [(0, 0)] * (coord_array_1.ndim - 1) + [(0, 1)]
        coord_1_z = np.pad(
            coord_array_1, pad_width=array_widths, mode="constant", constant_values=0
        )
        coord_2_z = np.pad(
            coord_array_2, pad_width=array_widths, mode="constant", constant_values=0
        )
        line_widths = (0, 1)
        l1_z = np.pad(self.line1_norm, line_widths, mode="constant", constant_values=0)
        l2_z = np.pad(self.line2_norm, line_widths, mode="constant", constant_values=0)
        # The selected points will have different cross product signs. Need to
        # shift points relative to each line start if we have parallel lines.
        # Use ellipsis to handle arbitrary dimensions, indexing last dim with -1
        cross1_z = np.cross(l1_z, coord_1_z)[..., 2]
        cross2_z = np.cross(l2_z, coord_2_z)[..., 2]
        ok_cross = cross1_z * cross2_z < 0
        return ok_cross

    def _dot_product_ok(self, coord_array_1, coord_array_2):
        # Get dot product with each line.
        dot1 = np.sum(self.line1_norm[None, :] * coord_array_1, axis=-1)
        dot2 = np.sum(self.line2_norm[None, :] * coord_array_2, axis=-1)
        ok_dot = (dot1 > 0) & (dot2 > 0)
        return ok_dot

    def _apply_mask(self, array: NDArray, patch: dc.Patch, fill_value) -> NDArray:
        """Apply the mask to the output array."""
        cnorms_1, cnorms_2 = self._get_normalized_array_coord(array, patch)
        ok_cross = self._cross_product_ok(cnorms_1, cnorms_2)
        # For parallel lines only cross product is needed to define region.
        if self.parallel:
            mask = ok_cross
        else:
            ok_dot = self._dot_product_ok(cnorms_1, cnorms_2)
            mask = ok_cross & ok_dot
        array[mask] = fill_value
        return array


def _get_mute_geometry(patch, kwargs, relative=True):
    """
    Get (and validate) the muting parameters.

    Returns arrays of dims, axes, and values. Values has different meaning
    based on the number of dimensions.
    """
    # Validate we have dimension specifications
    if not kwargs or len(kwargs) > 2:
        msg = (
            "Mute requires one or two keyword arguments must be provided. "
            f"You passed {kwargs}"
        )
        raise ParameterError(msg)
    dim_ax_vals = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    dims = [x[0] for x in dim_ax_vals]
    axes = np.array([x[1] for x in dim_ax_vals])
    val_list = [x[2] for x in dim_ax_vals]
    if len(dims) == 1:
        geometry = _MuteGeometry1D.from_params(
            val_list, dims, axes, patch, relative=relative
        )
    elif len(dims) > 1:  # Dealing with lines.
        geometry = _MuteGeometry2D.from_params(
            val_list, dims, axes, patch, relative=relative
        )
    return geometry


@patch_function()
def mute(
    patch: PatchType,
    *,
    smooth=None,
    invert: bool = False,
    relative: bool = True,
    **kwargs,
) -> PatchType:
    """
    Mute (zero out) data in a region specified by one or more lines.

    Each dimension accepts two boundary specifications that define
    the mute region. Boundaries can be scalar values or arrays of values.

    Parameters
    ----------
    patch
        The patch instance.
    smooth
        Parameter controlling smoothing of the mute envelope. Defines the sigma
        Can be:
        - None: sharp mute
        - float (0.0-1.0): fraction of dimension range (e.g., 0.01 = 1%)
          which is applied independently to each dimension involved in the
          mute.
        - int: Indicates number of samples for each dimension.
        - Quantity with units, indicates values along a single dimension.
          Only applicable if a single dimension is specified.
        - dict: {dim: taper_value} for dimension-specific smooth
          values which can be any of the above.
    invert
        If True, invert the taper such that the values outside the defined region
        are set to 0.
    relative
        If True (default), values are relative to coordinate edges.
        Positive values are offsets from start, negative from end.
    **kwargs
        Dimension specifications as (boundary_1, boundary_2) pairs.
        Each boundary can be:
        - Scalar: constant value (defines plane perpendicular to dimension)
        - Array/list: coordinates defining line/curve/surface
        - None or ...: edge of coordinate (min or max)

    Examples
    --------
    >>> import dascore as dc
    >>> from scipy.ndimage import gaussian_filter
    >>>
    >>> patch = dc.get_example_patch().full(1)
    >>>
    >>> # Mute first 0.5s (relative to start by default)
    >>> muted = patch.mute(time=(0, 0.5))
    >>>
    >>> # Mute everything except middle section
    >>> kept = patch.mute(time=(0.2, -0.2), invert=True)
    >>>
    >>> # 1D Mute with smoothed absolute units for time.
    >>> muted = patch.mute(time=(0.2, 0.8), smooth=0.02 * dc.units.s)
    >>>
    >>> # Classic first break mute: mute early arrivals
    >>> # Line from (t=0, d=0) to (t=0.3, d=300) defines velocity=1000 m/s
    >>> muted = patch.mute(
    ...     time=(0, [0, 0.3]),
    ...     distance=(None, [0, 300]),
    ...     smooth=0.02,
    ... )
    >>>
    >>> # Mute late arrivals: from velocity line to end
    >>> muted = patch.mute(
    ...     time=([0, 0.3], None),
    ...     distance=([0, 300], 0),
    ... )
    >>>
    >>> # Mute wedge between two velocity lines
    >>> muted = patch.mute(
    ...     time=([0, 0.375], [0, 0.25]),
    ...     distance=([0, 300], [0, 300]),
    ... )
    >>>
    >>> # Mute wedge outside two velocity lines
    >>> muted = patch.mute(
    ...     time=([0, 0.375], [0, 0.25]),
    ...     distance=([0, 300], [0, 300]),
    ...     invert=True,
    ... )
    >>>
    >>> # Apply custom tapering
    >>> ones = patch.full(1.0)
    >>> envelope = ones.mute(
    ...     time=([0, 0.375], [0, 0.25]),
    ...     distance=([0, 300], [0, 300]),
    ... )
    >>> # Knock down edges with rolling mean along the time dimension.
    >>> smooth = envelope.rolling(time=5, samples=True).mean()
    >>> # Then multiply the two patches.
    >>> result = patch * smooth

    Notes
    -----
    - By relative=True (the default) means values are offsets from coordinate
      edges. Use relative=False for absolute coordinate values.

    - Currently, mute doesn't support more than 2 dimensions.

    - For more control over boundary smoothing, use a patch with one values
      then apply custom tapering/smoothing before multiplying with the
      original patch. See example section for more details.

    See Also
    --------
    - [`Patch.select`](`dascore.Patch.select`)
    - [`Patch.taper_range`](`dascore.Patch.taper_range`)
    - [`Patch.gaussian_filter`](`dascore.proc.filter.gaussian_filter`)
    """
    # Get geometry object to set up the problem.
    geo = _get_mute_geometry(patch, kwargs, relative)
    # Initialize the output array which shares the dimensionality of the
    # patch (so it will broadcast) but is as flat as possible.
    out_shape = [patch.shape[ax] if ax in geo.axes else 1 for ax in range(patch.ndim)]
    out = np.zeros(out_shape) if invert else np.ones(out_shape)
    fill_val = 1 if invert else 0
    # Easy path for 1D mute.
    out = geo._apply_mask(out, patch, fill_val)
    # Apply smoothing if requested.
    if smooth is not None:
        out = geo._apply_smoothing(out, smooth, patch)
    return patch.update(data=patch.data * out)
