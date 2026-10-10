"""Module for re-sampling patches."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from pydantic import ConfigDict

import dascore as dc
import dascore.compat as compat
from dascore.core.processor import PatchProcessor, _via_numpy
from dascore.exceptions import FilterValueError, ParameterError
from dascore.units import get_filter_units
from dascore.utils.array_api import array_namespace
from dascore.utils.imports import lazy_import
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import (
    _require_evenly_sampled,
    drop_associated_coords,
    get_dim_axis_value,
    get_start_stop_step,
)
from dascore.utils.time import dtype_time_like, to_int, to_timedelta64

scipy_decimate = lazy_import("scipy.signal", "decimate")


def _apply_scipy_decimation(data, factor, ftype, axis):
    """Apply decimation along an axis."""
    try:
        data = scipy_decimate(data, factor, ftype=ftype, axis=axis)
    except ValueError as e:
        msg = (
            "Scipy decimation failed. This can happen for dimensions with "
            "few elements. Consider setting filter_type to False. The raised "
            f"exception was {e}"
        )
        raise FilterValueError(msg)
    return data


# `copy` shadows pydantic's deprecated `BaseModel.copy`, which nothing calls.
with suppress_warnings(UserWarning, message='Field name "copy"'):

    class _DecimateFields(PatchProcessor):
        """The parameters of `Decimate`, kept apart for the warning above."""

        name = None
        filter_type: Any = "iir"
        copy: Any = True


class Decimate(_DecimateFields):
    """
    Decimate a patch along a dimension.

    Parameters
    ----------
    patch
        The patch to decimate.
    filter_type
        Anti-aliasing filter: `"iir"`, `"fir"`, or None. None disables
        pre-filtering and may cause aliasing.
    copy
        Copy the sliced data so it does not retain the original array. Applies
        only when `filter_type` is None.
    **kwargs
        Dimension and factor; `time=10` decimates the time axis by 10.

    Notes
    -----
    - Uses `scipy.signal.decimate` when `filter_type` is specified; otherwise,
      takes every nth sample along the dimension.

    - If the decimation dimension is small, this can fail due to lack of
      padding values.

    - Coordinates measured on the decimated dimension are decimated with
      it: taking every nth value of a dimension takes every nth value of
      everything indexed by it.

    - With a filter, missing samples (holes in the step) raise; use
      [split_gaps](`dascore.Patch.split_gaps`) or
      [fill_gaps](`dascore.Patch.fill_gaps`) first.

    See Also
    --------
    [resample](`dascore.Patch.resample`)
        Change sampling to a specified interval or number of samples, rather
        than by an integer decimation factor.

    Examples
    --------
    # Simple example using iir
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> decimated_irr = patch.decimate(time=10, filter_type='iir')
    >>> # Example using fir along distance dimension
    >>> decimated_fir = patch.decimate(distance=10, filter_type='fir')
    """

    model_config = ConfigDict(extra="allow")

    def get_metadata(self, meta):
        """Return the decimated coordinates, the axis, factor and slices."""
        extras = self.model_extra or {}
        dim, axis, factor = get_dim_axis_value(meta, kwargs=extras)[0]
        coords, slices = meta.coords.decimate(**{dim: int(factor)})
        if self.filter_type:
            _require_evenly_sampled(meta, dim, "filtered decimate")
            coords = coords._update_grid(dim)
            # scipy's own refusal of a factor which is not an integer.
            operator.index(factor)
        plan = {"axis": axis, "factor": int(factor), "slices": slices}
        return meta.new(coords=coords), plan

    def numpy_kernel(self, data, *, axis, factor, slices):
        """Return the data filtered and decimated by scipy, or sliced."""
        if self.filter_type:
            ftype = self.filter_type
            return _apply_scipy_decimation(data, factor, ftype=ftype, axis=axis)
        # Copying releases the reference to the parent array.
        return np.array(data[slices]) if self.copy else data[slices]

    def kernel(self, data, *, axis, factor, slices):
        """Return the data sliced on its own backend; filtered by numpy."""
        if self.filter_type:
            numpy_decimate = _via_numpy(type(self).numpy_kernel, "decimate")
            return numpy_decimate(self, data, axis=axis, factor=factor, slices=slices)
        xp = array_namespace(data)
        return xp.asarray(data[slices], copy=True) if self.copy else data[slices]


def _interpolate_associated(cm, dim, coord_num, samples_num, kind) -> dict:
    """
    Interpolate the coordinates which ride the interpolated dimension.

    A coordinate of numbers is a function of the dimension, so it is
    interpolated the way the data is. Anything else is dropped, by
    updating it to None: a label has nothing between its values, and a
    time does not survive the trip through floating point -- a nanosecond
    of the present is 1.6e18 of them, where the nearest float64 is
    hundreds of nanoseconds away. Dropped explicitly, because
    interpolating onto the same number of samples in different places
    would otherwise leave the old values sitting on the new ones.
    """
    out = {}
    for name, coord_dims in cm.dim_map.items():
        coord = cm.coord_map[name]
        if name == dim or dim not in coord_dims:
            continue
        # Asked before the number test, not after: numpy counts a
        # timedelta64 as a number, and interpolating one gives back a
        # float in whatever resolution it was stored in.
        if dtype_time_like(coord.dtype) or not np.issubdtype(coord.dtype, np.number):
            out[name] = None
            continue
        func = compat.interp1d(
            coord_num,
            coord.values,
            axis=coord_dims.index(dim),
            kind=kind,
            fill_value="extrapolate",
        )
        values = func(samples_num)
        out[name] = (coord_dims, dc.core.get_coord(data=values, units=coord.units))
    return out


class Interpolate(PatchProcessor):
    """
    Set coordinates of patch along a dimension using interpolation.

    Parameters
    ----------
    patch
        The patch object to which interpolation is applied.
    kind
        The type of interpolation. See Notes for more details.
        If a string, the following are supported:
            linear - linear interpolation between a pair of points.
            nearest - use the nearest sample for interpolation.
        If an int, it specifies the order of spline to use. EG 1 is a linear
            spline, 2 is quadratic, 3 is cubic, etc.

    **kwargs
        Used to specify dimension and interpolation values. Use a value of
        None to "snap" coordinate to evenly sampled points along coordinate.

    Notes
    -----
    Uses `scipy.interpolate.interp1d`; see its documentation for interpolation
    details.

    Coordinates measured on the interpolated dimension are interpolated
    with it where they are numbers, and dropped otherwise: a label has
    nothing between its values, and a time does not survive the trip
    through floating point.

    Interpolation fills across coordinate holes: samples requested inside
    a gap are interpolated from the samples on either side of it.

    See Also
    --------
    [Patch.snap_coords](`dascore.Patch.snap_coords`)
        Snap coordinates to evenly sampled values without interpolating data.
    [resample](`dascore.Patch.resample`)
        Resample data to a target sampling interval or number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # up-sample time coordinate
    >>> time = patch.coords.get_array('time')
    >>> time_step = patch.get_coord("time").step
    >>> new_time = np.arange(time.min(), time.max(), 0.5 * time_step)
    >>> patch_uptime = patch.interpolate(time=new_time)
    >>> # interpolate unevenly sampled dim to evenly sampled
    >>> patch = dc.get_example_patch("wacky_dim_coords_patch")
    >>> patch_time_even = patch.interpolate(time=None)
    """

    kind: Any = "linear"

    model_config = ConfigDict(extra="allow")

    def get_metadata(self, meta):
        """Return the new coordinates, the axis, and both sets of positions."""
        extras = self.model_extra or {}
        dim, axis, samples = get_dim_axis_value(meta, kwargs=extras)[0]
        cm = meta.coords
        if samples is None:
            samples = cm.coord_map[dim].snap().values
        # interp1d does not support datetime64.
        coord_num = to_int(cm.get_array(dim))
        samples_num = to_int(samples)
        if np.asarray(samples_num).dtype.kind not in "biufc":
            # Text or objects: let scipy refuse them, as it did unplanned.
            _interp(coord_num, coord_num, samples_num, 0, self.kind)
        coord_new = dc.core.get_coord(data=samples, snap=True)
        # evaluated at the labels the samples become, which the released
        # snap may move
        samples_num = to_int(coord_new.values)
        updates = {dim: (cm.dim_map[dim], coord_new)}
        updates |= _interpolate_associated(cm, dim, coord_num, samples_num, self.kind)
        plan = {"axis": axis, "coord": coord_num, "samples": samples_num}
        return meta.new(coords=cm._update_grid(dim, **updates)), plan

    def numpy_kernel(self, data, *, axis, coord, samples):
        """Return the data interpolated by scipy at the new positions."""
        return _interp(data, coord, samples, axis, self.kind)


def _interp(data, coord, samples, axis, kind):
    """Return data sampled at `coord` interpolated onto `samples` along an axis."""
    func = compat.interp1d(coord, data, axis=axis, kind=kind, fill_value="extrapolate")
    return func(samples)


class Resample(PatchProcessor):
    """
    Resample along a single dimension using Fourier Method and interpolation.

    The dimension which should be resampled is passed as kwargs. The key
    is the dimension name and the value is the new sampling period.

    Since Fourier methods only support adding or removing an integer number
    of frequency bins, the exact desired sampling rate is often not achievable
    with resampling alone. If the fourier resampling doesn't produce the exact
    result, an interpolation (see [interpolate](`dascore.Patch.interpolate`))
    is used to achieve the desired sampling rate.

    Parameters
    ----------
    patch
        The patch to resample.
    window
        The Fourier-domain window that tapers the Fourier spectrum. See
        scipy.signal.resample for details. Only used if method == 'fft'.
    interp_kind
        The interpolation type if output of fourier resampling doesn't produce
        exactly the right sampling rate.
    samples
        If true, the values in kwargs represent the number of samples along
        The specified dimension.
    **kwargs
        keyword arguments to specify dimension and new sampling value. Units
        can also be used to specify sampling_period or frequency.

    Notes
    -----
    - Unless `samples` is `True`, this function requires a sampling_period.
    - The resulting Patch can be slightly shorter than the input Patch.
    - Coordinates associated with the resampled dimension are dropped because
      resampling cannot safely infer their new values. A `DASCoreWarning` names
      any coordinates which were dropped.

    Examples
    --------
    >>> # resample a patch along time dimension to 10 ms
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> new = patch.resample(time=np.timedelta64(10, 'ms'))
    >>> # Resample time dimension to 50 Hz
    >>> from dascore.units import Hz
    >>> new = patch.resample(time=(50 * Hz))
    >>> # Resample distance dimension to a sampling period of 15m
    >>> from dascore.units import m
    >>> new = patch.resample(distance=15 * m)
    >>> # Resample time axis such that there are 50 samples total
    >>> new = patch.resample(time=50, samples=True)

    See Also
    --------
    [decimate](`dascore.Patch.decimate`)
    [interpolate](`dascore.Patch.interpolate`)
    """

    window: Any = None
    interp_kind: Any = "linear"
    samples: Any = False

    model_config = ConfigDict(extra="allow")

    def get_metadata(self, meta):
        """Return the resampled coordinates, and the length and positions to hit."""
        dim, axis, value = get_dim_axis_value(meta, kwargs=self.model_extra or {})[0]
        coord = meta.get_coord(dim, require_sorted=True, require_evenly_sampled=True)
        new_step = None
        if not self.samples:
            step = coord.step
            coord_units = dc.get_quantity(coord.units)
            # inverse coord unit to trick filter units into giving correct units.
            if coord_units is not None:
                coord_units = 1 / coord_units
            new_step, _ = get_filter_units(value, value, to_unit=coord_units)
            if new_step is None:
                msg = (
                    f"resample requires a sampling period for dimension {dim!r}; "
                    f"got {value!r}. Pass samples=True to resample by length."
                )
                raise ParameterError(msg)
            # nasty hack so that ints/floats get converted to seconds.
            if isinstance(step, np.timedelta64):
                new_step = to_timedelta64(new_step)
            new_len = meta.shape[axis] * (step / new_step)
        else:
            new_len = value
        num = int(np.round(new_len))
        # The positions scipy's resample gives for its output.
        start = coord[0]
        new_coord = start + (coord[1] - start) * (len(coord) / num) * np.arange(num)
        cm = drop_associated_coords(meta.coords, dim, "Resampling")
        out = meta.new(coords=cm._update_grid(dim, **{dim: new_coord}))
        plan = {"axis": axis, "num": num, "coord": None, "samples": None}
        # Interpolate if new sampling rate is not very close to desired sampling rate.
        if not self.samples and not np.isclose(new_len, num):
            start, stop, _ = get_start_stop_step(out, dim)
            new_coord = np.arange(start, stop, new_step)
            interp = Interpolate(kind=self.interp_kind, **{dim: new_coord})
            out, interp_plan = interp.get_metadata(out)
            plan |= {"coord": interp_plan["coord"], "samples": interp_plan["samples"]}
        return out, plan

    def numpy_kernel(self, data, *, axis, num, coord, samples):
        """Return the data resampled by scipy, then interpolated if needed."""
        data = compat.resample(data, num, axis=axis, window=self.window)
        if coord is not None:
            data = _interp(data, coord, samples, axis, self.interp_kind)
        return data
