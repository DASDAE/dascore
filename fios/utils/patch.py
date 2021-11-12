"""
Utilities for working with the Patch class.
"""
import collections
import copy
import functools
import inspect
from typing import (Any, Callable, Dict, Literal, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import pandas as pd

from fios.constants import DEFAULT_PATCH_ATTRS, PatchType
from fios.exceptions import PatchAttributeError, PatchDimError
from fios.utils.mapping import FrozenDict
from fios.utils.misc import append_func
from fios.utils.time import to_datetime64, to_timedelta64

attr_type = Union[Dict[str, Any], str, Sequence[str], None]


class Coords:
    """A wrapper around xarray coords for a bit more intuitive access."""

    def __init__(self, coords):
        self._coords = coords

    def __getitem__(self, item):
        """Return the raw numpy array."""
        out = self._coords[item]
        return getattr(out, "values", out)

    def __str__(self):
        return str(self._coords)

    __repr__ = __str__

    def get(self, item):
        """Return item or None if not in coord. Same as dict.get"""
        return self._coords.get(item)

    def __iter__(self):
        return self._coords.__iter__()

    @property
    def timedelta64(self):
        """Return time deltas of time dimension."""
        time = self._coords["time"]
        return time - time[0]

    @property
    def datetime64(self):
        """Return datetime64 of time dimension."""
        return self._coords["time"]


def _shallow_copy(patch: PatchType) -> PatchType:
    """
    Shallow copy patch so data array, attrs, and history can be changed.

    Note
    ----
    This is an internal function because Patch should not be immutable with
    the public APIs.
    """
    dar = patch._data_array.copy(deep=False)  # dont copy data and such
    attrs = dict(dar.attrs)
    attrs["history"] = list(attrs.get("history", []))
    dar.attrs = attrs
    return patch.__class__(dar)


def _func_and_kwargs_str(func: Callable, patch, *args, **kwargs) -> str:
    """
    Get a str rep of the function and input args.
    """
    callargs = inspect.getcallargs(func, patch, *args, **kwargs)
    callargs.pop("patch", None)
    callargs.pop("self", None)
    kwargs_ = callargs.pop("kwargs", {})
    arguments = []
    arguments += [f"{k}={repr(v)}" for k, v in callargs.items() if v is not None]
    arguments += [f"{k}={repr(v)}" for k, v in kwargs_.items() if v is not None]
    arguments.sort()
    out = f"{func.__name__}("
    if arguments:
        out += f"{','.join(arguments)}"
    return out + ")"


def _get_history_str(patch: PatchType, func, *args, _history="full", **kwargs) -> str:
    """
    Log history of a function being called on a Patch object.

    Parameters
    ----------
    patch
        The patch which will track history.
    func
        A callable which takes the patch as the first argument.
    *args
        The arguments passed to the function.
    _history
        String specifying how the history is to be recorded.
    **kwargs
        kwargs for func.
    """
    if _history is None:
        return None
    if _history == "full":
        history_str = _func_and_kwargs_str(func, patch, *args, **kwargs)
    else:
        history_str = str(func.__name__)
    return history_str


def check_patch_dims(
    patch: PatchType, required_dims: Optional[Tuple[str, ...]]
) -> PatchType:
    """
    Check if a patch object has required dimensions, else raise PatchDimError.

    Parameters
    ----------
    patch
        The input patch
    required_dims
        A tuple of required dimensions.
    """
    if required_dims is None:
        return patch
    dim_set = set(patch.dims)
    missing_dims = set(required_dims) - set(patch.dims)
    if not set(required_dims).issubset(dim_set):
        msg = f"patch is missing required dims: {tuple(missing_dims)}"
        raise PatchDimError(msg)
    return patch


def check_patch_attrs(patch: PatchType, required_attrs: attr_type) -> PatchType:
    """
    Check for expected attributes.

    Parameters
    ----------
    patch
        The patch to validate
    required_attrs
        The expected attrs. Can be a sequence or mapping. If sequence, only
        check that attrs exist. If mapping also check that values are equal.
    """
    if required_attrs is None:
        return patch
    # test that patch attr mapping is equal
    patch_attrs = patch.attrs
    if isinstance(required_attrs, Mapping):
        sub = {i: patch_attrs[i] for i in required_attrs}
        if sub != dict(required_attrs):
            msg = f"Patch's attrs {sub} are not required attrs: {required_attrs}"
            raise PatchAttributeError(msg)
    else:
        missing = set(required_attrs) - set(patch.attrs)
        if missing:
            msg = f"Patch is missing the following attributes: {missing}"
            raise PatchAttributeError(msg)
    return patch


def patch_function(
    required_dims: Optional[Union[Tuple[str, ...], Callable]] = None,
    required_attrs: attr_type = None,
    history: Literal["full", "method_name", None] = "full",
):
    """
    Decorator to mark a function as a patch method.

    Parameters
    ----------
    required_dims
        A tuple of dimensions which must be found in the Patch.
    required_attrs
        A dict of attributes which must be found in the Patch and whose
        values must be equal to those provided.
    history
        Specifies how to track history on Patch.
            Full - Records function name and str version of input arguments.
            method_name - Only records method name. Useful if args are long.
            None - Function call is not recorded in history attribute.

    Examples
    -------
    1. A patch method which requires dimensions (time, distance)
    >>> @fios.patch_function(dims=('time', 'distance'))
    >>> def do_something(patch):
    >>>     ...   # raises a PatchCoordsError if patch doesn't have time,
    >>>     #  distance

    2. A patch method which requires an attribute 'data_type' == 'DAS'
    >>> @fios.patch_function(required_attrs={'data_type': 'DAS'})
    >>> def do_another_thing(patch):
    >>>     ...  # raise PatchAttributeError if patch doesn't have attribute
    >>>     # called "data_type" or its values is not equal to "DAS".
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _func(patch, *args, **kwargs):
            check_patch_dims(patch, required_dims)
            check_patch_attrs(patch, required_attrs)
            hist_str = _get_history_str(patch, func, *args, _history=history, **kwargs)
            out: PatchType = func(patch, *args, **kwargs)
            # attach history string. Consider something a bit less hacky.
            if hist_str:
                out = _shallow_copy(out)
                out._data_array.attrs["history"].append(hist_str)
            return out

        return _func

    if callable(required_dims):  # the decorator is used without parens
        return patch_function()(required_dims)

    return _wrapper


def copy_attrs(attrs) -> dict:
    """Make a copy of the attributes dict so it can be mutated."""
    out = dict(attrs)
    if "history" in out:
        out["history"] = list(out["history"])
    return out


class _AttrsCoordsMixer:
    """Class for handling complex interactions between attrs and coords."""

    # a dict of {key: List[func]} for special handling of set attrs.
    set_attr_funcs = collections.defaultdict(list)
    # a dict of {key: func} for special handling of missing attributes.
    missing_attr_funcs = {}
    _missing_attr_keys = frozenset()
    # a dict of {attr: (expected_type, func)}
    _expected_types = dict(
        dt=(np.timedelta64, to_timedelta64),
        time_min=(np.datetime64, to_datetime64),
        time_max=(np.datetime64, to_datetime64),
    )

    def __init__(self, attrs, coords, dims):
        """Init with attrs and coords"""
        self.attrs = attrs
        self.coords = coords
        self.dims = dims
        self._original_attrs = attrs
        self._original_coords = coords
        # fill missing values with default or implicit values
        self._set_default_attrs()
        self._set_attr_types()
        self._condition_coords()
        # infer any attr values from coordinates
        self._update_attrs_from_coords(self._missing_attr_keys)

    def update_attrs(self, **kwargs):
        """
        Update a coordinate.

        Parameters
        ----------
        **kwargs
            The coordinates to update.
        """
        for key, value in kwargs.items():
            attr = self.copied_attrs
            attr[key] = value
            # apply special processing for this key
            if key in self.set_attr_funcs:
                for func in self.set_attr_funcs[key]:
                    func(self)

    @append_func(set_attr_funcs["time_min"])
    def _update_time_max_from_time_min(self):
        """Update end time based on new start time."""
        td = self._original_attrs["time_max"] - self._original_attrs["time_min"]
        if not pd.isnull(td):
            attrs = self.copied_attrs
            attrs["time_max"] = self.attrs["time_min"] + td

    @append_func(set_attr_funcs["time_min"])
    def _update_coords_time_min(self):
        """Update coordinate time for new starttime"""
        time = self.coords.get("time", None)
        time_min = self.attrs["time_min"]
        if time is not None and np.min(time) != time_min:
            td = time - time[0]
            self.copied_coords["time"] = time_min + td

    @append_func(set_attr_funcs["time_max"])
    def _update_time_min_from_time_max(self):
        """Update start time based on new end time."""
        td = self._original_attrs["time_max"] - self._original_attrs["time_min"]
        if not pd.isnull(td):
            attrs = self.copied_attrs
            attrs["time_min"] = self.attrs["time_max"] - td

    @append_func(set_attr_funcs["time_max"])
    def _update_coords_time_max(self):
        """Update coordinate time for new starttime"""
        time = self.coords.get("time", None)
        time_max = self.attrs["time_max"]
        if time is not None and np.max(time) != time_max:
            td = time - time[-1]
            self.copied_coords["time"] = time_max + td

    @functools.cached_property
    def copied_attrs(self):
        """Make a copy of the attrs before mutating."""
        self.attrs = copy_attrs(self.attrs)
        return self.attrs

    @functools.cached_property
    def copied_coords(self):
        """Make a copy of the coords before mutating."""
        coords = self.coords
        if isinstance(coords, Coords):  # unpack coordinates
            coords = coords._coords
        # TODO I don't like having this deep copy but shallow still caused
        # mutations so I will leave it for now.
        coords = copy.deepcopy(coords)
        self.coords = coords
        return coords

    def __call__(self, *args, **kwargs):
        return self.attrs, self.coords

    def _set_default_attrs(self):
        """Get the attribute dict, add required keys if not yet defined."""
        # add default values if they are not in out or attrs yet
        missing = set(DEFAULT_PATCH_ATTRS) - set(self.attrs)
        if not missing:
            return
        self._missing_attr_keys = missing
        out = copy_attrs(self.attrs)
        for key in missing:
            value = DEFAULT_PATCH_ATTRS[key]
            out[key] = value if not callable(value) else value()
        self.attrs = out

    # --- Coordinate bits

    def _update_attrs_from_coords(self, fill_keys=None):
        """Fill in missing attributes that can be inferred."""
        fill_keys = fill_keys if fill_keys else set(self.attrs)
        # iterate each dimension, fill in missing values with data from coords
        for dim in self.dims:
            array = self.coords[dim]
            # make sure its not an xarray thing
            array = getattr(array, "values", array)
            if f"{dim}_min" in fill_keys:
                self.copied_attrs[f"{dim}_min"] = np.min(array)
            if f"{dim}_max" in fill_keys:
                self.copied_attrs[f"{dim}_max"] = np.max(array)
        # add dims column
        if not len(self.attrs['dims']):
            self.copied_attrs['dims'] = ','.join(self.dims)

    def update_coords(self, **kwargs):
        """Update the coordinates based on kwarg inputs."""
        for key, value in kwargs.items():
            self.copied_coords[key] = value
        self._condition_coords()
        self._update_attrs_from_coords()

    def _condition_coords(self):
        """
        Condition the coordinates before using them.

        This is mainly to enforce common time conventions
        """
        coords = self.coords
        if coords is None or (time := coords.get("time")) is None:
            return
        start_time = self.attrs["time_min"] or np.datetime64("1970-01-01")
        # Convert non-datetime into time deltas
        if not np.issubdtype(time.dtype, np.datetime64):
            td = to_timedelta64(time)
            time = start_time + td
        self.copied_coords["time"] = time

    def _set_attr_types(self):
        """Make sure time attrs are expected types"""
        for key, (expected_type, func) in self._expected_types.items():
            val = self.attrs[key]
            if not isinstance(val, expected_type):
                self.copied_attrs[key] = func(val)
