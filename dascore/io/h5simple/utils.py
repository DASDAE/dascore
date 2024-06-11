"""Utilities for terra15."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.core import get_coord

# --- Getting format/version

DATA_ARRAY_NAMES = frozenset(["raw", "data"])
TIME_ARRAY_NAMES = frozenset(("timestamp", "time", "timestamps"))
OTHER_COORD_ARRAY_NAMES = frozenset(("channels", "distance"))

FILE_FORMAT_ATTR_NAMES = frozenset(("__format__", "file_format", "format"))
DEFAULT_ATTRS = frozenset(("CLASS", "PYTABLES_FORMAT_VERSION", "TITLE", "VERSION"))


def _maybe_trim_data(cm, data, kwargs):
    """Maybe use kwargs to trim data array."""
    new_cm, new_data = cm.select(array=data, **kwargs)
    return new_cm, new_data


def _get_attrs_coords_and_data(h5, snap, fiber_io):
    """Return attrs, coordinate manager, and data node."""
    attrs = h5.root._v_attrs
    attr_names = set(attrs._v_attrnames) - DEFAULT_ATTRS
    attr_dict = {x: getattr(attrs, x) for x in attr_names}
    attr_dict["file_version"] = fiber_io.version
    attr_dict["file_format"] = fiber_io.name
    cm, data = _get_cm_and_data(h5, snap, dims=attr_dict.get("dims"))
    attr_dict["dims"] = cm.dims
    return attr_dict, cm, data


def _get_coord(v, snap, name):
    """Get the coord values from a node."""
    if snap:
        start = v[0] if name != "time" else dc.to_datetime64(v[0])
        stop = v[-1] if name != "time" else dc.to_datetime64(v[-1])
        duration = stop - start
        step = duration / (len(v) - 1)
        coord = get_coord(min=start, max=stop + step, step=step)
        assert len(coord) == len(v)
    else:
        values = v[:] if name != "time" else dc.to_datetime64(v[:])
        coord = get_coord(data=values)
    return coord


def _fill_coords(coord_shape_dict, other_nodes, data_node):
    """
    Fill missing coordinate with "channel".

    This is needed because the foresee data on pubdas only specify time;
    we have to fill in channel number.
    """
    missing_shape = set(data_node.shape) - set(coord_shape_dict)
    assert len(missing_shape) == 1, "can only fill one missing coord."
    shape = next(iter(missing_shape))
    coord_shape_dict[shape] = "channel"
    other_nodes["channel"] = np.arange(shape)
    return other_nodes, coord_shape_dict


def _get_coords_and_dims(data_node, time_node, other_nodes, snap=True, dims=None):
    """Get dims tuple and coord dict."""
    if dims:
        dims = dims if not isinstance(dims, str) else dims.split(",")
    else:  # ascertain dims from shape
        can_guess_shape = len(data_node.shape) == len(set(data_node.shape))
        assert can_guess_shape, "Cant determine dims; shape values not unique!"
        assert len(time_node.shape) == 1, "time node has more than one dimension!"
        # get a dict of {coord_name: shape} for 1d coords.
        coord_shape_dict = {
            len(v): x for x, v in other_nodes.items() if len(v.shape) == 1
        }
        coord_shape_dict[len(time_node)] = "time"
        # need to fill some dims
        if len(coord_shape_dict) != len(data_node.shape):
            other_nodes, coord_shape_dict = _fill_coords(
                coord_shape_dict,
                other_nodes,
                data_node,
            )

        dims = tuple(coord_shape_dict[x] for x in data_node.shape)
    other_nodes["time"] = time_node
    coords = {i: _get_coord(v, snap=snap, name=i) for i, v in other_nodes.items()}
    return dims, coords


def _get_cm_and_data(h5, snap=False, dims=None):
    """Extract coordinate manager and data node."""
    array_names = {x.name for x in h5.list_nodes("/") if hasattr(x, "shape")}
    data_node_name = array_names & DATA_ARRAY_NAMES
    time_node_name = array_names & TIME_ARRAY_NAMES
    other_node_names = array_names - data_node_name - time_node_name

    assert len(data_node_name) == 1, f"{h5} doesn't have exactly one data node."
    assert len(time_node_name) == 1, f"{h5} doesn't have exactly one time node"

    data_node = getattr(h5.root, next(iter(data_node_name)))
    time_node = getattr(h5.root, next(iter(time_node_name)))
    other_nodes = {x: getattr(h5.root, x) for x in other_node_names}

    dims, coords = _get_coords_and_dims(data_node, time_node, other_nodes, snap, dims)
    return dc.core.get_coord_manager(coords, dims=dims), data_node


def _is_h5simple(h5):
    """Determine if open h5 file is simple H5."""
    has_arrays = _has_required_arrays(h5)
    version_ok = _no_format_or_simple_specified(h5)
    if has_arrays and version_ok:
        return True
    return False


def _has_required_arrays(h5):
    """Determine if h5 file has required arrays to be h5 simple."""
    array_names = set(h5)
    data_node = array_names & DATA_ARRAY_NAMES
    time_node = array_names & TIME_ARRAY_NAMES
    return bool(data_node) and bool(time_node)


def _no_format_or_simple_specified(h5):
    """Ensure no other format is specified, or that simpleH5 is."""
    attrs = h5.attrs
    attr_names = set(attrs)
    file_format = attr_names & FILE_FORMAT_ATTR_NAMES
    format = getattr(attrs, next(iter(file_format))) if file_format else "h5simple"
    if format == "h5simple":
        return True
    return False
