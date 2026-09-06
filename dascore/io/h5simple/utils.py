"""Utilities for simple h5 files."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import STORAGE_PROVENANCE_ATTRS
from dascore.core import get_coord
from dascore.io.utils import get_exact_coord
from dascore.utils.misc import _maybe_unpack, unbyte

# --- Getting format/version

DATA_ARRAY_NAMES = frozenset(["raw", "data"])
TIME_ARRAY_NAMES = frozenset(("timestamp", "time", "timestamps"))
OTHER_COORD_ARRAY_NAMES = frozenset(("channels", "distance"))

FILE_FORMAT_ATTR_NAMES = frozenset(("__format__", "file_format", "format"))
DEFAULT_ATTRS = frozenset(("CLASS", "PYTABLES_FORMAT_VERSION", "TITLE", "VERSION"))


def _get_attrs_coords_and_data(h5, snap):
    """Return attrs, coordinate manager, and data node."""
    attrs = h5.attrs
    # This format has no header schema, so every root attr is copied. Two
    # kinds must not be: storage provenance, which belongs to the spool,
    # and the format discriminator, which says which reader to use. A file
    # carrying either would otherwise pass it on as a patch attr -- and
    # only scan used to drop them, so scan and read disagreed.
    skip = DEFAULT_ATTRS | FILE_FORMAT_ATTR_NAMES | set(STORAGE_PROVENANCE_ATTRS)
    attr_names = set(attrs) - skip
    attr_dict = {x: unbyte(attrs[x]) for x in attr_names}
    cm, data = _get_cm_and_data(h5, snap, dims=attr_dict.get("dims"))
    attr_dict.pop("dims", None)
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
        coord = get_exact_coord(values)
    return coord


def _name_missing_dim(coord_shape_dict, data_node) -> int:
    """
    Name the one axis no node accounts for "channel", and return its length.

    This is needed because the foresee data on pubdas only specify time;
    we have to fill in channel number.
    """
    missing_shape = set(data_node.shape) - set(coord_shape_dict)
    assert len(missing_shape) == 1, "can only fill one missing coord."
    shape = next(iter(missing_shape))
    coord_shape_dict[shape] = "channel"
    return shape


def _get_dims(data_node, time_node, other_nodes, dims=None):
    """
    Get the dims tuple, and the coord nodes any filling added.

    The format does not have to state its dimensions; when it does not,
    each is named by the node whose length matches that axis. Only node
    shapes are read here, never their values.
    """
    if dims:
        stated = tuple(dims.split(",")) if isinstance(dims, str) else tuple(dims)
        return stated, other_nodes
    can_guess_shape = len(data_node.shape) == len(set(data_node.shape))
    assert can_guess_shape, "Cant determine dims; shape values not unique!"
    assert len(time_node.shape) == 1, "time node has more than one dimension!"
    # get a dict of {coord_name: shape} for 1d coords.
    coord_shape_dict = {len(v): x for x, v in other_nodes.items() if len(v.shape) == 1}
    coord_shape_dict[len(time_node)] = "time"
    # need to fill some dims
    if len(coord_shape_dict) != len(data_node.shape):
        _name_missing_dim(coord_shape_dict, data_node)
    return tuple(coord_shape_dict[x] for x in data_node.shape), other_nodes


def _get_coords_and_dims(data_node, time_node, other_nodes, snap=True, dims=None):
    """Get dims tuple and coord dict."""
    dims, other_nodes = _get_dims(data_node, time_node, other_nodes, dims)
    other_nodes["time"] = time_node
    # the filled axis is named but not built: only coordinates need it
    if "channel" in dims and "channel" not in other_nodes:
        length = data_node.shape[dims.index("channel")]
        other_nodes["channel"] = np.arange(length)
    coords = {i: _get_coord(v, snap=snap, name=i) for i, v in other_nodes.items()}
    return dims, coords


def _get_nodes(h5):
    """Return the file's data node, time node, and other array nodes."""
    root_nodes = {name: node for name, node in h5.items() if hasattr(node, "shape")}
    array_names = set(root_nodes)
    data_node_name = array_names & DATA_ARRAY_NAMES
    time_node_name = array_names & TIME_ARRAY_NAMES
    other_node_names = array_names - data_node_name - time_node_name

    assert len(data_node_name) == 1, f"{h5} doesn't have exactly one data node."
    assert len(time_node_name) == 1, f"{h5} doesn't have exactly one time node"

    return (
        root_nodes[next(iter(data_node_name))],
        root_nodes[next(iter(time_node_name))],
        {x: root_nodes[x] for x in other_node_names},
    )


def _get_dims_and_data(h5):
    """Return the data node's dims and the node itself, reading no values."""
    dims_attr = unbyte(h5.attrs["dims"]) if "dims" in h5.attrs else None
    data_node, time_node, other_nodes = _get_nodes(h5)
    dims, _ = _get_dims(data_node, time_node, other_nodes, dims_attr)
    return dims, data_node


def _get_cm_and_data(h5, snap=False, dims=None):
    """Extract coordinate manager and data node."""
    data_node, time_node, other_nodes = _get_nodes(h5)
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
    names = set(attrs) & FILE_FORMAT_ATTR_NAMES
    # Every name that states a format has to state this one; a file which
    # names two disagreeing formats belongs to neither.
    return all(unbyte(_maybe_unpack(attrs[name])) == "h5simple" for name in names)
