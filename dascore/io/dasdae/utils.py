"""DASDAE format utilities."""

from __future__ import annotations

import numpy as np
from tables import NodeError

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.utils.misc import suppress_warnings
from dascore.utils.time import to_int

# --- Functions for writing DASDAE format


def _create_or_get_group(h5, group, name):
    """Create a new group or get existing."""
    try:
        group = h5.create_group(group, name)
    except NodeError:
        group = getattr(group, name)
    return group


def _create_or_squash_array(h5, group, name, data):
    """Create a new array, if it exists delete and re-create."""
    try:
        array = h5.create_array(group, name, data)
    except NodeError:
        old_node = getattr(group, name)
        h5.remove_node(old_node)
        array = h5.create_array(group, name, data)
    return array


def _write_meta(hfile, file_version):
    """Write metadata to hdf5 file."""
    attrs = hfile.root._v_attrs
    attrs["__format__"] = "DASDAE"
    attrs["__DASDAE_version__"] = file_version
    attrs["__dascore__version__"] = dc.__version__


def _save_attrs_and_dims(patch, patch_group):
    """Save the attributes."""
    # copy attrs to group attrs
    # TODO will need to test if objects are serializable
    attr_dict = patch.attrs.model_dump(exclude_unset=True)
    for i, v in attr_dict.items():
        patch_group._v_attrs[f"_attrs_{i}"] = v
    patch_group._v_attrs["_dims"] = ",".join(patch.dims)


def _save_array(data, name, group, h5):
    """Save an array to a group, handle datetime flubbery."""
    # handle datetime conversions
    is_dt = np.issubdtype(data.dtype, np.datetime64)
    is_td = np.issubdtype(data.dtype, np.timedelta64)
    if is_dt or is_td:
        data = to_int(data)
    array_node = _create_or_squash_array(h5, group, name, data)
    array_node._v_attrs["is_datetime64"] = is_dt
    array_node._v_attrs["is_timedelta64"] = is_td


def _save_coords(patch, patch_group, h5):
    """Save coordinates."""
    cm = patch.coords
    for name, coord in cm.coord_map.items():
        dims = cm.dim_map[name]
        # First save coordinate arrays
        data = coord.values
        save_name = f"_coord_{name}"
        _save_array(data, save_name, patch_group, h5)
        # then save dimensions of coordinates
        save_name = f"_cdims_{name}"
        patch_group._v_attrs[save_name] = ",".join(dims)


def _save_patch(patch, wave_group, h5, name):
    """Save the patch to disk."""
    patch_group = _create_or_get_group(h5, wave_group, name)
    _save_attrs_and_dims(patch, patch_group)
    _save_coords(patch, patch_group, h5)
    # add data
    if patch.data.shape:
        _create_or_squash_array(h5, patch_group, "data", patch.data)


# --- Functions for reading


def _get_attrs(patch_group):
    """Get the saved attributes form the group attrs."""
    out = {}
    attrs = [x for x in patch_group._v_attrs._f_list() if x.startswith("_attrs_")]
    for attr_name in attrs:
        key = attr_name.replace("_attrs_", "")
        val = patch_group._v_attrs[attr_name]
        # need to unpack one value arrays
        if isinstance(val, np.ndarray) and not val.shape:
            val = np.asarray([val])[0]
        out[key] = val
    with suppress_warnings(DeprecationWarning):
        return PatchAttrs(**out)


def _read_array(table_array):
    """Read an array into numpy."""
    data = table_array[:]
    if table_array._v_attrs["is_datetime64"]:
        data = data.view("datetime64[ns]")
    if table_array._v_attrs["is_timedelta64"]:
        data = data.view("timedelta64[ns]")
    return data


def _get_coords(patch_group, dims, attrs2):
    """Get the coordinates from a patch group."""
    coord_dict = {}  # just store coordinates here
    coord_dim_dict = {}  # stores {coord_name: ((dims, ...), coord)}
    for coord in [x for x in patch_group if x.name.startswith("_coord_")]:
        name = coord.name.replace("_coord_", "")
        array = _read_array(coord)
        coord = get_coord(
            data=array,
            units=getattr(attrs2, f"{name}_units", None),
            step=getattr(attrs2, f"{name}_step", None),
        )
        coord_dict[name] = coord
    # associates coordinates with dimensions
    c_dims = [x for x in patch_group._v_attrs._f_list() if x.startswith("_cdims")]
    for coord_name in c_dims:
        name = coord_name.replace("_cdims_", "")
        value = patch_group._v_attrs[coord_name]
        assert name in coord_dict, "Should already have loaded coordinate array"
        coord_dim_dict[name] = (tuple(value.split(".")), coord_dict[name])
        # add dimensions to coordinates that have them.
    cm = get_coord_manager(coord_dim_dict, dims=dims)
    return cm


def _get_dims(patch_group):
    """Get the dims tuple from the patch group."""
    dims = patch_group._v_attrs["_dims"]
    if not dims:
        out = ()
    else:
        out = tuple(dims.split(","))
    return out


def _read_patch(patch_group, **kwargs):
    """Read a patch group, return Patch."""
    attrs = _get_attrs(patch_group)
    dims = _get_dims(patch_group)
    coords = _get_coords(patch_group, dims, attrs)
    # Note, previously this was wrapped with try, except (Index, KeyError)
    # and the data = np.array(None) in except block. Not sure, why, removed
    # try except.
    if kwargs:
        coords, data = coords.select(array=patch_group["data"], **kwargs)
    else:
        data = patch_group["data"][:]
    return dc.Patch(data=data, coords=coords, dims=dims, attrs=attrs)


def _get_contents_from_patch_groups(h5, file_version, file_format="DASDAE"):
    """Get the contents from each patch group."""
    out = []
    for group in h5.iter_nodes("/waveforms"):
        contents = _get_patch_content_from_group(group)
        # populate file info
        contents["file_version"] = file_version
        contents["file_format"] = file_format
        contents["path"] = h5.filename
        # suppressing warnings because old dasdae files will issue warning
        # due to d_dim rather than dim_step. TODO fix test files in the future
        with suppress_warnings(DeprecationWarning):
            out.append(dc.PatchAttrs(**contents))
    return out


def _get_patch_content_from_group(group):
    """Get patch content from a single node."""
    attrs = group._v_attrs
    out = {}
    for key in attrs._f_list():
        value = getattr(attrs, key)
        new_key = key.replace("_attrs_", "")
        # need to unpack 0 dim arrays.
        if isinstance(value, np.ndarray) and not value.shape:
            value = np.atleast_1d(value)[0]
        out[new_key] = value
    # rename dims
    out["dims"] = out.pop("_dims")
    return out
