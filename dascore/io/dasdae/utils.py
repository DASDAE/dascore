"""
DASDAE format utilities
"""
import numpy as np

import dascore as dc
from dascore.utils.time import to_number, to_timedelta64

# --- Functions for writing


def _write_meta(hfile):
    """Write metadata to hdf5 file."""
    attrs = hfile.root._v_attrs
    attrs["__format__"] = "DASDAE"
    attrs["__DASDAE_version__"] = dc.io.dasdae.__version__


def _generate_patch_node_name(patch):
    """Generates the name of the node."""

    def _format_datetime64(dt):
        """Format the datetime string in a sensible way."""
        out = str(np.datetime64(dt).astype("datetime64[ns]"))
        return out.replace(":", "_").replace("-", "_").replace(".", "_")

    attrs = patch.attrs
    start = _format_datetime64(attrs.get("time_min", ""))
    end = _format_datetime64(attrs.get("time_max", ""))
    net = attrs.get("network", "")
    sta = attrs.get("station", "")
    tag = attrs.get("tag", "")
    return f"DAS__{net}__{sta}__{tag}__{start}__{end}"


def _save_attrs_and_dim(patch, patch_group):
    """Save the attributes."""
    # copy attrs to group attrs
    # TODO will need to test if objects are serializable
    for i, v in patch.attrs.items():
        patch_group._v_attrs[f"_attrs_{i}"] = v
    patch_group._v_attrs["_dims"] = ",".join(patch.dims)


def _save_array(data, name, group, h5):
    """Save an array to a group, handle datetime flubbery."""
    # handle datetime conversions
    is_dt = np.issubdtype(data.dtype, np.datetime64)
    is_td = np.issubdtype(data.dtype, np.timedelta64)
    if is_dt or is_td:
        data = to_number(data)
    out = h5.create_array(
        group,
        name,
        data,
    )
    out._v_attrs["is_datetime64"] = is_dt
    out._v_attrs["is_timedelta64"] = is_td


def _save_coords(patch, patch_group, h5):
    """Save coordinates"""
    for name, coord in patch._data_array.coords.items():
        data = coord.values
        save_name = f"_coord_{name}"
        _save_array(data, save_name, patch_group, h5)


def _save_patch(patch, wave_group, h5):
    """Save the patch to disk."""
    name = _generate_patch_node_name(patch)
    patch_group = h5.create_group(wave_group, name)
    _save_attrs_and_dim(patch, patch_group)
    _save_coords(patch, patch_group, h5)
    # add data
    if patch.data.shape:
        h5.create_array(patch_group, "data", patch.data)


# --- Functions for reading


def _get_attrs(patch_group):
    """Get the saved attributes form the group attrs."""
    out = {}
    attrs = [x for x in patch_group._v_attrs._f_list() if x.startswith("_attrs_")]
    for attr_name in attrs:
        key = attr_name.replace("_attrs_", "")
        out[key] = patch_group._v_attrs[attr_name]
    return out


def _read_array(table_array):
    """Read an array into numpy."""
    data = table_array[:]
    if table_array._v_attrs["is_datetime64"]:
        data = data.astype("datetime64[ns]")
    if table_array._v_attrs["is_timedelta64"]:
        data = to_timedelta64(data)
    return data


def _get_coords(patch_group):
    """Get the coordinates from a patch group."""
    out = {}
    for coord in [x for x in patch_group if x.name.startswith("_coord_")]:
        name = coord.name.replace("_coord_", "")
        out[name] = _read_array(coord)
    return out


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
    coords = _get_coords(patch_group)
    try:
        data = patch_group["data"][:]
    except (IndexError, KeyError):
        data = np.array(None)
    return dc.Patch(data=data, coords=coords, dims=dims, attrs=attrs)
