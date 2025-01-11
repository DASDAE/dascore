"""DASDAE format utilities."""

from __future__ import annotations

import numpy as np
from tables import NodeError

import dascore as dc
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.utils.hdf5 import Empty
from dascore.utils.misc import suppress_warnings, unbyte
from dascore.utils.fs import get_uri
from dascore.utils.time import to_int

# --- Functions for writing DASDAE format


def _santize_pytables(some_dict):
    """Remove pytables names from a dict, remove any pickle-able things."""
    pytables_names = {"CLASS", "FLAVOR", "TITLE", "VERSION"}
    out = {}
    for i, v in some_dict.items():
        if i in pytables_names:
            continue
        try:
            val = unbyte(v)
        except ValueError:
            continue
        # Get rid of empty enum.
        if isinstance(val, Empty):
            val = ""
        out[i] = val
    return out


def _create_or_get_group(h5, group, name):
    """Create a new group or get existing."""
    try:
        group = h5.create_group(group, name)
    except NodeError:
        group = getattr(group, name)
    return group


def _create_or_squash_array(h5, group, name, data):
    """Create a new array, if it exists delete and re-create."""
    breakpoint()
    try:
        array = h5.create_array(group, name, data)
    except NodeError:
        old_node = getattr(group, name)
        h5.remove_node(old_node)
        array = h5.create_array(group, name, data)
    return array


def _write_meta(hfile, file_version):
    """Write metadata to hdf5 file."""
    attrs = hfile.attrs
    attrs["__format__"] = "DASDAE"
    attrs["__DASDAE_version__"] = file_version
    attrs["__dascore__version__"] = dc.__version__


def _save_attrs_and_dims(patch, patch_group):
    """Save the attributes."""
    # copy attrs to group attrs
    # TODO will need to test if objects are serializable
    attr_dict = patch.attrs.model_dump(exclude_unset=True)
    for i, v in attr_dict.items():
        patch_group.attrs[f"_attrs_{i}"] = v
    patch_group.attrs["_dims"] = ",".join(patch.dims)


def _save_array(data, name, group, h5):
    """Save an array to a group, handle datetime flubbery."""
    # handle datetime conversions
    is_dt = np.issubdtype(data.dtype, np.datetime64)
    is_td = np.issubdtype(data.dtype, np.timedelta64)
    if is_dt or is_td:
        data = to_int(data)
    array_node = _create_or_squash_array(h5, group, name, data)
    array_node.attrs["is_datetime64"] = is_dt
    array_node.attrs["is_timedelta64"] = is_td
    return array_node


def _save_coords(patch, patch_group, h5):
    """Save coordinates."""
    cm = patch.coords
    for name, coord in cm.coord_map.items():
        summary = coord.to_summary(name=name, dims=cm.dims[name]).model_dump(
            exclude_defaults=True
        )
        breakpoint()
        # First save coordinate arrays
        data = coord.values
        save_name = f"_coord_{name}"
        dataset = _save_array(data, save_name, patch_group, h5)
        dataset.attrs.update(summary)


def _save_patch(patch, wave_group, h5, name):
    """Save the patch to disk."""
    patch_group = _create_or_get_group(h5, wave_group, name)
    _save_attrs_and_dims(patch, patch_group)
    _save_coords(patch, patch_group, h5)
    # add data
    if patch.data.shape:
        _create_or_squash_array(h5, patch_group, "data", patch.data)


# --- Functions for reading


def _get_attrs(patch_group, path, format_name, format_version):
    """Get the saved attributes form the group attrs."""
    out = {}
    attrs = [x for x in patch_group.attrs if x.startswith("_attrs_")]
    tables_attrs = _santize_pytables(dict(patch_group.attrs))
    for key, value in tables_attrs.items():
        new_key = key.replace("_attrs_", "")
        # need to unpack 0 dim arrays.
        if isinstance(value, np.ndarray) and not value.shape:
            value = np.atleast_1d(value)[0]
        attrs[new_key] = value
    out["path"] = path
    out["format_name"] = format_name
    out["format_version"] = format_version
    return out


def _read_array(table_array):
    """Read an array into numpy."""
    data = table_array[:]
    if table_array.attrs["is_datetime64"]:
        data = data.view("datetime64[ns]")
    if table_array.attrs["is_timedelta64"]:
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
    c_dims = [x for x in patch_group.attrs if x.startswith("_cdims")]
    for coord_name in c_dims:
        name = coord_name.replace("_cdims_", "")
        value = patch_group.attrs[coord_name]
        assert name in coord_dict, "Should already have loaded coordinate array"
        coord_dim_dict[name] = (tuple(value.split(".")), coord_dict[name])
        # add dimensions to coordinates that have them.
    cm = get_coord_manager(coord_dim_dict, dims=dims)
    return cm


def _get_dims(patch_group):
    """Get the dims tuple from the patch group."""
    dims = patch_group.attrs["_dims"]
    if not dims:
        out = ()
    else:
        out = tuple(dims.split(","))
    return out


def _read_patch(patch_group, path, format_name, format_version, **kwargs):
    """Read a patch group, return Patch."""
    attrs = _get_attrs(patch_group, path, format_name, format_version)
    dims = _get_dims(patch_group)
    coords = _get_coords(patch_group, dims, attrs)
    if kwargs:
        coords, data = coords.select(array=patch_group["data"], **kwargs)
    else:
        data = patch_group["data"]
    return dc.Patch(data=data[:], coords=coords, dims=dims, attrs=attrs)


def _get_summary_from_patch_groups(h5, format_name="DASDAE"):
    """Get the contents from each patch group."""
    path = get_uri(h5)
    format_version = h5.attrs["__DASDAE_version__"]
    out = []
    for name, group in h5[("/waveforms")].items():
        contents = _get_patch_content_from_group(
            group,
            path=path,
            format_name=format_name,
            format_version=format_version,
        )
        # suppressing warnings because old dasdae files will issue warning
        # due to d_dim rather than dim_step.
        # TODO fix in parser.
        with suppress_warnings(DeprecationWarning):
            out.append(dc.PatchSummary(**contents))

    return out


def _get_coord_info(info, group):
    """Get the coord dictionary."""
    coords = {}
    coord_ds_names = tuple(x for x in group if x.startswith("_coord_"))
    for ds_name in coord_ds_names:
        name = ds_name.replace("_coord_", "")
        ds = group[ds_name]
        attrs = _santize_pytables(dict(ds.attrs))
        # Need to get old dimensions from c_dims in attrs.
        if "dims" not in attrs:
            attrs["dims"] = info.get(f"_cdims_{name}", name)
        # The summary info is not stored in attrs; need to read coord array.
        c_info = {}
        if "min" not in attrs:
            c_summary = (
                dc.core.get_coord(data=ds[:])
                .to_summary(name=name, dims=attrs["dims"])
                .model_dump(exclude_unset=True, exclude_defaults=True)
            )
            c_info.update(c_summary)

        c_info.update(
            {
                "dtype": ds.dtype.str,
                "shape": ds.shape,
                "name": name,
            }
        )
        coords[name] = c_info
    return coords


def _get_patch_content_from_group(group, path, format_name, format_version):
    """Get patch content from a single node."""
    # The attributes in the table.
    attrs = _get_attrs(group, path, format_name, format_version)
    # Get coord info
    coords = _get_coord_info(attrs, group)
    # Overwrite (or add) file-specific info.
    attrs["path"] = path
    attrs["file_format"] = format_name
    attrs["file_version"] = format_version
    # Add data info.
    data = group["data"]
    dims = attrs.pop("_dims", None)
    return dict(data=data, attrs=attrs, dims=dims, coords=coords)
