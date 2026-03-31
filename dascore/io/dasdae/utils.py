"""DASDAE format utilities.

See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for the
coord-summary fast path and string-serialization design notes used here.
"""

from __future__ import annotations

import numpy as np
from tables import NodeError

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import CoordSummary, get_coord
from dascore.core.summary import (
    PatchSummary,
    _normalize_coord_summary_dtype,
    coord_summary_from_available,
    coord_summary_from_metadata,
)
from dascore.utils.array import (
    convert_bytes_to_strings,
    convert_strings_to_bytes,
    is_string_byte_serializable_array,
)
from dascore.utils.attrs import separate_coord_info
from dascore.utils.misc import suppress_warnings
from dascore.utils.time import to_int

# Keys not counted as true kwargs for determining if patch is filtered/selected.
_KWARG_NON_KEYS = {"file_version", "file_format", "path", "source_patch_id"}


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
    """Save an array to a group, handling datetime and string values."""
    data = np.asarray(data)
    is_dt = np.issubdtype(data.dtype, np.datetime64)
    is_td = np.issubdtype(data.dtype, np.timedelta64)
    is_str = is_string_byte_serializable_array(data)
    original_string_dtype = str(data.dtype) if is_str else ""
    if is_dt or is_td:
        data = to_int(data)
    elif is_str:
        data = convert_strings_to_bytes(data)
    array_node = _create_or_squash_array(h5, group, name, data)
    array_node._v_attrs["is_datetime64"] = is_dt
    array_node._v_attrs["is_timedelta64"] = is_td
    array_node._v_attrs["is_string"] = is_str
    if is_str:
        array_node._v_attrs["original_string_dtype"] = original_string_dtype
    return array_node


def _save_coords(patch, patch_group, h5):
    """Save coordinates."""
    cm = patch.coords
    for name, coord in cm.coord_map.items():
        dims = cm.dim_map[name]
        # First save coordinate arrays
        data = coord.values
        save_name = f"_coord_{name}"
        array_node = _save_array(data, save_name, patch_group, h5)
        step = coord.step
        if step is not None:
            is_td = np.issubdtype(np.asarray(step).dtype, np.timedelta64)
            array_node._v_attrs["step"] = to_int(step) if is_td else step
            array_node._v_attrs["step_is_timedelta64"] = is_td
        if coord.units is not None:
            array_node._v_attrs["units"] = str(coord.units)
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
    with suppress_warnings(DeprecationWarning):
        for attr_name in attrs:
            key = attr_name.replace("_attrs_", "")
            val = patch_group._v_attrs[attr_name]
            # need to unpack one value arrays
            if isinstance(val, np.ndarray) and not val.shape:
                val = np.asarray([val])[0]
            out[key] = val
    return out


def _read_array(table_array):
    """Read an array into numpy."""
    data = table_array[:]
    if getattr(table_array._v_attrs, "is_datetime64", False):
        data = data.view("datetime64[ns]")
    if getattr(table_array._v_attrs, "is_timedelta64", False):
        data = data.view("timedelta64[ns]")
    if getattr(table_array._v_attrs, "is_string", False):
        original_dtype = getattr(table_array._v_attrs, "original_string_dtype", "")
        data = convert_bytes_to_strings(data, original_dtype)
    return data


def _translate_legacy_attrs(attrs):
    """Normalize legacy DASDAE attr payloads to flat coord metadata."""
    out = dict(attrs)
    coords = out.pop("coords", {})
    if hasattr(coords, "to_summary_dict"):
        coords = coords.to_summary_dict()
    for name, summary in coords.items():
        if hasattr(summary, "to_summary"):
            summary = summary.to_summary()
        if hasattr(summary, "model_dump"):
            summary = summary.model_dump()
        if not isinstance(summary, dict):
            continue
        for field in ("units", "step"):
            key = f"{name}_{field}"
            value = summary.get(field)
            if key not in out and value not in (None, ""):
                out[key] = value
    dims = out.get("dims", "")
    dims = tuple(dims.split(",")) if isinstance(dims, str) else tuple(dims or ())
    for name in dims:
        old_name = f"d_{name}"
        new_name = f"{name}_step"
        if new_name not in out and old_name in out:
            out[new_name] = out.pop(old_name)
    return out


def _get_coords(patch_group, dims, attrs2):
    """Get the coordinates from a patch group."""
    coord_dict = {}  # just store coordinates here
    coord_dim_dict = {}  # stores {coord_name: ((dims, ...), coord)}
    for coord in [x for x in patch_group if x.name.startswith("_coord_")]:
        name = coord.name.replace("_coord_", "")
        array = _read_array(coord)
        units = getattr(coord._v_attrs, "units", None)
        step = getattr(coord._v_attrs, "step", None)
        if getattr(coord._v_attrs, "step_is_timedelta64", False):
            step = np.timedelta64(step, "ns")
        coord = get_coord(
            data=array,
            units=units or attrs2.get(f"{name}_units", None),
            step=step if step is not None else attrs2.get(f"{name}_step", None),
        )
        coord_dict[name] = coord
    # associates coordinates with dimensions
    c_dims = [x for x in patch_group._v_attrs._f_list() if x.startswith("_cdims")]
    for coord_name in c_dims:
        name = coord_name.replace("_cdims_", "")
        value = patch_group._v_attrs[coord_name]
        assert name in coord_dict, "Should already have loaded coordinate array"
        coord_dim_dict[name] = (tuple(value.split(",")), coord_dict[name])
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
    attrs = _translate_legacy_attrs(_get_attrs(patch_group))
    dims = _get_dims(patch_group)
    coords = _get_coords(patch_group, dims, attrs)
    _, attr_info = separate_coord_info(attrs, dims=dims)
    attrs = PatchAttrs.from_dict(attr_info)
    # Note, previously this was wrapped with try, except (Index, KeyError)
    # and the data = np.array(None) in except block. Not sure, why, removed
    # try except.
    if not _kwargs_empty(kwargs):
        # We need to remove any coordinates from kwargs that are multi-dim
        # coords.
        cmap = coords.dim_map
        sub_kwargs = {
            i: v
            for i, v in kwargs.items()
            if v is not None
            and i not in _KWARG_NON_KEYS
            and ((i not in cmap) or (len(cmap[i]) == 1))
        }
        if sub_kwargs:
            coords, data = coords.select(array=patch_group["data"], **sub_kwargs)
        else:
            data = patch_group["data"][:]
    else:
        data = patch_group["data"][:]
    return dc.Patch(data=data, coords=coords, dims=dims, attrs=attrs)


def _get_contents_from_patch_groups(h5, file_version, file_format="DASDAE"):
    """Get the contents from each patch group."""
    out = []
    for group in h5.iter_nodes("/waveforms"):
        out.append(
            _get_patch_content_from_group(
                group,
                file_version=file_version,
                file_format=file_format,
            )
        )
    return out


def _kwargs_empty(kwargs) -> bool:
    """Determine if the keyword arguments are *effectively* empty."""
    # These keys get passed in from some spools, so don't count them.
    # We also only count keys whose values are not None.
    out = {
        i: v for i, v in kwargs.items() if v is not None and i not in _KWARG_NON_KEYS
    }
    return not bool(out)


def _read_array_sample(table_array, index):
    """Read one array sample and restore datetime-like dtypes when needed."""
    out = table_array[index]
    if getattr(table_array._v_attrs, "is_datetime64", False):
        out = np.asarray([out]).view("datetime64[ns]")[0]
    if getattr(table_array._v_attrs, "is_timedelta64", False):
        out = np.asarray([out]).view("timedelta64[ns]")[0]
    return out


def _get_coord_node_metadata(coord_node) -> dict[str, object]:
    """Extract normalized metadata from a stored coord node."""
    attrs = coord_node._v_attrs
    units = getattr(attrs, "units", None)
    step = getattr(attrs, "step", None)
    if getattr(attrs, "step_is_timedelta64", False):
        step = np.timedelta64(step, "ns")
    is_string = bool(getattr(attrs, "is_string", False))
    dtype = _normalize_coord_summary_dtype(
        str(coord_node.dtype).split("[")[0],
        is_datetime=bool(getattr(attrs, "is_datetime64", False)),
        is_timedelta=bool(getattr(attrs, "is_timedelta64", False)),
        is_string=is_string,
        original_dtype=getattr(attrs, "original_string_dtype", ""),
    )
    length = int(np.prod(coord_node.shape, dtype=int)) if coord_node.shape else 0
    return {
        "dtype": dtype,
        "units": units,
        "step": step,
        "is_string": is_string,
        "length": length,
    }


def _coord_summary_can_use_metadata_fast_path(meta: dict[str, object]) -> bool:
    """Return True when coord metadata is sufficient without reading the array."""
    return meta["step"] is not None and not meta["is_string"]


def _get_coord_summary_from_node(coord_node, dims):
    """Build a coord summary from a saved coord node without reading it all."""
    meta = _get_coord_node_metadata(coord_node)
    if _coord_summary_can_use_metadata_fast_path(meta):
        start = _read_array_sample(coord_node, 0)
        stop = _read_array_sample(coord_node, -1)
        return coord_summary_from_metadata(
            dims=dims,
            min=min(start, stop),
            max=max(start, stop),
            units=meta["units"],
            step=meta["step"],
            dtype=meta["dtype"],
            length=meta["length"],
            is_string=bool(meta["is_string"]),
        )
    data = _read_array(coord_node)
    return coord_summary_from_available(
        dims=dims,
        units=meta["units"],
        step=meta["step"],
        dtype=meta["dtype"],
        length=meta["length"],
        data=data,
        is_string=bool(meta["is_string"]),
    )


def _get_patch_content_from_group(group, file_version="", file_format="DASDAE"):
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
    out = _translate_legacy_attrs(out)
    dims = tuple(out["dims"].split(","))
    legacy_coords, attr_info = separate_coord_info(out, dims=dims)
    coord_map = {}
    for name, summary in legacy_coords.items():
        if {"min", "max"} <= set(summary):
            coord_map[name] = CoordSummary(**summary)
    for coord_node in [x for x in group if x.name.startswith("_coord_")]:
        name = coord_node.name.replace("_coord_", "")
        coord_dims = tuple(getattr(attrs, f"_cdims_{name}", "").split(","))
        summary = _get_coord_summary_from_node(coord_node, coord_dims)
        coord_map[name] = summary
    data_nodes = [x for x in group if x.name == "data"]
    dtype = str(data_nodes[0].dtype) if data_nodes else ""
    return PatchSummary.model_construct(
        attrs=PatchAttrs.from_dict(attr_info),
        coords=coord_map,
        dims=dims,
        shape=(),
        dtype=dtype,
        source_patch_id=group._v_name,
    )
