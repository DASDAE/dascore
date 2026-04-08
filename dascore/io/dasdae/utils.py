"""DASDAE format utilities.

See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for the
coord serialization and string-serialization design notes used here.
"""

from __future__ import annotations

import contextlib
import json
import pickle

import numpy as np

import dascore as dc
from dascore.config import get_config
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError
from dascore.io.core import _make_scan_payload
from dascore.utils.array import (
    convert_bytes_to_strings,
    convert_strings_to_bytes,
    is_string_byte_serializable_array,
)
from dascore.utils.attrs import separate_coord_info
from dascore.utils.misc import unbyte
from dascore.utils.time import to_int

# Keys not counted as true kwargs for determining if patch is filtered/selected.
_KWARG_NON_KEYS = {"file_version", "file_format", "path", "source_patch_id"}
_ATTR_PREFIX = "_attrs_"
_ATTR_TYPE_PREFIX = "_attr_type_"


# --- Functions for writing DASDAE format


def _write_meta(hfile, file_version):
    """Write metadata to hdf5 file."""
    hfile.attrs["__format__"] = "DASDAE"
    hfile.attrs["__DASDAE_version__"] = file_version
    hfile.attrs["__dascore__version__"] = dc.__version__


def _save_attrs_and_dims(patch, patch_group):
    """Save the attributes."""
    # copy attrs to group attrs
    # TODO will need to test if objects are serializable
    attr_dict = patch.attrs.model_dump(exclude_unset=True)
    for i, v in attr_dict.items():
        encoded, attr_type = _encode_attr_value(i, v)
        patch_group.attrs[f"{_ATTR_PREFIX}{i}"] = encoded
        if attr_type is not None:
            patch_group.attrs[f"{_ATTR_TYPE_PREFIX}{i}"] = attr_type
    patch_group.attrs["_dims"] = ",".join(patch.dims)


def _save_array(data, name, group):
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
    if name in group:
        # Overwrite the dataset in place when callers resave the same array node.
        del group[name]
    array_node = group.create_dataset(name, data=data)
    array_node.attrs["is_datetime64"] = is_dt
    array_node.attrs["is_timedelta64"] = is_td
    array_node.attrs["is_string"] = is_str
    if is_str:
        array_node.attrs["original_string_dtype"] = original_string_dtype
    return array_node


def _save_coords(patch, patch_group):
    """Save coordinates."""
    cm = patch.coords
    for name, coord in cm.coord_map.items():
        dims = cm.dim_map[name]
        # First save coordinate arrays
        data = coord.values
        save_name = f"_coord_{name}"
        array_node = _save_array(data, save_name, patch_group)
        step = coord.step
        if step is not None:
            is_td = np.issubdtype(np.asarray(step).dtype, np.timedelta64)
            array_node.attrs["step"] = to_int(step) if is_td else step
            array_node.attrs["step_is_timedelta64"] = is_td
        if coord.units is not None:
            array_node.attrs["units"] = str(coord.units)
        # then save dimensions of coordinates
        save_name = f"_cdims_{name}"
        patch_group.attrs[save_name] = ",".join(dims)


def _save_patch(patch, wave_group, name):
    """Save the patch to disk."""
    if name in wave_group:
        # Replace the entire patch group so stale datasets/attrs can't survive.
        del wave_group[name]
    patch_group = wave_group.create_group(name)
    _save_attrs_and_dims(patch, patch_group)
    _save_coords(patch, patch_group)
    # add data
    if patch.data.shape:
        _save_array(patch.data, "data", group=patch_group)


# --- Functions for reading


def _get_attrs(patch_group):
    """Get the saved attributes form the group attrs."""
    out = {}
    attrs = [x for x in patch_group.attrs if x.startswith(_ATTR_PREFIX)]
    for attr_name in attrs:
        key = attr_name.replace(_ATTR_PREFIX, "")
        val = _decode_attr_value(patch_group.attrs, key, patch_group.attrs[attr_name])
        # need to unpack one value arrays
        if isinstance(val, np.ndarray) and not val.shape:
            val = np.asarray([val])[0]
        out[key] = val
    return out


def _read_array(table_array):
    """Read an array into numpy."""
    data = table_array[:]
    attrs = table_array.attrs
    if attrs.get("is_datetime64"):
        data = data.view("datetime64[ns]")
    if attrs.get("is_timedelta64"):
        data = data.view("timedelta64[ns]")
    if attrs.get("is_string"):
        original_dtype = unbyte(attrs.get("original_string_dtype", ""))
        data = convert_bytes_to_strings(data, original_dtype)
    return data


def _read_array_sample(table_array, index):
    """Read one array sample and restore datetime-like dtypes when needed."""
    out = table_array[index]
    attrs = table_array.attrs
    if attrs.get("is_datetime64"):
        out = np.asarray([out]).view("datetime64[ns]")[0]
    if attrs.get("is_timedelta64"):
        out = np.asarray([out]).view("timedelta64[ns]")[0]
    if attrs.get("is_string"):
        original_dtype = unbyte(attrs.get("original_string_dtype", ""))
        out = convert_bytes_to_strings(np.asarray([out]), original_dtype)[0]
    return out


def _translate_legacy_attrs(attrs):
    """Normalize legacy DASDAE attr payloads to flat coord metadata."""
    out = dict(attrs)
    coords = out.pop("coords", {})
    if isinstance(coords, str):
        # Older DASDAE files stored the coord-summary payload as a pickled
        # string attr. Decode only this legacy coord metadata so scan/read can
        # recover units and steps without reviving general legacy attr unpickling.
        with contextlib.suppress(
            AttributeError,
            EOFError,
            KeyError,
            pickle.PickleError,
            TypeError,
            UnicodeError,
            ValueError,
        ):
            decoded = pickle.loads(coords.encode("latin1"))
            if (
                hasattr(decoded, "items")
                and not get_config().allow_dasdae_format_unpickle
            ):
                msg = (
                    "This DASDAE file contains legacy pickled coordinate metadata. "
                    "Unpickling DASDAE format metadata is disabled by default for "
                    "security. If you trust this file, enable legacy compatibility "
                    "with set_config(allow_dasdae_format_unpickle=True)."
                )
                raise InvalidFiberFileError(msg)
            coords = decoded
    if hasattr(coords, "to_summary_dict"):
        coords = coords.to_summary_dict()
    if not hasattr(coords, "items"):
        coords = {}
    for name, summary in coords.items():
        if hasattr(summary, "to_summary"):
            summary = summary.to_summary()
        if hasattr(summary, "model_dump"):
            summary = summary.model_dump()
        if not isinstance(summary, dict):
            continue
        for field in ("min", "max", "step", "units", "dtype", "len"):
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
    for coord in patch_group.values():
        name = coord.name.rsplit("/", maxsplit=1)[-1]
        if not name.startswith("_coord_"):
            continue
        name = name.replace("_coord_", "")
        node_attrs = coord.attrs
        units = node_attrs.get("units", None)
        node_step = node_attrs.get("step", None)
        if node_attrs.get("step_is_timedelta64", False):
            node_step = np.timedelta64(node_step, "ns")
        units = units or attrs2.get(f"{name}_units", None)
        step = node_step if node_step is not None else attrs2.get(f"{name}_step", None)
        shape = tuple(coord.shape)
        can_use_range_fast_path = (
            node_step is not None
            and not node_attrs.get("is_string", False)
            and len(shape) == 1
            and shape[0] > 0
        )
        if can_use_range_fast_path:
            start = _read_array_sample(coord, 0)
            stop = start + node_step * shape[0]
            coord = get_coord(start=start, stop=stop, step=node_step, units=units)
        else:
            array = _read_array(coord)
            coord = get_coord(data=array, units=units, step=step)
        coord_dict[name] = coord
    # associates coordinates with dimensions
    group_attrs = patch_group.attrs
    c_dims = [x for x in group_attrs if x.startswith("_cdims")]
    for coord_name in c_dims:
        name = coord_name.replace("_cdims_", "")
        value = unbyte(group_attrs[coord_name])
        assert name in coord_dict, "Should already have loaded coordinate array"
        coord_dim_dict[name] = (tuple(value.split(",")), coord_dict[name])
        # add dimensions to coordinates that have them.
    cm = get_coord_manager(coord_dim_dict, dims=dims)
    return cm


def _get_dims(patch_group):
    """Get the dims tuple from the patch group."""
    dims = unbyte(patch_group.attrs["_dims"])
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
    attr_info["_source_patch_id"] = patch_group.name.rsplit("/", maxsplit=1)[-1]
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


def _kwargs_empty(kwargs) -> bool:
    """Determine if the keyword arguments are *effectively* empty."""
    # These keys get passed in from some spools, so don't count them.
    # We also only count keys whose values are not None.
    out = {
        i: v for i, v in kwargs.items() if v is not None and i not in _KWARG_NON_KEYS
    }
    return not bool(out)


def _get_scan_payload_from_group(group):
    """Build one structured scan payload from a stored DASDAE patch group."""
    attrs = group.attrs
    out = {}
    # First recover the flat attr payload saved on the patch group itself.
    for key in attrs:
        if not key.startswith(_ATTR_PREFIX):
            continue
        value = _decode_attr_value(attrs, key.replace(_ATTR_PREFIX, ""), attrs[key])
        new_key = key.replace(_ATTR_PREFIX, "")
        # need to unpack 0 dim arrays.
        if isinstance(value, np.ndarray) and not value.shape:
            value = np.atleast_1d(value)[0]
        out[new_key] = unbyte(value)
    # rename dims
    out["dims"] = unbyte(attrs["_dims"])
    out = _translate_legacy_attrs(out)
    dims_str = out["dims"]
    dims = tuple(dims_str.split(",")) if dims_str else ()
    # Split flattened coord metadata from the remaining patch attrs.
    _, attr_info = separate_coord_info(out, dims=dims)
    coords = _get_coords(group, dims, out)
    # Data shape/dtype come from the stored data node without loading the array.
    data_node = group.get("data")
    dtype = str(data_node.dtype) if data_node is not None else ""
    shape = tuple(data_node.shape) if data_node is not None else ()
    return _make_scan_payload(
        attrs=PatchAttrs.from_dict(attr_info),
        coords=coords,
        dims=dims,
        shape=shape,
        dtype=dtype,
        source_patch_id=group.name.rsplit("/", maxsplit=1)[-1],
    )


def _encode_history_attr(value):
    """Serialize history as one flat JSON string for DASDAE storage."""
    if value in (None, "", (), []):
        return "[]", "history_json"
    if isinstance(value, str):
        payload = [value]
    else:
        payload = [str(item) for item in value]
    return json.dumps(payload), "history_json"


def _encode_attr_value(key, value):
    """Encode a patch attr into an HDF5-attr-safe representation."""
    if key == "history":
        return _encode_history_attr(value)
    if value is None:
        return "", "none"
    if isinstance(value, np.datetime64):
        return to_int(value), "datetime64[ns]"
    if isinstance(value, np.timedelta64):
        return to_int(value), "timedelta64[ns]"
    return value, None


def _decode_attr_value(attrs, key, value):
    """Decode one stored attr value using saved type metadata when present."""
    attr_type = unbyte(attrs.get(f"{_ATTR_TYPE_PREFIX}{key}", None))
    if attr_type is None:
        return _decode_legacy_attr_value(value)
    if attr_type == "none":
        return None
    if attr_type == "datetime64[ns]":
        return np.asarray([value], dtype="int64").view("datetime64[ns]")[0]
    if attr_type == "timedelta64[ns]":
        return np.asarray([value], dtype="int64").view("timedelta64[ns]")[0]
    if attr_type == "history_json":
        return tuple(json.loads(unbyte(value) or "[]"))
    return value


def _decode_legacy_attr_value(value):
    """Decode legacy DASDAE attrs still used by the shipped fixture files."""
    if value.__class__.__name__ == "Empty":
        return ""
    if isinstance(value, np.ndarray) and not value.shape:
        value = np.asarray([value])[0]
    if isinstance(value, np.bytes_ | bytes):
        try:
            return unbyte(value)
        except UnicodeDecodeError:
            return bytes(value).decode("latin1")
    return value


def _get_file_version(h5):
    """Return the DASDAE file version from a generic HDF5 handle."""
    return unbyte(h5.attrs.get("__DASDAE_version__", ""))


def _get_contents_from_patch_groups_generic(h5):
    """Get DASDAE scan summaries from a generic HDF5 handle."""
    waveforms = h5.get("waveforms")
    if waveforms is None:
        return []
    return [_get_scan_payload_from_group(group) for group in waveforms.values()]
