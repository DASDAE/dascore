"""DASDAE format utilities.

See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for the
coord serialization and string-serialization design notes used here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.io.core import STORED_PATCH_ID, make_scan_payload
from dascore.io.dasdae._compat import (
    NOT_DECODED,
    decode_pytables_attr,
    strip_legacy_coord_fields,
    translate_legacy_attrs,
)
from dascore.io.utils import get_exact_coord
from dascore.models.registry import get_model_tag, resolve_tagged_model
from dascore.utils.array import (
    convert_bytes_to_strings,
    convert_strings_to_bytes,
    is_string_byte_serializable_array,
)
from dascore.utils.misc import unbyte
from dascore.utils.pd import filter_df
from dascore.utils.time import to_int

# Keys not counted as true kwargs for determining if patch is filtered/selected.
_KWARG_NON_KEYS = {"file_version", "file_format", "path", "source_patch_key"}
_ATTR_PREFIX = "_attrs_"
_ATTR_TYPE_PREFIX = "_attr_type_"
# Root marker set on files whose patch attr namespace holds only true attrs.
# Files without it may mix flat coord metadata into attrs (see _compat).
_SEPARATE_ATTRS_KEY = "__attrs_coords_separate__"
# Names the attrs class a patch group holds. A sibling of the attr
# namespace rather than a member of it, since attrs allow extras and a
# patch may carry one spelled like this key.
_ATTRS_CLASS_KEY = "__attrs_class__"


# --- Functions for writing DASDAE format


def _write_meta(hfile, file_version):
    """Write metadata to hdf5 file."""
    hfile.attrs["__format__"] = "DASDAE"
    hfile.attrs["__DASDAE_version__"] = file_version
    hfile.attrs["__dascore__version__"] = dc.__version__
    # Mark the file as holding only true attrs (no flat coord metadata),
    # unless appending to a legacy file that already contains mixed patches.
    waveforms = hfile.get("waveforms")
    has_legacy_patches = (
        waveforms is not None
        and len(waveforms)
        and not hfile.attrs.get(_SEPARATE_ATTRS_KEY, False)
    )
    if not has_legacy_patches:
        hfile.attrs[_SEPARATE_ATTRS_KEY] = True


def _is_legacy_file(h5) -> bool:
    """Return True if the file may mix flat coord metadata into patch attrs."""
    return not h5.attrs.get(_SEPARATE_ATTRS_KEY, False)


def _is_legacy_group(patch_group, file_legacy: bool) -> bool:
    """
    Return True if a patch group may mix flat coord metadata into attrs.

    New patch groups appended to a legacy file carry their own marker, so
    they keep exact attr round-trips even though the file stays unmarked.
    """
    return file_legacy and not patch_group.attrs.get(_SEPARATE_ATTRS_KEY, False)


def _get_group_coord_names(patch_group) -> set[str]:
    """Get names of all dims/coords stored in a patch group."""
    names = set(_get_dims(patch_group))
    for key in patch_group:
        if key.startswith("_coord_"):
            names.add(key.removeprefix("_coord_"))
    return names


def _save_attrs_and_dims(patch, patch_group):
    """Save the attributes."""
    # copy attrs to group attrs
    # TODO will need to test if objects are serializable
    attr_dict = patch.attrs.model_dump(exclude_unset=True)
    # The ids are written. An older DASCore reads them as ordinary attrs
    # and then refuses to merge two patches whose ids differ -- which is
    # every pair -- so chunking such a spool there needs conflict="drop".
    # Worth it: a stored id is the only one which survives a move, and
    # everything else DASCore does with a patch already folds them.
    for i, v in attr_dict.items():
        encoded, attr_type = _encode_attr_value(i, v)
        patch_group.attrs[f"{_ATTR_PREFIX}{i}"] = encoded
        if attr_type is not None:
            patch_group.attrs[f"{_ATTR_TYPE_PREFIX}{i}"] = attr_type
    # Values are dumped one at a time rather than as one document, so the
    # class is recorded beside them rather than injected into them. A class
    # which cannot be named (see get_model_tag) is simply not named, which
    # reads back the way a file written before this did.
    if (tag := get_model_tag(type(patch.attrs))) is not None:
        patch_group.attrs[_ATTRS_CLASS_KEY] = tag
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
    # Per-group marker: groups appended to a legacy file are still written
    # in the separated-attrs form and must not be legacy-stripped on read.
    patch_group.attrs[_SEPARATE_ATTRS_KEY] = True
    _save_attrs_and_dims(patch, patch_group)
    _save_coords(patch, patch_group)
    # add data
    if patch.data.shape:
        _save_array(patch.data, "data", patch_group)


# --- Functions for reading


def _get_attrs(patch_group, legacy: bool = True):
    """Get the saved attributes from the group attrs."""
    out = {}
    attrs = [x for x in patch_group.attrs if x.startswith(_ATTR_PREFIX)]
    for attr_name in attrs:
        key = attr_name.removeprefix(_ATTR_PREFIX)
        val = _decode_attr_value(
            patch_group.attrs, key, patch_group.attrs[attr_name], legacy=legacy
        )
        # need to unpack one value arrays
        if isinstance(val, np.ndarray) and not val.shape:
            val = np.asarray([val])[0]
        out[key] = val
    return out


def _get_attrs_class(patch_group) -> type[PatchAttrs]:
    """
    Return the attrs class a patch group names, or the base class.

    A file written before the class was recorded names nothing, and one
    written by a format which is no longer installed names something
    unresolvable; both read as plain attrs, which is what such a file
    always used to give.
    """
    tag = unbyte(patch_group.attrs.get(_ATTRS_CLASS_KEY, None))
    return resolve_tagged_model(tag or None, default=PatchAttrs)


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


def _get_coords(patch_group, dims, attrs2, snap=True):
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
            if snap or np.ndim(array) != 1:
                coord = get_coord(data=array, units=units, step=step)
            else:
                coord = get_exact_coord(array, units=units)
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


def _matches_attr_filters(attrs, kwargs):
    """Return True if attrs match any applicable attr filters in kwargs."""

    def is_nullish(value):
        """Return True if value is a scalar nullish query value."""
        is_null = pd.isnull(value)
        return bool(is_null) if not hasattr(is_null, "__len__") else False

    query = {
        x: y
        for x, y in kwargs.items()
        if x not in _KWARG_NON_KEYS and not x.startswith("_") and not is_nullish(y)
    }
    if not query:
        return True
    attr_df = pd.DataFrame([attrs])
    return bool(filter_df(attr_df, ignore_bad_kwargs=True, **query)[0])


def _get_patch_attrs(patch_group, legacy: bool) -> dict:
    """Get the true patch attrs, cleaning legacy coord metadata if needed."""
    attrs = _get_attrs(patch_group, legacy=legacy)
    if legacy:
        dims = _get_dims(patch_group)
        coord_names = _get_group_coord_names(patch_group)
        attrs["dims"] = ",".join(dims)
        attrs = translate_legacy_attrs(attrs, coord_names)
        attrs = strip_legacy_coord_fields(attrs, coord_names)
    return attrs


def _read_patch(patch_group, legacy: bool = True, **kwargs):
    """Read a patch group, return Patch."""
    attrs = _get_attrs(patch_group, legacy=legacy)
    dims = _get_dims(patch_group)
    if legacy:
        attrs["dims"] = ",".join(dims)
        attrs = translate_legacy_attrs(attrs, _get_group_coord_names(patch_group))
        coords = _get_coords(patch_group, dims, attrs)
        attr_info = strip_legacy_coord_fields(attrs, set(coords.coord_map) | set(dims))
    else:
        coords = _get_coords(patch_group, dims, {})
        attr_info = attrs
    attr_info["_source_patch_key"] = patch_group.name.rsplit("/", maxsplit=1)[-1]
    # An id the file carries is the one which survived the round trip;
    # `read` prefers it to the one it would derive from the path.
    if stored := attr_info.get("patch_id", ""):
        attr_info[STORED_PATCH_ID] = stored
    attrs = _get_attrs_class(patch_group).from_dict(attr_info)
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


def _get_scan_payload_from_group(group, legacy: bool = True, snap=True):
    """Build one structured scan payload from a stored DASDAE patch group."""
    attrs = group.attrs
    out = {}
    # First recover the flat attr payload saved on the patch group itself.
    for key in attrs:
        if not key.startswith(_ATTR_PREFIX):
            continue
        new_key = key.removeprefix(_ATTR_PREFIX)
        value = _decode_attr_value(attrs, new_key, attrs[key], legacy=legacy)
        # need to unpack 0 dim arrays.
        if isinstance(value, np.ndarray) and not value.shape:
            value = np.atleast_1d(value)[0]
        out[new_key] = unbyte(value)
    dims = _get_dims(group)
    if legacy:
        out["dims"] = ",".join(dims)
        out = translate_legacy_attrs(out, _get_group_coord_names(group))
        coords = _get_coords(group, dims, out, snap=snap)
        attr_info = strip_legacy_coord_fields(out, set(coords.coord_map) | set(dims))
    else:
        coords = _get_coords(group, dims, {}, snap=snap)
        attr_info = out
    # Marked here as it is when the patch is read: an id the file carries
    # is the one which survived the round trip, and `scan` prefers it to
    # the one it would derive only when a format says it stored one.
    if stored := attr_info.get("patch_id", ""):
        attr_info[STORED_PATCH_ID] = stored
    # Data shape/dtype come from the stored data node without loading the array.
    data_node = group.get("data")
    dtype = str(data_node.dtype) if data_node is not None else ""
    shape = tuple(data_node.shape) if data_node is not None else ()
    return make_scan_payload(
        attrs=_get_attrs_class(group).from_dict(attr_info),
        coords=coords,
        dims=dims,
        shape=shape,
        dtype=dtype,
        source_patch_key=group.name.rsplit("/", maxsplit=1)[-1],
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


def _decode_attr_value(attrs, key, value, legacy: bool = True):
    """Decode one stored attr value using saved type metadata when present."""
    attr_type = unbyte(attrs.get(f"{_ATTR_TYPE_PREFIX}{key}", None))
    if attr_type is None:
        return _decode_legacy_attr_value(attrs, key, value, legacy=legacy)
    if attr_type == "none":
        return None
    if attr_type == "datetime64[ns]":
        return np.asarray([value], dtype="int64").view("datetime64[ns]")[0]
    if attr_type == "timedelta64[ns]":
        return np.asarray([value], dtype="int64").view("timedelta64[ns]")[0]
    if attr_type == "history_json":
        return tuple(json.loads(unbyte(value) or "[]"))
    return value


def _holds_pytables_payload(attrs, key, value) -> bool:
    """
    Whether a legacy attr holds a PyTables pickle rather than text.

    Nothing in the bytes separates the two -- a string attr of "N." is
    byte-identical to a pickled None. PyTables wrote real strings as UTF-8
    and pickled payloads as raw bytes, and HDF5 stores which of the two an
    attribute holds as its character set, so that is what decides here.
    """
    if not isinstance(value, np.bytes_ | bytes):
        return False
    # Only an h5py attrs manager exposes the character set; a plain
    # mapping of values cannot tell a payload from text.
    get_id = getattr(attrs, "get_id", None)
    if get_id is None:
        return False
    get_cset = getattr(get_id(f"{_ATTR_PREFIX}{key}").get_type(), "get_cset", None)
    return get_cset is not None and get_cset() == 0


def _decode_legacy_attr_value(attrs, key, value, legacy: bool = True):
    """
    Decode one attr value from a file written before attr types were stored.

    A legacy file may hold the value as a PyTables payload, which only
    ``legacy`` files are looked at for.
    """
    if value.__class__.__name__ == "Empty":
        return ""
    if isinstance(value, np.ndarray) and not value.shape:
        value = np.asarray([value])[0]
    if legacy and _holds_pytables_payload(attrs, key, value):
        decoded = decode_pytables_attr(bytes(value))
        if decoded is not NOT_DECODED:
            return decoded
    if isinstance(value, np.bytes_ | bytes):
        try:
            return unbyte(value)
        except UnicodeDecodeError:
            return bytes(value).decode("latin1")
    return value


def _get_file_version(h5):
    """Return the DASDAE file version from a generic HDF5 handle."""
    return unbyte(h5.attrs.get("__DASDAE_version__", ""))


def _get_contents_from_patch_groups_generic(h5, snap=True):
    """Get DASDAE scan summaries from a generic HDF5 handle."""
    waveforms = h5.get("waveforms")
    if waveforms is None:
        return []
    file_legacy = _is_legacy_file(h5)
    return [
        _get_scan_payload_from_group(
            group,
            legacy=_is_legacy_group(group, file_legacy),
            snap=snap,
        )
        for group in waveforms.values()
    ]
