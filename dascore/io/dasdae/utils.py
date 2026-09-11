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
from dascore.core.coords import (
    _EXACT_GRID_FIELDS,
    CoordMonotonicArray,
    CoordRange,
    CoordSegmented,
    _scalar_dtype,
    get_coord,
)
from dascore.core.summary import normalize_source_patch_key
from dascore.exceptions import PatchAttributeError
from dascore.io.core import STORED_PATCH_ID, make_scan_payload
from dascore.io.dasdae._compat import (
    NOT_DECODED,
    decode_pytables_attr,
    strip_legacy_coord_fields,
    translate_legacy_attrs,
)
from dascore.io.utils import get_exact_coord, resolve_keyed_source
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
    # appending never relabels a file below the version its groups need
    existing = _get_file_version(hfile)
    if existing and float(existing) > float(file_version):
        file_version = existing
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


def _raw(value, dtype):
    """A time as its integer ticks in the coordinate's unit; else itself."""
    array = np.asarray(value)
    if array.dtype.kind not in "mM":
        return value
    # a range's start may state a coarser unit than its dtype
    name = "start" if array.dtype.kind == "M" else "step"
    return array.astype(_scalar_dtype(dtype, name)).astype("int64")[()]


def _extended_float(coord) -> bool:
    """
    Whether the coordinate's floats are wider than a double.

    Judged by the scalar, not the item size: a long double stays a numpy
    scalar under ``item()`` even where it is only 64 bits wide.
    """
    dtype = np.dtype(coord.dtype)
    return dtype.kind == "f" and not isinstance(np.zeros((), dtype)[()].item(), float)


# Version 2 nodes state the coordinate class they hold, as every DASCore
# model states its class in a document (see dascore.models.registry).
_OBJECT_TYPE = "object_type"


def _save_coord(coord, name, group, compact: bool):
    """
    Save one coordinate node.

    Version 2 (``compact``) states each node's class: a range is written
    as its description and a segmented coordinate as a group of its
    segments, so a long acquisition costs a few numbers and no label is
    re-inferred on read; any other class, and version 1 throughout,
    writes its values.
    """
    if compact and isinstance(coord, CoordSegmented):
        node = group.create_group(name)
        for i, segment in enumerate(coord.segments):
            _save_coord(segment, str(i), node, compact)
    elif compact and isinstance(coord, CoordRange) and not _extended_float(coord):
        node = group.create_dataset(name, shape=(0,), dtype="int64")
        node.attrs["dtype"] = str(coord.dtype)
        node.attrs["start"] = _raw(coord.start, coord.dtype)
        node.attrs["length"] = len(coord)
        if coord._exact:
            for field in _EXACT_GRID_FIELDS:
                node.attrs[field] = getattr(coord, field)
        else:
            node.attrs["stop"] = _raw(coord.stop, coord.dtype)
            node.attrs["step"] = _raw(coord.step, coord.dtype)
    else:
        node = _save_array(coord.values, name, group)
        # Version 1 reads an array's step as a range to rebuild from its
        # first value, so only a range may state one there; version 2
        # reads it as the grid an array declares.
        step = coord.step
        if step is not None and (compact or isinstance(coord, CoordRange)):
            is_td = np.issubdtype(np.asarray(step).dtype, np.timedelta64)
            node.attrs["step"] = to_int(step) if is_td else step
            node.attrs["step_is_timedelta64"] = is_td
    if compact:
        node.attrs[_OBJECT_TYPE] = get_model_tag(type(coord))
    if coord.units is not None:
        node.attrs["units"] = str(coord.units)


def _save_coords(patch, patch_group, compact: bool):
    """Save coordinates and their dimensions."""
    cm = patch.coords
    for name, coord in cm.coord_map.items():
        _save_coord(coord, f"_coord_{name}", patch_group, compact)
        patch_group.attrs[f"_cdims_{name}"] = ",".join(cm.dim_map[name])


def _check_storable(patch):
    """Refuse a patch version 1 cannot store, before touching the file."""
    for name, coord in patch.coords.coord_map.items():
        if getattr(coord, "step_denominator", None) not in (None, 1):
            # Version 1 stores one whole-tick step and rebuilds the range
            # from it, which would quietly move every label off its grid.
            msg = (
                f"Coordinate {name!r} has a fractional step "
                f"({coord.step_exact}) which DASDAE format 1 cannot store."
            )
            raise NotImplementedError(msg)


def _save_patch(patch, wave_group, name, compact: bool = False):
    """Save the patch to disk."""
    if not compact:
        _check_storable(patch)
    if name in wave_group:
        # Replace the entire patch group so stale datasets/attrs can't survive.
        del wave_group[name]
    patch_group = wave_group.create_group(name)
    # Per-group marker: groups appended to a legacy file are still written
    # in the separated-attrs form and must not be legacy-stripped on read.
    patch_group.attrs[_SEPARATE_ATTRS_KEY] = True
    _save_attrs_and_dims(patch, patch_group)
    _save_coords(patch, patch_group, compact)
    # add data
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
    data = np.asarray(table_array[()])
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


def _read_range(node, units):
    """Rebuild a range from its version-2 description."""
    attrs = node.attrs
    dtype = np.dtype(unbyte(attrs["dtype"]))
    start = np.asarray(attrs["start"]).astype(dtype)[()]
    shape = (int(attrs["length"]),)
    if "step_numerator" in attrs:
        grid = {name: int(attrs[name]) for name in _EXACT_GRID_FIELDS}
        return CoordRange(start=start, shape=shape, units=units, **grid)
    stop = np.asarray(attrs["stop"]).astype(dtype)[()]
    step = attrs["step"]
    if dtype.kind in "mM":
        step = np.asarray(step).astype(_scalar_dtype(dtype, "step"))[()]
    elif isinstance(step, np.floating):
        # as the python float it was written from: a numpy scalar would
        # promote a float32 range to float64
        step = step.item()
    coord = CoordRange(start=start, stop=stop, step=step, units=units)
    # The stored fields are a validated range's own; deriving the count
    # from them again can move a float32 endpoint by a sample.
    return coord._construct(dict(start=start, stop=stop, step=step, shape=shape))


def _node_step(attrs):
    """The step an array node declares, or None."""
    step = attrs.get("step", None)
    if step is not None and attrs.get("step_is_timedelta64", False):
        step = np.timedelta64(step, "ns")
    return step


def _read_segment(node):
    """Rebuild one segment of a version-2 segmented coordinate."""
    units = node.attrs.get("units", None)
    if "start" in node.attrs:
        return _read_range(node, units)
    # the segments were settled exactly when written, so an array
    # segment is read as the values it holds, never snapped to a range
    values = _read_array(node)
    return CoordMonotonicArray(values=values, units=units, step=_node_step(node.attrs))


def _read_coord(node, name, attrs2, snap):
    """Rebuild one coordinate from its node."""
    node_attrs = node.attrs
    units = node_attrs.get("units", None) or attrs2.get(f"{name}_units", None)
    object_type = unbyte(node_attrs.get(_OBJECT_TYPE, ""))
    if object_type == "CoordSegmented":
        segments = [_read_segment(node[str(i)]) for i in range(len(node))]
        return CoordSegmented(segments=segments, units=units)
    if object_type == "CoordRange" and "start" in node_attrs:
        return _read_range(node, units)
    # any other class, a range too wide to describe, and every version 1
    # node hold their values
    node_step = _node_step(node_attrs)
    if object_type:
        # a version 2 array holds exactly its values; a step on it is the
        # grid it declares, never a range to rebuild
        array = _read_array(node)
        if node_step is not None:
            return get_coord(data=array, units=units, step=node_step)
        if snap or np.ndim(array) != 1:
            return get_coord(data=array, units=units)
        return get_exact_coord(array, units=units)
    step = node_step if node_step is not None else attrs2.get(f"{name}_step", None)
    shape = tuple(node.shape)
    can_use_range_fast_path = (
        node_step is not None
        and not node_attrs.get("is_string", False)
        and len(shape) == 1
        and shape[0] > 0
    )
    if can_use_range_fast_path:
        start = _read_array_sample(node, 0)
        stop = start + node_step * shape[0]
        return get_coord(start=start, stop=stop, step=node_step, units=units)
    array = _read_array(node)
    if snap or np.ndim(array) != 1:
        # A stored nominal step is a grid claim the values must meet, which
        # a legacy file's jittered values need not; it names the spacing
        # only for a single sample, where the values cannot.
        single = np.ndim(array) == 1 and len(array) == 1
        return get_coord(data=array, units=units, step=step if single else None)
    return get_exact_coord(array, units=units)


def _get_coords(patch_group, dims, attrs2, snap=True):
    """Get the coordinates from a patch group."""
    coord_dict = {}  # just store coordinates here
    coord_dim_dict = {}  # stores {coord_name: ((dims, ...), coord)}
    for node in patch_group.values():
        name = node.name.rsplit("/", maxsplit=1)[-1]
        if not name.startswith("_coord_"):
            continue
        name = name.removeprefix("_coord_")
        coord_dict[name] = _read_coord(node, name, attrs2, snap)
    # associates coordinates with dimensions
    group_attrs = patch_group.attrs
    c_dims = [x for x in group_attrs if x.startswith("_cdims")]
    for coord_name in c_dims:
        name = coord_name.replace("_cdims_", "")
        value = unbyte(group_attrs[coord_name])
        assert name in coord_dict, "Should already have loaded coordinate array"
        coord_dim_dict[name] = (tuple(value.split(",")), coord_dict[name])
        # add dimensions to coordinates that have them.
    data = patch_group.get("data")
    shape = tuple(data.shape) if data is not None else None
    cm = get_coord_manager(coord_dim_dict, dims=dims, shape=shape)
    return cm


def _get_dims(patch_group):
    """Get the dims tuple from the patch group."""
    dims = unbyte(patch_group.attrs["_dims"])
    if not dims:
        out = ()
    else:
        out = tuple(dims.split(","))
    return out


def _get_patch_group(h5, source_patch_key=""):
    """
    Return the one waveform group a source patch key names.

    A key names a direct child of the waveform group, never an h5py path.
    """
    waveforms = h5.get("waveforms", {})
    key = normalize_source_patch_key(source_patch_key)
    if "/" in key:
        # h5py would read the key as a path; a patch is a direct child
        raise PatchAttributeError(f"No patch named '{key}' in {h5.filename}.")
    return resolve_keyed_source(waveforms, key, where=str(h5.filename))


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
            data = patch_group["data"][()]
    else:
        data = patch_group["data"][()]
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
