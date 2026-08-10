"""Copy metadata from a DASDAE inventory onto a patch."""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np

import dascore as dc
from dascore.constants import INVENTORY_ATTRS, PatchType, attr_conflict_description
from dascore.core.coords import BaseCoord, get_coord
from dascore.core.inventory import (
    VALID_COORDINATE_LABELS,
    Interrogator,
    Inventory,
    ResolvedContext,
    _annotation_kind,
)
from dascore.exceptions import ParameterError, PatchError
from dascore.utils.attrs import validate_conflict
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import iterate
from dascore.utils.patch import patch_function
from dascore.utils.time import to_datetime64

# Attrs describing the data as it now stands rather than the system which
# recorded it. Processing functions maintain them, so blanket enrichment
# leaves them alone; naming one explicitly restores the as-acquired value.
_DATA_STATE_ATTRS = ("data_type", "data_category", "data_units")

# Observing-system facts which processing nevertheless maintains: decimating
# in time changes the sample rate, and in distance the channel spacing. They
# are excluded from blanket enrichment for the same reason as the data trio,
# and naming one restores the as-acquired value.
_PROCESSING_MAINTAINED_ATTRS = ("sample_rate", "spatial_interval")

_VALID_ON_MISSING = ("raise", "nan", "skip")

# Tracks whose fields enrich can project, and where their intervals live.
_TRACK_NAMES = ("optical_components", "geometry", "coupling")

# Track fields whose units the inventory documents.
_TRACK_FIELD_UNITS = {
    "coupling.depth": "m",
    "optical_components.optical_length": "m",
}

on_missing_description = """
on_missing
    What to do when an explicitly requested name is one the inventory does
    not define: "raise" (the default), "nan" to fill the dtype-appropriate
    missing marker, or "skip" to leave it off. Blanket requests copy what is
    applicable and never trigger it, and per-channel coverage gaps are always
    missing values rather than errors.
""".strip()


def _validate_on_missing(on_missing: str) -> str:
    """Ensure on_missing is a supported value."""
    if on_missing not in _VALID_ON_MISSING:
        msg = f"on_missing must be one of {_VALID_ON_MISSING}, got {on_missing!r}."
        raise ParameterError(msg)
    return on_missing


def _get_data_source_id(patch, data_source_id) -> str:
    """Return the id to resolve, requiring the patch and caller to agree."""
    patch_id = patch.attrs.data_source_id
    if data_source_id and patch_id and data_source_id != patch_id:
        msg = (
            f"The patch's data_source_id {patch_id!r} and the requested "
            f"{data_source_id!r} disagree; enrich resolves one data source."
        )
        raise PatchError(msg)
    out = data_source_id or patch_id
    if not out:
        msg = (
            "The patch has no data_source_id, so it names no inventory entry. "
            "Set one on the patch or pass data_source_id to enrich."
        )
        raise PatchError(msg)
    return out


def _get_resolution_times(patch, time):
    """
    Return the first and last instants the patch must resolve at.

    A patch whose time axis is no longer physical (a lag-time correlation,
    say) has no such instants of its own and takes an explicit one instead.
    Accepting both would be two spellings of the resolution instant.
    """
    coord = patch.coords.coord_map.get("time")
    physical = coord is not None and np.issubdtype(coord.dtype, np.datetime64)
    if physical:
        if time is not None:
            msg = (
                "The patch has a time coordinate, which is when it was "
                "recorded; enrich does not also accept a time argument."
            )
            raise PatchError(msg)
        return coord.min(), coord.max()
    if time is None:
        msg = (
            "The patch has no physical time coordinate, so enrich cannot tell "
            "which epoch applies; pass time to name the instant."
        )
        raise PatchError(msg)
    stamp = to_datetime64(time)
    return stamp, stamp


def _resolve_context(inventory, source_id, times) -> ResolvedContext:
    """
    Resolve the one acquisition and optical path covering the whole patch.

    Acquisition metadata is scalar per patch, so a patch straddling an epoch
    boundary has two answers to every question enrich asks.
    """
    first, last = times
    context = inventory.resolve(source_id, time=first)
    if first != last:
        other = inventory.resolve(source_id, time=last)
        if other.acquisition != context.acquisition:
            what = "acquisition"
        elif other.optical_path != context.optical_path:
            what = "optical path"
        else:
            what = ""
        if what:
            msg = (
                f"The patch spans a change of {what} for {source_id!r} "
                f"between {first} and {last}; select or split it first."
            )
            raise PatchError(msg)
    return context


def _get_interrogator(inventory, acquisition):
    """Return the resolved interrogator of an acquisition, if it has one."""
    value = acquisition.interrogator
    if isinstance(value, str):
        return inventory.get_resource(value)
    return value


def _get_system_attrs(inventory, context) -> dict:
    """Return the observing-system facts the resolved context defines."""
    interrogator = _get_interrogator(inventory, context.acquisition)
    out = {}
    for name in INVENTORY_ATTRS:
        prefix, _, field = name.rpartition(".")
        obj = interrogator if prefix == "interrogator" else context.acquisition
        value = getattr(obj, field, None)
        if _is_unset(value):
            continue
        out[name] = value
    return out


def _get_attr_values(inventory, context, attrs, on_missing) -> dict:
    """Return the attrs to copy, honoring on_missing for named requests."""
    if attrs is False or attrs is None:
        return {}
    system = _get_system_attrs(inventory, context)
    if attrs is True:
        return {
            i: v for i, v in system.items() if i not in _PROCESSING_MAINTAINED_ATTRS
        }
    available = dict(system)
    for name in _DATA_STATE_ATTRS:
        value = getattr(context.acquisition, name)
        if not _is_unset(value):
            available[name] = value
    out = {}
    for name in iterate(attrs):
        if name in available:
            out[name] = available[name]
        elif on_missing == "raise":
            msg = (
                f"The inventory defines no {name!r} for "
                f"{context.acquisition.code!r}; use on_missing to allow it."
            )
            raise PatchError(msg)
        elif on_missing == "nan":
            out[name] = _missing_marker(context, name)
    return out


def _missing_marker(context, name):
    """
    Return the missing marker matching the field's type.

    NaN is the marker a number can carry; nothing else has one, so None
    stands for "the inventory does not say".
    """
    prefix, _, field = name.rpartition(".")
    model = Interrogator if prefix == "interrogator" else type(context.acquisition)
    info = model.model_fields.get(field)
    annotation = None if info is None else info.annotation
    return np.nan if _is_numeric_annotation(annotation) else None


def _is_numeric_annotation(annotation) -> bool:
    """Return True when a field annotation admits a plain number."""
    if annotation is float or annotation is int:
        return True
    return any(_is_numeric_annotation(x) for x in get_args(annotation))


def _is_unset(value) -> bool:
    """
    Return True when an attr holds no information.

    NaN is how readers spell an unknown number, so a NaN placeholder is
    filled rather than treated as a value which disagrees.
    """
    if value is None or (isinstance(value, str) and not value):
        return True
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


def _values_equal(value1, value2) -> bool:
    """Return True when two attr values agree."""
    try:
        return bool(value1 == value2)
    except (TypeError, ValueError):
        return False


def _apply_conflicts(patch, new_attrs, conflicts) -> tuple[dict, list]:
    """
    Return the attrs to set and to drop, given the conflict policy.

    Filling an empty attr is never a conflict, and neither are equal values;
    a conflict is both sides holding different information.
    """
    current = dict(patch.attrs)
    updates, drops = {}, []
    for name, value in new_attrs.items():
        old = current.get(name, None)
        if _is_unset(old) or _values_equal(old, value):
            updates[name] = value
        elif conflicts == "raise":
            msg = (
                f"The patch's {name!r} is {old!r} but the inventory says "
                f"{value!r}. A disagreement here usually means the "
                "data_source_id resolved to the wrong place."
            )
            raise PatchError(msg)
        elif conflicts == "drop":
            drops.append(name)
        else:  # keep_first: enrichment puts the inventory's value first
            updates[name] = value
    return updates, drops


# The patch coordinates each map axis can be read from, in preference
# order. A patch reframed onto the path keeps the interrogator's meters
# under the explicit name, which is why it wins over plain distance.
_AXIS_COORDS = {
    "channel": ("channel",),
    "instrument_distance": ("instrument_distance", "distance"),
}


def _get_channel_axes(patch, acquisition) -> list[tuple[str, str]]:
    """
    Return every map axis the patch can be read on, with its coordinate.

    The map states its control points in whichever coordinates were
    measured, so the patch decides which of them is read. Guessing instead
    would be wrong by the channel spacing, silently.
    """
    dist_map = acquisition.distance_map
    if dist_map is None:
        msg = (
            f"Acquisition {acquisition.code!r} defines no distance_map, so "
            "its channels cannot be placed on the optical path."
        )
        raise PatchError(msg)
    out = []
    for axis in dist_map.axes:
        for name in _AXIS_COORDS[axis]:
            if name in patch.coords.coord_map:
                out.append((axis, name))
                break
    if out:
        return out
    wanted = sorted({x for axis in dist_map.axes for x in _AXIS_COORDS[axis]})
    msg = (
        f"Acquisition {acquisition.code!r} maps {list(dist_map.axes)} onto "
        f"path distance, so it needs one of the {wanted} coordinates, and "
        f"this patch has {sorted(patch.coords.coord_map)}. An acquisition "
        "whose patches carry interrogator meters is calibrated with a "
        "distance_map on the instrument_distance axis, one control point "
        "being enough to state an origin."
    )
    raise PatchError(msg)


def _get_channel_distances(patch, acquisition) -> tuple[str, str, np.ndarray]:
    """
    Return the channel coord name, its dimension, and optical distances.

    A patch which carries more than one of the map's coordinates must be
    consistent with the map about all of them. Picking one and moving on
    would answer a question the patch itself contradicts.
    """
    resolved = []
    for axis, name in _get_channel_axes(patch, acquisition):
        dims = patch.coords.dim_map[name]
        if len(dims) != 1:
            msg = f"The {name!r} coordinate must belong to exactly one dimension."
            raise PatchError(msg)
        values = patch.coords.coord_map[name].values
        distances = acquisition.channel_to_distance(values, axis=axis)
        resolved.append((name, dims[0], distances))
    first = resolved[0]
    for name, _, distances in resolved[1:]:
        if not _distances_agree(first[2], distances):
            msg = (
                f"The patch's {first[0]!r} and {name!r} coordinates place its "
                f"channels differently on the path ({first[2][0]} vs "
                f"{distances[0]} for the first channel). The patch and the "
                f"map of {acquisition.code!r} disagree; drop the coordinate "
                "which does not belong to this acquisition."
            )
            raise PatchError(msg)
    return first


def _distances_agree(first, second) -> bool:
    """Return True when two resolutions place every channel together."""
    if first.shape != second.shape:
        return False
    both_nan = np.isnan(first) & np.isnan(second)
    return bool(np.all(np.isclose(first, second, equal_nan=True) | both_nan))


def _fill_from_intervals(distances, intervals, values, kind):
    """
    Return per-distance values of an interval track.

    Coverage is half-open, matching the tracks themselves, with the
    outermost covered endpoint included so the last channel of a path is
    not silently uncovered. Point markers cover nothing.
    """
    fill = {"boolean": False, "numeric": np.nan}.get(kind, None)
    out = np.full(len(distances), fill, dtype=object)
    spans = [(lo, hi, value) for (lo, hi), value in zip(intervals, values) if lo < hi]
    claimed = np.zeros(len(distances), dtype=bool)
    for lo, hi, _ in spans:
        claimed |= (distances >= lo) & (distances < hi)
    for lo, hi, value in spans:
        covered = (distances >= lo) & (distances < hi)
        # The end of a coverage run belongs to the interval which ends
        # there unless another interval claims it, so the last channel of
        # a run is not left out by the half-open rule.
        covered |= (distances == hi) & ~claimed
        if isinstance(value, tuple | list):
            msg = (
                f"Cannot project the multi-valued {value!r} onto channels; "
                "a coordinate holds one value per channel."
            )
            raise PatchError(msg)
        if kind == "boolean":
            # Membership groups may overlap, so a channel belongs when any
            # covering interval says it does: the group is the union of its
            # true intervals.
            if value:
                out[covered] = True
        else:
            out[covered] = value
    if kind == "boolean":
        return out.astype(bool)
    if kind == "numeric":
        return np.asarray([np.nan if x is None else x for x in out], dtype=float)
    return out


def _get_annotation_coord(path, group, distances):
    """Return the coordinate values of one annotation group."""
    items = [x for x in path.annotations if x.group == group]
    if not items:
        return None
    kind = _annotation_kind(items[0].value)
    intervals = [x.interval for x in items]
    values = [x.value for x in items]
    return _fill_from_intervals(distances, intervals, values, kind)


def _get_track_coord(path, track, field, distances):
    """Return the coordinate values of one field of a typed track."""
    if track == "optical_components":
        items = path.optical_components
        intervals = list(path.component_intervals())
    else:
        items = getattr(path, track)
        intervals = [x.interval for x in items]
    if not items:
        return None
    values = [getattr(x, field, None) for x in items]
    if all(x is None for x in values):
        # Either the field is misspelled or no interval states it; both
        # mean the inventory defines nothing here.
        return None
    kinds = {_annotation_kind(x) for x in values if x is not None}
    kind = kinds.pop() if len(kinds) == 1 else "string"
    filled = _fill_from_intervals(distances, intervals, values, kind)
    if units := _TRACK_FIELD_UNITS.get(f"{track}.{field}"):
        return get_coord(data=filled, units=units)
    return filled


def _get_geometry_coord(inventory, path, label, distances):
    """Return one coordinate axis of the path geometry, with its units."""
    crs = inventory.coordinate_reference_system
    index = crs.axis_index(label)
    if not path.geometry:
        return None
    coords = path.coordinates_at(distances)
    if index >= coords.shape[1]:
        return None
    units = crs.units[index] if index < len(crs.units) else None
    return get_coord(data=coords[:, index], units=units)


def _get_blanket_coord_names(inventory, path) -> list[str]:
    """
    Return the coordinate names a blanket request copies.

    The geometry axes and the annotation groups: what the path says about
    each channel. Optical distance and the typed-track fields are asked for
    by name, since they restate what the patch's own axis and the inventory
    already record.
    """
    labels = inventory.coordinate_reference_system.coordinate_labels
    out = ["x", "y", "z"][: len(labels)] if path.geometry else []
    seen = dict.fromkeys(x.group for x in path.annotations)
    return out + [x for x in seen if x]


def _get_coord_values(inventory, path, name, distances, optical_distances):
    """Return the values of one requested coordinate, or None if undefined."""
    if name == "distance":
        # Optical path distance is in meters, as every path length is.
        return get_coord(data=optical_distances, units="m")
    track, _, field = name.partition(".")
    if track in _TRACK_NAMES and field:
        return _get_track_coord(path, track, field, distances)
    if name in VALID_COORDINATE_LABELS:
        return _get_geometry_coord(inventory, path, name, distances)
    return _get_annotation_coord(path, name, distances)


def _coords_equal(existing, values) -> bool:
    """Return True when a patch coordinate already holds these values."""
    other = values if isinstance(values, BaseCoord) else get_coord(data=values)
    if existing.units != other.units or existing.shape != other.shape:
        return False
    first, second = existing.values, other.values
    if first.dtype != second.dtype:
        return False
    equal = first == second
    if np.issubdtype(first.dtype, np.floating):
        equal |= np.isnan(first) & np.isnan(second)
    return bool(np.all(equal))


def _get_coords(inventory, context, patch, coords, on_missing) -> dict:
    """Return the coordinates to add to the patch."""
    path = context.optical_path
    if path is None:
        if coords is True:
            return {}
        msg = (
            f"No optical path is valid for {context.acquisition.code!r} at "
            "the patch's time, so no per-channel coordinates can be resolved."
        )
        raise PatchError(msg)
    blanket = coords is True
    names = _get_blanket_coord_names(inventory, path) if blanket else iterate(coords)
    if not names:
        # Nothing to project, so the patch needs no channel mapping.
        return {}
    channel_name, dim, distances = _get_channel_distances(patch, context.acquisition)
    out = {}
    for name in names:
        if name == channel_name:
            msg = (
                f"The patch's {name!r} coordinate is the one this acquisition "
                "maps onto the path, so enrich will not overwrite it with the "
                "path distance. Rename it first "
                f"(patch.rename_coords({name}='instrument_distance'))."
            )
            raise PatchError(msg)
        values = _get_coord_values(inventory, path, name, distances, distances)
        if (existing := patch.coords.coord_map.get(name)) is not None:
            if values is not None and _coords_equal(existing, values):
                continue  # re-enriching is a refresh, not a collision
            msg = (
                f"The patch already has a {name!r} coordinate which the "
                "inventory does not agree with; enrich will not overwrite "
                "it. Rename or drop it first."
            )
            raise PatchError(msg)
        if values is None:
            if blanket or on_missing == "skip":
                continue
            if on_missing == "raise":
                msg = (
                    f"The inventory defines no {name!r} for the optical path "
                    f"of {context.acquisition.code!r}; use on_missing to "
                    "allow it."
                )
                raise PatchError(msg)
            values = np.full(len(distances), np.nan)
        out[name] = (dim, values)
    return out


@patch_function()
@compose_docstring(
    conflict_desc=attr_conflict_description, on_missing_desc=on_missing_description
)
def enrich(
    patch: PatchType,
    inventory: Inventory,
    attrs: bool | tuple[str, ...] = True,
    coords: bool | tuple[str, ...] = True,
    data_source_id: str | None = None,
    time=None,
    on_missing: Literal["raise", "nan", "skip"] = "raise",
    conflicts: Literal["drop", "raise", "keep_first"] = "keep_first",
) -> PatchType:
    """
    Copy inventory metadata onto a patch.

    The patch resolves its inventory context from its ``data_source_id`` and
    its time, then the acquisition's channel map places each channel on the
    optical path so the path's tracks can be projected onto it. The patch
    keeps no reference to the inventory afterwards.

    Parameters
    ----------
    patch
        The patch to enrich.
    inventory
        The inventory to resolve against.
    attrs
        True (the default) to copy the observing-system facts the inventory
        is authoritative for, a tuple of names to copy exactly those, or
        False to copy none. The blanket form excludes `data_type`,
        `data_category`, and `data_units`, which describe the data as it now
        stands; naming one restores the as-acquired value.
    coords
        True (the default) to add the geometry axes and annotation groups of
        the resolved optical path, a tuple of names to add exactly those, or
        False to add none. Names may be `distance` for optical distance, a
        coordinate label the inventory's CRS defines, an annotation group, or
        a qualified track field such as `coupling.medium`.
    data_source_id
        The inventory identity to resolve, for a patch which does not carry
        one. Given both, the patch and this argument must agree.
    time
        The instant to resolve at, for a patch whose time axis is not
        physical. A patch with a real time coordinate resolves at its own
        time and passing this raises.
    {on_missing_desc}
    conflicts
        {conflict_desc}
        Enrichment combines the inventory's values with the patch's own, so
        the default `keep_first` lets the inventory win and re-enriching is
        a refresh. `raise` is the misresolution guard: a header disagreeing
        with the resolved acquisition usually means the `data_source_id`
        resolved to the wrong place.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.examples import inventory_patch_pair
    >>>
    >>> patch, inventory = inventory_patch_pair()
    >>> # Copy everything applicable.
    >>> enriched = patch.enrich(inventory)
    >>> # Or name what is wanted.
    >>> enriched = patch.enrich(
    ...     inventory, attrs=("gauge_length",), coords=("x", "y", "z"),
    ... )
    """
    validate_conflict(conflicts)
    _validate_on_missing(on_missing)
    source_id = _get_data_source_id(patch, data_source_id)
    times = _get_resolution_times(patch, time)
    context = _resolve_context(inventory, source_id, times)
    new_attrs = _get_attr_values(inventory, context, attrs, on_missing)
    updates, drops = _apply_conflicts(patch, new_attrs, conflicts)
    new_coords = {}
    if coords is not False and coords is not None:
        new_coords = _get_coords(inventory, context, patch, coords, on_missing)
    out = patch
    if drops:
        out = out.new(attrs=dc.PatchAttrs.from_dict(dict(out.attrs)).drop(*drops))
    if updates:
        out = out.update_attrs(**updates)
    if new_coords:
        out = out.update_coords(**new_coords)
    return out
