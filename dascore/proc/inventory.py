"""Copy metadata from a DASDAE inventory onto a patch."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import (
    INVENTORY_ATTRS,
    PatchType,
    enrich_attrs_description,
    enrich_conflicts_description,
    enrich_coords_description,
    enrich_on_missing_description,
)
from dascore.core.coords import BaseCoord, get_coord
from dascore.core.inventory import (
    DISTANCE_MAP_AXES,
    TRACK_IDENTITY_FIELDS,
    VALID_COORDINATE_LABELS,
    Interrogator,
    Inventory,
    ResolvedContext,
    _annotation_kind,
    interval_masks,
)
from dascore.exceptions import (
    InvalidInventoryError,
    InvalidSpoolQueryError,
    ParameterError,
    PatchError,
    UnresolvedPatchError,
)
from dascore.proc.coords import update_coords
from dascore.units import get_quantity_str
from dascore.utils.attrs import validate_conflict
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import iterate, validate_acquisition_key
from dascore.utils.models import values_equal
from dascore.utils.patch import patch_function
from dascore.utils.time import to_datetime64

# Attrs describing the data as it now stands rather than the system which
# recorded it. Processing functions maintain them, so blanket enrichment
# leaves them alone; naming one explicitly restores the as-acquired value.
_DATA_STATE_ATTRS = ("data_type", "data_category", "data_units")

# Facts the patch's own coordinates already state: the time coordinate has
# the sample rate and the distance coordinate the channel spacing, and
# decimating changes both. Nothing should be redundant between coords and
# attrs, so a blanket request leaves these alone; naming one restores the
# as-acquired value.
_COORD_REDUNDANT_ATTRS = ("sample_rate", "spatial_interval")

OnMissing = Literal["raise", "null", "ignore"]
_VALID_ON_MISSING = get_args(OnMissing)

# Tracks whose fields enrich can project. The inventory names them, so a
# track enrich can project and one selection can ask about are the same set.
_TRACK_NAMES = tuple(TRACK_IDENTITY_FIELDS)

# Track fields whose units the inventory documents.
_TRACK_FIELD_UNITS = {
    "coupling.depth": "m",
    "optical_components.optical_length": "m",
}


def _get_acquisition_key(patch, acquisition_key) -> str:
    """Return the id to resolve, requiring the patch and caller to agree."""
    patch_id = patch.attrs.acquisition_key
    if acquisition_key and patch_id and acquisition_key != patch_id:
        msg = (
            f"The patch's acquisition_key {patch_id!r} and the requested "
            f"{acquisition_key!r} disagree; enrich resolves one data source."
        )
        raise PatchError(msg)
    out = acquisition_key or patch_id
    if not out:
        msg = (
            "The patch has no acquisition_key, so it names no inventory entry. "
            "Set one on the patch or pass acquisition_key to enrich."
        )
        raise UnresolvedPatchError(msg)
    # A patch's own id is validated when its attrs are built, so only the
    # explicit argument can be malformed. That is the caller getting it
    # wrong rather than the inventory not describing the patch, and must
    # not be something on_unresolved can wave through.
    return validate_acquisition_key(out)


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
    # An id the inventory cannot resolve to exactly one entry says this
    # inventory does not describe this patch, which a spool may have a
    # policy about; being described twice, below, is not the same thing.
    try:
        context = inventory.resolve(source_id, time=first)
        other = context if first == last else inventory.resolve(source_id, time=last)
    except InvalidInventoryError as error:
        raise UnresolvedPatchError(str(error)) from error
    if first != last:
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


def _attr_owner(context, interrogator, name):
    """Return the object a possibly-dotted attr name belongs to, and its field."""
    prefix, _, field = name.rpartition(".")
    owner = interrogator if prefix == "interrogator" else context.acquisition
    return owner, field


def _epoch_bounds(inventory, acquisition_key: Sequence[str]) -> np.ndarray:
    """
    Return the instants at which resolving one key can change its answer.

    Every epoch bound along the key's branch, so two times falling between
    the same pair of them resolve identically at every level and one
    resolution serves both.
    """
    net_code, array_code, location, acq_code = acquisition_key
    times = []
    for network in inventory.networks:
        if network.code != net_code:
            continue
        times += [network.start_time, network.end_time]
        for array in network.fiber_arrays:
            if array.code != array_code:
                continue
            times += [array.start_time, array.end_time]
            times += [
                x
                for acq in array.acquisitions
                if acq.code == acq_code and acq.location_code == location
                for x in (acq.start_time, acq.end_time)
            ]
            times += [
                x
                for path in array.optical_paths
                if path.location_code == location
                for x in (path.start_time, path.end_time)
            ]
    stamps = [x for x in times if not np.isnat(x)]
    return np.unique(np.array(stamps, dtype="datetime64[ns]"))


def resolve_contexts(inventory, keys, starts, ends) -> np.ndarray:
    """
    Resolve index rows to the one inventory context each has, or to None.

    Each row is the half-open span of one patch. A row the inventory does
    not describe resolves to None, and so does one whose span crosses a
    change of acquisition or optical path — the inventory describes that
    row twice rather than not at all, and `conform_to_inventory` exists
    to subdivide it. A row crossing an epoch bound which changes neither
    still has one context, and gets it.

    Parameters
    ----------
    inventory
        The inventory to resolve against.
    keys
        Each row's acquisition_key.
    starts, ends
        Each row's first and last instant.

    Returns
    -------
    An object array of `ResolvedContext` or None, one per row.
    """
    epochs = resolve_row_epochs(inventory, keys, starts, ends)
    out = np.full(len(epochs), None, dtype=object)
    for row, epoch in enumerate(epochs):
        if epoch.described and not epoch.cuts and epoch.conflict is None:
            out[row] = epoch.context
    return out


class RowEpochs(NamedTuple):
    """
    How one index row sits against the epochs of its acquisition_key.

    Attributes
    ----------
    cuts
        The instants inside the row at which its optical path changes;
        empty for a row which stays within one epoch of everything.
    conflict
        The instant at which the row's *acquisition* changes, or None.
        Subdividing cannot rescue this: the pieces would describe one
        patch recorded under two configurations, which is a file that
        should not exist rather than one to reconcile.
    context
        What the row resolves to where it begins, or None where the
        inventory does not describe the row over its whole span. A row
        with no cuts and no conflict resolves to this throughout.
    """

    cuts: tuple
    conflict: Any
    context: ResolvedContext | None

    @property
    def described(self) -> bool:
        """Whether the inventory resolves this row, over its whole span."""
        return self.context is not None


# What a row whose key names no entry at all knows about its epochs.
_NO_EPOCHS = RowEpochs((), None, None)


def resolve_row_epochs(inventory, keys, starts, ends) -> list[RowEpochs]:
    """
    Report how each index row sits against the inventory's epochs.

    Where [`resolve_contexts`](`dascore.proc.inventory.resolve_contexts`)
    asks for the one context a row has and gives up on a row with two,
    this reports *where* the row's answers change, so a caller can split
    it into pieces which each have one. A row the inventory does not
    describe over its whole span is reported undescribed rather than
    partly described: the piece it does describe is not what the caller
    asked about, and keeping the edge simple is worth more than
    salvaging it.

    Parameters
    ----------
    inventory
        The inventory to resolve against.
    keys
        Each row's acquisition_key.
    starts, ends
        Each row's first and last instant — the instants themselves, so
        a row whose last instant falls exactly on a bound reaches into
        the epoch that bound opens.

    Returns
    -------
    A list of `RowEpochs`, one per row.
    """
    frame = pd.DataFrame(
        {
            "key": np.asarray(keys, dtype=object),
            "start": np.asarray(starts, dtype="datetime64[ns]"),
            "end": np.asarray(ends, dtype="datetime64[ns]"),
        }
    )
    out = [_NO_EPOCHS] * len(frame)
    for key, sub in frame.groupby("key", sort=False):
        # The empty string is how a patch spells "no identity"; it names
        # no entry, which is not the same as naming a missing one. A
        # malformed key names none either, and resolve would say so.
        codes = key.split(".") if isinstance(key, str) else []
        if len(codes) != 4:
            continue
        bounds = _epoch_bounds(inventory, codes)
        first, last = sub["start"].to_numpy(), sub["end"].to_numpy()
        # Epoch i runs from bounds[i - 1] up to bounds[i], so a row spans
        # the epochs from its start's index through its end's, and every
        # epoch after the first opens at the bound which begins it — an
        # instant which resolves that epoch for every row reaching it.
        starts_at = np.searchsorted(bounds, first, side="right")
        ends_at = np.searchsorted(bounds, last, side="right")
        # A row with no instant of its own says nothing about which epoch
        # applies, and resolving at NaT holds every epoch effective; that
        # is the whole inventory answering rather than one entry. A row
        # ending before it starts spans no epoch at all, which is the
        # same nothing to resolve against, reached by a different route.
        undated = np.isnat(first) | np.isnat(last) | (last < first)
        contexts: dict[int, ResolvedContext | None] = {}
        for position, (lo, hi) in enumerate(zip(starts_at, ends_at, strict=True)):
            if undated[position]:
                continue
            for epoch in range(int(lo), int(hi) + 1):
                if epoch not in contexts:
                    when = bounds[epoch - 1] if epoch else first[position]
                    contexts[epoch] = _try_resolve(inventory, key, when)
        for row, lo, hi, bad in zip(
            sub.index, starts_at, ends_at, undated, strict=True
        ):
            if bad:
                continue
            resolved = [contexts[epoch] for epoch in range(int(lo), int(hi) + 1)]
            if any(x is None for x in resolved):
                continue
            out[row] = _epoch_changes(resolved, bounds[int(lo) : int(hi)])
    return out


def _try_resolve(inventory, key, when) -> ResolvedContext | None:
    """Resolve one instant, or None where the inventory has no one entry."""
    try:
        return inventory.resolve(key, time=when)
    except InvalidInventoryError:
        return None


def _same(first, second) -> bool:
    """
    Return whether two resolutions say the same thing.

    Identity first, because one inventory hands out the same object for
    the same epoch and that costs nothing to check; only where the
    objects differ is it worth dumping them to compare by value. What
    matters to a patch is what the entry *says*, so an entity
    re-registered unchanged across a bound is not a change — the same
    call `_resolve_context` makes for a single patch.
    """
    return first is second or first == second


def _epoch_changes(resolved: list, boundaries) -> RowEpochs:
    """
    Reduce a row's consecutive contexts to what changes between them.

    Only the acquisition and the optical path are compared, exactly as
    `_resolve_context` compares them: the network and fiber array a
    patch hangs from say nothing about it that its acquisition does not,
    and a bound the answers do not change across is not a boundary this
    row crosses at all.
    """
    cuts = []
    # One boundary between each consecutive pair, so the three walk in
    # step -- `resolved[1:]` alone would leave the first sequence longer.
    for previous, current, boundary in zip(
        resolved[:-1], resolved[1:], boundaries, strict=True
    ):
        if not _same(previous.acquisition, current.acquisition):
            return RowEpochs(tuple(cuts), boundary, resolved[0])
        if not _same(previous.optical_path, current.optical_path):
            cuts.append(boundary)
    return RowEpochs(tuple(cuts), None, resolved[0])


def get_attr_values(inventory, contexts, name: str) -> list:
    """
    Return the value each resolved context states for one attr name.

    A context which does not state the name, and a row with no context at
    all, both give None: the inventory has no answer either way. Values
    are spelled the way a patch carrying the same fact spells them, so a
    selector matches an inventory answer and a stated header alike.
    """
    cache: dict[int, Any] = {}
    out = []
    for context in contexts:
        if context is None:
            out.append(None)
            continue
        if (value := cache.get(id(context), _UNCACHED)) is _UNCACHED:
            interrogator = _get_interrogator(inventory, context.acquisition)
            owner, field = _attr_owner(context, interrogator, name)
            value = getattr(owner, field, None)
            value = None if _is_unset(value) else _as_patch_value(value)
            cache[id(context)] = value
        out.append(value)
    return out


def _as_patch_value(value):
    """
    Return an inventory value in the spelling a patch attr would hold.

    The inventory models units as quantities and a patch carries the
    string it renders to, so comparing the two directly would never
    match — the same fact stored two ways.
    """
    if hasattr(value, "units") and hasattr(value, "dimensionality"):
        return get_quantity_str(value)
    return value


# A cache miss, told apart from a cached None (the inventory saying nothing).
_UNCACHED = object()


def _get_system_attrs(inventory, context) -> dict:
    """Return the observing-system facts the resolved context defines."""
    interrogator = _get_interrogator(inventory, context.acquisition)
    out = {}
    for name in INVENTORY_ATTRS:
        owner, field = _attr_owner(context, interrogator, name)
        value = getattr(owner, field, None)
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
        return {i: v for i, v in system.items() if i not in _COORD_REDUNDANT_ATTRS}
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
        elif on_missing == "null":
            out[name] = _missing_marker(context, name)
    return out


def _missing_marker(context, name):
    """
    Return the missing marker matching the field's type.

    NaN is the marker a number can carry; nothing else has one, so None
    stands for "the inventory does not say".
    """
    owner, field = _attr_owner(context, Interrogator, name)
    model = owner if isinstance(owner, type) else type(owner)
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
        if _is_unset(old) or values_equal(old, value):
            updates[name] = value
        elif conflicts == "raise":
            msg = (
                f"The patch's {name!r} is {old!r} but the inventory says "
                f"{value!r}. A disagreement here usually means the "
                "acquisition_key resolved to the wrong place."
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
# A map axis with no patch coordinate could never be read.
assert set(_AXIS_COORDS) == set(DISTANCE_MAP_AXES)


def _map_axis_coords(dist_map, names) -> list[tuple[str, str]]:
    """
    Return each map axis paired with the name it can be read on.

    One name per axis, the first of them present: the map states its
    control points in whichever coordinates were measured, so what is
    carried decides which is read. Guessing instead would be wrong by the
    channel spacing, silently. Stated once because a patch and an index
    row must agree about it — selection answering differently than
    enrichment would is the failure this whole path exists to avoid. No
    two axes share a name in `_AXIS_COORDS`, so a name appears once.
    """
    out = []
    for axis in dist_map.axes:
        for name in _AXIS_COORDS[axis]:
            if name in names:
                out.append((axis, name))
                break
    return out


def _readable_on(dist_map) -> list[str]:
    """The names a map could be read on, whichever axis answers."""
    return sorted({x for axis in dist_map.axes for x in _AXIS_COORDS[axis]})


def _get_channel_axes(patch, acquisition) -> list[tuple[str, str]]:
    """Return every map axis the patch can be read on, with its coordinate."""
    dist_map = acquisition.distance_map
    if dist_map is None:
        msg = (
            f"Acquisition {acquisition.code!r} defines no distance_map, so "
            "its channels cannot be placed on the optical path."
        )
        raise PatchError(msg)
    if out := _map_axis_coords(dist_map, patch.coords.coord_map):
        return out
    msg = (
        f"Acquisition {acquisition.code!r} maps {list(dist_map.axes)} onto "
        f"path distance, so it needs one of the {_readable_on(dist_map)} "
        f"coordinates, and this patch has {sorted(patch.coords.coord_map)}. "
        "An acquisition whose patches carry interrogator meters is "
        "calibrated with a distance_map on the instrument_distance axis, "
        "one control point being enough to state an origin."
    )
    raise PatchError(msg)


def _get_channel_distances(patch, acquisition) -> tuple[str, str, np.ndarray]:
    """
    Return the channel coord name, its dimension, and optical distances.

    A patch which carries more than one of the map's coordinates must be
    consistent with the map about all of them. Picking one and moving on
    would answer a question the patch itself contradicts.
    """
    resolved, failures = [], []
    for axis, name in _get_channel_axes(patch, acquisition):
        dims = patch.coords.dim_map[name]
        if len(dims) != 1:
            msg = f"The {name!r} coordinate must belong to exactly one dimension."
            raise PatchError(msg)
        values = patch.coords.coord_map[name].values
        try:
            distances = acquisition.channel_to_distance(values, axis=axis)
        except InvalidInventoryError as error:
            # An axis the map cannot be read on (a channel axis with no
            # spacing, say) leaves the others to answer.
            failures.append(f"{name!r}: {error}")
            continue
        resolved.append((name, dims[0], distances))
    if not resolved:
        joined = "; ".join(failures)
        msg = (
            "None of the patch's coordinates could be placed on the path by "
            f"{acquisition.code!r}: {joined}"
        )
        raise PatchError(msg)
    first = resolved[0]
    for name, dim, distances in resolved[1:]:
        # Sharing a dimension is what makes the comparison below meaningful:
        # two axes on different dimensions describe different things, and
        # projecting the tracks onto either would be a guess. It also makes
        # the two arrays the same length, since each is its dimension's.
        if dim != first[1]:
            msg = (
                f"The patch's {first[0]!r} and {name!r} coordinates belong to "
                f"different dimensions ({first[1]!r} and {dim!r}), so which "
                f"one is the channel axis of {acquisition.code!r} is ambiguous."
            )
            raise PatchError(msg)
        if not np.allclose(
            first[2], distances, rtol=0, atol=_DISTANCE_TOLERANCE, equal_nan=True
        ):
            offset = np.nanmax(np.abs(first[2] - distances))
            msg = (
                f"The patch's {first[0]!r} and {name!r} coordinates place its "
                f"channels up to {offset} m apart on the path. The patch and "
                f"the map of {acquisition.code!r} disagree; drop the "
                "coordinate which does not belong to this acquisition."
            )
            raise PatchError(msg)
    return first


# Two resolutions of the same channel should differ only by float noise,
# which is absolute: a relative tolerance would widen to a whole channel
# tens of kilometers down the fiber, where a disagreement matters most.
_DISTANCE_TOLERANCE = 1e-6


def _fill_from_intervals(distances, intervals, values, kind):
    """
    Return per-distance values of an interval track.

    Coverage follows `interval_masks`: half-open, with the end of each
    coverage run included so the last channel of a run is not silently
    uncovered, and point markers covering nothing.
    """
    fill = {"boolean": False, "numeric": np.nan}.get(kind, None)
    out = np.full(len(distances), fill, dtype=object)
    for value, covered in zip(
        values, interval_masks(distances, intervals), strict=True
    ):
        if not np.any(covered):
            continue
        # Anything with a length except a string holds more than one thing:
        # a tuple of measurements, or a mapping like the `extra_fields`
        # every track model carries. Caught here so the numeric branch
        # below cannot raise a bare TypeError out of float().
        if isinstance(value, Sized) and not isinstance(value, str):
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
    # A coordinate has to be one dtype. Leaving None in an object array beside
    # the strings makes a patch which cannot be written, chunked, or sorted,
    # so absence is the empty string here; there is no null in a str array.
    return np.asarray(["" if x is None else x for x in out], dtype=str)


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
    if all(_is_unset(x) for x in values):
        # The field is misspelled, or no interval states it, or every
        # interval leaves it at its empty default. All three mean the
        # inventory defines nothing here, and on_missing then rules --
        # rather than handing back a coordinate that is blank throughout.
        return None
    kinds = {_annotation_kind(x) for x in values if not _is_unset(x)}
    kind = kinds.pop() if len(kinds) == 1 else "string"
    filled = _fill_from_intervals(distances, intervals, values, kind)
    if units := _TRACK_FIELD_UNITS.get(f"{track}.{field}"):
        return get_coord(data=filled, units=units)
    return filled


def _get_geometry_coord(inventory, path, label, distances):
    """Return one coordinate axis of the path geometry, with its units."""
    crs = inventory.coordinate_reference_system
    # A label this CRS does not define is a name the inventory has no answer
    # for, which is on_missing's business rather than an error of its own --
    # a named annotation group the inventory lacks already behaves that way.
    try:
        index = crs.axis_index(label)
    except InvalidInventoryError:
        return None
    if not path.geometry:
        return None
    coords = path.coordinates_at(distances)
    if index >= coords.shape[1]:
        return None
    return get_coord(data=coords[:, index], units=crs.units[index])


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


def _get_coord_values(inventory, path, name, distances):
    """Return the values of one requested coordinate, or None if undefined."""
    if name == "distance":
        # The distances are already the optical path's, in meters.
        return get_coord(data=distances, units="m")
    track, _, field = name.partition(".")
    if track in _TRACK_NAMES:
        # A bare track name means the track's identity: which coupling
        # condition, which geometry segment, which component a channel
        # falls in. Every other field is asked for by its qualified name.
        return _get_track_coord(
            path, track, field or TRACK_IDENTITY_FIELDS[track], distances
        )
    if name in VALID_COORDINATE_LABELS:
        return _get_geometry_coord(inventory, path, name, distances)
    return _get_annotation_coord(path, name, distances)


# --- channel selection over index rows --------------------------------


def _channel_placement(dims: set[str], acquisition) -> tuple:
    """
    Return the dimension a row's channels are placed along, and its axis.

    The patch-level twin, `_get_channel_axes`, reads the map's axes off
    whichever of the patch's coordinates state them, dimensional or not.
    Here only a dimension will do: a plan trims a dimension, and what a
    non-dimensional coordinate says about the one it runs along is not in
    the index. Refusing rather than guessing is what keeps selection from
    answering differently than enrichment would.

    Returns
    -------
    A `(name, axis)` pair, or `(None, reason)` naming why there is none.
    """
    dist_map = acquisition.distance_map
    if dist_map is None:
        return None, (
            f"{acquisition.code!r} defines no distance_map, so its channels "
            "cannot be placed on the optical path"
        )
    found = _map_axis_coords(dist_map, dims)
    if not found:
        return None, (
            f"has dimensions {sorted(dims)}, and {acquisition.code!r} places "
            f"channels by one of {_readable_on(dist_map)}"
        )
    if len({name for _, name in found}) > 1:
        return None, (
            f"carries {sorted(name for _, name in found)} as separate "
            f"dimensions, so which of them {acquisition.code!r} places its "
            "channels by is ambiguous"
        )
    axis, name = found[0]
    return name, axis


def _undefined_mask(values) -> np.ndarray:
    """
    Return the channels an interval track states nothing about.

    `None` is how a query spells absence, and `_fill_from_intervals`
    decides how absence is stored: a string array has no null, so it is
    the empty string there, NaN in a numeric one, and a membership group
    is simply False where nothing includes it. The two must agree, which
    is why this reads as that function's mirror.
    """
    array = np.asarray(values)
    if array.dtype == bool:
        return ~array
    if np.issubdtype(array.dtype, np.number):
        return np.isnan(array)
    return array == ""


def _mask_pieces(mask: np.ndarray, grid: np.ndarray, low, high) -> list[tuple]:
    """
    Return the inclusive envelope of each run of kept channels.

    The row's own bounds are used at its ends rather than the grid's
    rebuilt ones. They are the same channel, but a float grid can land an
    ulp off, and a piece which does not compare equal to the row it
    covers would be trimmed on load instead of passing through.
    """
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    last = len(grid) - 1
    return [
        (
            low if start == 0 else grid[start],
            high if stop - 1 == last else grid[stop - 1],
        )
        for start, stop in zip(edges[::2], edges[1::2], strict=True)
    ]


def resolve_channel_pieces(
    inventory, contexts, frame, query, *, complement: bool = False
) -> tuple:
    """
    Return the channel dimension, each row's kept pieces, and refusals.

    The pieces are what a selection along the fiber keeps: one per
    contiguous run of matching channels, so a query a path answers in two
    places subdivides the row into two. A row with no context keeps
    nothing and says nothing about it — an inventory-backed selector
    silently does not match a patch the inventory is silent about, just
    as a patch lacking an attr is not selected on it.

    Every value is projected onto the channels the way `Patch.enrich`
    projects it and judged by the predicate the index applies to a stated
    attr, so a selector cannot mean one thing here and another in either.

    Parameters
    ----------
    inventory
        The inventory to resolve against.
    contexts
        Each row's resolved context, or None where it has none.
    frame
        The relation being selected; one row per patch.
    query
        The channel-level selectors, by inventory name.
    complement
        Keep the channels the query does *not* match. The mask runs along
        one dimension, so unlike a patch's rectangle its complement is
        exactly expressible; a row with no context keeps everything,
        being a row the selection never held.

    Returns
    -------
    A `(name, pieces, reasons)` triple. `reasons` holds a refusal per row
    and None elsewhere; when any row is refused `pieces` is None, since
    the caller raises rather than selecting, and `name` is whatever the
    rows which placed fine agreed on — possibly None.
    """
    name, placements, reasons = channel_placements(contexts, frame)
    if any(x is not None for x in reasons) or name is None:
        # Nothing was judged -- either a row must be refused, or no row
        # has a fiber at all -- so there are no pieces to report; the
        # caller reads which of the two off `reasons` and `name`.
        return name, None, reasons
    out, unusable = [], []
    for row in _placed_rows(contexts, placements, frame, name):
        unusable.append(row.reason)
        if row.grid is None:
            # A row the inventory is silent about is not selected, and so
            # its complement keeps it whole; one it cannot place is refused
            # above rather than answered for.
            out.append([(row.low, row.high)] if complement and not row.reason else [])
            continue
        mask = np.ones(len(row.grid), dtype=bool)
        for wanted, selector in query.items():
            mask &= _channel_matches(
                inventory, row.context.optical_path, wanted, selector, row.distances
            )
        out.append(_mask_pieces(~mask if complement else mask, *row.grid_bounds))
    return name, out, [x or y for x, y in zip(reasons, unusable, strict=True)]


def channel_placements(contexts, frame) -> tuple:
    """
    Return the one dimension a relation's channels run along, and per row
    how each is placed on it, or why it cannot be.
    """
    placements, reasons = [], []
    for context, dims in zip(contexts, frame["dims"], strict=True):
        placement = (None, None)
        if context is not None:
            placement = _channel_placement(set(dims.split(",")), context.acquisition)
        placements.append(placement)
        # The second half of a placement is the axis when there is one and
        # the reason there is not, so only a nameless one carries a reason.
        placed = context is not None and not placement[0]
        reasons.append(placement[1] if placed else None)
    named = {x for x, _ in placements if x is not None}
    if len(named) > 1:
        joined = ", ".join(sorted(named))
        msg = (
            "The patches of this spool place their channels along different "
            f"dimensions ({joined}), so an operation along the fiber has no "
            "one dimension to work on. Select them apart first."
        )
        raise InvalidSpoolQueryError(msg)
    return next(iter(named), None), placements, reasons


class PlacedRow(NamedTuple):
    """One index row's channels, placed on its optical path."""

    context: Any
    low: Any
    high: Any
    grid: np.ndarray | None
    distances: np.ndarray | None
    reason: str | None

    @property
    def grid_bounds(self) -> tuple:
        """The arguments `_mask_pieces` reads a mask against."""
        return self.grid, self.low, self.high


def _placed_rows(contexts, placements, frame, name: str):
    """
    Yield each row's channel grid and where it lands on the optical path.

    A row with no context, or no usable step to rebuild its grid with,
    yields no grid; the two are different answers, so only the second
    carries a reason to refuse it with.
    """
    steps = frame[f"{name}_step"].to_numpy()
    bounds = zip(frame[f"{name}_min"], frame[f"{name}_max"], strict=True)
    # Keyed by identity because that is what is cheap: sibling epochs
    # resolve to one shared context object, and `__eq__` on an inventory
    # model dumps the whole subtree. A miss only recomputes.
    cache: dict[tuple, tuple] = {}
    for context, (dim, axis), (low, high), step in zip(
        contexts, placements, bounds, steps, strict=True
    ):
        if context is None or dim is None:
            yield PlacedRow(context, low, high, None, None, None)
            continue
        if pd.isnull(step) or not step:
            # Which channels are which is decided on the sample grid, and
            # the step is its only description. Guessing would trim the
            # wrong channels silently, which is worse than saying so.
            reason = "states no channel spacing to place its channels on"
            yield PlacedRow(context, low, high, None, None, reason)
            continue
        # Envelopes are value-ordered whatever the coordinate's orientation,
        # so the grid is walked by the step's magnitude -- a reverse-sorted
        # patch states a negative one, and counting samples with it would
        # give none at all.
        step = abs(step)
        key = (id(context), axis, low, high, step)
        if (placed := cache.get(key)) is None:
            grid = np.arange(round((high - low) / step) + 1) * step + low
            placed = (grid, context.acquisition.channel_to_distance(grid, axis=axis))
            cache[key] = placed
        grid, distances = placed
        yield PlacedRow(context, low, high, grid, distances, None)


def resolve_split_pieces(inventory, contexts, frame, name, keep) -> tuple:
    """
    Return the channel dimension, each row's pieces by value, and refusals.

    Splitting expands the spool into one patch per value a group takes
    along each row, so a row is answered with a list of `(value, piece)`
    pairs rather than pieces alone. A value the group states in two
    places gives that value two pieces, so a row's pieces are disjoint
    but neither contiguous nor in envelope order — they arrive grouped by
    value, and the values are sorted rather than the envelopes.

    Parameters
    ----------
    inventory
        The inventory to resolve against.
    contexts
        Each row's resolved context, or None where it has none.
    frame
        The relation being split; one row per patch.
    name
        The inventory-derived coordinate to split on.
    keep
        Decides which values to emit, by the value itself.

    Returns
    -------
    A `(dim, rows, reasons)` triple, `rows` holding the `(value, piece)`
    pairs of each row in value order.
    """
    dim, placements, reasons = channel_placements(contexts, frame)
    if any(x is not None for x in reasons) or dim is None:
        return dim, None, reasons
    out, unusable = [], []
    for row in _placed_rows(contexts, placements, frame, dim):
        unusable.append(row.reason)
        out.append([])
        if row.grid is None:
            continue
        path = row.context.optical_path
        values, _ = _channel_values(inventory, path, name, row.distances)
        if values is None:
            # A group this path defines nowhere puts none of its channels
            # anywhere, so the row contributes no output at all.
            continue
        for value in _split_values(values):
            if not keep(value):
                continue
            pieces = _mask_pieces(np.asarray(values) == value, *row.grid_bounds)
            out[-1].extend((value, piece) for piece in pieces)
    return dim, out, [x or y for x, y in zip(reasons, unusable, strict=True)]


def _split_values(values) -> list:
    """
    Return the distinct values a group takes, in a stable order.

    Every kind splits, one output per value: a categorical group by its
    strings, a membership group into the channels it includes and those
    it does not, and a numeric one by each distinct measurement. Sorted
    so the outputs of a spool do not depend on which channel came first.

    Absence is not a value, so the channels a group says nothing about
    make no output of their own — with the one exception a membership
    group is: `False` there means "not in this group", which is a
    statement about every channel rather than the absence of one.
    """
    array = np.asarray(values)
    if array.dtype == bool:
        return sorted(set(array.tolist()))
    return sorted(set(array[~_undefined_mask(array)].tolist()))


def _channel_values(inventory, path, name, distances) -> tuple:
    """
    Return one name's value per channel, with the units it carries.

    A path is what states anything along the fiber, so an acquisition
    without one -- a valid inventory, describing a system which simply
    projects nothing -- has no values, exactly as a path which defines
    the name nowhere has none.
    """
    values = (
        None if path is None else _get_coord_values(inventory, path, name, distances)
    )
    if values is None:
        return None, None
    if not isinstance(values, BaseCoord):
        return values, None
    # The projection carries the units the inventory documents for the
    # field, and dropping them would refuse the unit-bearing selectors
    # the index accepts against a stated attr of the same kind.
    units = None if values.units is None else {"num": get_quantity_str(values.units)}
    return values.values, units


def _channel_matches(inventory, path, name, selector, distances) -> np.ndarray:
    """Return the channels one selector matches, as the index would."""
    from dascore.io.index.query import evaluate_attr_predicate  # noqa: PLC0415

    values, units = _channel_values(inventory, path, name, distances)
    if values is None:
        # A name this path defines nowhere states nothing about any of its
        # channels, so it matches none of them -- listing a name is not
        # promising a value for it.
        return np.zeros(len(distances), dtype=bool)
    if selector is None:
        # `None` is the query spelling of the undefined marker, which is a
        # value here rather than the "select everything" a bare None means
        # of an attr: a channel the track says nothing about is a channel.
        return _undefined_mask(values)
    return evaluate_attr_predicate(list(values), name, selector, units)


def _coords_equal(existing, values) -> bool:
    """Return True when a patch coordinate already holds these values."""
    other = values if isinstance(values, BaseCoord) else get_coord(data=values)
    if existing.units != other.units or existing.shape != other.shape:
        return False
    first, second = existing.values, other.values
    # Kind, not the exact dtype: a string array's width is fixed by the
    # longest value it happens to hold, so a patch sliced down to the short
    # values keeps the wider dtype and would otherwise never match a fresh
    # projection of itself -- re-enriching a selection would stop being a
    # refresh and start raising.
    if first.dtype.kind != second.dtype.kind:
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
        values = _get_coord_values(inventory, path, name, distances)
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
            if blanket or on_missing == "ignore":
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
    attrs_desc=enrich_attrs_description,
    coords_desc=enrich_coords_description,
    on_missing_desc=enrich_on_missing_description,
    conflicts_desc=enrich_conflicts_description,
)
def enrich(
    patch: PatchType,
    inventory: Inventory,
    attrs: bool | tuple[str, ...] = True,
    coords: bool | tuple[str, ...] = True,
    acquisition_key: str | None = None,
    time=None,
    on_missing: OnMissing = "raise",
    conflicts: Literal["drop", "raise", "keep_first"] = "keep_first",
) -> PatchType:
    """
    Copy inventory metadata onto a patch.

    The patch resolves its inventory context from its ``acquisition_key`` and
    its time, then the acquisition's channel map places each channel on the
    optical path so the path's tracks can be projected onto it. The patch
    keeps no reference to the inventory afterwards.

    Parameters
    ----------
    patch
        The patch to enrich.
    inventory
        The inventory to resolve against.
    {attrs_desc}
    {coords_desc}
    acquisition_key
        The inventory identity to resolve, for a patch which does not carry
        one. Given both, the patch and this argument must agree.
    time
        The instant to resolve at, for a patch whose time axis is not
        physical. A patch with a real time coordinate resolves at its own
        time and passing this raises.
    {on_missing_desc}
    {conflicts_desc}

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
    if on_missing not in _VALID_ON_MISSING:
        msg = f"on_missing must be one of {_VALID_ON_MISSING}, got {on_missing!r}."
        raise ParameterError(msg)
    source_id = _get_acquisition_key(patch, acquisition_key)
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
        # The raw function: enrich is the operation worth recording, and the
        # nested entry would paste a rendered repr of every added coordinate
        # into the history of every patch enriched.
        out = update_coords.raw_function(out, **new_coords)
    return out
