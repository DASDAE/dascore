"""Copy metadata from a DASDAE inventory onto a patch."""

from __future__ import annotations

from typing import get_args

import numpy as np

import dascore as dc
from dascore.constants import (
    ENRICH_CONFLICT,
    INVENTORY_ATTRS,
    ON_MISSING,
    PatchType,
    enrich_attrs_description,
    enrich_conflict_description,
    enrich_coords_description,
    enrich_on_missing_description,
)
from dascore.core._spool_inventory import (
    COORD_REDUNDANT_ATTRS,
    DATA_STATE_ATTRS,
    VALID_ON_MISSING,
    attr_owner,
    get_coord_values,
    get_interrogator,
    is_unset,
    map_axis_coords,
    readable_on,
    to_axis_units,
    validate_enrich_conflict,
    validate_enrich_selection,
)
from dascore.core.coords import BaseCoord, get_coord
from dascore.core.inventory import (
    Interrogator,
    Inventory,
    ResolvedContext,
    axis_columns,
)
from dascore.exceptions import (
    InvalidInventoryError,
    ParameterError,
    PatchError,
    UnresolvedPatchError,
)
from dascore.models import values_equal
from dascore.proc.coords import update_coords
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import iterate, validate_acquisition_key, warn_or_raise
from dascore.utils.patch import patch_function
from dascore.utils.time import to_datetime64


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


def _get_system_attrs(inventory, context) -> dict:
    """Return the observing-system facts the resolved context defines."""
    interrogator = get_interrogator(inventory, context.acquisition)
    out = {}
    for name in INVENTORY_ATTRS:
        owner, field = attr_owner(context, interrogator, name)
        value = getattr(owner, field, None)
        if is_unset(value):
            continue
        out[name] = value
    return out


def _get_attr_values(inventory, context, attrs, on_missing) -> dict:
    """Return the attrs to copy, honoring on_missing for named requests."""
    if attrs is False:
        return {}
    system = _get_system_attrs(inventory, context)
    if attrs is True:
        return {i: v for i, v in system.items() if i not in COORD_REDUNDANT_ATTRS}
    available = dict(system)
    for name in DATA_STATE_ATTRS:
        value = getattr(context.acquisition, name)
        if not is_unset(value):
            available[name] = value
    out = {}
    for name in iterate(attrs):
        if name in available:
            out[name] = available[name]
        elif on_missing == "null":
            out[name] = _missing_marker(context, name)
        else:
            _report_missing(context, name, on_missing)
    return out


def _report_missing(context, name, on_missing, subject: str = "") -> None:
    """Raise or warn about a requested name the inventory does not define."""
    msg = (
        f"The inventory defines no {name!r} for {subject}"
        f"{context.acquisition.code!r}; use on_missing to allow it."
    )
    warn_or_raise(msg, PatchError, behavior=on_missing)


def _missing_marker(context, name):
    """
    Return the missing marker matching the field's type.

    NaN is the marker a number can carry; nothing else has one, so None
    stands for "the inventory does not say".
    """
    owner, field = attr_owner(context, Interrogator, name)
    model = owner if isinstance(owner, type) else type(owner)
    info = model.model_fields.get(field)
    annotation = None if info is None else info.annotation
    return np.nan if _is_numeric_annotation(annotation) else None


def _is_numeric_annotation(annotation) -> bool:
    """Return True when a field annotation admits a plain number."""
    if annotation is float or annotation is int:
        return True
    return any(_is_numeric_annotation(x) for x in get_args(annotation))


def _apply_conflict(patch, new_attrs, conflict) -> tuple[dict, list]:
    """
    Return the attrs to set and to drop, given the conflict policy.

    Filling an empty attr is never a conflict, and neither are equal values;
    a conflict is both sides holding different information.
    """
    current = dict(patch.attrs)
    updates, drops = {}, []
    for name, value in new_attrs.items():
        old = current.get(name, None)
        if is_unset(old) or values_equal(old, value):
            updates[name] = value
        elif conflict == "raise":
            msg = (
                f"The patch's {name!r} is {old!r} but the inventory says "
                f"{value!r}. A disagreement here usually means the "
                "acquisition_key resolved to the wrong place."
            )
            raise PatchError(msg)
        elif conflict == "drop":
            drops.append(name)
        elif conflict == "keep_last":
            # The inventory is asked to correct the header rather than to
            # agree with it, so its value is the one which stands.
            updates[name] = value
        # keep_first: the patch stated it first, so the patch keeps it.
    return updates, drops


def _get_channel_axes(patch, acquisition) -> list[tuple[str, str]]:
    """Return every map axis the patch can be read on, with its coordinate."""
    dist_map = acquisition.distance_map
    if dist_map is None:
        msg = (
            f"Acquisition {acquisition.code!r} defines no distance_map, so "
            "its channels cannot be placed on the optical path."
        )
        raise PatchError(msg)
    if out := map_axis_coords(dist_map, patch.coords.coord_map):
        return out
    msg = (
        f"Acquisition {acquisition.code!r} maps {list(dist_map.axes)} onto "
        f"path distance, so it needs one of the {readable_on(dist_map)} "
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
        coord = patch.coords.coord_map[name]
        values = to_axis_units(coord.values, coord.units, axis, name)
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


def _get_blanket_coord_names(inventory, path) -> list[str]:
    """
    Return the coordinate names a blanket request copies.

    The geometry columns, the label groups, and how each channel is
    coupled: what the path says about each channel. Optical distance and
    the other typed tracks are asked for by name, since they restate what
    the patch's own axis and the inventory already record; a coupling
    condition states something about the fiber's surroundings which
    nothing else does, and is the first thing a deployment comparing
    grouted with hanging fiber has to select on.
    """
    crs = inventory.coordinate_reference_system
    axis_names = crs.coordinate_labels
    # The axes are copied under their canonical names, and only where some
    # segment actually places the fiber; the rest come under their own.
    axes = {x for segment in path.geometry for x in axis_columns(segment, crs)}
    out = ["x", "y", "z"][: len(axis_names)] if axes else []
    out += [x for x in path.geometry_columns() if x not in axes]
    seen = dict.fromkeys(x.group for x in path.labels)
    out += [x for x in seen if x]
    # A path with no coupling conditions resolves to nothing here, and a
    # blanket request asks only for what the path itself states.
    return out + (["coupling"] if path.coupling else [])


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
        values = get_coord_values(inventory, path, name, distances)
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
            # A blanket request asks for the names the path itself lists,
            # so one of those without values would be the inventory
            # disagreeing with itself rather than a gap to have a policy about.
            assert not blanket, f"blanket name {name!r} resolved to nothing"
            if on_missing != "null":
                _report_missing(context, name, on_missing, "the optical path of ")
                continue
            values = np.full(len(distances), np.nan)
        out[name] = (dim, values)
    return out


@patch_function()
@compose_docstring(
    attrs_desc=enrich_attrs_description,
    coords_desc=enrich_coords_description,
    on_missing_desc=enrich_on_missing_description,
    conflict_desc=enrich_conflict_description,
)
def enrich(
    patch: PatchType,
    inventory: Inventory,
    attrs: bool | tuple[str, ...] = True,
    coords: bool | tuple[str, ...] = True,
    acquisition_key: str | None = None,
    time=None,
    on_missing: ON_MISSING = "raise",
    conflict: ENRICH_CONFLICT = "keep_first",
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
    {conflict_desc}

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
    validate_enrich_conflict(conflict)
    if on_missing not in VALID_ON_MISSING:
        msg = f"on_missing must be one of {VALID_ON_MISSING}, got {on_missing!r}."
        raise ParameterError(msg)
    validate_enrich_selection(attrs, coords)
    source_id = _get_acquisition_key(patch, acquisition_key)
    times = _get_resolution_times(patch, time)
    context = _resolve_context(inventory, source_id, times)
    new_attrs = _get_attr_values(inventory, context, attrs, on_missing)
    updates, drops = _apply_conflict(patch, new_attrs, conflict)
    new_coords = {}
    if coords is not False:
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
