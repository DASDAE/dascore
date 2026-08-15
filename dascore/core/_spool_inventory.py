"""
Spool-side inventory machinery.

Everything a spool needs to answer questions from an attached DASDAE
inventory lives here: resolving index rows to their inventory contexts,
placing channels on the optical path, selecting and splitting along the
fiber, and the reference type a spool carries an unread inventory as.
`dascore.proc.inventory` keeps the patch-level verb (`Patch.enrich`) and
imports the shared projection primitives from this module, so a selector
means one thing whether a patch or an index row answers it.

This module is internal; nothing here is public API.
"""

from __future__ import annotations

import inspect
import threading
import warnings
from collections.abc import Sequence, Sized
from itertools import pairwise
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from pydantic import ValidationError

import dascore as dc
from dascore.core.coords import BaseCoord, get_coord
from dascore.core.inventory import (
    DISTANCE_MAP_AXES,
    TRACK_IDENTITY_FIELDS,
    VALID_COORDINATE_LABELS,
    Inventory,
    ResolvedContext,
)
from dascore.core.inventory_loader import BLESSED_NAME, find_inventory
from dascore.exceptions import (
    InvalidInventoryError,
    InvalidSpoolError,
    InvalidSpoolQueryError,
    MissingOptionalDependencyError,
    ParameterError,
    PatchError,
    UnresolvedPatchError,
)
from dascore.units import get_quantity_str
from dascore.utils.intervals import interval_masks, value_kind
from dascore.utils.misc import iterate

# One vocabulary for both fiber verbs. What the quiet option leaves
# behind still differs -- enrich leaves the patch as it was, conform
# removes it -- but that is the verb's business rather than the value's.
VALID_ON_UNRESOLVED = ("raise", "warn", "ignore")

# Written once because both fiber verbs refuse for it, and two spellings
# would let them start explaining the same refusal differently.
UNPLACEABLE = (
    "are described by the inventory but cannot have their channels placed "
    "along the fiber"
)

UNRESOLVED_WARNING = (
    "The attached inventory does not describe every patch in this spool, and "
    "those it does not describe were not enriched. Use on_unresolved='raise' "
    "to see which, 'ignore' to silence this, or remove them from the spool "
    "with Spool.conform_to_inventory."
)


class InventoryRef:
    """
    An inventory a spool has been pointed at but has not read.

    Attaching states where the inventory is; the read happens at the
    first question only an inventory can answer, and then once. The
    holder is shared rather than copied, because a spool copy-constructs
    from its parent (`select`, `sort`, `chunk` each return a new one), so
    a spool sliced ten ways still reads its inventory a single time and
    two views of one parent can never disagree about what it says.

    An inventory is an input, not a cache, so it is never re-read behind
    the caller's back: a file which changes under a running program is a
    new input rather than a stale one, and re-attaching is how the
    program says to read it again -- `Spool.attach_inventory()` with no
    argument for the one a directory carries, the same path again for
    any other. A read which failed is not a read, though, and is tried
    again next time: an unreadable inventory is a thing to go and fix,
    and holding the failure would mean the fix could not be seen.
    """

    def __init__(self, path, blessed: bool = False):
        # Anchored now, while the working directory is still the one the
        # caller named it from: the read happens later, and a relative
        # path would then be resolved against wherever the program had
        # got to -- another directory, or another process entirely.
        # For a blessed reference this is the directory, not the file:
        # which of the two forms the directory carries is decided when it
        # is read, so discovery stays a stat and its complaints wait.
        self.path = path.absolute()
        self.blessed = blessed
        self._inventory: Inventory | None = None
        self._lock = threading.Lock()

    def __getstate__(self):
        """A lock cannot be pickled, and a fresh one is what a copy wants."""
        return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def resolve(self) -> Inventory:
        """Return the inventory, reading it if this is the first ask."""
        # Held across the read, not merely around the assignment: threads
        # mapping over one spool all reach this at once, and reading a
        # large authoring directory once per worker is the cost this
        # whole class exists to avoid.
        with self._lock:
            if self._inventory is None:
                self._inventory = self._read()
            return self._inventory

    def _read(self) -> Inventory:
        """Read the inventory, saying where an unreadable one came from."""
        try:
            source = self.path
            if self.blessed:
                source = find_inventory(self.path)
                if source is None:
                    msg = "nothing is there now, though there was on opening"
                    raise InvalidInventoryError(msg)
            return dc.inventory(source)
        except (
            InvalidInventoryError,
            MissingOptionalDependencyError,
            ValidationError,
            OSError,
        ) as error:
            # Surfacing from inside select() or enrich(), the failure has
            # to say which file it means and how the spool came to have
            # it -- most sharply when nobody chose it by hand.
            where = (
                f"the inventory {self.path} carries under the name "
                f"{BLESSED_NAME!r}, attached when the spool was opened"
                if self.blessed
                else f"the inventory attached to this spool from {self.path}"
            )
            msg = f"Could not read {where}: {error}"
            raise InvalidInventoryError(msg) from error

    def __eq__(self, other) -> bool:
        """
        Whether this and another attachment are the same one.

        An attachment is compared as the thing it is -- a place, or a
        value -- rather than by what reading it would produce. Comparing
        never reads, which is what keeps `==` and `+` from doing file
        I/O, from raising out of an unreadable inventory, and from
        answering differently depending on whether something happened to
        read it first. So a place equals the same place, a value equals
        an equal value, and a place is no value until someone asks.
        """
        if isinstance(other, InventoryRef):
            return (self.path, self.blessed) == (other.path, other.blessed)
        if isinstance(other, Inventory):
            return False
        return NotImplemented

    # Defined because __eq__ is: a reference is spool state, never a key.
    __hash__ = None


def _combine_state(values, label):
    """Return the one value two spools agree on, or None if neither has one."""
    present = [x for x in values if x is not None]
    if not present:
        return None
    if len(present) == 2 and present[0] != present[1]:
        msg = (
            f"The spools carry different {label}, which have no combined "
            "meaning. Attach one inventory to the combined spool instead, "
            "or drop an operand's with Spool.remove_inventory -- which is "
            "also the answer when neither was attached by hand and each "
            "directory simply carries its own."
        )
        raise InvalidSpoolError(msg)
    return present[0]


def combine_inventories(first, second) -> tuple:
    """
    Return the (inventory, enrich kwargs) a union of two spools carries.

    The two halves carry over independently: an inventory attached to one
    operand still describes the patches it came with, and so does the
    enrichment set up from it — which is why attaching the same inventory
    to the other operand cannot turn a working union into an error. Two
    operands answering either question differently have no single answer.
    """
    # At call time only; importing Spool at module top would be circular.
    from dascore.core.spool import Spool  # noqa: PLC0415

    inventory = _combine_state(
        [getattr(x, "_inventory", None) for x in (first, second)], "inventories"
    )
    enrichment = _combine_state(
        [x._enrichment() if isinstance(x, Spool) else None for x in (first, second)],
        "enrich arguments",
    )
    # Enrichment is only reachable through an inventory, so it cannot
    # outlive one; a union carrying arguments and nothing to apply them
    # from would enrich nothing while claiming to.
    assert inventory is not None or enrichment is None
    return inventory, enrichment


def validate_enrich_selection(attrs, coords) -> None:
    """Refuse None, the retired spelling of enrich's False off switch."""
    for label, value in (("attrs", attrs), ("coords", coords)):
        if value is None:
            msg = (
                f"enrich's {label} must be True, False, or a collection of "
                "names; pass False to copy none."
            )
            raise ParameterError(msg)


def normalize_enrich_kwargs(kwargs) -> dict:
    """
    Canonicalize enrich arguments, rejecting any Patch.enrich would.

    Two spools which enrich identically have to compare equal, so an
    argument stated explicitly at its own default, or given as a list
    where a tuple would do, must reach the same stored form.
    """
    signature = inspect.signature(dc.Patch.enrich)
    valid = set(signature.parameters) - {"patch", "inventory"}
    if bad := sorted(set(kwargs) - valid):
        msg = (
            f"Spool.enrich got unknown argument(s) {bad}; it passes "
            f"{sorted(valid)} through to Patch.enrich."
        )
        raise ParameterError(msg)
    bound = signature.bind_partial(**kwargs)
    bound.apply_defaults()
    # Refused on this call rather than on whichever patch is pulled first.
    validate_enrich_selection(
        bound.arguments.get("attrs", False), bound.arguments.get("coords", False)
    )
    # A collection of names means what it holds, not which container holds it.
    return {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in bound.arguments.items()
        if name in valid
    }


# --- planning-frame helpers -------------------------------------------


def resolution_columns(frame: pd.DataFrame) -> list | None:
    """
    Return the columns a relation resolves against an inventory with.

    Resolution needs an identity and the instants to resolve it at, so a
    relation missing either gets None here. A spool whose patches carry no
    `acquisition_key` has no identity to offer, and one whose time axis
    is not physical — lag times from a correlation, say — has no
    instants, which is the same thing `Patch.enrich` refuses to guess at.
    """
    columns = [frame.get(x) for x in ("acquisition_key", "time_min", "time_max")]
    physical = all(column is not None for column in columns) and all(
        np.issubdtype(column.dtype, np.datetime64) for column in columns[1:]
    )
    return columns if physical else None


def _first_few(items, limit: int = 5) -> str:
    """Name a few of the offending rows, and say how many went unnamed."""
    listed = ", ".join(str(x) for x in items[:limit])
    extra = len(items) - limit
    return listed if extra <= 0 else f"{listed} (and {extra} more)"


def report_unconformed(rows: pd.DataFrame, on_unresolved: str) -> None:
    """
    Say what conforming is about to drop, as loudly as it was asked to.

    Naming the files is the whole value of the loud policies: an
    inventory which covers an archive apart from a handful of patches is
    reporting a gap in itself as often as a stray file.
    """
    if on_unresolved == "ignore":
        return
    paths = list(rows["source_path"])
    advice = (
        "Pass on_unresolved='warn' to drop them with a warning, or "
        "'ignore' to drop them silently."
        if on_unresolved == "raise"
        else "Pass on_unresolved='ignore' to silence this, or 'raise' to "
        "fail on the gap instead."
    )
    msg = (
        f"The inventory does not describe {len(paths)} patch(es) in this "
        f"spool: {_first_few(paths)}. {advice}"
    )
    if on_unresolved == "raise":
        raise UnresolvedPatchError(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


def refuse_rows(source_rows: pd.DataFrame, reasons, summary: str) -> None:
    """
    Raise naming the patches an inventory verb cannot handle, and why.

    Every refusal these verbs make has the same shape — a few patches out
    of an archive, each with its own particular; naming them is the whole
    value, since the fix is nearly always to one file or one inventory
    entry. `reasons` holds a particular per row and None where the row is
    fine, so a caller judges rows without also formatting them.
    """
    named = [
        f"{path} ({reason})"
        for path, reason in zip(source_rows["source_path"], reasons, strict=True)
        if reason is not None
    ]
    if not named:
        return
    msg = f"{len(named)} patch(es) {summary}: {_first_few(named)}."
    raise PatchError(msg)


def acquisition_conflicts(epochs) -> list:
    """
    Name the patches whose acquisition changes partway through.

    Subdividing cannot reconcile these the way it reconciles a change of
    optical path — see `RowEpochs.conflict` for why. `on_unresolved` does
    not wave them through either: the inventory describes such a patch
    twice rather than not at all.
    """
    return [None if x.conflict is None else f"at {x.conflict}" for x in epochs]


def unsubdividable(rows: pd.DataFrame, pieces, name: str) -> list:
    """
    Name the patches which must be split but state no sampling interval.

    The pieces are found on the patch's own sample grid, which its step
    is the only description of: a piece ends one step short of where the
    next begins. Without a step both pieces would have to claim the
    boundary value, and envelopes are inclusive, so a sample landing
    there would appear in both. Duplicating a sample is a worse answer
    than saying so — the caller asked for the metadata to be reconciled,
    not for the data to be restructured.
    """
    return [
        f"at {row_pieces[0]}" if row_pieces and (pd.isnull(step) or not step) else None
        for step, row_pieces in zip(rows[f"{name}_step"], pieces, strict=True)
    ]


def _unstated(values) -> np.ndarray:
    """
    Return a mask of the entries which state no value.

    A patch says it does not know a name by leaving it null, which a
    string column spells as the empty string; both are what an attached
    inventory is asked to fill in.
    """
    series = pd.Series(np.asarray(values, dtype=object))
    return (series.isna() | series.eq("")).to_numpy()


def drops_samples(rows: pd.DataFrame, pieces, name: str) -> bool:
    """
    Return whether any row keeps less than the whole of itself.

    Read off the pieces rather than taken on trust, because getting it
    wrong is silent either way: a plan wrongly called lossless loads back
    the samples it dropped, and one wrongly called lossy only forgoes a
    collapse. Conform's pieces always cover their row, so it answers
    False here without the caller having to say so.
    """
    lows = rows[f"{name}_min"].to_numpy()
    highs = rows[f"{name}_max"].to_numpy()
    steps = rows[f"{name}_step"].to_numpy()
    for row_pieces, low, high, step in zip(pieces, lows, highs, steps, strict=True):
        if not row_pieces:
            continue  # a row kept out entirely leaves no outputs to collapse
        if row_pieces[0][0] != low or row_pieces[-1][1] != high:
            return True
        # Conform's pieces meet exactly one step apart, so they cover the
        # row between them; a selection's have the channels it dropped in
        # the gaps, which is the whole difference.
        gap = abs(step)
        if any(b[0] != a[1] + gap for a, b in pairwise(row_pieces)):
            return True
    return False


def check_stampable(name: str, rows: pd.DataFrame) -> None:
    """
    Refuse a stamp which would overwrite the plan's own bookkeeping.

    An annotation group may be named anything the inventory does not
    reserve, and the stamp is assigned onto the outputs — so a group
    called `output_id` would replace the column binding each output to
    its members, and one called `time_min` an envelope. Overwriting a
    carried attr is fine and is how re-splitting restamps; these are not
    attrs.
    """
    envelopes = {
        x for x in rows.columns if x.rsplit("_", 1)[-1] in {"min", "max", "step"}
    }
    if name not in {"output_id", "dims", *envelopes} and not name.startswith("_"):
        return
    msg = (
        f"{name!r} is how the spool itself describes a patch, so stamping "
        "it would overwrite what binds each output to the data it came "
        "from. Rename the group, or pass stamp=False to expand by it "
        "without recording the value."
    )
    raise InvalidSpoolQueryError(msg)


def stated_channels(channels: dict) -> dict:
    """
    Return the channel selectors which actually select something.

    A bare `...` selects everything here as everywhere, so it asks for no
    trimming at all — but the name still had to be recognized as the
    inventory's, or the index would be left to complain that it has never
    heard of it. `None` is not dropped beside it: on a coordinate the
    fiber defines it spells the undefined marker, which is a statement
    about which channels to keep rather than the absence of one.
    """
    return {name: value for name, value in channels.items() if value is not Ellipsis}


def glob_filter(include, exclude):
    """
    Return a predicate deciding which split values to keep.

    The patterns are matched against each value written as a string, so
    one vocabulary covers every kind a group can hold: `"hole_*"` reads a
    categorical group, and a membership group is `"True"` and `"False"`.
    Globs mean what they mean everywhere else here, which is what SQLite
    means by them rather than what `fnmatch` does.
    """
    from dascore.io.index.query import glob_to_regex  # noqa: PLC0415

    patterns = tuple(
        None if spec is None else [glob_to_regex(str(x)) for x in iterate(spec)]
        for spec in (include, exclude)
    )

    def keep(value) -> bool:
        wanted, unwanted = patterns
        text = str(value)
        if unwanted is not None and any(x.match(text) for x in unwanted):
            # Excluding wins, so naming a family and carving one out of it
            # reads in either order.
            return False
        return wanted is None or any(x.match(text) for x in wanted)

    return keep


def match_resolved(values, name: str, selector, units=None) -> np.ndarray:
    """Return which of the values an inventory states match a selector."""
    from dascore.io.index.query import evaluate_attr_predicate  # noqa: PLC0415

    values = np.asarray(values, dtype=object)
    out = np.zeros(len(values), dtype=bool)
    # A name the inventory has no answer for is not one it can select on,
    # and the predicate would be comparing against the missing marker.
    stated = ~_unstated(values)
    if stated.any():
        out[stated] = evaluate_attr_predicate(values[stated], name, selector, units)
    return out


# --- shared projection primitives -------------------------------------
# Used by `Patch.enrich` (dascore.proc.inventory) and by the row-level
# machinery below, so a selector or a projection means the same thing
# whether a patch or an index row answers it.


def get_interrogator(inventory, acquisition):
    """Return the resolved interrogator of an acquisition, if it has one."""
    value = acquisition.interrogator
    if isinstance(value, str):
        return inventory.get_resource(value)
    return value


def attr_owner(context, interrogator, name):
    """Return the object a possibly-dotted attr name belongs to, and its field."""
    prefix, _, field = name.rpartition(".")
    owner = interrogator if prefix == "interrogator" else context.acquisition
    return owner, field


def is_unset(value) -> bool:
    """
    Return True when an attr holds no information.

    NaN is how readers spell an unknown number, so a NaN placeholder is
    filled rather than treated as a value which disagrees.
    """
    if value is None or (isinstance(value, str) and not value):
        return True
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


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


# The patch coordinates each map axis can be read from, in preference
# order. A patch reframed onto the path keeps the interrogator's meters
# under the explicit name, which is why it wins over plain distance.
AXIS_COORDS = {
    "channel": ("channel",),
    "instrument_distance": ("instrument_distance", "distance"),
}
# A map axis with no patch coordinate could never be read.
assert set(AXIS_COORDS) == set(DISTANCE_MAP_AXES)


def map_axis_coords(dist_map, names) -> list[tuple[str, str]]:
    """
    Return each map axis paired with the name it can be read on.

    One name per axis, the first of them present: the map states its
    control points in whichever coordinates were measured, so what is
    carried decides which is read. Guessing instead would be wrong by the
    channel spacing, silently. Stated once because a patch and an index
    row must agree about it — selection answering differently than
    enrichment would is the failure this whole path exists to avoid. No
    two axes share a name in `AXIS_COORDS`, so a name appears once.
    """
    out = []
    for axis in dist_map.axes:
        for name in AXIS_COORDS[axis]:
            if name in names:
                out.append((axis, name))
                break
    return out


def readable_on(dist_map) -> list[str]:
    """The names a map could be read on, whichever axis answers."""
    return sorted({x for axis in dist_map.axes for x in AXIS_COORDS[axis]})


# Tracks whose fields enrich can project. The inventory names them, so a
# track enrich can project and one selection can ask about are the same set.
_TRACK_NAMES = tuple(TRACK_IDENTITY_FIELDS)

# Track fields whose units the inventory documents.
_TRACK_FIELD_UNITS = {
    "coupling.depth": "m",
    "optical_components.optical_length": "m",
}


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
    kind = value_kind(items[0].value)
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
    if all(is_unset(x) for x in values):
        # The field is misspelled, or no interval states it, or every
        # interval leaves it at its empty default. All three mean the
        # inventory defines nothing here, and on_missing then rules --
        # rather than handing back a coordinate that is blank throughout.
        return None
    kinds = {value_kind(x) for x in values if not is_unset(x)}
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


def get_coord_values(inventory, path, name, distances):
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


# --- epoch resolution over index rows ---------------------------------


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

    @property
    def settled(self) -> ResolvedContext | None:
        """The one context this row resolves to throughout, or None."""
        whole = self.described and not self.cuts and self.conflict is None
        return self.context if whole else None


# What a row whose key names no entry at all knows about its epochs.
NO_EPOCHS = RowEpochs((), None, None)


def resolve_contexts(inventory, keys, starts, ends) -> np.ndarray:
    """
    Resolve index rows to the one inventory context each has, or to None.

    Each row is the half-open span of one patch. A row the inventory does
    not describe resolves to None, and so does one whose span crosses a
    change of acquisition or optical path — the inventory describes that
    row twice rather than not at all, and `conform_to_inventory` exists
    to subdivide it. A row crossing an epoch bound which changes neither
    still has one context, and gets it.
    """
    epochs = resolve_row_epochs(inventory, keys, starts, ends)
    out = np.full(len(epochs), None, dtype=object)
    for row, epoch in enumerate(epochs):
        out[row] = epoch.settled
    return out


def resolve_row_epochs(inventory, keys, starts, ends) -> list[RowEpochs]:
    """
    Report how each index row sits against the inventory's epochs.

    Where `resolve_contexts` asks for the one context a row has and gives
    up on a row with two, this reports *where* the row's answers change,
    so a caller can split it into pieces which each have one. A row the
    inventory does not describe over its whole span is reported
    undescribed rather than partly described: the piece it does describe
    is not what the caller asked about, and keeping the edge simple is
    worth more than salvaging it.

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
    out = [NO_EPOCHS] * len(frame)
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
    call `Patch.enrich` makes for a single patch.
    """
    return first is second or first == second


def _epoch_changes(resolved: list, boundaries) -> RowEpochs:
    """
    Reduce a row's consecutive contexts to what changes between them.

    Only the acquisition and the optical path are compared, exactly as
    `Patch.enrich` compares them: the network and fiber array a patch
    hangs from say nothing about it that its acquisition does not, and a
    bound the answers do not change across is not a boundary this row
    crosses at all.
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


# A cache miss, told apart from a cached None (the inventory saying nothing).
_UNCACHED = object()


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
            interrogator = get_interrogator(inventory, context.acquisition)
            owner, field = attr_owner(context, interrogator, name)
            value = getattr(owner, field, None)
            value = None if is_unset(value) else _as_patch_value(value)
            cache[id(context)] = value
        out.append(value)
    return out


# --- channel selection over index rows --------------------------------


def _channel_placement(dims: set[str], acquisition) -> tuple:
    """
    Return the dimension a row's channels are placed along, and its axis.

    The patch-level twin in `dascore.proc.inventory` reads the map's axes
    off whichever of the patch's coordinates state them, dimensional or
    not. Here only a dimension will do: a plan trims a dimension, and
    what a non-dimensional coordinate says about the one it runs along is
    not in the index. Refusing rather than guessing is what keeps
    selection from answering differently than enrichment would.

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
    found = map_axis_coords(dist_map, dims)
    if not found:
        return None, (
            f"has dimensions {sorted(dims)}, and {acquisition.code!r} places "
            f"channels by one of {readable_on(dist_map)}"
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
    name, placements, reasons = _channel_placements(contexts, frame)
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


def _channel_placements(contexts, frame) -> tuple:
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
    dim, placements, reasons = _channel_placements(contexts, frame)
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
        None if path is None else get_coord_values(inventory, path, name, distances)
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
