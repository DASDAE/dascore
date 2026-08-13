"""
Load a DASDAE inventory from an authoring directory.

The authoring format splits an inventory along its natural grain: small
heterogeneous objects (acquisitions, interrogators, cables) live in YAML or
JSON files matching the models, while long row-shaped track data lives in
CSV files a field crew can maintain as a spreadsheet. A directory of these
files is itself a loadable inventory, and ``to_yaml`` exports the
single-file interchange artifact for shipping beside a data archive.

This module reads the object half. The track tables, and the optical path
epoch directories which hold them, are refused by name rather than
skipped, so a directory cannot load as an entity silently missing the
tracks its own files state.

The contract, in one line: **file declares object_type, container agrees, name
implies identity, envelope implies version.** Every object file states what
it is, its container checks that statement rather than supplying it, its
name decides which entity it is, and the top-level ``inventory.yaml``
versions the whole document. An address restated inside a file must agree
with the name; there is never a precedence rule between two spellings of
one fact.

Loading is strict about near-misses and indifferent to clean misses:
anything which claims to participate in a convention and gets it wrong
raises, while anything which does not participate -- photos, field notes,
deployment logs -- is ignored where it lies.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from dascore.core.inventory import (
    Acquisition,
    Cable,
    Enclosure,
    ExternalResource,
    FiberArray,
    Interrogator,
    Inventory,
    Network,
    OpticalMeasurement,
    Station,
    _overlapping_epochs,
)
from dascore.exceptions import (
    InvalidInventoryError,
    MissingOptionalDependencyError,
)
from dascore.models import InventoryModel, TimeRangedModel
from dascore.models.registry import TAG_FIELD
from dascore.utils.misc import check_code, optional_import
from dascore.utils.time import to_datetime64

# One data model stands behind all three spellings, so they are accepted
# identically -- but one identity may only be spelled once.
_OBJECT_SUFFIXES = (".yaml", ".yml", ".json")

# The object file of an entity directory, and the envelope at the root.
_ATTRS_STEM = "attrs"
_ENVELOPE_STEM = "inventory"

# Separates an entity's name from the epoch it starts.
_EPOCH_MARKER = "@"

# The reserved container stem: unlike an attribute table, these directories
# address a child entity whose name carries its epoch.
_PATH_STEM = "path"

# The document-level collections, which live in the directory structure
# rather than in the envelope.
_ENVELOPE_COLLECTIONS = ("networks", "resources")

# Address levels which name a containing entity rather than a field of the
# entity being named.
_ADDRESS_LEVELS = ("network", "fiber_array")


class _Container(NamedTuple):
    """How one top-level directory maps entry names onto models."""

    models: tuple[type[InventoryModel], ...]
    # The dotted tokens an entry name holds, outermost first. A token which
    # names a field of the model states that field; the rest are
    # _ADDRESS_LEVELS naming the entity which contains it.
    identity: tuple[str, ...]
    # Collections whose members are addressed by their own names, in their
    # own container. A file here may not state them: assembly would replace
    # whatever it said, so stating them is a fact the format cannot keep.
    supplied: tuple[str, ...] = ()

    @property
    def epochs(self) -> bool:
        """Whether names here may carry an epoch, which needs a time range."""
        return all(issubclass(x, TimeRangedModel) for x in self.models)


_CONTAINERS: Mapping[str, _Container] = {
    "resources": _Container(
        (Interrogator, Cable, Enclosure, ExternalResource, OpticalMeasurement),
        ("resource_id",),
    ),
    "networks": _Container((Network,), ("code",), ("fiber_arrays", "stations")),
    "fiber_arrays": _Container((FiberArray,), ("network", "code"), ("acquisitions",)),
    "stations": _Container((Station,), ("network", "code")),
    "acquisitions": _Container(
        (Acquisition,), ("network", "fiber_array", "location_code", "code")
    ),
}


class _Entry(NamedTuple):
    """One loaded entity, with where it came from and what contains it."""

    # A resource or one of the time-ranged entities a network contains.
    # Loosely typed because those two halves share only their base class,
    # while assembling needs the fields of whichever half it holds.
    model: Any
    source: Path
    # This entry's _ADDRESS_LEVELS tokens, outermost first.
    address: tuple[str, ...]


def _model_names() -> frozenset[str]:
    """
    Return the name of every inventory model.

    A file declaring one of these is claiming to be part of an inventory,
    which is what makes it a near-miss rather than field material when it
    turns up somewhere unrecognized.
    """

    def walk(model):
        # Scoped to dascore's own models, so that what a caller happens to
        # have subclassed and imported cannot change which files this
        # format calls a near-miss.
        if model.__module__.startswith("dascore."):
            yield model.__name__
        for sub in model.__subclasses__():
            yield from walk(sub)

    return frozenset(walk(InventoryModel)) | {Inventory.__name__}


def _quote(path: Path) -> str:
    """
    Name a file for an error message.

    Its container is included because a bare name is ambiguous across
    containers and the full path is noise the reader already knows.
    """
    return str(Path(path.parent.name) / path.name)


def _object_suffix(path: Path) -> str | None:
    """
    Return a path's object-file suffix, or None if it has none.

    Matched without regard to case: a shouted ``DAS.L001.YAML`` is the file
    ``DAS.L001.yaml`` would be, and skipping it for its spelling would load
    an inventory silently missing whatever it named.
    """
    suffix = path.suffix.casefold()
    return suffix if suffix in _OBJECT_SUFFIXES else None


def _entry_name(path: Path) -> str:
    """
    Return the address an entry's name states.

    ``Path.stem`` cannot be used: an address is full of dots, so it would
    read ``DAS.L001`` as the stem ``DAS`` with a suffix.
    """
    if path.is_dir() or (suffix := _object_suffix(path)) is None:
        return path.name
    return path.name[: -len(suffix)]


def _read_object(path: Path) -> dict[str, Any]:
    """
    Parse one YAML or JSON object file into a mapping.

    Both spellings share one data model, so the suffix picks the parser
    and decides nothing else.
    """
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        msg = f"Could not read {_quote(path)}: {error}."
        raise InvalidInventoryError(msg) from error
    if _object_suffix(path) == ".json":
        try:
            data = json.loads(text)
        except ValueError as error:
            msg = f"Could not parse JSON from {_quote(path)}: {error}."
            raise InvalidInventoryError(msg) from error
    else:
        yaml = optional_import("yaml", required_for="YAML inventory serialization")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            msg = f"Could not parse YAML from {_quote(path)}: {error}."
            raise InvalidInventoryError(msg) from error
    if not isinstance(data, Mapping):
        msg = f"{_quote(path)} holds no mapping, so it defines no object."
        raise InvalidInventoryError(msg)
    return dict(data)


def _declared_type(path: Path) -> str | None:
    """
    Return the model type a file declares, or None if it declares none.

    This tells field material from a misfiled object, so a file which does
    not parse simply is not an object: a YAML-suffixed file under a photos
    directory owes this format nothing. Nor does an unreadable one make an
    inventory unloadable -- without PyYAML installed, a JSON inventory must
    still load past whatever YAML happens to lie beside it.
    """
    try:
        data = _read_object(path)
    except (InvalidInventoryError, MissingOptionalDependencyError):
        return None
    declared = data.get(TAG_FIELD)
    return declared if isinstance(declared, str) else None


# The date is extended ISO 8601 and the time is basic, because ':' is not a
# legal filename character on Windows.
_EPOCH_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?:T(?P<time>\d{6}(?:\.\d+)?))?")


def _parse_epoch(text: str, source: Path):
    """
    Parse the timestamp an entity name states after its ``@``.

    All inventory timestamps are UTC, so a designator saying so -- or
    saying otherwise -- is refused rather than ignored.
    """
    where = f"in the name of {_quote(source)}"
    date, _, time = text.partition("T")
    if text.endswith("Z") or "+" in text or "-" in time:
        msg = (
            f"Epoch timestamp {text!r} {where} carries a timezone designator. "
            "Inventory timestamps are UTC and take none."
        )
        raise InvalidInventoryError(msg)
    unreadable = (
        f"Could not read an epoch timestamp from {text!r} {where}. Expected a "
        "date (2024-06-01) or a date and an ISO basic time "
        "(2024-05-12T103000, fractional seconds as T103000.12)"
    )
    if (match := _EPOCH_RE.fullmatch(text)) is None:
        raise InvalidInventoryError(f"{unreadable}.")
    time = match["time"] or "000000"
    # A datetime64 keeps nanoseconds, and quietly drops whatever is finer,
    # so a name stating more precision than it can hold would load as an
    # instant other than the one it names.
    if len(time.partition(".")[2]) > 9:
        msg = (
            f"Epoch timestamp {text!r} {where} states a time finer than the "
            "nanosecond an inventory timestamp keeps."
        )
        raise InvalidInventoryError(msg)
    try:
        parsed = to_datetime64(f"{date}T{time[:2]}:{time[2:4]}:{time[4:]}")
    except ValueError as error:
        # The pattern admits digits which name no instant, e.g. a 13th month.
        raise InvalidInventoryError(f"{unreadable}: {error}") from error
    # A nanosecond timestamp spans about 1678 to 2262 and numpy wraps around
    # silently past either end, which would put this epoch centuries from
    # where its name says and quietly misfile every child of it.
    if not str(parsed).startswith(date):
        msg = (
            f"Epoch timestamp {text!r} {where} is outside the range a "
            f"nanosecond timestamp can represent; it would read as {parsed}."
        )
        raise InvalidInventoryError(msg)
    return parsed


def _split_epoch(source: Path, epochs_allowed: bool):
    """Split an entry name into the address and the epoch it starts."""
    name, marker, epoch_text = _entry_name(source).partition(_EPOCH_MARKER)
    if not marker:
        return name, None
    if not epochs_allowed:
        msg = (
            f"{_quote(source)} names an epoch with {_EPOCH_MARKER!r}, but "
            f"{source.parent.name} have none."
        )
        raise InvalidInventoryError(msg)
    if _EPOCH_MARKER in epoch_text:
        msg = (
            f"{_quote(source)} states more than one {_EPOCH_MARKER!r}; a name "
            "carries at most one epoch."
        )
        raise InvalidInventoryError(msg)
    if not epoch_text:
        msg = f"{_quote(source)} ends in {_EPOCH_MARKER!r} but names no epoch."
        raise InvalidInventoryError(msg)
    return name, _parse_epoch(epoch_text, source)


def _pick_model(data: dict, container: _Container, source: Path):
    """
    Return the model a file declares, checking its container agrees.

    The container never supplies the type: a file which does not say what
    it is has not participated in the format, and one which says something
    its container cannot hold is misfiled rather than reinterpreted.

    The tag is left in the data. Every model reads its own, so the file's
    statement is checked a second time by the object it builds.
    """
    declared = data.get(TAG_FIELD)
    legal = tuple(x.__name__ for x in container.models)
    # Not merely absent: a type which is not a name at all -- a list, a
    # mapping -- names no model either, and must not reach the lookups below.
    if not isinstance(declared, str):
        msg = (
            f"{_quote(source)} declares no {TAG_FIELD}. Every object file "
            f"states what it is, e.g. '{TAG_FIELD}: {legal[0]}'."
        )
        raise InvalidInventoryError(msg)
    for model in container.models:
        if model.__name__ == declared:
            return model
    known = "an inventory model" if declared in _model_names() else "unknown"
    msg = (
        f"{_quote(source)} declares {TAG_FIELD} {declared!r} ({known}), which "
        f"{source.parent.name} cannot hold. Expected one of {legal}."
    )
    raise InvalidInventoryError(msg)


def _refuse_supplied(data: dict, fields, source: Path, held_by: str) -> None:
    """
    Refuse a collection which the directory structure supplies.

    The hierarchy materializes from names, so the members of these
    collections are addressed in their own container. Assembly replaces
    whatever a file said about them, which would make a stated one vanish
    -- and vanish only once some other file happened to address the same
    entity, so the file's own meaning would depend on its neighbours.
    """
    for field in fields:
        if field in data:
            msg = (
                f"{_quote(source)} states {field}, which live in the directory "
                f"structure rather than in {held_by}."
            )
            raise InvalidInventoryError(msg)


def _build(model, data: dict, source: Path):
    """Validate one mapping into its model, naming the file if it fails."""
    if "schema_version" in data:
        msg = (
            f"{_quote(source)} states a schema_version. The envelope versions "
            "the document exactly once; individual files never do."
        )
        raise InvalidInventoryError(msg)
    try:
        return model(**data)
    except Exception as error:
        msg = f"Could not read {model.__name__} from {_quote(source)}: {error}"
        raise InvalidInventoryError(msg) from error


def _apply_identity(data: dict, container: _Container, name: str, source: Path):
    """
    Fill the fields an entry's name states, and return its address.

    A name is an address, so the tokens it holds are facts about the
    entity. A file may restate them, but a restatement which disagrees is
    an error rather than an override.
    """
    # A single-token identity is the whole name: a resource_id is free-form
    # and may hold the dots an address would have split on.
    tokens = name.split(".") if len(container.identity) > 1 else [name]
    if len(tokens) != len(container.identity):
        spelled = ".".join(f"<{x}>" for x in container.identity)
        msg = (
            f"{_quote(source)} holds {len(tokens)} dot separated tokens, but "
            f"its name is an address of {len(container.identity)}: {spelled}."
        )
        raise InvalidInventoryError(msg)
    address = []
    for token, field in zip(tokens, container.identity, strict=True):
        if field in _ADDRESS_LEVELS:
            # Checked here because the entity this token names is built
            # later, from every address which mentions it, and by then
            # there is no one file to blame for the token being illegal.
            try:
                check_code(token)
            except InvalidInventoryError as error:
                msg = f"{_quote(source)} names {field} {token!r}: {error}"
                raise InvalidInventoryError(msg) from error
            address.append(token)
            continue
        stated = data.setdefault(field, token)
        if stated != token:
            msg = (
                f"{_quote(source)} states {field}={stated!r} but its name says "
                f"{token!r}. A restated address must agree with the name."
            )
            raise InvalidInventoryError(msg)
    return tuple(address)


def _load_entry(entry: Path, data_source: Path, container: _Container) -> _Entry:
    """Load one entry of a container from its object file."""
    data = _read_object(data_source)
    model = _pick_model(data, container, data_source)
    _refuse_supplied(data, container.supplied, data_source, "an object file")
    name, epoch = _split_epoch(entry, container.epochs)
    address = _apply_identity(data, container, name, data_source)
    if epoch is not None and "start_time" not in data:
        data["start_time"] = epoch
    built = _build(model, data, data_source)
    if epoch is not None and built.start_time != epoch:
        msg = (
            f"{_quote(data_source)} states start_time {built.start_time} but "
            f"its name says {epoch}. A restated address must agree with the "
            "name."
        )
        raise InvalidInventoryError(msg)
    return _Entry(built, data_source, address)


def _refuse_tracks(entity: Path) -> None:
    """
    Refuse the parts of an entity directory which cannot be read yet.

    Track tables and optical path epochs are the next piece of this
    format. Refusing them by name beats loading an entity which silently
    lacks the tracks its own directory states.
    """
    for child in sorted(entity.iterdir()):
        if child.name.startswith("."):
            continue
        stem = child.name.partition(_EPOCH_MARKER)[0].partition(".")[0].casefold()
        if child.is_dir() and stem == _PATH_STEM:
            msg = (
                f"{_quote(child)} is an optical path epoch, which cannot be "
                "read yet. State optical_paths in the entity's "
                f"{_ATTRS_STEM} file for now."
            )
            raise InvalidInventoryError(msg)
        if child.suffix.casefold() == ".csv":
            msg = (
                f"{_quote(child)} is a track table, which cannot be read yet. "
                f"State {_entry_name(child)} in the entity's {_ATTRS_STEM} "
                "file for now."
            )
            raise InvalidInventoryError(msg)


def _attrs_file(entity: Path) -> Path:
    """Return the object file of an entity directory, refusing strays."""
    found = []
    for child in sorted(entity.iterdir()):
        if child.is_dir() or child.name.startswith("."):
            continue
        if _object_suffix(child) is None:
            continue
        if _entry_name(child).casefold() != _ATTRS_STEM:
            # Only a file claiming to be an inventory object is misfiled
            # here; a field note which happens to be YAML is no more
            # participating than one nested a directory deeper, which the
            # stray walk already steps over.
            if _declared_type(child) in _model_names():
                msg = (
                    f"{_quote(child)} declares an inventory object, but an "
                    f"entity directory holds only its {_ATTRS_STEM} file "
                    "and its tracks."
                )
                raise InvalidInventoryError(msg)
            continue
        found.append(child)
    if not found:
        msg = (
            f"{_quote(entity)} holds no {_ATTRS_STEM} file, so it states "
            "nothing about the entity it names."
        )
        raise InvalidInventoryError(msg)
    if len(found) > 1:
        spelled = ", ".join(x.name for x in found)
        msg = (
            f"{_quote(entity)} spells its object file more than once "
            f"({spelled}); one identity is spelled once."
        )
        raise InvalidInventoryError(msg)
    return found[0]


def _collide(first: Path, second: Path) -> str:
    """Explain why two entries of one container name one identity."""
    if _entry_name(first) != _entry_name(second):
        return "differ only by case, which a case-insensitive filesystem cannot hold"
    if first.is_dir() != second.is_dir():
        return "spell one identity as both a file and a directory"
    return "spell one identity with two extensions"


def _container_entries(directory: Path) -> list[Path]:
    """
    Return the entries of a container directory, one per identity.

    Identity is unique per container regardless of spelling, so two names
    differing only by case, or one name spelled as both a file and a
    directory, are two spellings of one thing rather than two things.

    Case is folded here and nowhere else, because the rule is about the
    filesystem rather than about identity: a case-insensitive one cannot
    hold both files. Codes themselves stay case-sensitive, as they are to
    the models and to ``Inventory.resolve``, so a directory which folded
    them would disagree with what it loads.
    """
    seen: dict[str, Path] = {}
    for child in sorted(directory.iterdir()):
        if child.name.startswith("."):
            # A resource id is free-form, so a leading dot could be meant as
            # part of a name rather than as a hidden file. The filesystem
            # disagrees, and an entity nothing can see would go missing in
            # silence, so the ambiguity is refused rather than resolved.
            if (
                _object_suffix(child) is not None
                and _declared_type(child) in _model_names()
            ):
                msg = (
                    f"{_quote(child)} declares an inventory object, but a name "
                    "beginning with '.' is hidden, so it cannot name one."
                )
                raise InvalidInventoryError(msg)
            continue
        if child.is_file() and _object_suffix(child) is None:
            if child.suffix.casefold() == ".csv":
                msg = (
                    f"{_quote(child)} is a track table outside an entity "
                    "directory; a table lives beside the attrs file of the "
                    "entity whose tracks it holds."
                )
                raise InvalidInventoryError(msg)
            continue
        key = _entry_name(child).casefold()
        if (first := seen.get(key)) is not None:
            msg = f"{_quote(first)} and {_quote(child)} {_collide(first, child)}."
            raise InvalidInventoryError(msg)
        seen[key] = child
    return list(seen.values())


def _load_container(directory: Path, container: _Container, root: Path):
    """Load every entry of one top-level container directory."""
    out = []
    for child in _container_entries(directory):
        if child.is_dir():
            _refuse_tracks(child)
            data_source = _attrs_file(child)
            # An entity directory holds its own attrs and tracks; an object
            # filed inside it belongs to a container and is not loaded from
            # here, so it has to be refused rather than stepped over.
            _refuse_stray_objects(child, root, skip=data_source)
        else:
            data_source = child
        out.append(_load_entry(child, data_source, container))
    return out


def _check_epoch_duplicates(entries: list[_Entry]) -> None:
    """
    Refuse two entries whose epochs of one entity overlap in time.

    Epoch-name uniqueness is temporal rather than textual, so
    ``@2024-06-01`` and ``@2024-06-01T000000`` are two spellings of one
    epoch. The models reach the same verdict through the assembled tree,
    but name the entity; the author can only act on the files, so the
    overlap is found here while both are still in hand. The rule itself
    is the models', borrowed rather than restated, since a check which
    caught only identical instants would let a partial overlap through to
    the fileless message this exists to replace.
    """
    sources = {id(x.model): x.source for x in entries}
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(*entry.address, getattr(entry.model, "location_code", ""))].append(
            entry.model
        )
    for models in grouped.values():
        for _, first, second in _overlapping_epochs(models, lambda x: x.code):
            msg = (
                f"{_quote(sources[id(first)])} and {_quote(sources[id(second)])} "
                "state epochs of one entity which overlap in time."
            )
            raise InvalidInventoryError(msg)
    return None


def _escapes(child, parent) -> str:
    """
    Say how a child's validity leaves its parent epoch, or "" if it fits.

    Half-open at both ends, so a child ending exactly where its parent
    does fits: the instant they share belongs to neither. An unset bound
    is unbounded, and unbounded is outside any epoch which states that
    end -- which is how a child with no epoch of its own escapes a parent
    which has one.
    """
    if not pd.isnull(parent.start_time) and (
        pd.isnull(child.start_time) or child.start_time < parent.start_time
    ):
        return "starts before"
    if not pd.isnull(parent.end_time) and (
        pd.isnull(child.end_time) or child.end_time > parent.end_time
    ):
        return "runs past"
    return ""


def _place(children: list[_Entry], parents: list[_Entry], kind: str):
    """
    Group children by the parent epoch each one falls in, parents-aligned.

    An epoch is chosen the way every other resolution chooses one, by
    time. A child falling in no epoch of its container is misfiled, and
    one falling in several -- including one whose own start is unset while
    its container has more than one epoch -- is ambiguous rather than
    resolved.

    Starting inside an epoch is not enough to belong to it: a child which
    runs on past its container's epoch would be reachable only before the
    boundary, since resolution past it selects the next epoch, which does
    not hold the child. That is a contradiction in the directory rather
    than a choice to make on the author's behalf.
    """
    out: list[list[_Entry]] = [[] for _ in parents]
    for child in children:
        matches = [
            index
            for index, parent in enumerate(parents)
            if parent.model.is_effective_at(child.model.start_time)
        ]
        if len(matches) != 1:
            start = child.model.start_time
            when = "at any time" if pd.isnull(start) else f"at {start}"
            named = ".".join(child.address)
            msg = (
                f"{_quote(child.source)} belongs to {kind} {named!r}, which "
                f"has {len(matches)} epochs effective {when}."
            )
            raise InvalidInventoryError(msg)
        parent = parents[matches[0]]
        if escape := _escapes(child.model, parent.model):
            span = f"{parent.model.start_time} to {parent.model.end_time}"
            msg = (
                f"{_quote(child.source)} is valid from {child.model.start_time} "
                f"to {child.model.end_time}, so it {escape} the {kind} epoch "
                f"holding it, which runs {span}. State it once per epoch it "
                "spans."
            )
            raise InvalidInventoryError(msg)
        out[matches[0]].append(child)
    return out


def _full_address(entry: _Entry) -> tuple[str, ...]:
    """
    Return the address naming this entry's own entity.

    A child's ``address`` is exactly its parent's full address, and that is
    what lets a flat directory nest: an acquisition addressed ``DAS.L001``
    belongs to the fiber array whose own address is ``DAS.L001``.
    """
    return (*entry.address, entry.model.code)


def _by_parent(entries: list[_Entry]) -> dict[tuple, list[_Entry]]:
    """Group entries under the full address of the entity containing them."""
    out = defaultdict(list)
    for entry in entries:
        out[entry.address].append(entry)
    return out


def _by_self(entries: list[_Entry]) -> dict[tuple, list[_Entry]]:
    """Group entries, which may be epochs of one entity, by their own address."""
    out = defaultdict(list)
    for entry in entries:
        out[_full_address(entry)].append(entry)
    return out


def _fill_arrays(arrays: list[_Entry], acquisitions: list[_Entry]) -> list[_Entry]:
    """
    Put every acquisition in the fiber array epoch which held it.

    An array named only by an acquisition's address exists, the same way a
    network named only by an array's address does; the hierarchy is never
    built by nesting.
    """
    by_parent = _by_parent(acquisitions)
    out = []
    for address, entries in _by_self(arrays).items():
        placed = _place(by_parent.pop(address, []), entries, "fiber array")
        for entry, acqs in zip(entries, placed, strict=True):
            if not acqs:
                out.append(entry)
                continue
            models = tuple(x.model for x in acqs)
            out.append(entry._replace(model=entry.model.new(acquisitions=models)))
    # Whatever is left is addressed to an array no file declares, which the
    # address itself brings into being.
    for address, entries in by_parent.items():
        model = FiberArray(
            code=address[-1], acquisitions=tuple(x.model for x in entries)
        )
        out.append(_Entry(model, entries[0].source, address[:-1]))
    return out


def _assemble(entries: Mapping[str, list[_Entry]]) -> tuple[Network, ...]:
    """
    Build the network tree a flat directory addresses.

    The hierarchy materializes from names rather than from nesting, so a
    network mentioned by an address exists whether or not a file declares
    it.
    """
    arrays = _by_parent(
        _fill_arrays(entries.get("fiber_arrays", []), entries.get("acquisitions", []))
    )
    stations = _by_parent(entries.get("stations", []))
    networks = _by_self(entries.get("networks", []))
    for address in (*arrays, *stations):
        networks.setdefault(address, [])
    out = []
    for address, network_entries in networks.items():
        if not network_entries:
            # A network named only by an address is an undated container.
            code = address[-1]
            network_entries = [_Entry(Network(code=code), Path(code), ())]
        by_array = _place(arrays.get(address, []), network_entries, "network")
        by_station = _place(stations.get(address, []), network_entries, "network")
        for entry, arrays_here, stations_here in zip(
            network_entries, by_array, by_station, strict=True
        ):
            out.append(
                entry.model.new(
                    fiber_arrays=tuple(x.model for x in arrays_here),
                    stations=tuple(x.model for x in stations_here),
                )
            )
    return tuple(out)


def _load_envelope(root: Path) -> dict[str, Any] | None:
    """
    Read the document-level singletons from the envelope.

    The envelope is optional while authoring -- a bare directory of fiber
    arrays and acquisitions loads with defaults -- but it is the one place
    the document's version and CRS may be stated.
    """
    # Scanned rather than constructed from the known suffixes, so that a
    # shouted spelling is found and, when both are present, reported as the
    # two spellings of one envelope that they are.
    found = [
        child
        for child in sorted(root.iterdir())
        if child.is_file()
        and _object_suffix(child) is not None
        and _entry_name(child).casefold() == _ENVELOPE_STEM
    ]
    # None rather than an empty mapping: an envelope stating only its
    # own type says the directory is an inventory, which is a different
    # fact from there being no envelope at all.
    if not found:
        return None
    if len(found) > 1:
        spelled = ", ".join(x.name for x in found)
        msg = f"The envelope is spelled more than once ({spelled})."
        raise InvalidInventoryError(msg)
    source = found[0]
    data = _read_object(source)
    if (declared := data.get(TAG_FIELD)) != Inventory.__name__:
        msg = (
            f"{_quote(source)} declares {TAG_FIELD} {declared!r}; the envelope "
            f"declares '{TAG_FIELD}: {Inventory.__name__}'."
        )
        raise InvalidInventoryError(msg)
    _refuse_supplied(data, _ENVELOPE_COLLECTIONS, source, "the envelope")
    # Validated here, discarding the result, so that a typo'd key or an
    # unreadable value names its file like every other error this format
    # raises rather than surfacing as a bare pydantic error at the end.
    try:
        Inventory(**data)
    except Exception as error:
        msg = f"Could not read the envelope from {_quote(source)}: {error}"
        raise InvalidInventoryError(msg) from error
    return data


def _refuse_stray_objects(start: Path, root: Path, skip: Path | None = None) -> None:
    """
    Refuse a model-declaring file which nothing contains.

    A typo like ``aquisitions/`` must not quietly load an inventory with
    no acquisitions, and neither must an object filed one level too deep,
    inside an entity directory which holds only its own attrs and tracks.
    Everything which declares nothing stays welcome where it lies.
    """
    known = _model_names()

    def check(path: Path):
        if path.name.startswith(".") or path == skip:
            return
        # A symlink is not part of the format, and one pointing at an
        # ancestor walks the inventory a second time -- where its own files
        # are found with nothing containing them, so a valid directory
        # reports itself as stray, and the walk recurses until it cannot.
        if path.is_symlink():
            return
        if path.is_dir():
            for child in sorted(path.iterdir()):
                check(child)
        elif _object_suffix(path) is not None:
            if (declared := _declared_type(path)) in known:
                msg = (
                    f"{path.relative_to(root)} declares {TAG_FIELD} "
                    f"{declared!r} but "
                    "nothing contains it. An inventory object lives in one of "
                    f"{tuple(_CONTAINERS)}, named for the entity it is."
                )
                raise InvalidInventoryError(msg)

    check(start)


def _check_strays(root: Path) -> None:
    """Refuse a stray object anywhere outside a recognized container."""
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name in _CONTAINERS:
            continue
        if child.is_file() and _entry_name(child).casefold() == _ENVELOPE_STEM:
            continue
        _refuse_stray_objects(child, root)


def load_directory(path: str | os.PathLike) -> Inventory:
    """
    Load an inventory from a directory in the authoring format.

    Parameters
    ----------
    path
        The inventory directory.
    """
    root = Path(path)
    envelope = _load_envelope(root)
    entries = {
        name: _load_container(root / name, container, root)
        for name, container in _CONTAINERS.items()
        if (root / name).is_dir()
    }
    _check_strays(root)
    # After the stray check, which knows why a directory holds no container:
    # a mistyped `aquisitions/` can say so, rather than being reported as a
    # directory holding nothing.
    if not entries and envelope is None:
        # Every directory would otherwise be an empty inventory, so a
        # mistyped path would load rather than say it holds nothing.
        msg = (
            f"{root} holds no inventory: it has neither an envelope nor any "
            f"of the containers {tuple(_CONTAINERS)}."
        )
        raise InvalidInventoryError(msg)
    for name, container in _CONTAINERS.items():
        if container.epochs:
            _check_epoch_duplicates(entries.get(name, []))
    resources = {x.model.resource_id: x.model for x in entries.get("resources", [])}
    networks = _assemble(entries)
    return Inventory(**(envelope or {}), resources=resources, networks=networks).check()


def inventory(source: Inventory | str | os.PathLike | None = None) -> Inventory:
    """
    Load or create a DASDAE inventory.

    Parameters
    ----------
    source
        An existing Inventory (returned as is), a directory in the
        authoring format, a YAML file path or YAML text, or None for an
        empty inventory.

    Examples
    --------
    >>> import dascore as dc
    >>> empty = dc.inventory()
    >>> assert dc.inventory(empty) is empty
    """
    if source is None:
        return Inventory()
    if isinstance(source, Inventory):
        return source
    if isinstance(source, str | os.PathLike):
        if os.path.isdir(source):
            return load_directory(source)
        return Inventory.from_yaml(source)
    msg = f"Could not get an inventory from {source!r}."
    raise InvalidInventoryError(msg)
