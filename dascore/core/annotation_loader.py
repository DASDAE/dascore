"""
Read annotation sets from storage.

A set is stored either as a directory naming what it holds -- always
``annotations.csv``, one row per annotation; ``attrs`` where it states its
own dimensions and provenance; ``vertices.csv`` where any path or polygon
needs one -- or as a bare table whose dimensions the caller states.

A directory of those directories is a collection, and reads as one set
whose ``set`` column names which of them each row came from: one return
type, so nothing downstream has to ask which layout it was handed. What a
set declares only for itself -- its dimensions, its provenance, its
documented columns -- is kept under ``attrs.sets``, and a row's identity is
still the ``id`` it already had, so an id two sets share is refused rather
than qualified by the set it came from.

CSV has no types, so this module decides what each column holds before the
models see it: a dimension column is numbers or times, a ``basis`` cell is
the JSON document its curve dumps, and every other cell is read the way it
was written. Tables are read strictly, through
[`read_table`](`dascore.utils.tables.read_table`), and the neutral errors
that raises are named as annotation errors here, at the one boundary which
knows the format.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Collection, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from dascore.core.annotations import (
    _END,
    _START,
    _VERTEX_COLUMNS,
    ANNOTATION_STEM,
    ATTRS_STEM,
    OBJECT_SUFFIXES,
    RESERVED_COLUMNS,
    TABLE_SUFFIX,
    VERTEX_STEM,
    AnnotationSet,
    _text,
)
from dascore.exceptions import InvalidAnnotationError, ParameterError
from dascore.models.registry import TAG_FIELD
from dascore.utils.misc import iterate, optional_import
from dascore.utils.paths import quote_path
from dascore.utils.tables import parse_cell, read_table
from dascore.utils.time import to_datetime64

# What an attrs file declares itself to be; the model writes its own tag.
_SET_TAG = "AnnotationSetAttrs"

# Columns whose cells stay text however they are spelled: an id which
# looks like a number is still the label the vertices name it by. Derived
# rather than listed, so a column added to the reserved set is text by
# default rather than silently falling through to `parse_cell`. `value`
# holds whichever kind its group holds, and `basis` holds a document.
_TEXT_COLUMNS = frozenset(RESERVED_COLUMNS) - {"value", "basis"}

# The column a vertex states its place in the order by; a number.
_ORDINAL = _VERTEX_COLUMNS[1]


def _read_object(path: Path) -> dict[str, Any]:
    """Parse one YAML or JSON object file into a mapping."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        msg = f"Could not read {quote_path(path)}: {error}."
        raise ParameterError(msg) from error
    # The suffix is casefolded, as the inventory casefolds its own: an
    # attrs.JSON is the file attrs.json would be, and reading it as YAML
    # for its spelling would fail on a document which is not wrong. The
    # stem is not: the part names are exact, as every other name a format
    # reserves is.
    if path.suffix.casefold() == OBJECT_SUFFIXES[0]:
        try:
            data = json.loads(text)
        except ValueError as error:
            msg = f"Could not parse JSON from {quote_path(path)}: {error}."
            raise ParameterError(msg) from error
    else:
        yaml = optional_import("yaml", required_for="YAML annotation storage")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            msg = f"Could not parse YAML from {quote_path(path)}: {error}."
            raise ParameterError(msg) from error
    if not isinstance(data, Mapping):
        msg = f"{quote_path(path)} holds no mapping, so it states no attributes."
        raise ParameterError(msg)
    return dict(data)


def _entries(directory: Path) -> list[Path]:
    """
    Return what a directory holds, as this format's own error.

    ``iterdir`` and the ``exists`` calls in the scans raise ``OSError`` -- a
    directory whose permissions were tightened is the ordinary case -- and
    that would leave `annotations` unwrapped, where every other failure to
    read a stored set arrives as an annotation error.
    """
    try:
        return sorted(directory.iterdir())
    except OSError as error:
        msg = f"Could not read {quote_path(directory)}: {error}."
        raise ParameterError(msg) from error


def _one_spelling(directory: Path, stem: str, suffixes: Sequence[str]) -> Path | None:
    """Return the one file a stem names, or None; two spellings raise."""
    found = [
        x
        for x in _entries(directory)
        if x.stem == stem and x.suffix.casefold() in suffixes
    ]
    if len(found) > 1:
        listed = ", ".join(sorted(x.name for x in found))
        msg = (
            f"{quote_path(directory)} states {stem} more than once: {listed}. "
            "A set spells each of its parts once."
        )
        raise ParameterError(msg)
    return found[0] if found else None


def _read_attrs(directory: Path) -> dict[str, Any]:
    """Return the attributes a set directory states, which may be none."""
    path = _one_spelling(directory, ATTRS_STEM, OBJECT_SUFFIXES)
    if path is None:
        return {}
    data = _read_object(path)
    declared = data.pop(TAG_FIELD, None)
    if declared is not None and declared != _SET_TAG:
        msg = (
            f"{quote_path(path)} declares {declared!r}, but the attributes of "
            f"an annotation set declare {_SET_TAG!r}."
        )
        raise ParameterError(msg)
    return data


def _read_dimension(series: pd.Series, path: Path) -> pd.Series:
    """
    Read a dimension column as the numbers or times its cells state.

    Numbers are tried first because every datetime spelling this writes is
    an ISO string, which is not a number, while seconds from the epoch are
    a number a distance column would lose to a date.
    """
    stated = series.notna()
    if not stated.any():
        return series
    with suppress(TypeError, ValueError):
        return pd.to_numeric(series)
    try:
        values = to_datetime64(series[stated].to_numpy(dtype=str))
    except (TypeError, ValueError) as error:
        msg = (
            f"The column {series.name!r} of {quote_path(path)} states neither "
            f"numbers nor times: {error}."
        )
        raise ParameterError(msg) from error
    out = pd.Series(
        np.datetime64("NaT", "ns"), index=series.index, dtype="datetime64[ns]"
    )
    out[stated] = values
    return out


def _read_ordinal(series: pd.Series, path: Path) -> pd.Series:
    """Read the vertex order column as the numbers it states."""
    try:
        return pd.to_numeric(series)
    except (TypeError, ValueError) as error:
        msg = (
            f"{quote_path(path)} has a non-numeric {_ORDINAL}: {error}. A vertex "
            "states its place in the order as a number."
        )
        raise ParameterError(msg) from error


def _read_basis(series: pd.Series, path: Path) -> pd.Series:
    """Read a basis column as the documents its cells hold."""

    def read(cell):
        if not isinstance(cell, str):
            return cell
        try:
            return json.loads(cell)
        except ValueError as error:
            msg = (
                f"A basis in {quote_path(path)} is not a JSON document: {error}. "
                "A stored basis is what its curve dumps."
            )
            raise ParameterError(msg) from error

    return series.map(read)


def _dimension_spellings(dims: Sequence[str]) -> frozenset[str]:
    """Every column name a declared dimension may be spelled with."""
    return frozenset(x for dim in dims for x in (dim, f"{dim}{_START}", f"{dim}{_END}"))


def _read_cells(
    frame: pd.DataFrame, dims: Sequence[str], path: Path, ordered: bool = False
) -> pd.DataFrame:
    """
    Read a table's text cells as the values each column holds.

    Only the vertices are ordered, so only they read ``seq`` as the number
    it is: the annotations table does not reserve that name, and an
    annotation carrying its own ``seq`` means whatever it says.
    """
    spellings = _dimension_spellings(dims)
    out = {}
    for name in frame.columns:
        series = frame[name]
        if str(name) in spellings:
            out[name] = _read_dimension(series, path)
        elif ordered and str(name) == _ORDINAL:
            out[name] = _read_ordinal(series, path)
        elif str(name) == "basis":
            out[name] = _read_basis(series, path)
        elif str(name) in _TEXT_COLUMNS or series.isna().all():
            # A column no row states says nothing about what it holds, and
            # reading its emptiness as a type would invent one.
            out[name] = series
        else:
            out[name] = series.map(_read_extra)
    return pd.DataFrame(out)


def _read_extra(cell):
    """
    Read one cell of a column the set does not model.

    A cell reading 'nan' or 'inf' parses as a float which every later
    reader treats as unset, so the value would be deleted rather than
    retyped; those stay the text the table plainly states.
    """
    if not isinstance(cell, str):
        return cell
    value = parse_cell(cell)
    if isinstance(value, float) and not math.isfinite(value):
        return cell
    return value


def _read_set_table(
    path: Path, dims: Sequence[str], what: str, ordered: bool = False
) -> pd.DataFrame | None:
    """
    Read one of a set's tables, with its cells typed.

    A table stating nothing at all is what a set of no annotations writes,
    so it reads back as none rather than as a table with no columns.
    """
    if _is_blank(path):
        return None
    return _read_cells(read_table(path, what=what), dims, path, ordered=ordered)


def _is_blank(path: Path) -> bool:
    """Whether a table holds nothing but whitespace."""
    try:
        return not path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        # Unreadable is the table reader's to name, with its own message.
        return False


def _refuse_stray_tables(directory: Path, known: Collection[str], what: str) -> None:
    """
    Refuse a table whose name names no part of what a directory holds.

    A ``vertexes.csv`` beside an ``annotations.csv`` claims to participate
    in this convention and gets it wrong, which is worth more than being
    quietly skipped. Hidden names are not tables here, as they are not sets:
    an editor's ``.annotations.csv.swp`` or a half-copied ``.annotations.csv``
    is a companion the directory keeps, and taking a directory down for one
    would make a stray sync file fatal.
    """
    stray = sorted(
        x.name
        for x in _entries(directory)
        if not x.name.startswith(".")
        and x.suffix.casefold() == TABLE_SUFFIX
        and x.stem not in known
    )
    if stray:
        msg = f"{quote_path(directory)} holds the table(s) {', '.join(stray)}, {what}"
        raise ParameterError(msg)


def _refuse_overrides(what: str, **stated) -> None:
    """
    Refuse an argument a source states for itself.

    Silently dropping one is the worse failure: a caller passing
    ``dims=patch.dims`` to whatever it was handed would get the source's
    dimensions from one kind of source and its own from another, with
    nothing said either way.
    """
    given = sorted(k for k, v in stated.items() if v is not None)
    if given:
        msg = (
            f"{', '.join(given)} was given for {what}, which states it. "
            "Read it and change it, rather than reading it as something else."
        )
        raise ParameterError(msg)


def _load_directory(directory: Path, dims, **kwargs) -> AnnotationSet:
    """Load the set a directory holds, or the sets a directory of them does."""
    # Both are the directory's to state, and passing them through would
    # reach AnnotationSet twice as a bare TypeError.
    _refuse_overrides(
        "a set directory",
        attrs=kwargs.pop("attrs", None),
        vertices=kwargs.pop("vertices", None),
    )
    attrs = _read_attrs(directory)
    # A directory which states annotations is a set, and nothing below it is
    # looked at: a folder someone kept beside its tables -- a backup of an
    # attrs file, an older copy of the set -- is no more this format's
    # business than a notes.txt is, and reading the directory as a
    # collection because of one would refuse a set which is complete.
    if _states_annotations(directory):
        return _load_set(directory, attrs, dims, **kwargs)
    if children := _child_sets(directory):
        return _load_collection(directory, children, attrs, dims, **kwargs)
    return _load_set(directory, attrs, dims, **kwargs)


def _states_annotations(path: Path) -> bool:
    """Whether a directory states annotations of its own; a file states none."""
    if not path.is_dir():
        return False
    return _one_spelling(path, ANNOTATION_STEM, (TABLE_SUFFIX,)) is not None


def _child_sets(directory: Path) -> list[Path]:
    """
    Return the set directories a directory of sets holds, in name order.

    Only reached for a directory which states no annotations itself. The
    sets are found first and nothing is refused until at least one is: a
    directory holding none of them is not a collection at all -- it is the
    data, or a directory carrying its annotations under the hidden name --
    and complaining about what sits in it would refuse a directory this
    format has no claim on.

    Once it is a collection, a child which states attributes and no
    annotations is half a set and says so, and one holding sets of its own
    is refused, since sets loaded together are one collection rather than a
    tree. Anything else is left alone -- the data the sets describe, a
    folder of figures -- as is a directory holding only vertices, which is
    never looked for without the annotations it belongs to.
    """
    # Hidden names are skipped as the file scanner skips them: a
    # `.inventory` beside the sets describes the data, not the annotations.
    children = [
        x for x in _entries(directory) if x.is_dir() and not x.name.startswith(".")
    ]
    out = [x for x in children if _states_annotations(x)]
    if not out:
        return []
    for child in (x for x in children if x not in out):
        if nested := [x.name for x in _entries(child) if _states_annotations(x)]:
            msg = (
                f"{quote_path(child)} holds the set(s) {', '.join(nested)}. Sets "
                "loaded together are one collection, not a tree of them."
            )
            raise ParameterError(msg)
        if _one_spelling(child, ATTRS_STEM, OBJECT_SUFFIXES) is not None:
            msg = (
                f"{quote_path(child)} states the attributes of a set but no "
                f"{ANNOTATION_STEM}{TABLE_SUFFIX}, so it states no annotations."
            )
            raise ParameterError(msg)
    _refuse_colliding_names(directory, out)
    return out


def _refuse_colliding_names(directory: Path, children: Sequence[Path]) -> None:
    """
    Refuse set names which differ only in case.

    The name is the label: it is the `set` column and the key in
    ``attrs.sets``, so two which fold together are one set on a filesystem
    which folds them and two on this one. A symlinked set is allowed, by
    contrast -- a collection is a convenience for reading, not an authored
    identity, so pointing one at a set kept elsewhere is a fair use of it.
    """
    folded: dict[str, str] = {}
    for child in children:
        first = folded.setdefault(child.name.casefold(), child.name)
        if first != child.name:
            msg = (
                f"{quote_path(directory)} holds the sets {first} and {child.name}, "
                "whose names differ only in case. A set name is the label its "
                "rows carry, so it must name one set on any filesystem."
            )
            raise ParameterError(msg)


def _load_collection(
    directory: Path, children: Sequence[Path], attrs: Mapping, dims, **kwargs
) -> AnnotationSet:
    """
    Load the sets a directory of them holds, as one set.

    Each child is read in its own dimensions, then merged by `_merge_sets`.
    The refusals here are the ones only a collection can hit: sets stated
    twice, a table beside them, and dimensions given for a set which
    declares its own.
    """
    if attrs.get("sets"):
        msg = (
            f"{quote_path(directory)} states sets in its attributes and holds "
            "them in directories. A collection states each of its sets once."
        )
        raise ParameterError(msg)
    _refuse_stray_tables(
        directory,
        (),
        "which name no set. A set is a directory here, so a table beside them "
        "states nothing; a bare table is read on its own.",
    )
    if attrs.get("dims"):
        _refuse_overrides("a directory of sets stating its own dimensions", dims=dims)
    # The caller's dimensions, else the ones stated beside the sets, stand in
    # for a child which declares none; a child which declares its own is read
    # in those, and is refused the argument as it would be on its own, rather
    # than having it dropped where nobody can see it happen.
    default = dims if dims is not None else attrs.get("dims")
    loaded = {}
    for child in children:
        child_attrs = _read_attrs(child)
        if child_attrs.get("dims"):
            _refuse_overrides(
                f"{quote_path(child)}, which states its own dimensions", dims=dims
            )
        loaded[child.name] = _load_set(
            child, child_attrs, None if child_attrs.get("dims") else default
        )
    return _merge_sets(loaded, attrs, **kwargs)


def _merge_sets(
    loaded: Mapping[str, AnnotationSet], attrs: Mapping, **kwargs
) -> AnnotationSet:
    """Build the one set the sets loaded together make."""
    stated = [str(x) for x in iterate(attrs.get("dims") or ())]
    # The collection's own dimensions first, then each set's in the order
    # the sets were read, which is their names' -- so the order is stable
    # for one directory, though renaming a set can change it.
    dims = tuple(
        dict.fromkeys([*stated, *(x for one in loaded.values() for x in one.dims)])
    )
    frames = {name: _labeled(one, name) for name, one in loaded.items()}
    _refuse_undeclared_dims(loaded, frames, dims)
    _refuse_mixed_spellings(frames, dims)
    _refuse_mixed_kinds(frames, dims, "an annotation")
    frame = pd.concat(frames.values(), ignore_index=True, sort=False)
    _refuse_shared_ids(frame)
    vertices = {
        name: drawn
        for name, one in loaded.items()
        if not (drawn := one.to_vertices()).empty
    }
    _refuse_mixed_vertices(vertices)
    _refuse_mixed_kinds(vertices, dims, "a vertex")
    document = dict(attrs)
    document["dims"] = dims
    document["sets"] = {name: one.attrs for name, one in loaded.items()}
    return AnnotationSet(
        frame,
        vertices=pd.concat(vertices.values(), ignore_index=True, sort=False)
        if vertices
        else None,
        attrs=document,
        **kwargs,
    )


def _labeled(one: AnnotationSet, name: str) -> pd.DataFrame:
    """
    Return one set's annotations, saying which set each row came from.

    Only the label is added. What the set states for itself stays in its
    attributes, where `attrs.sets` keeps it and a row reads it back through
    the label -- writing any of it into every row would store one fact
    twice, in two places which can then disagree.
    """
    frame = one.to_dataframe()
    if "set" in frame.columns:
        msg = (
            f"The set {name!r} states a set column, so it is already a "
            "collection -- sets saved flat, most likely. A collection is not a "
            "member of another one; read it on its own, or spread it back out."
        )
        raise ParameterError(msg)
    frame["set"] = name
    return frame


def _refuse_undeclared_dims(
    loaded: Mapping[str, AnnotationSet], frames: Mapping[str, pd.DataFrame], dims
) -> None:
    """
    Refuse a column which is a dimension in one set and something else in
    another.

    The collection's dimensions are the union of its sets', so a set which
    holds a column another set declares as a dimension would have that
    column read as a coordinate it never claimed: its extra would stop
    being an extra and start bounding a region.
    """
    for name, frame in frames.items():
        undeclared = set(dims) - set(loaded[name].dims)
        for dim in sorted(undeclared):
            spelled = [dim, f"{dim}{_START}", f"{dim}{_END}"]
            held = sorted(x for x in spelled if x in frame.columns)
            if not held:
                continue
            claims = sorted(k for k, v in loaded.items() if dim in v.dims)
            msg = (
                f"The set {name} holds {', '.join(held)} without declaring "
                f"{dim!r} a dimension, and {', '.join(claims)} declares it one. "
                "One column states one thing, so the sets cannot be read "
                "together until they agree on what it is."
            )
            raise ParameterError(msg)


def _refuse_mixed_spellings(frames: Mapping[str, pd.DataFrame], dims) -> None:
    """
    Refuse a dimension two sets spell differently.

    A set holds one spelling of a dimension, so a merged table cannot hold
    both a bare ``time`` and a ``time_start``/``time_end`` pair; the
    constructor refuses that too, but without naming the sets. Neither
    spelling stands in for the other: a half-open range of no width holds
    nothing, so a point is not a range and cannot be rewritten as one.
    """
    for dim in dims:
        points = [name for name, x in frames.items() if dim in x.columns]
        ranges = [name for name, x in frames.items() if f"{dim}{_START}" in x.columns]
        if points and ranges:
            msg = (
                f"The dimension {dim!r} is spelled as a point in "
                f"{', '.join(points)} and as a range in {', '.join(ranges)}, so "
                "the sets state one thing two ways and cannot be read together."
            )
            raise ParameterError(msg)


def _refuse_mixed_vertices(vertices: Mapping[str, pd.DataFrame]) -> None:
    """
    Refuse sets which draw their vertices in different dimensions.

    A vertex states every dimension its table names, so a curve drawn in
    distance and time cannot sit in one table beside one drawn in distance
    alone: the second would leave a column empty and be half a shape. The
    annotations merge because a bound may be unstated; a vertex may not.
    """
    spelled = {
        name: tuple(sorted(set(map(str, frame.columns)) - set(_VERTEX_COLUMNS)))
        for name, frame in vertices.items()
    }
    if len(set(spelled.values())) < 2:
        return
    listed = "; ".join(f"{name} in {', '.join(dims)}" for name, dims in spelled.items())
    msg = (
        f"The sets loaded together draw their vertices in different dimensions "
        f"({listed}). A vertex states every dimension its table names, so these "
        "cannot be read as one table."
    )
    raise ParameterError(msg)


def _refuse_mixed_kinds(frames: Mapping[str, pd.DataFrame], dims, what: str) -> None:
    """
    Refuse a dimension two sets state in different kinds of value.

    Each set is read in its own dimensions, so one may state ``time`` as
    seconds and another as dates; concatenating those gives a column of
    both, which every later comparison -- a bound against a bound, a sort,
    a select -- either refuses far from here or gets wrong quietly. The
    kinds are compared rather than the dtypes: a column of whole numbers
    and one of floats say the same kind of thing.
    """
    kinds: dict[str, dict[str, str]] = {}
    for name, frame in frames.items():
        for dim in dims:
            for column in (dim, f"{dim}{_START}", f"{dim}{_END}"):
                if column not in frame.columns:
                    continue
                kind = _kind(frame[column])
                seen = kinds.setdefault(column, {})
                if kind != "nothing" and seen and kind not in seen:
                    stated, first = next(iter(seen.items()))
                    msg = (
                        f"The sets state {what}'s {column} in different kinds of "
                        f"value: {kind} in {name}, {stated} in {first}. One column "
                        "holds one kind, so the sets cannot be read together."
                    )
                    raise ParameterError(msg)
                if kind != "nothing":
                    seen.setdefault(kind, name)


def _kind(series: pd.Series) -> str:
    """Name the kind of value a column holds, as a reader would say it."""
    kind = getattr(series.dtype, "kind", "O")
    if series.isna().all():
        # A column no row states says nothing about its kind, so it agrees
        # with whatever the other sets state.
        return "nothing"
    return {"M": "times", "m": "durations", "O": "text", "T": "text", "U": "text"}.get(
        kind, "numbers"
    )


def _refuse_shared_ids(frame: pd.DataFrame) -> None:
    """
    Refuse an id which names a row in more than one set.

    Each set was built before this, so its own ids are already unique;
    what is left is a collision between sets, which would make the id no
    longer an address into the collection. The merged set would refuse it
    anyway, but without saying which sets collided.
    """
    if "id" not in frame.columns:
        return
    # Blank spelled as the set spells it, so this cannot drift from what
    # `_check_ids` counts as an unstated id.
    ids = frame["id"].map(_text)
    shared = (ids != "") & ids.duplicated(keep=False)
    if not shared.any():
        return
    first = ids[shared].iloc[0]
    named = sorted(set(frame.loc[ids == first, "set"]))
    msg = (
        f"The annotation id {first} names a row in {' and '.join(named)}. Ids are "
        "unique across the sets loaded together, so one id names one annotation."
    )
    raise ParameterError(msg)


def _load_set(directory: Path, attrs: Mapping, dims, **kwargs) -> AnnotationSet:
    """Load the set a directory holds."""
    # Dimensions are the directory's once it has stated them: reading the
    # cells against other dimensions would type them differently and
    # build a set which is not the one stored here.
    if attrs.get("dims"):
        _refuse_overrides("a directory stating its own dimensions", dims=dims)
    _refuse_stray_tables(
        directory,
        (ANNOTATION_STEM, VERTEX_STEM),
        f"which name no part of a set. A set states {ANNOTATION_STEM}"
        f"{TABLE_SUFFIX} and, where it has vertices, {VERTEX_STEM}{TABLE_SUFFIX}.",
    )
    table = _one_spelling(directory, ANNOTATION_STEM, (TABLE_SUFFIX,))
    if table is None:
        msg = (
            f"{quote_path(directory)} holds no {ANNOTATION_STEM}{TABLE_SUFFIX}, "
            "so it states no annotations."
        )
        raise ParameterError(msg)
    stated = _declared_dims(attrs, dims, directory)
    frame = _read_set_table(table, stated, "no annotations")
    # Found as the annotations table is: a set spells each of its parts
    # once, and a `vertices.CSV` beside a `vertices.csv` is two spellings of
    # one part rather than a table nobody reads.
    vertex_path = _one_spelling(directory, VERTEX_STEM, (TABLE_SUFFIX,))
    vertices = None
    if vertex_path is not None:
        vertices = _read_set_table(vertex_path, stated, "no vertices", ordered=True)
    # The read dimensions rather than the given ones: they are the same
    # names, already a tuple, so a file spelling one dimension as a bare
    # string builds the set the same way the constructor would.
    return AnnotationSet(frame, dims=stated, vertices=vertices, attrs=attrs, **kwargs)


def _load_file(path: Path, dims, **kwargs) -> AnnotationSet:
    """Load the set a bare table holds."""
    if path.suffix.lower() != TABLE_SUFFIX:
        msg = (
            f"{quote_path(path)} is not a table an annotation set is read from. "
            f"A bare set is a {TABLE_SUFFIX} file; a set with vertices is a "
            "directory."
        )
        raise ParameterError(msg)
    stated = _declared_dims({}, dims, path)
    return AnnotationSet(
        _read_set_table(path, stated, "no annotations"), dims=stated, **kwargs
    )


def _declared_dims(attrs: Mapping, dims, source: Path) -> tuple[str, ...]:
    """
    Return the dimensions to read a source in: the caller's, else its own.

    The cells cannot be read before this is known -- which columns hold
    times rather than text is exactly what a dimension decides -- so a
    source stating none fails here rather than as a puzzling column later.
    """
    stated = dims if dims is not None else attrs.get("dims")
    if not stated:
        msg = (
            f"{quote_path(source)} states no dimensions, and none were given. "
            "Annotations are read in the dimensions they are stated in: write "
            f"them in {ATTRS_STEM}{OBJECT_SUFFIXES[0]} or pass "
            "dims=('distance', 'time')."
        )
        raise ParameterError(msg)
    # Through `iterate`, as the set itself reads them: a lone string is one
    # dimension rather than a sequence of its own letters, and typing the
    # cells against eight one-character dimensions would fail somewhere far
    # from here. Refusing it instead would leave the same input accepted
    # in memory and rejected from a file.
    return tuple(str(x) for x in iterate(stated))


def annotations(
    source: AnnotationSet | str | os.PathLike | Any = None,
    dims: Sequence[str] | None = None,
    **kwargs,
) -> AnnotationSet:
    """
    Load annotations from whatever holds them.

    The one door every source goes through, as
    [`dascore.spool`](`dascore.spool`) is for patches: a set comes back
    from a set, a directory, a table on disk, or anything a dataframe can
    be built from.

    Parameters
    ----------
    source
        An `AnnotationSet`, a dataframe of one row per annotation, or a path
        to a CSV table, a set directory, or a directory of set directories.
    dims
        The patch dimensions the annotations are stated in. Required unless
        the source states them itself, and overriding it where it does.
    **kwargs
        Passed to [`AnnotationSet`](`dascore.core.annotations.AnnotationSet`).
        A source already holding what one states -- a set, or a directory
        holding its own attributes and vertices -- refuses it rather than
        dropping it.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> frame = pd.DataFrame({"group": ["event"], "distance": [10.0]})
    >>> picks = dc.annotations(frame, dims=("distance",))
    >>> len(picks)
    1

    A set is handed straight back, so a function taking either a set or a
    path may simply call this on whatever it was given.

    >>> dc.annotations(picks) is picks
    True

    Sets stored side by side read as one, each row saying which set it came
    from.

    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder) / "sets"
    ...     _ = picks.save(root / "hand")
    ...     _ = picks.save(root / "phasenet")
    ...     together = dc.annotations(root)
    >>> len(together), together[0].set, sorted(together.attrs.sets)
    (2, 'hand', ['hand', 'phasenet'])
    """
    if isinstance(source, AnnotationSet):
        # A built set states everything these would override, and building
        # it again from its frame would quietly drop whatever the overrides
        # did not restate.
        _refuse_overrides("a set which is already built", dims=dims, **kwargs)
        return source
    if isinstance(source, str | os.PathLike):
        path = Path(source)
        try:
            if path.is_dir():
                return _load_directory(path, dims, **kwargs)
            if path.exists():
                return _load_file(path, dims, **kwargs)
        # ValidationError too: a stored document which does not build the
        # models is a bad file, and it is named as one here rather than
        # arriving as pydantic's report on a call the caller did not make.
        except (ParameterError, ValidationError) as error:
            raise InvalidAnnotationError(str(error)) from error
        msg = f"{quote_path(path)} does not exist, so it holds no annotations."
        raise InvalidAnnotationError(msg)
    return AnnotationSet(source, dims=dims, **kwargs)
