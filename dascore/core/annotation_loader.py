"""Read annotation sets from storage.

A set is either a directory holding an ``annotations`` table, optional
``attrs``, a ``features`` table where the set has features, and ``bases.json``
where it has bases; or a bare annotations table whose dimensions the caller
supplies. A directory of set directories loads as one set with a ``set``
column on both tables. Per-set dimensions, provenance, and documented columns
remain under ``attrs.sets``. Feature ids, and annotation ids where stated, are
unique across the sets loaded together.

A table may declare dimensions in a ``# dims: distance, time`` comment above
its header. Data directories may store annotations under ``.annotations``.
Columns beginning with an underscore are ignored.

CSV cells are typed before validation: dimension columns use the set's
reader, order columns are numbers, and reserved columns stay text. Table
errors are exposed as annotation errors.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from dascore.core.annotations import (
    _MAX,
    _MIN,
    ANNOTATION_STEM,
    ATTRS_STEM,
    BASES_STEM,
    DIMS_KEY,
    FEATURE_STEM,
    OBJECT_SUFFIXES,
    ORDINAL_COLUMNS,
    RESERVED_COLUMNS,
    TABLE_SUFFIXES,
    AnnotationSet,
    _combine,
    _is_text_dtype,
    _Tables,
    _text,
    read_dimension,
    read_ordinal,
)
from dascore.exceptions import InvalidAnnotationError, ParameterError
from dascore.models.registry import TAG_FIELD
from dascore.utils.documents import read_document
from dascore.utils.misc import iterate
from dascore.utils.paths import quote_path
from dascore.utils.tables import (
    drop_private_columns,
    parse_cell,
    read_parquet,
    read_parquet_metadata,
    read_table,
)

# What an attrs file declares itself to be; the model writes its own tag.
_SET_TAG = "AnnotationSetAttrs"

# Reserved columns stay text however they are spelled: an id which looks
# like a number is still a label. The order columns are numbers.
_TEXT_COLUMNS = frozenset(RESERVED_COLUMNS) - set(ORDINAL_COLUMNS)

# What a table may carry above its header: comment lines, one of which may
# declare the dimensions the table is stated in.
_COMMENT = "#"
_DIMS_PRAGMA = "dims"

# The suffix which names the parquet encoding, for the branches which read
# a table rather than merely find one.
PARQUET_SUFFIX = TABLE_SUFFIXES[1]

# Store a data directory's annotations in `.annotations/` (a set or
# collection of sets) or `.annotations.csv`. Hidden names keep the default
# file scanner from treating annotations as data.
BLESSED_NAME = ".annotations"


def _read_object(path: Path, holds: str = "states no attributes") -> dict[str, Any]:
    """Parse one YAML or JSON object file into a mapping."""
    # Suffixes are case-insensitive, as in inventories; reserved stems are exact.
    is_json = path.suffix.casefold() == OBJECT_SUFFIXES[0]
    return read_document(
        path,
        "json" if is_json else "yaml",
        error=ParameterError,
        holds=holds,
    )


def _entries(directory: Path) -> list[Path]:
    """
    List directory entries, wrapping OSError as ParameterError.

    The loader translates this into an annotation error, as it does other failures to
    read stored sets.
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
    if retired := sorted(set(_RETIRED_ATTRS) & set(data)):
        replaced = ", ".join(f"{x} (now {_RETIRED_ATTRS[x]})" for x in retired)
        msg = (
            f"{quote_path(path)} states {replaced}, which an earlier layout "
            "wrote; rewrite the set with io.save."
        )
        raise InvalidAnnotationError(msg)
    return data


# Attributes an earlier layout wrote, and what states them now.
_RETIRED_ATTRS = {
    "history": "data_id",
    "columns": "annotation_columns and feature_columns",
}


def _read_dimension(series: pd.Series, path: Path) -> pd.Series:
    """Read a dimension column as the set reads one, naming the table."""
    return read_dimension(series, f" of {quote_path(path)}")


def _read_ordinal(series: pd.Series, path: Path) -> pd.Series:
    """Read an order column as the set reads one, naming the table."""
    return read_ordinal(series, f" of {quote_path(path)}")


def _read_bases(directory: Path) -> dict[str, Any] | None:
    """Return the curve documents a set directory states, or None."""
    path = _one_spelling(directory, BASES_STEM, OBJECT_SUFFIXES)
    if path is None:
        return None
    return _read_object(path, holds="states no bases")


def _dimension_spellings(dims: Sequence[str]) -> frozenset[str]:
    """Every column name a declared dimension may be spelled with."""
    return frozenset(x for dim in dims for x in (dim, f"{dim}{_MIN}", f"{dim}{_MAX}"))


def _is_text(series: pd.Series) -> bool:
    """Whether a column holds text, whichever way pandas is spelling it."""
    return getattr(series.dtype, "kind", "") in "OTU"


def _read_cells(
    frame: pd.DataFrame,
    dims: Sequence[str],
    path: Path,
    ordered: bool = False,
    typed: bool = False,
    text: Collection[str] = (),
) -> pd.DataFrame:
    """
    Read a table's text cells as the values each column holds.

    A column the set declares as text is left as the text it states:
    "001" read as a number would not be the label it was written as.

    Only the annotations table is ordered, so only it reads ``seq``,
    ``part`` and ``ring`` as numbers.

    A typed table -- parquet -- states what each column holds, so only its
    text columns are read further, and only as far as a dimension: text
    elsewhere is text, since a format with a boolean of its own would have
    used one. A dimension or order column which arrives as a type it cannot
    hold is refused rather than trusted.
    """
    spellings = _dimension_spellings(dims)
    out = {}
    for name in frame.columns:
        series = frame[name]
        if typed and not _is_text(series):
            # The file stated what this column holds, so nothing here has
            # to work it out from a spelling -- only check that what it
            # states is a thing the column is allowed to hold.
            if str(name) in spellings:
                _check_kind(series, name, path, "iufMm", "numbers, times or durations")
            elif ordered and str(name) in ORDINAL_COLUMNS:
                _check_kind(series, name, path, "iuf", "a number")
            out[name] = series
        elif str(name) in spellings:
            out[name] = _read_dimension(series, path)
        elif ordered and str(name) in ORDINAL_COLUMNS:
            out[name] = _read_ordinal(series, path)
        elif str(name) in text:
            out[name] = series
        elif typed:
            # Text in a typed table is text: a cell reading 'true' in a
            # format which has a boolean is the word, not the boolean.
            out[name] = series
        elif str(name) in _TEXT_COLUMNS or series.isna().all():
            # A column no row states says nothing about what it holds, and
            # reading its emptiness as a type would invent one.
            out[name] = series
        else:
            out[name] = series.map(_read_extra)
    # The index the table was read with: a file whose every column is the
    # author's own still stated rows, and building from the columns alone
    # would drop them where nothing would say they had gone.
    return pd.DataFrame(out, index=frame.index)


def _check_kind(series: pd.Series, name, path: Path, kinds: str, what: str) -> None:
    """Refuse a typed column whose type the field it names cannot be."""
    if series.dtype.kind in kinds:
        return
    msg = (
        f"The column {str(name)!r} of {quote_path(path)} holds "
        f"{series.dtype}, where it states {what}."
    )
    raise ParameterError(msg)


def _read_extra(cell):
    """
    Read one cell of a column the set does not model.

    A cell reading 'nan' or 'inf' stays the text the table plainly states:
    `parse_cell` reads only what a table spells a number with, and a
    non-finite one -- which every later reader treats as unset -- is not
    among them.
    """
    return parse_cell(cell) if isinstance(cell, str) else cell


def _read_set_table(
    path: Path,
    dims: Sequence[str],
    what: str,
    ordered: bool = False,
    skip: int = 0,
    text: Collection[str] = (),
) -> pd.DataFrame | None:
    """
    Read one of a set's tables, with its cells typed.

    A table stating nothing at all is what a set of no annotations writes
    -- a blank CSV, or a parquet file with no columns -- so it reads back
    as none rather than as a table which states nothing.
    """
    if _is_parquet(path):
        frame, _ = read_parquet(path, what=what, empty=True)
        if not len(frame.columns):
            return None
        frame = _kept_columns(frame, path)
        return _read_cells(frame, dims, path, ordered=ordered, typed=True, text=text)
    if _is_blank(path):
        return None
    frame = _kept_columns(read_table(path, what=what, skip=skip), path)
    return _read_cells(frame, dims, path, ordered=ordered, text=text)


def _kept_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Return a table without the columns its author kept for themselves.

    Read before any cell is: what a private column holds is not this
    format's to type, to check against a declaration, or to refuse. A
    table of nothing else states rows no column of the set can hold, and
    is named here rather than left to the set, which would no longer know
    which file they were in.
    """
    kept = drop_private_columns(frame)
    if len(kept.index) and not len(kept.columns):
        msg = (
            f"{quote_path(path)} states rows and no column but its author's "
            "own; a header beginning with an underscore is read by nothing."
        )
        raise ParameterError(msg)
    return kept


def _is_parquet(path: Path) -> bool:
    """Whether a table's name says it is parquet rather than CSV."""
    return path.suffix.casefold() == PARQUET_SUFFIX


def _read_table_dims(path: Path) -> tuple[tuple[str, ...] | None, int]:
    """
    Return the dimensions a table declares for itself, and the CSV lines
    above its header.

    Each encoding declares them where it can: a CSV in a comment above the
    header, a parquet file in the metadata its footer holds.
    """
    if not _is_parquet(path):
        return _read_pragma(path)
    stated = read_parquet_metadata(path)
    return _stated_dims(stated.get(DIMS_KEY), path), 0


def _stated_dims(document: str | None, path: Path) -> tuple[str, ...] | None:
    """Read the dimensions a parquet file states in its metadata."""
    if document is None:
        return None
    try:
        stated = json.loads(document)
    except ValueError as error:
        msg = (
            f"{quote_path(path)} states {DIMS_KEY} as {document!r}, which is not "
            f"a JSON document: {error}."
        )
        raise ParameterError(msg) from error
    names = tuple(str(x) for x in iterate(stated))
    if not names:
        msg = (
            f"{quote_path(path)} states {DIMS_KEY} but names none; a table which "
            "declares its dimensions names them."
        )
        raise ParameterError(msg)
    return names


def _read_pragma(path: Path) -> tuple[tuple[str, ...] | None, int]:
    """
    Return the dimensions a table declares above its header, and the lines
    to skip to reach that header.

    A table which states its dimensions nowhere else may declare them in a
    comment above its header::

        # dims: distance, time
        # picked by hand
        group,time_min,time_max

    Nothing above the header is skipped unless one of those lines is the
    declaration. A column name may begin with the comment mark -- `# note`
    is a name a set can hold and this library writes unquoted -- and eating
    that header would promote the first row of data to the header with
    nothing said. Where a table does declare its dimensions its author has
    opted into the convention, and further comments beside the declaration
    are skipped with it.

    A comment because a CSV has nowhere else to put this. A reader told
    ``comment="#"`` skips the line; one which is not told reads it as the
    header, which is the cost of the convention and the reason `to_csv`
    does not write it.
    """
    stated: tuple[str, ...] | None = None
    skip = 0
    with _readable(path) as stream:
        for line in stream:
            bare = line.strip()
            if bare and not bare.startswith(_COMMENT):
                break
            skip += 1
            name, _, rest = bare.removeprefix(_COMMENT).partition(":")
            if name.strip().casefold() != _DIMS_PRAGMA:
                continue
            if stated is not None:
                msg = (
                    f"{quote_path(path)} declares {_DIMS_PRAGMA} more than once "
                    "above its header; a table states its dimensions once."
                )
                raise ParameterError(msg)
            stated = tuple(x.strip() for x in rest.split(",") if x.strip())
            if not stated:
                msg = (
                    f"{quote_path(path)} declares {_DIMS_PRAGMA} above its "
                    "header but names none; write '# dims: distance, time'."
                )
                raise ParameterError(msg)
    # No declaration, so there was no preamble to skip: whatever those lines
    # were, the first of them is this table's header.
    return (stated, skip) if stated is not None else (None, 0)


@contextmanager
def _readable(path: Path):
    """
    Open a table for the scan above its header, naming what cannot be read.

    Swallowing the failure would be worse than reporting it: a table which
    does not decode has no dimensions to find, and the caller would go on to
    advise writing the very line the file already holds.
    """
    try:
        with path.open(encoding="utf-8-sig") as stream:
            yield stream
    except (OSError, UnicodeDecodeError) as error:
        msg = f"Could not read {quote_path(path)}: {error}."
        raise ParameterError(msg) from error


def _is_blank(path: Path) -> bool:
    """Whether a table holds nothing but whitespace."""
    with _readable(path) as stream:
        return not stream.read().strip()


def _tables(directory: Path) -> list[Path]:
    """
    Every file in a directory whose name says it is a table.

    Hidden names are not tables here, as they are not sets: an editor's
    ``.annotations.csv.swp`` or a half-copied ``.annotations.csv`` is a
    companion the directory keeps, and taking a directory down for one would
    make a stray sync file fatal.
    """
    return [
        x
        for x in _entries(directory)
        if not x.name.startswith(".") and x.suffix.casefold() in TABLE_SUFFIXES
    ]


def _refuse_stray_tables(directory: Path, known: Collection[str], what: str) -> None:
    """
    Refuse a table whose name names no part of what a directory holds.

    A ``feature.csv`` beside an ``annotations.csv`` claims to participate
    in this convention and gets it wrong, which is worth more than being
    quietly skipped.
    """
    stray = sorted(x.name for x in _tables(directory) if x.stem not in known)
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
    # A directory which states annotations is a set, and nothing below it is
    # looked at: a folder someone kept beside its tables -- a backup of an
    # attrs file, an older copy of the set -- is no more this format's
    # business than a notes.txt is, and reading the directory as a
    # collection because of one would refuse a set which is complete.
    children: list[Path] = []
    if not _states_annotations(directory):
        children = _child_sets(directory)
        # A directory which states nothing itself may still carry
        # annotations, under the hidden name, as a directory of data carries
        # its inventory. Nothing of its own is read on this path: a data
        # directory's attrs.json is about the data, and what the caller
        # states is the carried table's to take, since it states none of it.
        if not children and (carried := find_annotations(directory)) is not None:
            return _load_path(carried, dims, **kwargs)
    _refuse_stated(directory, kwargs)
    attrs = _read_attrs(directory)
    if children:
        return _load_collection(directory, children, attrs, dims, **kwargs)
    return _load_set(directory, attrs, dims, **kwargs)


def _given_attrs(kwargs: Mapping) -> Mapping:
    """Return what a caller stated for a source which states nothing itself."""
    attrs = kwargs.get("attrs")
    if attrs is None:
        return {}
    return attrs if isinstance(attrs, Mapping) else attrs.model_dump()


def _refuse_stated(directory: Path, kwargs: dict) -> None:
    """
    Refuse attributes, features or bases given for a directory which states them.

    Refused for a directory which holds a set or the sets, not for one
    carrying a bare `.annotations.csv`: that table states neither, so a
    caller has the same say over what it holds as it has passing the table
    itself. Consumed rather than merely refused, since what a directory
    states would otherwise reach `AnnotationSet` twice as a bare TypeError.
    """
    _refuse_overrides(
        f"{quote_path(directory)}, which states them",
        attrs=kwargs.pop("attrs", None),
        features=kwargs.pop("features", None),
        bases=kwargs.pop("bases", None),
    )


def _load_path(path: Path, dims, **kwargs) -> AnnotationSet:
    """Load whichever of the two shapes a path holds."""
    if path.is_dir():
        return _load_directory(path, dims, **kwargs)
    return _load_file(path, dims, **kwargs)


def _states_annotations(path: Path) -> bool:
    """
    Whether a directory states annotations of its own, in either encoding.

    A file states none: the scans below walk what a directory holds, and a
    plain file among them is not a set which spelled itself oddly.
    """
    if not path.is_dir():
        return False
    return _one_spelling(path, ANNOTATION_STEM, TABLE_SUFFIXES) is not None


def find_annotations(directory: str | os.PathLike) -> Path | None:
    """
    Return what a directory of data carries its annotations under, or None.

    Only the name is judged, never the contents, as
    [find_inventory](`dascore.core.inventory_loader.find_inventory`) judges
    the inventory's: a hidden ``.annotations/`` holds the set, or the sets,
    a directory of data was annotated with, and ``.annotations.csv`` is the
    bare-table spelling of the same thing. Hidden, so the file scanner does
    not read it as data, and so a directory which states a visible
    ``annotations.csv`` is a set rather than something carrying one.

    Raises rather than answering where a directory says two things at once
    -- both spellings present -- or where what sits under the name is the
    wrong kind of thing for it, which is a misspelling of the convention
    rather than a file which owes it nothing. `InvalidAnnotationError`, as
    `find_inventory` raises the inventory's own: this is a door callers use
    directly, so it fails the way the rest of `dc.annotations` fails.

    Parameters
    ----------
    directory
        The directory to look in.
    """
    root = Path(directory)
    tree = root / BLESSED_NAME
    named = ", ".join(tree.with_suffix(x).name for x in TABLE_SUFFIXES)
    # The suffix is matched without regard to case, as every other table
    # this format finds is: a `.annotations.PARQUET` is the file the lower
    # case name would be, and reading one and not the other would make the
    # carried name mean less than the visible one.
    tables = [
        x
        for x in _entries(root)
        if x.stem == BLESSED_NAME and x.suffix.casefold() in TABLE_SUFFIXES
    ]
    found = [x for x in (tree, *tables) if x.exists()]
    if not found:
        return None
    if len(found) > 1:
        listed = ", ".join(x.name for x in found)
        msg = (
            f"{quote_path(root)} carries annotations more than once: {listed}. "
            "A directory states what it carries once; keep the one it means."
        )
        raise InvalidAnnotationError(msg)
    (only,) = found
    if only == tree and not only.is_dir():
        msg = (
            f"{quote_path(only)} is a file. The annotations a directory carries "
            f"are the set directory {BLESSED_NAME}/, or a bare table: {named}."
        )
        raise InvalidAnnotationError(msg)
    if only != tree and only.is_dir():
        msg = (
            f"{quote_path(only)} is a directory. A set held as a directory is "
            f"named {BLESSED_NAME}/; a name with a suffix spells a bare table."
        )
        raise InvalidAnnotationError(msg)
    return only


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
    folder of figures -- as is a directory holding only features, which are
    never looked for without the annotations they belong to.
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
                f"{ANNOTATION_STEM} table, so it states no annotations."
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
    # for a child which declares none. A child which declares its own -- in
    # its attributes or above its table -- is read in those, and refuses the
    # standing-in dimensions as it would refuse them on its own, rather than
    # having them dropped where nobody can see it happen.
    default = dims if dims is not None else attrs.get("dims")
    given = "dims" if dims is not None else "the dimensions stated beside the sets"
    loaded = {}
    for child in children:
        child_attrs = _read_attrs(child)
        if _declares_dims(child, child_attrs):
            if default is not None:
                msg = (
                    f"{given} was given for {quote_path(child)}, which states its "
                    "own. Read it and change it, rather than reading it as "
                    "something else."
                )
                raise ParameterError(msg)
            loaded[child.name] = _load_set(child, child_attrs, None)
            continue
        loaded[child.name] = _load_set(child, child_attrs, default)
    return _merge_sets(loaded, attrs, **kwargs)


def _declares_dims(directory: Path, attrs: Mapping) -> bool:
    """
    Whether a stored set states its own dimensions, however it states them.

    A set declares them in its attributes or above its table, and the two
    spellings mean the same thing here: either way the set has said what it
    is read in, so the collection's own dimensions are not its to take.
    """
    if attrs.get("dims"):
        return True
    table = _one_spelling(directory, ANNOTATION_STEM, TABLE_SUFFIXES)
    return table is not None and _read_table_dims(table)[0] is not None


def _merge_sets(
    loaded: Mapping[str, AnnotationSet], attrs: Mapping, **kwargs
) -> AnnotationSet:
    """Build the one set the sets loaded together make, table by table."""
    stated = [str(x) for x in iterate(attrs.get("dims") or ())]
    # The collection's own dimensions first, then each set's in name order.
    dims = tuple(
        dict.fromkeys([*stated, *(x for one in loaded.values() for x in one.dims)])
    )
    frames = {name: _labeled(one.annotations, name) for name, one in loaded.items()}
    # Features only where a set has some, so none leaves the table bare.
    tables = {
        name: _labeled(one.features, name)
        for name, one in loaded.items()
        if len(one.features)
    }
    _refuse_undeclared_dims(loaded, frames, dims)
    document = dict(attrs)
    document["dims"] = dims
    document["sets"] = {name: one.attrs for name, one in loaded.items()}
    parts = {
        name: _Tables(frames[name], tables.get(name), one.bases)
        for name, one in loaded.items()
    }
    return _combine(parts, dims, attrs=document, **kwargs)


def _labeled(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Return one set's table, saying which set each row came from.

    Only the label is added: what the set states for itself stays in
    ``attrs.sets``, which a row reaches through the label.
    """
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
    column read as a coordinate it never claimed.
    """
    for name, frame in frames.items():
        undeclared = set(dims) - set(loaded[name].dims)
        for dim in sorted(undeclared):
            spelled = [dim, f"{dim}{_MIN}", f"{dim}{_MAX}"]
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


def _load_set(directory: Path, attrs: Mapping, dims, **kwargs) -> AnnotationSet:
    """Load the set a directory holds."""
    # A directory which states its dimensions is read in them only.
    if attrs.get("dims"):
        _refuse_overrides("a directory stating its own dimensions", dims=dims)
    _refuse_stray_tables(
        directory,
        (ANNOTATION_STEM, FEATURE_STEM),
        f"which name no part of a set. A set states {ANNOTATION_STEM} and, "
        f"where it has features, {FEATURE_STEM}, each as a "
        f"{' or a '.join(TABLE_SUFFIXES)} table.",
    )
    table = _one_spelling(directory, ANNOTATION_STEM, TABLE_SUFFIXES)
    if table is None:
        msg = (
            f"{quote_path(directory)} holds no {ANNOTATION_STEM} table and no "
            f"{BLESSED_NAME}, so it states no annotations and carries none."
        )
        raise ParameterError(msg)
    declared, skip = _read_table_dims(table)
    stated = _declared_dims(attrs, dims, directory, declared, table)
    own, inherited = _stated_dtypes(attrs, "annotation_columns")
    frame = _read_set_table(
        table,
        stated,
        "no annotations",
        ordered=True,
        skip=skip,
        text=_text_columns(own, inherited),
    )
    frame = _settle_dtypes(frame, own, inherited, table)
    features = None
    if (path := _one_spelling(directory, FEATURE_STEM, TABLE_SUFFIXES)) is not None:
        own, inherited = _stated_dtypes(attrs, "feature_columns")
        features = _read_set_table(
            path,
            (),
            "no features",
            skip=_undeclared(path),
            text=_text_columns(own, inherited),
        )
        features = _settle_dtypes(features, own, inherited, path)
    return AnnotationSet(
        frame,
        features=features,
        bases=_read_bases(directory),
        dims=stated,
        attrs=attrs,
        **kwargs,
    )


def _restore_dtypes(frame, dtypes: Mapping[str, str], path: Path):
    """
    Give each column back the dtype the set declares for it.

    A CSV states no types, so a `category` or an `Int64` column comes
    back as whatever its cells parse as; the declaration beside it is
    what says which the column holds, and the set checks it on building.
    Parquet keeps its own types, so the cast changes nothing there. Text
    is left as it arrived: every spelling of it satisfies the check, and
    casting would change which spelling a column carries. A declaration
    which is not one is left for the set's own validation to refuse.
    """
    restored = {}
    for name, dtype in dtypes.items():
        if frame is None or name not in frame.columns or _is_text_dtype(dtype):
            continue
        try:
            restored[name] = frame[name].astype(dtype)
        except (TypeError, ValueError) as error:
            msg = (
                f"The column {name!r} of {quote_path(path)} declares the dtype "
                f"{dtype}, which its cells cannot be read as: {error}."
            )
            raise ParameterError(msg) from error
    return frame.assign(**restored) if restored else frame


def _declared_dtypes(columns: Mapping | None) -> dict[str, str]:
    """The dtype each declared column states, where the declaration is one."""
    out = {}
    for name, spec in (columns or {}).items():
        dtype = (
            spec.get("dtype")
            if isinstance(spec, Mapping)
            else getattr(spec, "dtype", None)
        )
        if dtype:
            out[name] = dtype
    return out


def _text_columns(own: Mapping, inherited: Mapping) -> frozenset[str]:
    """The columns declared as text, which a table leaves as it reads them."""
    dtypes = {**{k: v for k, (v, _) in inherited.items()}, **own}
    return frozenset(k for k, v in dtypes.items() if _is_text_dtype(v))


def _stated_dtypes(attrs: Mapping, key: str, own=None) -> tuple[dict, dict]:
    """
    Return the set's own declared dtypes, and those the sets saved flat
    into it agree on, each with the names of the sets declaring it.
    Children which disagree leave the column to inference; two spellings
    of text agree.
    """
    stated = _declared_dtypes(attrs.get(key) if own is None else own)
    merged: dict[str, tuple[str, set[str]]] = {}
    clashing = set()
    for label, child in (attrs.get("sets") or {}).items():
        for name, dtype in _declared_dtypes(_child_field(child, key)).items():
            first, owners = merged.setdefault(name, (dtype, set()))
            if first != dtype and not (_is_text_dtype(first) and _is_text_dtype(dtype)):
                clashing.add(name)
            owners.add(label)
    inherited = {
        k: v for k, v in merged.items() if k not in clashing and k not in stated
    }
    return stated, inherited


def _settle_dtypes(frame, own: Mapping, inherited: Mapping, path: Path):
    """
    Restore declared dtypes. A child's declaration holds for a column only
    where every row stating it came from a child declaring it, and the
    other children's blanks fit the dtype; otherwise the column is read as
    an undeclared one would be.
    """
    if frame is None:
        return frame
    applied = dict(own)
    reread = {}
    for name, (dtype, owners) in inherited.items():
        if name not in frame.columns:
            continue
        labels = frame["set"].map(_text) if "set" in frame.columns else None
        stated = frame[name].notna()
        if labels is not None and set(labels[stated]) <= owners:
            # Other children's rows hold blanks, which the dtype must too.
            if set(labels) <= owners or _holds_blank(dtype):
                applied[name] = dtype
                continue
        if _is_text_dtype(dtype) and not _is_parquet(path):
            reread[name] = frame[name].map(_read_extra)
    frame = frame.assign(**reread) if reread else frame
    return _restore_dtypes(frame, applied, path)


def _holds_blank(dtype: str) -> bool:
    """Whether a dtype holds a missing value; int64 and bool do not."""
    try:
        return bool(pd.Series([None], dtype=dtype).isna().all())
    except (TypeError, ValueError):
        return False


def _child_field(child, key: str):
    """Read one field of a child's attributes, a document or a model."""
    return child.get(key) if isinstance(child, Mapping) else getattr(child, key, None)


def _load_file(path: Path, dims, **kwargs) -> AnnotationSet:
    """Load the set a bare table holds."""
    if path.suffix.casefold() not in TABLE_SUFFIXES:
        named = " or ".join(TABLE_SUFFIXES)
        msg = (
            f"{quote_path(path)} is not a table an annotation set is read from. "
            f"A bare set is a {named} file; a set with features is a directory."
        )
        raise ParameterError(msg)
    declared, skip = _read_table_dims(path)
    # A bare table states no attributes of its own, so a caller may hand it
    # some -- and the dimensions they name are the ones its cells are read
    # in, since nothing can be read before that is known.
    given = _given_attrs(kwargs)
    stated = _declared_dims(given, dims, path, declared, path)
    # An empty mapping is an override which clears the declarations, as
    # the set reads it, so only an absent one falls back to the attrs.
    columns = kwargs.get("annotation_columns")
    if columns is None:
        columns = given.get("annotation_columns")
    own, inherited = _stated_dtypes(given, "annotation_columns", own=columns or {})
    frame = _read_set_table(
        path,
        stated,
        "no annotations",
        ordered=True,
        skip=skip,
        text=_text_columns(own, inherited),
    )
    frame = _settle_dtypes(frame, own, inherited, path)
    return AnnotationSet(frame, dims=stated, **kwargs)


def _undeclared(path: Path) -> int:
    """
    Refuse a features table which declares dimensions, and return the lines
    above its header -- none, since it declares none.

    Features hold no coordinates, so a declaration there is misplaced.
    """
    declared, skip = _read_table_dims(path)
    if declared is not None:
        where = (
            f"states {DIMS_KEY}"
            if _is_parquet(path)
            else f"declares {_DIMS_PRAGMA} above its header"
        )
        msg = (
            f"{quote_path(path)} {where}. Features hold no coordinates; the "
            "set states its dimensions once, with its annotations."
        )
        raise ParameterError(msg)
    return skip


def _declared_dims(
    attrs: Mapping,
    dims,
    source: Path,
    declared: tuple[str, ...] | None = None,
    table: Path | None = None,
) -> tuple[str, ...]:
    """
    Return the dimensions to read a source in: the caller's, else its own.

    The cells cannot be read before this is known -- which columns hold
    times rather than text is exactly what a dimension decides -- so a
    source stating none fails here rather than as a puzzling column later.

    Where a table declares them above its header, restating them is allowed
    if the two agree and refused if they differ: there is no precedence rule
    between two spellings of one fact.
    """
    stated = dims if dims is not None else attrs.get("dims")
    if declared is not None:
        if stated is None:
            stated = declared
        elif tuple(str(x) for x in iterate(stated)) != declared:
            spelled = "was given" if dims is not None else "is stated in its attributes"
            named = table or source
            where = f"in its {DIMS_KEY}" if _is_parquet(named) else "above its header"
            msg = (
                f"{quote_path(named)} declares the dimensions "
                f"{', '.join(declared)} {where}, but "
                f"{', '.join(str(x) for x in iterate(stated))} {spelled}. "
                "Dimensions may be spelled twice where the two agree."
            )
            raise ParameterError(msg)
    if not stated:
        msg = (
            f"{quote_path(source)} states no dimensions, and none were given. "
            "Annotations are read in the dimensions they are stated in: write "
            f"them in {ATTRS_STEM}{OBJECT_SUFFIXES[0]}, in a "
            f"'{_COMMENT} {_DIMS_PRAGMA}: distance, time' line above the table, "
            "or pass dims=('distance', 'time')."
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

    Accept an existing set, dataframe-compatible data, a table, a set directory, a
    collection directory, or a data directory carrying ``.annotations``.

    Parameters
    ----------
    source
        An `AnnotationSet`, a dataframe of one row per annotation, or a path
        to a CSV table, a set directory, a directory of set directories, or a
        directory of data carrying either under ``.annotations``.
    dims
        The patch dimensions the annotations are stated in. Required unless
        the source states them itself -- in its attributes, or in a
        ``# dims: distance, time`` line above its table. A source which
        states them takes them again only if the two agree.
    **kwargs
        Passed to [`AnnotationSet`](`dascore.core.annotations.AnnotationSet`).
        A source already holding what one states -- a set, or a directory
        holding its own attributes, features or bases -- refuses it rather than
        dropping it.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> frame = pd.DataFrame({"phase": ["P"], "distance": [10.0]})
    >>> picks = dc.annotations(frame, dims=("distance",))
    >>> len(picks)
    1

    Existing sets are returned unchanged, so callers can accept sets or paths.

    >>> dc.annotations(picks) is picks
    True

    Sets stored side by side read as one, each row saying which set it came
    from.

    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder) / "sets"
    ...     _ = picks.io.save(root / "hand")
    ...     _ = picks.io.save(root / "phasenet")
    ...     together = dc.annotations(root)
    >>> len(together), sorted(together.annotations["set"])
    (2, ['hand', 'phasenet'])

    A directory of data is read as the annotations it carries, so the
    directory a spool was opened on is a path this takes too.

    >>> with tempfile.TemporaryDirectory() as folder:
    ...     data = Path(folder)
    ...     _ = picks.io.save(data / ".annotations")
    ...     carried = dc.annotations(data)
    >>> carried == picks
    True
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
            if path.exists():
                return _load_path(path, dims, **kwargs)
        # ValidationError too: a stored document which does not build the
        # models is a bad file, and it is named as one here rather than
        # arriving as pydantic's report on a call the caller did not make.
        except (ParameterError, ValidationError) as error:
            raise InvalidAnnotationError(str(error)) from error
        msg = f"{quote_path(path)} does not exist, so it holds no annotations."
        raise InvalidAnnotationError(msg)
    return AnnotationSet(source, dims=dims, **kwargs)
