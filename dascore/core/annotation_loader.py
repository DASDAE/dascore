"""
Read annotation sets from storage.

A set is stored either as a directory naming what it holds -- ``attrs``
stating the dimensions and provenance, ``annotations.csv`` holding one row
per annotation, and ``vertices.csv`` where any path or polygon needs one --
or as a bare table whose dimensions the caller states.

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
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dascore.core.annotations import (
    _END,
    _START,
    _VERTEX_COLUMNS,
    ANNOTATION_STEM,
    ATTRS_STEM,
    TABLE_SUFFIX,
    VERTEX_STEM,
    AnnotationSet,
)
from dascore.exceptions import InvalidAnnotationError, ParameterError
from dascore.models.registry import TAG_FIELD
from dascore.utils.misc import optional_import
from dascore.utils.paths import quote_path
from dascore.utils.tables import parse_cell, read_table
from dascore.utils.time import to_datetime64

# The spellings an attrs file takes; the table stems come from the set,
# which writes the names this reads.
_OBJECT_SUFFIXES = (".yaml", ".yml", ".json")

# What an attrs file declares itself to be; the model writes its own tag.
_SET_TAG = "AnnotationSetAttrs"

# Columns whose cells stay text however they are spelled: an id which
# looks like a number is still the label the vertices name it by.
_TEXT_COLUMNS = frozenset(
    {"id", "group", "tags", "parent", "geometry", "acquisition_key"}
)

# The column a vertex states its place in the order by; a number.
_ORDINAL = _VERTEX_COLUMNS[1]


def _read_object(path: Path) -> dict[str, Any]:
    """Parse one YAML or JSON object file into a mapping."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        msg = f"Could not read {quote_path(path)}: {error}."
        raise ParameterError(msg) from error
    if path.suffix == ".json":
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


def _one_spelling(directory: Path, stem: str, suffixes: Sequence[str]) -> Path | None:
    """Return the one file a stem names, or None; two spellings raise."""
    found = [x for x in (directory / f"{stem}{y}" for y in suffixes) if x.exists()]
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
    path = _one_spelling(directory, ATTRS_STEM, _OBJECT_SUFFIXES)
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


def _read_cells(frame: pd.DataFrame, dims: Sequence[str], path: Path) -> pd.DataFrame:
    """Read a table's text cells as the values each column holds."""
    spellings = _dimension_spellings(dims)
    out = {}
    for name in frame.columns:
        series = frame[name]
        if str(name) in spellings:
            out[name] = _read_dimension(series, path)
        elif str(name) == _ORDINAL:
            out[name] = _read_ordinal(series, path)
        elif str(name) == "basis":
            out[name] = _read_basis(series, path)
        elif str(name) in _TEXT_COLUMNS:
            out[name] = series
        else:
            out[name] = series.map(lambda x: parse_cell(x) if isinstance(x, str) else x)
    return pd.DataFrame(out)


def _read_set_table(path: Path, dims: Sequence[str], what: str) -> pd.DataFrame | None:
    """
    Read one of a set's tables, with its cells typed.

    A table stating nothing at all is what a set of no annotations writes,
    so it reads back as none rather than as a table with no columns.
    """
    if _is_blank(path):
        return None
    return _read_cells(read_table(path, what=what), dims, path)


def _is_blank(path: Path) -> bool:
    """Whether a table holds nothing but whitespace."""
    try:
        return not path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        # Unreadable is the table reader's to name, with its own message.
        return False


def _refuse_stray_tables(directory: Path) -> None:
    """
    Refuse a table whose name names no part of a set.

    A ``vertexes.csv`` beside an ``annotations.csv`` claims to participate
    in this convention and gets it wrong, which is worth more than being
    quietly skipped.
    """
    known = {ANNOTATION_STEM, VERTEX_STEM}
    stray = sorted(
        x.name for x in directory.glob(f"*{TABLE_SUFFIX}") if x.stem not in known
    )
    if stray:
        msg = (
            f"{quote_path(directory)} holds the table(s) {', '.join(stray)}, which "
            f"name no part of a set. A set states {ANNOTATION_STEM}{TABLE_SUFFIX} "
            f"and, where it has vertices, {VERTEX_STEM}{TABLE_SUFFIX}."
        )
        raise ParameterError(msg)


def _load_directory(directory: Path, dims, **kwargs) -> AnnotationSet:
    """Load the set a directory holds."""
    attrs = _read_attrs(directory)
    _refuse_stray_tables(directory)
    table = directory / f"{ANNOTATION_STEM}{TABLE_SUFFIX}"
    if not table.exists():
        msg = (
            f"{quote_path(directory)} holds no {ANNOTATION_STEM}{TABLE_SUFFIX}, "
            "so it states no annotations."
        )
        raise ParameterError(msg)
    stated = _declared_dims(attrs, dims, directory)
    frame = _read_set_table(table, stated, "no annotations")
    vertex_path = directory / f"{VERTEX_STEM}{TABLE_SUFFIX}"
    vertices = None
    if vertex_path.exists():
        vertices = _read_set_table(vertex_path, stated, "no vertices")
    return AnnotationSet(frame, dims=dims, vertices=vertices, attrs=attrs, **kwargs)


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
    Return the dimensions a source states, from its attrs or the caller.

    The cells cannot be read before this is known -- which columns hold
    times rather than text is exactly what a dimension decides -- so a
    source stating none fails here rather than as a puzzling column later.
    """
    stated = dims if dims is not None else attrs.get("dims")
    if not stated:
        msg = (
            f"{quote_path(source)} states no dimensions, and none were given. "
            "Annotations are read in the dimensions they are stated in: write "
            f"them in {ATTRS_STEM}{_OBJECT_SUFFIXES[0]} or pass "
            "dims=('distance', 'time')."
        )
        raise ParameterError(msg)
    return tuple(str(x) for x in stated)


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
        An `AnnotationSet`, a path to a set directory or a CSV table, or a
        dataframe of one row per annotation.
    dims
        The patch dimensions the annotations are stated in. Required unless
        the source states them itself.
    **kwargs
        Passed to [`AnnotationSet`](`dascore.core.annotations.AnnotationSet`).

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> frame = pd.DataFrame({"group": ["event"], "distance": [10.0]})
    >>> picks = dc.annotations(frame, dims=("distance",))
    >>> len(picks)
    1
    """
    if isinstance(source, AnnotationSet):
        return source
    if isinstance(source, str | os.PathLike):
        path = Path(source)
        try:
            if path.is_dir():
                return _load_directory(path, dims, **kwargs)
            if path.exists():
                return _load_file(path, dims, **kwargs)
        except ParameterError as error:
            raise InvalidAnnotationError(str(error)) from error
        msg = f"{quote_path(path)} does not exist, so it holds no annotations."
        raise InvalidAnnotationError(msg)
    return AnnotationSet(source, dims=dims, **kwargs)
