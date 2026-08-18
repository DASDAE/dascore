"""
Utilities for reading strict CSV tables.

A table written by hand is read the way it was written: every cell arrives
as text, the reader refuses a row which is not its header wide, and the
value of a cell is decided by what the cell says rather than by whatever
pandas inferred from the rows it happened to see.

These raise `ParameterError` and name the file with
[quote_path](`dascore.utils.paths.quote_path`); a format which wants its own
error type wraps them once, at whatever boundary reads its tables.
"""

from __future__ import annotations

import csv
import datetime
import json
from collections.abc import Mapping, Sized
from pathlib import Path

import numpy as np
import pandas as pd

from dascore.exceptions import ParameterError
from dascore.utils.misc import optional_import, to_str
from dascore.utils.paths import quote_path
from dascore.utils.time import to_datetime64

# The metadata key a parquet file names its document columns in.
DOCUMENT_KEY = "dascore:documents"


def read_table(path: Path, what: str = "nothing", skip: int = 0) -> pd.DataFrame:
    r"""
    Read one strict CSV table.

    Every cell arrives as text and the caller coerces it, so a column's
    meaning is the field's rather than whatever pandas inferred from the
    rows it happened to see. Only a truly empty cell is null: an empty
    cell means unset, and a document which writes ``NA`` means the string.

    Parameters
    ----------
    path
        The CSV file to read.
    what
        What a table with no columns fails to state, for that error message.
    skip
        Lines above the header, for a format which states something of its
        own before its table. Read past before the table is parsed at all,
        so nothing downstream has to agree about what those lines were. Row
        numbers in errors still count from the top of the file, so they name
        the line a reader would look at.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from dascore.utils.tables import read_table
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "coupling.csv"
    ...     _ = path.write_text("group,value\nrail,true\n")
    ...     frame = read_table(path)
    >>> list(frame.columns), frame["value"][0]
    (['group', 'value'], 'true')
    """
    # The header is read first and by itself, for two reasons: pandas
    # renames a repeated column rather than refusing it, so by the time a
    # frame exists the second one is `coupling_type.1` and the clash cannot
    # be seen; and it raises its own error for a file with no columns,
    # which would arrive before this one could say what was expected.
    #
    # One open, which pandas then reads on from: `skiprows` would have it
    # skip *rows*, and a quote in a skipped line can make one row span the
    # rest of the file, leaving pandas a different header than was checked
    # here. Both readers therefore also decode alike, which a locale-encoded
    # read or a byte order mark reaching only one of them would break.
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for _ in range(skip):
                stream.readline()
            position = stream.tell()
            reader = csv.reader(stream)
            header = next(reader, [])
            if header:
                # Streamed rather than listed: a table is the part of this
                # format meant to grow, and holding every cell as a python
                # object beside the frame pandas builds would cost several
                # times what the frame itself does.
                _check_widths(reader, header, path, start=skip + 2)
            _check_header(header, path, what)
            stream.seek(position)
            # index_col=False so that no column is ever read as an index; the
            # row widths above already agree, and this keeps them agreeing.
            return pd.read_csv(
                stream,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
                index_col=False,
            )
    # csv.Error too: a cell longer than csv.field_size_limit stops the
    # header scan, and without this it would leave this function as a bare
    # _csv.Error rather than as whatever the caller's format raises.
    except (OSError, UnicodeDecodeError, csv.Error) as read_error:
        msg = f"Could not read {quote_path(path)}: {read_error}."
        raise ParameterError(msg) from read_error


def _check_header(header: list[str], path: Path, what: str) -> None:
    """Refuse a table with no columns, or one which names a column twice."""
    if not header:
        msg = f"{quote_path(path)} has no columns, so it states {what}."
        raise ParameterError(msg)
    repeated = sorted({x for x in header if header.count(x) > 1})
    if repeated:
        msg = (
            f"{quote_path(path)} names {', '.join(repeated)} more than once; one "
            "column states one field."
        )
        raise ParameterError(msg)


def _check_widths(reader, header: list[str], path: Path, start: int = 2) -> None:
    """
    Refuse a row which is not its header wide.

    Pandas refuses neither a wide row nor a narrow one: by default the
    surplus cell pushes the first column into the index, so every value in
    the row shifts one field left and lands in its neighbour's meaning. A
    row states one cell per column or it is not a row.
    """
    for number, row in enumerate(reader, start=start):
        if row and len(row) != len(header):
            msg = (
                f"{quote_path(path)} row {number} states {len(row)} cells where "
                f"its header names {len(header)} columns."
            )
            raise ParameterError(msg)


def write_parquet(frame: pd.DataFrame, path, metadata: Mapping | None = None) -> None:
    """
    Write a dataframe as one parquet file, keeping the values it holds.

    Parquet stores types, so a column comes back as what it was written as
    rather than as text a reader has to guess at. A column with no single
    type -- one holding both text and booleans, or a model, or a nested
    mapping -- has no parquet type of its own; each of its cells is written
    as a JSON document instead, and the file names those columns in its
    metadata so a reader gets the values back rather than their spelling.

    Parameters
    ----------
    frame
        The table to write.
    path
        Where to write it.
    metadata
        Key-value strings for the file's own metadata, which
        [read_parquet](`dascore.utils.tables.read_parquet`) hands back. A
        format states here what its table cannot say in a column.

    Examples
    --------
    Needs pyarrow, which the doctest run does not have, so this one is not
    executed; `tests/test_utils/test_tables.py` runs the same round trip.

    >>> import tempfile  # doctest: +SKIP
    >>> from pathlib import Path
    >>> import pandas as pd
    >>> from dascore.utils.tables import read_parquet, write_parquet
    >>> frame = pd.DataFrame({"group": ["rail"], "value": [True]})
    >>> with tempfile.TemporaryDirectory() as folder:  # doctest: +SKIP
    ...     path = Path(folder) / "coupling.parquet"
    ...     write_parquet(frame, path, {"dascore:dims": "distance"})
    ...     out, stated = read_parquet(path)
    >>> out.equals(frame), stated["dascore:dims"]  # doctest: +SKIP
    (True, 'distance')
    """
    write_parquet_table(parquet_table(frame, metadata), path)


def parquet_table(frame: pd.DataFrame, metadata: Mapping | None = None):
    """
    Return a dataframe as the parquet table it is written from.

    Spelled out before anything is written, so a caller storing several
    tables at once can prepare them all before it touches the directory.
    See [write_parquet](`dascore.utils.tables.write_parquet`), which is
    this and the write together.
    """
    arrow = optional_import("pyarrow", required_for="parquet tables")
    spelled, documents = {}, []
    for label, name in zip(frame.columns, _named(frame.columns), strict=True):
        series = frame[label]
        if _one_type(series):
            spelled[name] = series
            continue
        documents.append(name)
        spelled[name] = series.map(_document)
    stated = {str(k): str(v) for k, v in (metadata or {}).items()}
    if reserved := sorted(x for x in stated if _is_reserved(x)):
        msg = (
            f"The metadata key(s) {', '.join(reserved)} are not a caller's to "
            f"state: {DOCUMENT_KEY} is what this writer names its own document "
            "columns in, and pandas and ARROW: are what the file already holds "
            "for the reader which wrote them."
        )
        raise ParameterError(msg)
    if documents:
        stated[DOCUMENT_KEY] = json.dumps(documents)
    table = arrow.Table.from_pandas(
        pd.DataFrame(spelled, index=frame.index), preserve_index=False
    )
    # Added to what pyarrow wrote rather than replacing it: the pandas key
    # it puts there is what a pandas reader uses to rebuild an index and the
    # dtypes it can, and dropping it would make this file say less than
    # pyarrow wrote.
    kept = {**(table.schema.metadata or {}), **stated}
    return table.replace_schema_metadata(kept)


def write_parquet_table(table, path) -> None:
    """Write a prepared parquet table, as `parquet_table` returns it."""
    parquet = optional_import("pyarrow.parquet", required_for="parquet tables")
    parquet.write_table(table, path)


def read_parquet(
    path, what: str = "nothing", empty: bool = False
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Read one parquet file, and whatever it states about itself.

    Returns the table and its metadata, with the columns
    [write_parquet](`dascore.utils.tables.write_parquet`) stored as JSON
    documents read back as the values they hold.

    Parameters
    ----------
    path
        The parquet file to read.
    what
        What a table with no columns fails to state, for that error message.
    empty
        Whether a table with no columns is allowed, for a format in which
        that is how something holding nothing is written.
    """
    parquet = optional_import("pyarrow.parquet", required_for="parquet tables")
    try:
        table = parquet.read_table(path)
        stated = _stated_metadata(table.schema.metadata)
        # Inside the block as well: pandas metadata another writer left is
        # read here, and a malformed one raises where nothing else would
        # name the file it came from.
        frame = table.to_pandas()
    except Exception as error:
        # Any error pyarrow raises: it reports a truncated file, an
        # unreadable one and something which is not parquet at all through
        # several types of its own, and the caller's format names them all
        # the same way.
        msg = f"Could not read {quote_path(path)}: {error}."
        raise ParameterError(msg) from error
    if not len(frame.columns) and not empty:
        msg = f"{quote_path(path)} has no columns, so it states {what}."
        raise ParameterError(msg)
    for name in _document_columns(stated.pop(DOCUMENT_KEY, "[]"), path):
        if name not in frame.columns:
            msg = (
                f"{quote_path(path)} names {name!r} as a column of documents, "
                "and holds no such column."
            )
            raise ParameterError(msg)
        # Held as object: the cells are whatever their documents state, and
        # letting pandas re-infer a type from them would hand back a column
        # of a type the file never said it had.
        read = [_read_document(x, name, path) for x in frame[name]]
        frame[name] = pd.Series(read, index=frame.index, dtype=object)
    return frame, stated


def read_parquet_metadata(path) -> dict[str, str]:
    """
    Return what a parquet file states about itself, reading no rows.

    The footer alone, so a caller which needs what a file says before it
    decides how to read the table -- which dimensions its columns are in,
    say -- does not pay for the table to find out.

    Parameters
    ----------
    path
        The parquet file to read.
    """
    parquet = optional_import("pyarrow.parquet", required_for="parquet tables")
    try:
        schema = parquet.read_schema(path)
    except Exception as error:
        msg = f"Could not read {quote_path(path)}: {error}."
        raise ParameterError(msg) from error
    return _stated_metadata(schema.metadata)


def _document_columns(stated: str, path) -> list[str]:
    """Read the columns a file names as documents, refusing what it cannot mean."""
    try:
        names = json.loads(stated)
    except ValueError as error:
        msg = (
            f"{quote_path(path)} states {DOCUMENT_KEY} as {stated!r}, which is "
            f"not a JSON document: {error}."
        )
        raise ParameterError(msg) from error
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        msg = (
            f"{quote_path(path)} states {DOCUMENT_KEY} as {stated!r}; it names "
            "the columns which hold documents, so it is a list of names."
        )
        raise ParameterError(msg)
    return names


def _named(columns) -> list[str]:
    """
    Return a table's column names as parquet holds them, which is as text.

    A frame may label a column with anything hashable and a CSV writes
    whatever that prints as, but arrow takes names only as strings -- so
    they are spelled here rather than left to fail deep inside a conversion.
    Two labels which spell alike would name one column, which is the same
    refusal a frame naming a column twice already gets.
    """
    named = [str(x) for x in columns]
    if len(set(named)) != len(named):
        repeated = sorted({x for x in named if named.count(x) > 1})
        msg = (
            f"The column(s) {', '.join(repeated)} are named more than once once "
            "their names are spelled as text, which is how parquet holds them; "
            "one column states one thing."
        )
        raise ParameterError(msg)
    return named


def _is_reserved(key: str) -> bool:
    """Whether a metadata key names something the file states for itself."""
    return key == DOCUMENT_KEY or key == "pandas" or key.startswith("ARROW:")


def _one_type(series: pd.Series) -> bool:
    """Whether a column holds one type parquet has a column shape for."""
    if series.dtype != object:
        return True
    # Text is the one type an object column may still hold: pandas gives
    # some string columns object dtype, and a whole column of text is
    # exactly what parquet stores as a string column.
    return all(isinstance(x, str) for x in series if _is_stated(x))


def _is_stated(value) -> bool:
    """Whether a cell states anything, whatever kind of thing it holds."""
    if value is None:
        return False
    if isinstance(value, Sized) and not isinstance(value, str):
        # A container states itself; asking pandas whether it is null
        # answers once per element, which has no truth value.
        return True
    return not pd.isna(value)


def _document(value):
    """Spell one cell of a column parquet has no single type for."""
    if not _is_stated(value):
        return None
    return json.dumps(_documented(value), default=str)


def _documented(value):
    """
    Return a value as the json types it is made of.

    A missing value nested inside one becomes null rather than the text of
    whatever spelling of missing it was: `pd.NA` written as "<NA>" would
    come back as a string a reader takes for a value.

    Numpy's scalars are the ones that matter: `np.int64` is not an `int`
    and `np.bool_` is not a `bool`, so json falls back to spelling them as
    text -- and a column of numbers written by numpy would come back as a
    column of strings, which is exactly what this encoding exists not to
    do. A time has no json type at all and is spelled as DASCore spells
    every time, which reads back as one.

    A value none of this can spell -- a `Decimal`, a `bytes`, an object of
    someone's own class -- is written as its text, which is what the CSV
    encoding does with it too. That is the one place this loses a type, and
    the reason a column parquet has a type for is never sent this way.
    """
    if not _is_stated(value):
        return None
    # At nanoseconds, whatever resolution the value arrived in: a set holds
    # its times at nanoseconds, and one spelling per instant is what makes
    # two writes of one set the same file.
    if isinstance(value, np.datetime64):
        return to_str(np.datetime64(value, "ns"))
    if isinstance(value, np.timedelta64):
        return to_str(np.timedelta64(value, "ns"))
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, np.generic):
        return _documented(value.item())
    if isinstance(value, datetime.datetime | datetime.date | pd.Timestamp):
        return _documented(to_datetime64(value))
    if isinstance(value, Mapping):
        return {str(k): _documented(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset | np.ndarray):
        return [_documented(x) for x in value]
    return value


def _read_document(cell, name: str, path) -> object:
    """Read one cell of a column the file names as documents."""
    if not isinstance(cell, str):
        # Every spelling of a null becomes the one which was written: a
        # column of documents holds values, and None is the value pandas
        # hands back for a cell parquet stored as null.
        return None if not _is_stated(cell) else cell
    try:
        return json.loads(cell)
    except ValueError as error:
        msg = (
            f"The column {name!r} of {quote_path(path)} holds "
            f"{cell!r}, which is not a JSON document: {error}."
        )
        raise ParameterError(msg) from error


def _stated_metadata(metadata) -> dict[str, str]:
    """Return the key-value metadata a file states, as text."""
    out = {}
    for key, value in (metadata or {}).items():
        # Decoded leniently, keys as well as values: this is another
        # writer's metadata as often as it is ours, and a key which is not
        # UTF-8 is something to report rather than something to hide.
        name = key.decode(errors="replace")
        # What pyarrow writes for itself, which is not the caller's to read.
        if name == "pandas" or name.startswith("ARROW:"):
            continue
        out[name] = value.decode(errors="replace")
    return out


def row_cells(row) -> dict[str, str]:
    """
    Return a row's stated cells, an empty one meaning unset.

    Examples
    --------
    >>> import pandas as pd
    >>> from dascore.utils.tables import row_cells
    >>> frame = pd.DataFrame({"group": ["rail"], "value": [None]})
    >>> row_cells(frame.iloc[0])
    {'group': 'rail'}
    """
    return {str(k): v for k, v in row.items() if not pd.isnull(v)}


def require_columns(frame: pd.DataFrame, needed, path: Path) -> None:
    """
    Refuse a table which does not carry a column it is read by.

    Parameters
    ----------
    frame
        The table to check.
    needed
        The column names required; a None is ignored, so a caller may pass
        an optional column straight through.
    path
        The file the table was read from, for the error message.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> from dascore.utils.tables import require_columns
    >>> frame = pd.DataFrame({"sequence": [1]})
    >>> require_columns(frame, ["sequence", None], Path("t/x.csv")) is None
    True
    """
    missing = [x for x in needed if x is not None and x not in frame.columns]
    if missing:
        msg = (
            f"{quote_path(path)} states no {', '.join(missing)} column, which its "
            "rows are read by."
        )
        raise ParameterError(msg)


def require_stated(frame: pd.DataFrame, needed, path: Path) -> None:
    """
    Refuse a blank cell in a column the table is read by.

    A column which orders or groups the rows decides where each one goes,
    so a row leaving it empty has no place. Left to pandas the row would
    simply disappear -- a null sorts last, and a null grouping key drops
    its row from every group.

    Parameters
    ----------
    frame
        The table to check.
    needed
        The column names which must be stated by every row; a None is
        ignored.
    path
        The file the table was read from, for the error message.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> from dascore.utils.tables import require_stated
    >>> frame = pd.DataFrame({"sequence": [1, 2]})
    >>> require_stated(frame, ["sequence"], Path("t/x.csv")) is None
    True
    """
    for column in needed:
        if column is None:
            continue
        empty = [
            str(n) for n, ok in enumerate(frame[column].notna(), start=2) if not ok
        ]
        if empty:
            msg = (
                f"{quote_path(path)} leaves {column} empty at row(s) "
                f"{', '.join(empty)}, so those rows state no place."
            )
            raise ParameterError(msg)


def ordered_rows(frame: pd.DataFrame, column: str | None, path: Path) -> pd.DataFrame:
    """
    Return the rows in the order the named column states, if any.

    A table which names one is read by it rather than by row position, so
    re-sorting a spreadsheet cannot change what it means. A table which
    names none keeps the order it was written in.

    Parameters
    ----------
    frame
        The table to order.
    column
        The numeric column stating the order, or None to keep the written
        order.
    path
        The file the table was read from, for the error message.

    Examples
    --------
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> from dascore.utils.tables import ordered_rows
    >>> frame = pd.DataFrame({"sequence": ["2", "1"], "name": ["b", "a"]})
    >>> list(ordered_rows(frame, "sequence", Path("t/x.csv"))["name"])
    ['a', 'b']
    """
    if column is None:
        return frame
    try:
        keys = pd.to_numeric(frame[column])
    except (TypeError, ValueError) as convert_error:
        msg = f"{quote_path(path)} has a non-numeric {column}: {convert_error}."
        raise ParameterError(msg) from convert_error
    return frame.assign(**{column: keys}).sort_values(column, kind="stable")


def parse_cell(text: str):
    """
    Read a cell's value the way its own text states it.

    A CSV has no types, so a value which may be a string, a boolean or a
    number is decided by what was written. A value which is genuinely a
    string but looks like one of the others is the one thing this spelling
    cannot express; that value is authored in YAML, where the types are
    explicit.

    Examples
    --------
    >>> from dascore.utils.tables import parse_cell
    >>> parse_cell("True"), parse_cell("1e3"), parse_cell("1.5"), parse_cell("car")
    (True, 1000, 1.5, 'car')
    """
    if (folded := text.strip().casefold()) in ("true", "false"):
        return folded == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    # int(number) rather than int(text): 1e3 is integral, and only the
    # number knows that -- the text raises.
    return int(number) if number.is_integer() and "." not in text else number
