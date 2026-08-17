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
from pathlib import Path

import pandas as pd

from dascore.exceptions import ParameterError
from dascore.utils.paths import quote_path


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
