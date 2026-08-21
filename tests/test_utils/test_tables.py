"""Tests for the strict CSV table reader."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from dascore.core.annotations import Line
from dascore.exceptions import ParameterError
from dascore.utils.tables import (
    DOCUMENT_KEY,
    ordered_rows,
    parquet_table,
    parse_cell,
    read_parquet,
    read_parquet_metadata,
    read_table,
    require_columns,
    require_stated,
    row_cells,
    write_parquet,
)
from dascore.utils.time import to_datetime64, to_timedelta64

try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None


def _write(path, text: str, name: str = "table.csv", encoding: str = "utf-8"):
    """Write a table and return its path."""
    out = path / name
    out.write_text(text, encoding=encoding)
    return out


class TestReadTable:
    """Reading is strict about shape and indifferent to pandas' inference."""

    def test_every_cell_is_text(self, tmp_path):
        """A column's type is the caller's business, not pandas' guess."""
        path = _write(tmp_path, "sequence,value\n1,2.0\n")
        frame = read_table(path)
        assert list(frame["sequence"]) == ["1"]
        assert list(frame["value"]) == ["2.0"]

    def test_only_empty_is_null(self, tmp_path):
        """A cell saying NA means the string; a blank one means unset."""
        path = _write(tmp_path, "a,b\nNA,\n")
        frame = read_table(path)
        assert frame["a"][0] == "NA"
        assert pd.isnull(frame["b"][0])

    def test_byte_order_mark(self, tmp_path):
        """A mark written by a spreadsheet is not part of the first header."""
        path = _write(tmp_path, "a,b\n1,2\n", encoding="utf-8-sig")
        assert list(read_table(path).columns) == ["a", "b"]

    def test_no_columns_refused(self, tmp_path):
        """An empty file states nothing."""
        path = _write(tmp_path, "")
        with pytest.raises(ParameterError, match="no columns, so it states nothing"):
            read_table(path)

    def test_no_columns_names_what_is_missing(self, tmp_path):
        """The caller's format names what a column-less file fails to state."""
        path = _write(tmp_path, "")
        with pytest.raises(ParameterError, match="states no track"):
            read_table(path, what="no track")

    def test_repeated_column_refused(self, tmp_path):
        """One column states one field; pandas would silently rename it."""
        path = _write(tmp_path, "a,a\n1,2\n")
        with pytest.raises(ParameterError, match="more than once"):
            read_table(path)

    def test_wide_row_refused(self, tmp_path):
        """A surplus cell would shift every value one field left."""
        path = _write(tmp_path, "a,b\n1,2,3\n")
        with pytest.raises(ParameterError, match="3 cells"):
            read_table(path)

    def test_narrow_row_refused(self, tmp_path):
        """A missing cell is refused rather than filled in."""
        path = _write(tmp_path, "a,b\n1\n")
        with pytest.raises(ParameterError, match="1 cells"):
            read_table(path)

    def test_missing_file(self, tmp_path):
        """An unreadable file raises the caller's error, not OSError."""
        with pytest.raises(ParameterError, match="Could not read"):
            read_table(tmp_path / "absent.csv")

    def test_oversized_cell_refused(self, tmp_path):
        """A cell past the csv module's limit stops the scan, not the caller."""
        limit = csv.field_size_limit()
        path = _write(tmp_path, "a,b\n" + "x" * (limit + 1) + ",2\n")
        with pytest.raises(ParameterError, match="field larger than field limit"):
            read_table(path)

    def test_skipped_lines_are_not_the_header(self, tmp_path):
        """A format may state something of its own above its table."""
        path = _write(tmp_path, "# dims: time\na,b\n1,2\n")
        frame = read_table(path, skip=1)
        assert list(frame.columns) == ["a", "b"]

    def test_skipped_lines_still_count(self, tmp_path):
        """A row is named by the line a reader would look at."""
        path = _write(tmp_path, "# dims: time\na,b\n1,2,3\n")
        with pytest.raises(ParameterError, match="row 3"):
            read_table(path, skip=1)


class TestRowCells:
    """Only stated cells are reported."""

    def test_unset_dropped(self):
        """A blank cell means unset rather than a null value."""
        frame = pd.DataFrame({"a": ["x"], "b": [None]})
        assert row_cells(frame.iloc[0]) == {"a": "x"}


class TestRequireColumns:
    """A table must carry the columns it is read by."""

    def test_present(self, tmp_path):
        """Nothing happens when every needed column is there."""
        frame = pd.DataFrame({"sequence": ["1"]})
        assert require_columns(frame, ["sequence"], tmp_path / "t.csv") is None

    def test_none_ignored(self, tmp_path):
        """A None names no column, so it is not missing."""
        frame = pd.DataFrame({"sequence": ["1"]})
        assert require_columns(frame, [None], tmp_path / "t.csv") is None

    def test_missing_refused(self, tmp_path):
        """A missing column is named in the error."""
        frame = pd.DataFrame({"a": ["1"]})
        with pytest.raises(ParameterError, match="no sequence column"):
            require_columns(frame, ["sequence"], tmp_path / "t.csv")


class TestRequireStated:
    """A row leaving an ordering or grouping cell blank has no place."""

    def test_stated(self, tmp_path):
        """Nothing happens when every row states the column."""
        frame = pd.DataFrame({"sequence": ["1", "2"]})
        assert require_stated(frame, ["sequence"], tmp_path / "t.csv") is None

    def test_none_ignored(self, tmp_path):
        """A None names no column, so nothing is required."""
        frame = pd.DataFrame({"sequence": ["1"]})
        assert require_stated(frame, [None], tmp_path / "t.csv") is None

    def test_blank_refused(self, tmp_path):
        """The refused row is named by its line in the file."""
        frame = pd.DataFrame({"sequence": ["1", None, "3"]})
        with pytest.raises(ParameterError, match="row\\(s\\) 3"):
            require_stated(frame, ["sequence"], tmp_path / "t.csv")


class TestOrderedRows:
    """Rows are read in the order their column states."""

    def test_sorted_numerically(self, tmp_path):
        """Text digits order as numbers, not as strings."""
        frame = pd.DataFrame({"sequence": ["10", "2"], "name": ["b", "a"]})
        out = ordered_rows(frame, "sequence", tmp_path / "t.csv")
        assert list(out["name"]) == ["a", "b"]

    def test_stable(self, tmp_path):
        """Rows sharing a key keep the order they were written in."""
        frame = pd.DataFrame({"sequence": ["1", "1"], "name": ["b", "a"]})
        out = ordered_rows(frame, "sequence", tmp_path / "t.csv")
        assert list(out["name"]) == ["b", "a"]

    def test_no_column_keeps_order(self, tmp_path):
        """A table naming no ordering column is left as written."""
        frame = pd.DataFrame({"name": ["b", "a"]})
        out = ordered_rows(frame, None, tmp_path / "t.csv")
        assert list(out["name"]) == ["b", "a"]

    def test_non_numeric_refused(self, tmp_path):
        """An ordering column which is not numeric orders nothing."""
        frame = pd.DataFrame({"sequence": ["first"]})
        with pytest.raises(ParameterError, match="non-numeric sequence"):
            ordered_rows(frame, "sequence", tmp_path / "t.csv")


class TestParseCell:
    """A cell's value is decided by what the cell says."""

    @pytest.mark.parametrize("text", ["true", "True", " TRUE "])
    def test_true(self, text):
        """Booleans are read regardless of case or padding."""
        assert parse_cell(text) is True

    def test_false(self):
        """False is a boolean, not the string."""
        assert parse_cell("false") is False

    def test_integer(self):
        """A whole number is an int."""
        out = parse_cell("5")
        assert out == 5 and isinstance(out, int)

    def test_exponent_is_integral(self):
        """1e3 is integral even though its text is not."""
        out = parse_cell("1e3")
        assert out == 1000 and isinstance(out, int)

    def test_float_keeps_its_point(self):
        """A number written with a point stays a float."""
        out = parse_cell("2.0")
        assert out == 2.0 and isinstance(out, float)

    def test_text(self):
        """Anything else is the string it was written as."""
        assert parse_cell("car") == "car"

    def test_a_whole_number_wider_than_a_float(self):
        """A nanosecond epoch is 19 digits, which a float cannot hold."""
        out = parse_cell("1600000000123456789")
        assert out == 1600000000123456789 and isinstance(out, int)

    # The digits are full width, which `float` reads and a table never writes.
    @pytest.mark.parametrize("text", ["1_000", "\uff11\uff12\uff13", "nan", "inf"])
    def test_what_a_table_does_not_spell_a_number_with(self, text):
        """Python reads these as numbers; a table's cell does not state one."""
        assert parse_cell(text) == text


def _forge(frame: pd.DataFrame, path, documents: str) -> None:
    """Write a parquet file whose document footer this library did not write."""
    table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    kept = {**(table.schema.metadata or {}), DOCUMENT_KEY: documents}
    pyarrow.parquet.write_table(table.replace_schema_metadata(kept), path)


@pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
class TestParquet:
    """Parquet keeps what a column holds, and what the file says about itself."""

    def test_types_survive(self, tmp_path):
        """A column comes back as what it was written as."""
        frame = pd.DataFrame({"a": [1.5], "b": [True], "c": ["text"]})
        path = tmp_path / "table.parquet"
        write_parquet(frame, path)
        out, _ = read_parquet(path)
        assert out.equals(frame)

    def test_a_column_of_no_one_type(self, tmp_path):
        """A column parquet has no shape for is written as documents."""
        frame = pd.DataFrame({"value": ["car", True, 3]})
        path = tmp_path / "table.parquet"
        write_parquet(frame, path)
        out, _ = read_parquet(path)
        assert list(out["value"]) == ["car", True, 3]

    def test_a_nested_cell(self, tmp_path):
        """A mapping is a document, and reads back as the mapping it was."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame({"meta": [{"a": [1, 2]}, None]}), path)
        out, _ = read_parquet(path)
        assert list(out["meta"]) == [{"a": [1, 2]}, None]

    def test_a_column_of_text_stays_a_column_of_text(self, tmp_path):
        """Text is a type parquet has, so it is not written as documents."""
        frame = pd.DataFrame({"note": pd.Series(["a", None], dtype=object)})
        assert frame["note"].dtype == object  # the branch this is about
        table = parquet_table(frame)
        assert DOCUMENT_KEY.encode() not in (table.schema.metadata or {})
        # Whichever width of string arrow picks, it is a string column.
        assert "string" in str(table.schema.field("note").type)
        path = tmp_path / "table.parquet"
        write_parquet(frame, path)
        out, _ = read_parquet(path)
        assert out["note"][0] == "a" and pd.isna(out["note"][1])

    def test_a_cell_holding_a_list(self, tmp_path):
        """A container states itself; asking pandas answers once per element."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame({"tags": [["a", "b"], None]}), path)
        out, _ = read_parquet(path)
        assert list(out["tags"]) == [["a", "b"], None]

    def test_numbers_numpy_made(self, tmp_path):
        """A numpy scalar is the number it holds, not the text str() gives."""
        path = tmp_path / "table.parquet"
        frame = pd.DataFrame({"value": [np.int64(3), "text", np.bool_(True)]})
        write_parquet(frame, path)
        out, _ = read_parquet(path)
        assert list(out["value"]) == [3, "text", True]

    def test_a_time_of_any_spelling(self, tmp_path):
        """A time has no JSON type, so every spelling of one is written alike."""
        path = tmp_path / "table.parquet"
        stamp = pd.Timestamp("2020-01-01T00:00:01")
        # Three spellings of one instant, which arrive at different
        # resolutions: a Timestamp reads back at microseconds, the others at
        # nanoseconds, and the file holds one of them.
        frame = pd.DataFrame(
            {"when": [stamp, stamp.to_pydatetime(), stamp.to_datetime64(), "text"]}
        )
        write_parquet(frame, path)
        out, _ = read_parquet(path)
        written = list(out["when"])
        assert written[3] == "text"
        assert len(set(written[:3])) == 1
        assert to_datetime64(written[0]) == np.datetime64("2020-01-01T00:00:01")

    def test_a_duration_and_a_model(self, tmp_path):
        """Neither has a JSON type; a model dumps itself, a duration is text."""
        path = tmp_path / "table.parquet"
        line = Line(start={"distance": 0.0}, end={"distance": 10.0})
        frame = pd.DataFrame({"cell": [np.timedelta64(5, "s"), line, "text"]})
        write_parquet(frame, path)
        out, _ = read_parquet(path)
        held = list(out["cell"])
        assert to_timedelta64(held[0]) == np.timedelta64(5, "s")
        assert Line(**held[1]) == line

    def test_a_duration_of_any_spelling(self, tmp_path):
        """A duration is written alike however it arrived, as a time is."""
        path = tmp_path / "table.parquet"
        span = pd.Timedelta(seconds=5)
        frame = pd.DataFrame(
            {"span": [span, span.to_pytimedelta(), span.to_numpy(), "text"]}
        )
        write_parquet(frame, path)
        out, _ = read_parquet(path)
        written = list(out["span"])
        assert written[3] == "text"
        assert len(set(written[:3])) == 1
        assert to_timedelta64(written[0]) == np.timedelta64(5, "s")

    def test_a_missing_value_inside_a_document(self, tmp_path):
        """Missing is missing, not the text of whichever spelling it arrived in."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame({"meta": [{"score": pd.NA, "n": 1}, "text"]}), path)
        out, _ = read_parquet(path)
        assert list(out["meta"]) == [{"score": None, "n": 1}, "text"]

    def test_a_document_column_keeps_its_own_type(self, tmp_path):
        """A column of documents holds what its cells hold, not what pandas infers."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame({"code": pd.Series([1, 2], dtype=object)}), path)
        out, _ = read_parquet(path)
        assert out["code"].dtype == object
        assert list(out["code"]) == [1, 2]

    def test_metadata_round_trips(self, tmp_path):
        """What a format states about its table comes back to it."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame({"a": [1]}), path, {"dascore:dims": '["time"]'})
        _, stated = read_parquet(path)
        assert stated == {"dascore:dims": '["time"]'}

    def test_metadata_without_the_table(self, tmp_path):
        """The footer alone, for a caller which only needs what it says."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame({"a": [1]}), path, {"dascore:dims": '["time"]'})
        assert read_parquet_metadata(path) == {"dascore:dims": '["time"]'}

    def test_no_columns_refused(self, tmp_path):
        """An empty table states nothing, as an empty CSV does."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame(), path)
        with pytest.raises(ParameterError, match="states no track"):
            read_parquet(path, what="no track")

    def test_a_file_which_is_not_parquet(self, tmp_path):
        """Whatever pyarrow raises, the caller sees its own error."""
        path = _write(tmp_path, "a,b\n1,2\n", name="table.parquet")
        with pytest.raises(ParameterError, match="Could not read"):
            read_parquet(path)
        with pytest.raises(ParameterError, match="Could not read"):
            read_parquet_metadata(path)

    @pytest.mark.parametrize("key", [DOCUMENT_KEY, "pandas", "ARROW:schema"])
    def test_a_key_the_file_states_for_itself(self, key, tmp_path):
        """A caller states neither this writer's keys nor the format's own."""
        path = tmp_path / "table.parquet"
        with pytest.raises(ParameterError, match="not a caller's to state"):
            write_parquet(pd.DataFrame({"a": [1]}), path, {key: '["a"]'})

    def test_a_column_named_something_other_than_text(self, tmp_path):
        """Arrow names a column with text, so a label is spelled as text."""
        path = tmp_path / "table.parquet"
        write_parquet(pd.DataFrame({"distance": [1.0], 7: ["x"]}), path)
        out, _ = read_parquet(path)
        assert list(out.columns) == ["distance", "7"]

    def test_two_labels_which_spell_alike(self, tmp_path):
        """One column states one thing, whatever the two labels were."""
        path = tmp_path / "table.parquet"
        frame = pd.DataFrame({7: ["a"], "7": ["b"]})
        with pytest.raises(ParameterError, match="named more than once"):
            write_parquet(frame, path)

    def test_a_file_whose_pandas_metadata_is_not_json(self, tmp_path):
        """Another writer's metadata is read here, so a bad one is named here."""
        path = tmp_path / "table.parquet"
        table = pyarrow.Table.from_pandas(
            pd.DataFrame({"a": [1]}), preserve_index=False
        )
        kept = {**(table.schema.metadata or {}), b"pandas": b"not json"}
        pyarrow.parquet.write_table(table.replace_schema_metadata(kept), path)
        with pytest.raises(ParameterError, match="Could not read"):
            read_parquet(path)

    def test_a_document_column_which_is_not_there(self, tmp_path):
        """A file naming a column it does not hold says so."""
        path = tmp_path / "table.parquet"
        _forge(pd.DataFrame({"a": [1]}), path, '["b"]')
        with pytest.raises(ParameterError, match="holds no such column"):
            read_parquet(path)

    def test_a_document_which_does_not_parse(self, tmp_path):
        """A cell a file names as a document is read as one, or named."""
        path = tmp_path / "table.parquet"
        _forge(pd.DataFrame({"a": ["{oops"]}), path, '["a"]')
        with pytest.raises(ParameterError, match="not a JSON document"):
            read_parquet(path)
