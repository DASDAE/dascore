"""Tests for reading and writing stored annotation sets."""

from __future__ import annotations

import datetime
import json
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None

import dascore as dc
from dascore.core.annotation_loader import find_annotations
from dascore.core.annotations import DIMS_KEY, Line, Moveout, _one_file
from dascore.exceptions import InvalidAnnotationError, ParameterError
from dascore.utils.tables import DOCUMENT_KEY, write_parquet

DIMS = ("distance", "time")

TIMES = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]")


def _folds_case() -> bool:
    """Return True if this filesystem holds two case variants as one file."""
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        (directory / "CaseProbe").write_text("")
        return (directory / "caseprobe").exists()


# Asked once, at collection, as the inventory's tests ask it: Windows and
# most macOS checkouts fold case, so a directory named for another with a
# different case cannot exist there to be refused.
FOLDS_CASE = _folds_case()


def _denies_access() -> bool:
    """Return True if a directory can be made unreadable by chmod."""
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name) / "locked"
        directory.mkdir()
        directory.chmod(0o000)
        try:
            list(directory.iterdir())
            return False
        except OSError:
            return True
        finally:
            directory.chmod(0o755)


# Windows keeps a directory listable whatever its mode, and root reads
# everything, so neither can be shown the failure this names.
DENIES_ACCESS = _denies_access()


def _extras(annotations) -> list[dict]:
    """Return the extras of every lone row, in row order."""
    return [dict(x.extra) for x in annotations if not x.id]


def _bounds(annotations) -> list[dict]:
    """Return the bounds of every lone row, in row order."""
    return [dict(x.geometry.bounds) for x in annotations if not x.id]


@pytest.fixture(scope="module")
def curve() -> Moveout:
    """A moveout a path may be drawn from."""
    return Moveout(
        apex_distance=100.0,
        apex_time=np.datetime64("2020-01-01T00:00:05"),
        velocity=1500.0,
        standoff=30.0,
        distance_min=0.0,
        distance_max=200.0,
    )


@pytest.fixture(scope="module")
def regions() -> dc.AnnotationSet:
    """A set of lone regions, which a bare table can hold."""
    frame = pd.DataFrame(
        {
            "id": ["r1", "r2"],
            "distance_min": [120.0, 10.0],
            "distance_max": [340.0, 60.0],
            "time_min": [
                np.datetime64("2020-01-01T00:00:10"),
                np.datetime64("2020-01-01T00:00:20"),
            ],
            "time_max": [
                np.datetime64("2020-01-01T00:00:12"),
                np.datetime64("2020-01-01T00:00:22"),
            ],
            "note": ["traffic", "walker"],
            "score": [0.9, 0.2],
            "checked": [True, False],
        }
    )
    return dc.AnnotationSet(
        frame, dims=DIMS, acquisition_key="NET.ARR.00.das", data_id="decimated"
    )


@pytest.fixture(scope="module")
def with_features(curve) -> dc.AnnotationSet:
    """A hand-drawn path, a path drawn only by a curve, and a lone box."""
    frame = pd.DataFrame(
        {
            "feature_id": ["p1", "p1", "p1", None],
            "distance": [10.0, 95.0, 185.0, np.nan],
            "time": [
                np.datetime64("2020-01-01T00:00:00.1"),
                np.datetime64("2020-01-01T00:00:01"),
                np.datetime64("2020-01-01T00:00:01.9"),
                None,
            ],
            "distance_min": [np.nan, np.nan, np.nan, 5.0],
            "distance_max": [np.nan, np.nan, np.nan, 15.0],
            "speed": [3.0, 4.0, 5.0, np.nan],
        }
    )
    features = pd.DataFrame(
        {
            "id": ["p1", "p2"],
            "geometry": ["path", "path"],
            "basis": [None, "arrival"],
            "vehicle": ["car", None],
        }
    )
    return dc.AnnotationSet(
        frame, features=features, bases={"arrival": curve}, dims=DIMS
    )


@pytest.fixture(scope="module")
def picks() -> dc.AnnotationSet:
    """A set of time ranges made by a picker, on its own acquisition."""
    frame = pd.DataFrame(
        {
            "id": ["m1", "m2"],
            "phase": ["p", "s"],
            "time_min": [
                np.datetime64("2020-01-01T00:00:01"),
                np.datetime64("2020-01-01T00:00:03"),
            ],
            "time_max": [
                np.datetime64("2020-01-01T00:00:02"),
                np.datetime64("2020-01-01T00:00:04"),
            ],
            "score": [0.4, 0.6],
        }
    )
    return dc.AnnotationSet(
        frame,
        dims=("time",),
        acquisition_key="NET.ARR.00.fast",
        creation_info={"author": "phasenet"},
    )


def _path_set(feature="p9", distance=(1000.0, 1100.0, 1200.0), dims=DIMS):
    """A set holding one path in distance (and time, where declared)."""
    columns = {"feature_id": [feature] * len(distance), "distance": list(distance)}
    if "time" in dims:
        start = "2020-01-01T00:00:05"
        columns["time"] = pd.date_range(start, periods=len(distance), freq="s")
    features = pd.DataFrame({"id": [feature], "geometry": ["path"]})
    return dc.AnnotationSet(pd.DataFrame(columns), features=features, dims=dims)


class TestRoundTrip:
    """A set written out and read back is the set it was."""

    def test_regions_through_a_directory(self, regions, tmp_path):
        """Bounds, extras and provenance all survive a directory."""
        regions.io.save(tmp_path / "picks")
        loaded = dc.annotations(tmp_path / "picks")
        assert loaded == regions
        assert loaded.attrs.data_id == "decimated"

    def test_regions_through_a_bare_table(self, regions, tmp_path):
        """A set of regions is a table, and its dims are stated again."""
        path = tmp_path / "picks.csv"
        regions.io.to_csv(path)
        loaded = dc.annotations(path, dims=DIMS)
        assert loaded.annotations.equals(regions.annotations)

    def test_implied_features_through_a_bare_table(self, tmp_path):
        """Groups a feature_id implies come back from the table alone."""
        frame = pd.DataFrame({"feature_id": ["e1", "e1", None], "time": [1.0, 2, 3]})
        out = dc.AnnotationSet(frame, dims=DIMS)
        path = tmp_path / "picks.csv"
        out.io.to_csv(path)
        assert dc.annotations(path, dims=DIMS) == out

    def test_implied_features_are_written_explicitly(self, tmp_path):
        """A directory states every feature, implied ones included."""
        frame = pd.DataFrame({"feature_id": ["e1", "e1"], "time": [1.0, 2.0]})
        out = dc.AnnotationSet(frame, dims=DIMS)
        directory = out.io.save(tmp_path / "picks")
        assert "e1" in (directory / "features.csv").read_text()
        assert dc.annotations(directory) == out

    def test_whole_number_ids(self, tmp_path):
        """Ids written as numbers name the same rows once read back."""
        frame = pd.DataFrame({"id": [1, 2], "distance": [1.0, 2.0]})
        out = dc.AnnotationSet(frame, dims=DIMS)
        assert dc.annotations(out.io.save(tmp_path / "picks")) == out

    def test_a_column_stating_nothing(self, tmp_path):
        """A column no row states holds the same nothing after a write."""
        frame = pd.DataFrame({"id": ["a"], "note": [None], "distance": [1.0]})
        out = dc.AnnotationSet(frame, dims=DIMS)
        assert dc.annotations(out.io.save(tmp_path / "picks")) == out

    def test_a_set_of_no_annotations(self, tmp_path):
        """A set with columns and no rows is still that set."""
        frame = pd.DataFrame({"id": [], "distance_min": [], "distance_max": []})
        out = dc.AnnotationSet(frame, dims=DIMS)
        assert dc.annotations(out.io.save(tmp_path / "picks")) == out

    def test_order_columns(self, tmp_path):
        """seq, part and ring come back as the whole numbers they were."""
        frame = pd.DataFrame(
            {
                "feature_id": ["p"] * 4 + [None],
                "part": [1, 1, 0, 0, None],
                "distance": [5.0, 6.0, 0.0, 1.0, 9.0],
            }
        )
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"]})
        out = dc.AnnotationSet(frame, features=features, dims=DIMS)
        loaded = dc.annotations(out.io.save(tmp_path / "picks"))
        assert loaded == out
        assert loaded.annotations["seq"].dtype == "Int64"

    def test_the_tables_a_writer_wrote(self, with_features, tmp_path):
        """The tables a set writes build that set again, unread by the loader."""
        directory = with_features.io.save(tmp_path / "picks")
        rebuilt = dc.AnnotationSet(
            pd.read_csv(directory / "annotations.csv"),
            features=pd.read_csv(directory / "features.csv"),
            bases=json.loads((directory / "bases.json").read_text()),
            dims=with_features.dims,
        )
        assert rebuilt == with_features

    def test_features_and_bases(self, with_features, curve, tmp_path):
        """Features and the curves they name both survive."""
        loaded = dc.annotations(with_features.io.save(tmp_path / "picks"))
        assert loaded == with_features
        assert loaded["p2"].basis == curve
        assert loaded["p1"].extra == {"vehicle": "car"}

    def test_extras_keep_their_kind(self, regions, tmp_path):
        """A cell written as a number or a boolean reads back as one."""
        extras = _extras(dc.annotations(regions.io.save(tmp_path / "picks")))
        assert extras[0]["score"] == 0.9
        assert extras[0]["checked"] is True
        assert extras[1]["checked"] is False

    def test_times_keep_their_type(self, regions, tmp_path):
        """A time endpoint reads back as a time, not as its text."""
        loaded = dc.annotations(regions.io.save(tmp_path / "picks"))
        start, _ = _bounds(loaded)[0]["time"]
        assert isinstance(start, np.datetime64)

    def test_a_line_basis(self, tmp_path):
        """A line with time endpoints survives bases.json as the curve it is."""
        line = Line(
            start={"distance": 0.0, "time": np.datetime64("2020-01-01")},
            end={"distance": 50.0, "time": np.datetime64("2020-01-01")},
        )
        features = pd.DataFrame({"id": ["p1"], "geometry": ["path"], "basis": ["l"]})
        out = dc.AnnotationSet(None, features=features, bases={"l": line}, dims=DIMS)
        loaded = dc.annotations(out.io.save(tmp_path / "picks"))
        assert loaded.bases["l"] == line
        assert loaded == out

    def test_a_line_of_python_datetimes(self, tmp_path):
        """Python datetimes are held as numpy times, so a reload is equal."""
        start = datetime.datetime(2020, 1, 1)
        line = Line(
            start={"distance": 0.0, "time": start},
            end={"distance": 50.0, "time": start + datetime.timedelta(seconds=2)},
        )
        assert line.vertices(3)["time"].dtype == np.dtype("datetime64[ns]")
        features = pd.DataFrame({"id": ["p1"], "geometry": ["path"], "basis": ["l"]})
        out = dc.AnnotationSet(None, features=features, bases={"l": line}, dims=DIMS)
        loaded = dc.annotations(out.io.save(tmp_path / "picks"))
        assert loaded.bases["l"] == line
        assert loaded == out

    def test_reserved_text_columns_of_numbers(self, tmp_path):
        """A name or data_id written as numbers is text on both sides of a save."""
        frame = pd.DataFrame(
            {
                "feature_id": ["a", None],
                "time": [1.0, 2.0],
                "name": [1, 2],
                "data_id": [10, 20],
            }
        )
        features = pd.DataFrame({"id": ["a"], "name": [3], "data_id": [30]})
        out = dc.AnnotationSet(frame, features=features, dims=DIMS)
        assert list(out.annotations["name"]) == ["1", "2"]
        assert dc.annotations(out.io.save(tmp_path / "picks")) == out

    def test_a_duration_basis(self, tmp_path):
        """A line over an offset reads back from bases.json as durations."""
        line = Line(
            start={"offset": np.timedelta64(1, "s")},
            end={"offset": np.timedelta64(3, "s")},
        )
        out = dc.AnnotationSet(None, dims=("offset",), bases={"l": line})
        loaded = dc.annotations(out.io.save(tmp_path / "picks"))
        assert isinstance(loaded.bases["l"].start["offset"], np.timedelta64)
        assert loaded == out

    def test_columns_of_both_tables(self, tmp_path):
        """What each table documents about its columns survives."""
        frame = pd.DataFrame({"feature_id": ["e"], "time": [1.0], "phase": ["P"]})
        out = dc.AnnotationSet(
            frame,
            dims=DIMS,
            annotation_columns={"phase": {"description": "P or S"}},
            feature_columns={"magnitude": {"units": "m"}},
        )
        loaded = dc.annotations(out.io.save(tmp_path / "picks"))
        assert loaded.attrs == out.attrs

    def test_an_unstated_bound(self, tmp_path):
        """An empty dimension cell reads back as unconstrained."""
        frame = pd.DataFrame(
            {
                "time": [np.nan, 3.0],
                "distance_min": [1.0, np.nan],
                "distance_max": [2.0, np.nan],
            }
        )
        annotations = dc.AnnotationSet(frame, dims=DIMS)
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert loaded == annotations
        assert "distance" not in _bounds(loaded)[1]


class TestDeclaredDtypes:
    """A CSV states no types; the declaration beside it gives them back."""

    @pytest.fixture
    def typed(self):
        """A set declaring a categorical and a nullable integer column."""
        frame = pd.DataFrame(
            {
                "distance_min": [0.0, 1.0],
                "distance_max": [1.0, 2.0],
                "kind": pd.Series(["a", "b"], dtype="category"),
                "count": pd.Series([1, None], dtype="Int64"),
            }
        )
        columns = {"kind": {"dtype": "category"}, "count": {"dtype": "Int64"}}
        return dc.AnnotationSet(frame, dims=("distance",), annotation_columns=columns)

    def test_declared_dtypes_survive_a_csv(self, typed, tmp_path):
        """The loaded set holds what the saved one declared, and equals it."""
        loaded = dc.annotations(typed.io.save(tmp_path / "typed"))
        frame = loaded.annotations
        assert frame["kind"].dtype.name == "category"
        assert frame["count"].dtype.name == "Int64"
        assert loaded == typed

    def test_declared_feature_dtypes_survive_a_csv(self, tmp_path):
        """The features table restores its own declarations."""
        frame = pd.DataFrame({"feature_id": ["a", "b"], "distance": [1.0, 2.0]})
        features = pd.DataFrame(
            {"id": ["a", "b"], "rank": pd.Series([1, None], dtype="Int64")}
        )
        out = dc.AnnotationSet(
            frame,
            features=features,
            dims=("distance",),
            feature_columns={"rank": {"dtype": "Int64"}},
        )
        loaded = dc.annotations(out.io.save(tmp_path / "typed"))
        assert loaded.features["rank"].dtype.name == "Int64"
        assert loaded == out

    def test_declared_dtypes_survive_parquet(self, typed, tmp_path):
        """Parquet keeps the types itself; the declaration then changes nothing."""
        pytest.importorskip("pyarrow")
        loaded = dc.annotations(typed.io.save(tmp_path / "typed", format="parquet"))
        assert loaded == typed

    def test_a_bare_table_given_its_declarations(self, typed, tmp_path):
        """A CSV states no attrs; a caller handing it the columns gets them back."""
        path = tmp_path / "typed.csv"
        typed.io.to_csv(path)
        columns = typed.attrs.annotation_columns
        loaded = dc.annotations(path, dims=("distance",), annotation_columns=columns)
        assert loaded.annotations["count"].dtype.name == "Int64"

    def test_a_declared_text_dtype_keeps_blank_cells_unset(self, tmp_path):
        """Text is not cast on reload, so a blank stays a blank, not the word 'nan'."""
        frame = pd.DataFrame(
            {"distance_min": [0.0, 1.0], "distance_max": [1.0, 2.0], "n": ["a", None]}
        )
        built = dc.AnnotationSet(
            frame, dims=("distance",), annotation_columns={"n": {"dtype": "str"}}
        )
        extras = _extras(dc.annotations(built.io.save(tmp_path / "set")))
        assert extras[0]["n"] == "a"
        assert "n" not in extras[1]

    def test_a_declared_text_column_keeps_its_text(self, tmp_path):
        """A cell a table would read as a number, a truth value or a date
        stays the text it was written as when the column is declared text.
        """
        frame = pd.DataFrame(
            {
                "distance_min": [0.0, 1.0, 2.0],
                "distance_max": [1.0, 2.0, 3.0],
                "label": ["001", "true", "2020-01-01"],
            }
        )
        columns = {"label": {"dtype": "str"}}
        built = dc.AnnotationSet(frame, dims=("distance",), annotation_columns=columns)
        loaded = dc.annotations(built.io.save(tmp_path / "set"))
        assert [x["label"] for x in _extras(loaded)] == ["001", "true", "2020-01-01"]
        assert loaded == built
        path = tmp_path / "set.csv"
        built.io.to_csv(path)
        bare = dc.annotations(path, dims=("distance",), annotation_columns=columns)
        assert [x["label"] for x in _extras(bare)] == ["001", "true", "2020-01-01"]

    def test_an_empty_columns_override_restores_nothing(self, typed, tmp_path):
        """`annotation_columns={}` clears the declarations, so nothing is restored."""
        path = tmp_path / "typed.csv"
        typed.io.to_csv(path)
        loaded = dc.annotations(
            path, dims=("distance",), attrs=typed.attrs, annotation_columns={}
        )
        assert not loaded.attrs.annotation_columns
        assert loaded.annotations["count"].dtype.name != "Int64"

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ({"dtype": "not-a-dtype"}, "cannot be read as"),
            (None, "annotation_columns"),
            ({"dtype": "Int64"}, "cannot be read as"),
        ],
    )
    def test_a_declaration_edited_badly(self, spec, match, tmp_path):
        """A bad declaration in a stored attrs file is refused on reload."""
        frame = pd.DataFrame({"distance_min": [0.0], "distance_max": [1.0], "n": ["x"]})
        directory = dc.AnnotationSet(frame, dims=("distance",)).io.save(
            tmp_path / "set"
        )
        attrs_path = directory / "attrs.json"
        document = json.loads(attrs_path.read_text())
        document["annotation_columns"] = {"n": spec}
        attrs_path.write_text(json.dumps(document))
        with pytest.raises(InvalidAnnotationError, match=match):
            dc.annotations(directory)


class TestWhatATableCannotSay:
    """A CSV has no types, and these are the corners where that shows."""

    def test_an_order_column_on_a_lone_row(self, tmp_path):
        """An order column is reserved for ordered members, stored or not."""
        path = tmp_path / "picks.csv"
        path.write_text("phase,distance,seq\na,1.0,third\n")
        with pytest.raises(InvalidAnnotationError, match="is not numeric"):
            dc.annotations(path, dims=("distance",))
        path.write_text("phase,distance,seq\na,1.0,3\n")
        with pytest.raises(InvalidAnnotationError, match="orders the members"):
            dc.annotations(path, dims=("distance",))

    def test_an_empty_cell_is_unset(self, tmp_path):
        """A table cannot tell an empty cell from an empty string."""
        frame = pd.DataFrame({"note": ["", "b"], "distance": [1.0, 2.0]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        assert dc.annotations(annotations.io.save(tmp_path / "picks")) == annotations
        assert pd.isna(annotations.annotations["note"][0])

    def test_a_datetime_extra_reads_back_as_text(self, tmp_path):
        """Only a declared dimension is known to hold times, so only it is read
        as one; an extra keeps the text it was written as.
        """
        frame = pd.DataFrame(
            {"distance": [1.0], "when": [np.datetime64("2020-01-01T00:00:00")]}
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert _extras(loaded)[0]["when"] == "2020-01-01T00:00:00.000000000"

    def test_a_value_column_is_an_extra(self, tmp_path):
        """`value` is no longer modelled, so a table reads it as it reads any cell."""
        frame = pd.DataFrame({"value": ["P", "true"], "distance": [1.0, 2.0]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert [x["value"] for x in _extras(loaded)] == ["P", True]

    def test_an_int_beside_a_blank_stays_an_int(self, tmp_path):
        """An unset cell must not make the writer spell an int as a float."""
        frame = pd.DataFrame(
            {"value": pd.Series([None, 5], dtype=object), "distance": [1.0, 2.0]}
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        text = annotations.io.to_csv()
        assert "\n5,2.0" in text and "5.0" not in text
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert [x.get("value") for x in _extras(loaded)] == [None, 5]

    def test_a_blank_column_round_trips(self, tmp_path):
        """A column no row states reloads equal to what was saved."""
        frame = pd.DataFrame({"value": [None, None], "distance": [1.0, 2.0]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        assert dc.annotations(annotations.io.save(tmp_path / "picks")) == annotations

    def test_a_non_finite_looking_extra_stays_text(self, tmp_path):
        """A cell reading 'nan' is text, not a value which then vanishes."""
        frame = pd.DataFrame({"distance": [1.0], "note": ["nan"]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert _extras(loaded)[0]["note"] == "nan"

    def test_an_extra_some_rows_leave_blank(self, tmp_path):
        """A blank cell is unset; the rows which state one still read."""
        frame = pd.DataFrame({"distance": [1.0, 2.0], "note": ["seen", None]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        extras = _extras(dc.annotations(annotations.io.save(tmp_path / "picks")))
        assert extras[0]["note"] == "seen"
        assert "note" not in extras[1]

    def test_a_numeric_looking_extra_reads_as_a_number(self, tmp_path):
        """A cell is read the way its own text states it, as every table is."""
        frame = pd.DataFrame({"distance": [1.0], "zip": ["01234"]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert _extras(loaded)[0]["zip"] == 1234


class TestDurationDimensions:
    """A duration is a coordinate CSV has no spelling for."""

    @staticmethod
    def _set():
        """A set whose dimension is an offset from something."""
        spans = np.array([1, 3], dtype="timedelta64[s]")
        frame = pd.DataFrame(
            {"id": ["a"], "offset_min": spans[:1], "offset_max": spans[1:]}
        )
        return dc.AnnotationSet(frame, dims=("offset",))

    def test_csv_refused(self, tmp_path):
        """Written as text nothing reads it back, so it is not written."""
        with pytest.raises(ParameterError, match="no spelling for"):
            self._set().io.save(tmp_path / "picks")

    def test_to_csv_refused(self):
        """The bare table says the same thing."""
        with pytest.raises(ParameterError, match="no spelling for"):
            self._set().io.to_csv()

    @pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
    def test_parquet_keeps_it(self, tmp_path):
        """Parquet has a type for a duration, so the set stores."""
        out = self._set()
        saved = out.io.save(tmp_path / "picks", format="parquet")
        assert dc.annotations(saved) == out

    @pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
    def test_a_path_drawn_over_one(self, tmp_path):
        """A path over an offset keeps its members and its curve."""
        line = Line(
            start={"offset": np.timedelta64(1, "s")},
            end={"offset": np.timedelta64(3, "s")},
        )
        frame = pd.DataFrame(
            {
                "feature_id": ["p"] * 2,
                "offset": np.array([1, 3], dtype="timedelta64[s]"),
            }
        )
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"], "basis": ["l"]})
        out = dc.AnnotationSet(
            frame, features=features, bases={"l": line}, dims=("offset",)
        )
        saved = out.io.save(tmp_path / "curve", format="parquet")
        loaded = dc.annotations(saved)
        assert loaded == out
        assert loaded["p"].basis == line


class TestSavingOverASet:
    """Writing states the whole directory, not only the parts it has."""

    def test_stale_parts_are_cleared(self, with_features, regions, tmp_path):
        """A set without features or bases leaves none behind for the next read."""
        directory = tmp_path / "picks"
        with_features.io.save(directory)
        regions.io.save(directory)
        assert not (directory / "features.csv").exists()
        assert not (directory / "bases.json").exists()
        assert dc.annotations(directory) == regions

    def test_a_retired_vertices_table_is_superseded(self, regions, tmp_path):
        """Saving over an old set clears its vertices table, so it reads back."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "vertices.csv").write_text("id,seq,distance\na,0,1.0\n")
        regions.io.save(directory)
        assert not (directory / "vertices.csv").exists()
        assert dc.annotations(directory) == regions

    def test_a_hand_authored_yaml_is_superseded(self, tmp_path):
        """Saving a set read from YAML does not leave two attrs files."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "attrs.yaml").write_text(yaml.safe_dump({"dims": list(DIMS)}))
        (directory / "annotations.csv").write_text("note,distance\nnoise,1.0\n")
        loaded = dc.annotations(directory)
        loaded.io.save(directory)
        assert not (directory / "attrs.yaml").exists()
        assert dc.annotations(directory) == loaded

    def test_a_hand_authored_bases_yaml(self, with_features, tmp_path):
        """Bases may be authored as YAML too, and a save supersedes it."""
        directory = with_features.io.save(tmp_path / "picks")
        document = json.loads((directory / "bases.json").read_text())
        (directory / "bases.json").unlink()
        (directory / "bases.yaml").write_text(yaml.safe_dump(document))
        loaded = dc.annotations(directory)
        assert loaded == with_features
        loaded.io.save(directory)
        assert not (directory / "bases.yaml").exists()

    def test_a_file_owing_this_format_nothing_is_left(self, regions, tmp_path):
        """Only the spellings a set claims are cleared."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.bak").write_text("mine")
        regions.io.save(directory)
        assert (directory / "attrs.bak").read_text() == "mine"

    def test_a_shouted_suffix_is_read_and_superseded(self, regions, tmp_path):
        """One data model stands behind a suffix however it is spelled."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").rename(directory / "attrs.JSON")
        loaded = dc.annotations(directory)
        assert loaded == regions
        loaded.io.save(directory)
        # A case-insensitive filesystem holds both names in one file, so only
        # the count is this format's to say.
        assert (
            len([x for x in directory.iterdir() if x.stem.casefold() == "attrs"]) == 1
        )
        assert dc.annotations(directory) == regions

    def test_one_file_under_two_names(self, tmp_path):
        """The question the supersede pass asks; the test above is where it bites."""
        path = tmp_path / "attrs.json"
        path.write_text("{}")
        assert _one_file(path, tmp_path / "." / "attrs.json")
        assert not _one_file(path, tmp_path / "attrs.yaml")


class TestTheDoor:
    """Everything a set may be loaded from goes through one function."""

    def test_a_set_is_itself(self, regions):
        """Loading a set which is already loaded hands it back."""
        assert dc.annotations(regions) is regions

    def test_a_set_refuses_overrides(self, regions):
        """Silently dropping them would make one door mean two things."""
        with pytest.raises(ParameterError, match="already built"):
            dc.annotations(regions, dims=("time", "distance"))
        with pytest.raises(ParameterError, match="already built"):
            dc.annotations(regions, acquisition_key="N.A.00.das")

    @pytest.mark.parametrize(
        "given",
        [{"attrs": {"dims": DIMS}}, {"features": pd.DataFrame()}, {"bases": {}}],
    )
    def test_a_directory_refuses_what_it_states(self, regions, tmp_path, given):
        """A directory holds its own attributes, features and bases."""
        directory = regions.io.save(tmp_path / "picks")
        with pytest.raises(InvalidAnnotationError, match="which states them"):
            dc.annotations(directory, **given)

    def test_a_dataframe(self):
        """A frame becomes a set, as the constructor makes one."""
        frame = pd.DataFrame({"note": ["a"], "distance": [1.0]})
        assert len(dc.annotations(frame, dims=("distance",))) == 1

    def test_nothing(self):
        """A set of nothing is still a set."""
        assert len(dc.annotations(dims=("distance",))) == 0

    def test_a_path_which_is_not_there(self, tmp_path):
        """A path naming nothing says so, rather than reading nothing."""
        with pytest.raises(InvalidAnnotationError, match="does not exist"):
            dc.annotations(tmp_path / "missing", dims=DIMS)

    def test_a_file_which_is_not_a_table(self, tmp_path):
        """Only a table is a bare set."""
        path = tmp_path / "picks.txt"
        path.write_text("note\na\n")
        with pytest.raises(InvalidAnnotationError, match="not a table"):
            dc.annotations(path, dims=DIMS)

    def test_errors_are_annotation_errors(self, tmp_path):
        """The neutral errors the table reader raises are named here."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "annotations.csv").write_text("note\nnoise,extra\n")
        with pytest.raises(InvalidAnnotationError, match="states 2 cells"):
            dc.annotations(directory, dims=DIMS)

    def test_a_blank_table(self, tmp_path):
        """A table stating nothing at all is a set of nothing."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "annotations.csv").write_text("\n")
        assert dc.annotations(directory, dims=DIMS) == dc.AnnotationSet(dims=DIMS)

    @pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
    def test_a_parquet_table_of_no_columns(self, tmp_path):
        """The same holds for a parquet file stating no column."""
        path = tmp_path / "picks.parquet"
        pyarrow.parquet.write_table(pyarrow.table({}), path)
        assert len(dc.annotations(path, dims=DIMS)) == 0

    def test_a_set_of_none(self, tmp_path):
        """A set of no annotations writes an empty table and reads back."""
        empty = dc.annotations(dims=DIMS)
        assert dc.annotations(empty.io.save(tmp_path / "picks")) == empty


class TestDeclaringDimensions:
    """Cells cannot be read before the dimensions are known."""

    def test_stated_by_the_attrs(self, regions, tmp_path):
        """A directory states its own dimensions."""
        assert dc.annotations(regions.io.save(tmp_path / "picks")).dims == DIMS

    def test_stated_by_the_caller(self, regions, tmp_path):
        """A bare table has the caller state them."""
        path = tmp_path / "picks.csv"
        regions.io.to_csv(path)
        assert dc.annotations(path, dims=DIMS).dims == DIMS

    def test_stated_by_neither(self, regions, tmp_path):
        """A source stating none fails saying how to state them."""
        path = tmp_path / "picks.csv"
        regions.io.to_csv(path)
        with pytest.raises(InvalidAnnotationError, match="states no dimensions"):
            dc.annotations(path)

    @pytest.mark.parametrize("source", ["file", "caller"])
    def test_stated_as_a_bare_string(self, tmp_path, source):
        """One dimension may be a lone string, not a sequence of its letters."""
        directory = tmp_path / "picks"
        directory.mkdir()
        stated = '{"dims": "distance"}' if source == "file" else "{}"
        (directory / "attrs.json").write_text(stated)
        (directory / "annotations.csv").write_text("note,distance\na,1.0\n")
        dims = None if source == "file" else "distance"
        loaded = dc.annotations(directory, dims=dims)
        assert loaded.dims == ("distance",)
        assert _bounds(loaded)[0]["distance"] == (1.0, 1.0)

    def test_a_document_which_does_not_build(self, tmp_path):
        """A bad stored document is named as a bad file, not as a bad call."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "attrs.json").write_text('{"dims": ["distance"], "n": 1}')
        (directory / "annotations.csv").write_text("note,distance\na,1.0\n")
        with pytest.raises(InvalidAnnotationError, match="Extra inputs"):
            dc.annotations(directory)

    @pytest.mark.parametrize(
        "field, value, now",
        [("history", ["decimate"], "data_id"), ("columns", {}, "annotation_columns")],
    )
    def test_a_retired_attrs_field(self, regions, tmp_path, field, value, now):
        """A set written with a retired field names what replaced it."""
        directory = regions.io.save(tmp_path / "picks")
        document = json.loads((directory / "attrs.json").read_text())
        document[field] = value
        (directory / "attrs.json").write_text(json.dumps(document))
        with pytest.raises(InvalidAnnotationError, match=f"{field}.*{now}"):
            dc.annotations(directory)

    def test_a_directory_which_states_them_refuses_others(self, regions, tmp_path):
        """Reading the cells against other dimensions would type them
        differently and build a set which is not the one stored.
        """
        directory = regions.io.save(tmp_path / "picks")
        with pytest.raises(InvalidAnnotationError, match="its own dimensions"):
            dc.annotations(directory, dims=("time", "distance"))

    def test_a_directory_which_states_none_takes_them(self, regions, tmp_path):
        """Where a directory states none, the caller's are the only ones."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").unlink()
        loaded = dc.annotations(directory, dims=("time", "distance"))
        assert loaded.dims == ("time", "distance")


class TestTheAttrsFile:
    """What a set directory says about itself."""

    def test_yaml_spelling(self, regions, tmp_path):
        """One data model stands behind both spellings."""
        directory = regions.io.save(tmp_path / "picks")
        document = json.loads((directory / "attrs.json").read_text())
        (directory / "attrs.yaml").write_text(yaml.safe_dump(document))
        (directory / "attrs.json").unlink()
        assert dc.annotations(directory) == regions

    def test_two_spellings(self, regions, tmp_path):
        """A set spells each of its parts once."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.yml").write_text("{}")
        with pytest.raises(InvalidAnnotationError, match="more than once"):
            dc.annotations(directory)

    def test_the_wrong_object(self, regions, tmp_path):
        """A file declaring another model is a misfiled object."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").write_text(
            '{"object_type": "Inventory", "dims": ["time"]}'
        )
        with pytest.raises(InvalidAnnotationError, match="declares 'Inventory'"):
            dc.annotations(directory)

    def test_which_is_not_a_mapping(self, regions, tmp_path):
        """A document stating a list defines no attributes."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").write_text('["distance", "time"]')
        with pytest.raises(InvalidAnnotationError, match="no mapping"):
            dc.annotations(directory)

    def test_which_does_not_parse(self, regions, tmp_path):
        """Unparseable YAML names the file rather than the parser."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").unlink()
        (directory / "attrs.yaml").write_text("dims: [\n")
        with pytest.raises(InvalidAnnotationError, match="Could not parse YAML"):
            dc.annotations(directory)

    def test_bad_json(self, regions, tmp_path):
        """Unparseable JSON names the file too."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").write_text("{")
        with pytest.raises(InvalidAnnotationError, match="Could not parse JSON"):
            dc.annotations(directory)

    def test_which_cannot_be_read(self, regions, tmp_path):
        """A file which does not decode names itself, not the codec."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").write_bytes(b'{"dims": ["\xff\xfe"]}')
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(directory)

    def test_no_attrs_file(self, regions, tmp_path):
        """A directory without one is read on the caller's dimensions."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "attrs.json").unlink()
        assert dc.annotations(directory, dims=DIMS).dims == DIMS


class TestTheTables:
    """What a set directory holds, and what it may not."""

    def test_no_annotations_table(self, tmp_path):
        """A directory without one states no annotations."""
        directory = tmp_path / "picks"
        directory.mkdir()
        with pytest.raises(InvalidAnnotationError, match="no annotations table"):
            dc.annotations(directory, dims=DIMS)

    def test_a_table_which_cannot_be_read(self, regions, tmp_path):
        """A table which does not decode is the table reader's to name."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / "annotations.csv").write_bytes(b"note\n\xff\xfe\n")
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(directory)

    def test_features_named_in_another_case(self, with_features, tmp_path):
        """Every part of a set is found the same way, so none is skipped."""
        directory = with_features.io.save(tmp_path / "picks")
        (directory / "features.csv").rename(directory / "features.CSV")
        assert dc.annotations(directory) == with_features

    @pytest.mark.skipif(FOLDS_CASE, reason="this filesystem holds one of the two")
    def test_features_spelled_twice(self, with_features, tmp_path):
        """A set spells each of its parts once, features included."""
        directory = with_features.io.save(tmp_path / "picks")
        text = (directory / "features.csv").read_text()
        (directory / "features.CSV").write_text(text)
        with pytest.raises(InvalidAnnotationError, match="states features more than"):
            dc.annotations(directory)

    @pytest.mark.parametrize("name", ["feature.csv", "vertices.csv"])
    def test_a_stray_table(self, regions, tmp_path, name):
        """A near-miss, or the retired vertices table, raises rather than hides."""
        directory = regions.io.save(tmp_path / "picks")
        (directory / name).write_text("id,seq\n")
        with pytest.raises(InvalidAnnotationError, match=name.replace(".", r"\.")):
            dc.annotations(directory)

    def test_bases_which_are_not_json(self, with_features, tmp_path):
        """bases.json is a document, and names itself where it is not one."""
        directory = with_features.io.save(tmp_path / "picks")
        (directory / "bases.json").write_text("{oops")
        with pytest.raises(InvalidAnnotationError, match="Could not parse JSON"):
            dc.annotations(directory)

    def test_bases_which_are_not_curves(self, with_features, tmp_path):
        """A document which parses but names no curve is still refused."""
        directory = with_features.io.save(tmp_path / "picks")
        (directory / "bases.json").write_text('{"arrival": {}}')
        with pytest.raises(InvalidAnnotationError, match="basis 'arrival'"):
            dc.annotations(directory)

    def test_a_basis_key_naming_nothing(self, with_features, tmp_path):
        """A features row naming a basis bases.json lacks is refused."""
        directory = with_features.io.save(tmp_path / "picks")
        (directory / "bases.json").unlink()
        with pytest.raises(InvalidAnnotationError, match="not among the bases"):
            dc.annotations(directory)

    def test_a_non_numeric_seq(self, tmp_path):
        """A member states its place in the order as a number."""
        directory = _path_set().io.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        header, first, *rest = table.read_text().splitlines()
        seq = header.split(",").index("seq")
        cells = first.split(",")
        cells[seq] = "first"
        table.write_text("\n".join([header, ",".join(cells), *rest]) + "\n")
        with pytest.raises(InvalidAnnotationError, match="is not numeric"):
            dc.annotations(directory)

    def test_a_dimension_which_is_neither(self, regions, tmp_path):
        """A dimension column holds numbers or times, and says so."""
        directory = regions.io.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        table.write_text(table.read_text().replace("120.0", "far"))
        with pytest.raises(InvalidAnnotationError, match="neither numbers, times"):
            dc.annotations(directory)

    def test_a_row_no_dimension_states(self, tmp_path):
        """A row whose every dimension cell is empty is nowhere."""
        path = tmp_path / "picks.csv"
        path.write_text("note,time_min,time_max\nquiet,,\n")
        with pytest.raises(InvalidAnnotationError, match="state no dimension"):
            dc.annotations(path, dims=("time",))

    def test_a_minute_resolution_time(self, tmp_path):
        """Numpy writes only the fields a unit carries, and they all read back."""
        frame = pd.DataFrame(
            {
                "time_min": [np.datetime64("2020-01-01T12:30")],
                "time_max": [np.datetime64("2020-01-01T12:35")],
            }
        )
        annotations = dc.AnnotationSet(frame, dims=("time",))
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert loaded == annotations
        assert isinstance(_bounds(loaded)[0]["time"][0], np.datetime64)

    @pytest.mark.parametrize("blank", [None, ""])
    def test_a_dimension_some_rows_leave_blank(self, blank):
        """Times as text beside empty cells read as times and as unset."""
        frame = pd.DataFrame(
            {
                "time_min": ["2020-01-01", blank],
                "time_max": ["2020-01-02", blank],
                "distance": [np.nan, 1.0],
            }
        )
        bounds = _bounds(dc.AnnotationSet(frame, dims=("time", "distance")))
        assert bounds[0]["time"][0] == np.datetime64("2020-01-01")
        assert "time" not in bounds[1]

    def test_a_text_dimension_column_agrees_with_its_region(self):
        """The frame and the geometry built from it say the same thing."""
        frame = pd.DataFrame({"time_min": ["2020-01-01"], "time_max": ["2020-01-02"]})
        out = dc.AnnotationSet(frame, dims=("time",))
        held = out.annotations["time_min"][0]
        assert isinstance(held, pd.Timestamp | np.datetime64)
        assert _bounds(out)[0]["time"][0] == np.datetime64("2020-01-01")

    def test_an_id_which_looks_like_a_number(self, tmp_path):
        """A feature id is the label its members name it by, never a number."""
        annotations = _path_set(feature="1", dims=("distance",))
        loaded = dc.annotations(annotations.io.save(tmp_path / "picks"))
        assert loaded.features["id"][0] == "1"
        assert loaded.annotations["feature_id"][0] == "1"
        assert loaded == annotations


class TestWriting:
    """How a set spells itself out."""

    def test_to_csv_returns_text(self, regions):
        """The text comes back whether or not it is written."""
        assert regions.io.to_csv().splitlines()[0].startswith("id,distance_min")

    def test_to_csv_refuses_features(self, with_features):
        """A bare table holds only annotations."""
        with pytest.raises(ParameterError, match="holds features or bases"):
            with_features.io.to_csv()

    def test_to_csv_refuses_bases(self):
        """An unused basis still has no place in a bare table."""
        line = Line(start={"distance": 0.0}, end={"distance": 1.0})
        out = dc.AnnotationSet(
            pd.DataFrame({"distance": [1.0]}), dims=DIMS, bases={"l": line}
        )
        with pytest.raises(ParameterError, match="holds features or bases"):
            out.io.to_csv()

    def test_save_makes_the_directory(self, regions, tmp_path):
        """Saving into a directory which is not there makes it."""
        directory = regions.io.save(tmp_path / "deep" / "picks")
        assert directory.is_dir()

    def test_save_writes_only_what_is_needed(self, regions, tmp_path):
        """A set without features or bases states neither."""
        directory = regions.io.save(tmp_path / "picks")
        assert {x.name for x in directory.iterdir()} == {
            "attrs.json",
            "annotations.csv",
        }

    def test_save_writes_what_it_holds(self, with_features, tmp_path):
        """A set with features and bases states all four parts."""
        directory = with_features.io.save(tmp_path / "picks")
        assert {x.name for x in directory.iterdir()} == {
            "attrs.json",
            "annotations.csv",
            "features.csv",
            "bases.json",
        }

    def test_save_over_itself(self, regions, tmp_path):
        """Saving twice into one directory rewrites it."""
        regions.io.save(tmp_path / "picks")
        assert dc.annotations(regions.io.save(tmp_path / "picks")) == regions

    def test_times_are_written_unambiguously(self, regions):
        """A time is written the way DASCore writes every datetime."""
        assert "2020-01-01T00:00:10.000000000" in regions.io.to_csv()

    def test_a_nested_extra_is_written_as_its_document(self):
        """A cell a table has no column shape for is written as text."""
        frame = pd.DataFrame({"distance": [1.0], "meta": [{"n": 1}]})
        text = dc.AnnotationSet(frame, dims=("distance",)).io.to_csv()
        assert '{""n"": 1}' in text

    def test_a_frozen_nested_extra_is_one_document(self):
        """A mapping held frozen inside another is written as JSON, not as text."""
        frame = pd.DataFrame({"distance": [1.0], "meta": [{"a": {"b": [1]}}]})
        text = dc.AnnotationSet(frame, dims=("distance",)).io.to_csv()
        assert '{""a"": {""b"": [1]}}' in text

    def test_a_sequence_extra_is_comma_separated(self):
        """A sequence cell, held as a tuple, is written as comma-separated text."""
        frame = pd.DataFrame({"distance": [1.0], "codes": [["x", "y"]]})
        text = dc.AnnotationSet(frame, dims=("distance",)).io.to_csv()
        assert '"x, y"' in text

    def test_an_extra_json_cannot_spell(self):
        """A nested value with no json type is written as its text."""
        frame = pd.DataFrame({"distance": [1.0], "meta": [{"s": Decimal("1.5")}]})
        text = dc.AnnotationSet(frame, dims=("distance",)).io.to_csv()
        assert '{""s"": ""1.5""}' in text

    def test_the_attrs_name_their_model(self, regions, tmp_path):
        """The document says what it holds, as every stored object does."""
        directory = regions.io.save(tmp_path / "picks")
        text = (directory / "attrs.json").read_text()
        assert '"object_type": "AnnotationSetAttrs"' in text

    def test_bases_are_their_documents(self, with_features, curve, tmp_path):
        """bases.json maps each key to the document its curve dumps."""
        directory = with_features.io.save(tmp_path / "picks")
        document = json.loads((directory / "bases.json").read_text())
        assert document == {"arrival": curve.model_dump(mode="json")}


class TestCollections:
    """Sets stored side by side read as one set which says where each row came from."""

    @pytest.fixture
    def collection(self, regions, picks, tmp_path):
        """A directory holding two sets, each stating its own dimensions."""
        root = tmp_path / "sets"
        regions.io.save(root / "hand")
        picks.io.save(root / "phasenet")
        return root

    def test_a_dimension_one_set_leaves_blank(self, tmp_path):
        """A set stating no time beside one which does still holds times."""
        root = tmp_path / "sets"
        blank = pd.DataFrame(
            {"id": ["a"], "time_min": [None], "time_max": [None], "distance": [1.0]}
        )
        dc.AnnotationSet(blank, dims=DIMS).io.save(root / "one")
        stated = pd.DataFrame(
            {"id": ["b"], "time_min": TIMES[:1], "time_max": TIMES[1:2]}
        )
        dc.AnnotationSet(stated, dims=DIMS).io.save(root / "two")
        merged = dc.annotations(root)
        assert merged.annotations["time_min"].dtype == np.dtype("datetime64[ns]")
        assert dc.annotations(merged.io.save(tmp_path / "flat")) == merged

    def test_reads_as_one_set(self, collection, regions, picks):
        """Every set is in the table, in the dimensions all of them state."""
        loaded = dc.annotations(collection)
        assert len(loaded.annotations) == len(regions.annotations) + len(
            picks.annotations
        )
        assert loaded.dims == ("distance", "time")

    def test_names_the_set_each_row_came_from(self, collection):
        """The directory name is the label, in the set column and in attrs.sets."""
        loaded = dc.annotations(collection)
        assert {x.set for x in loaded} == {"hand", "phasenet"}
        assert sorted(loaded.attrs.sets) == ["hand", "phasenet"]

    def test_keeps_what_a_set_states_for_itself(self, collection, picks):
        """A child's dimensions and provenance survive without filling rows."""
        stated = dc.annotations(collection).attrs.sets["phasenet"]
        assert stated.dims == ("time",)
        assert stated.creation_info.author == "phasenet"
        assert stated.acquisition_key == picks.attrs.acquisition_key

    def test_a_row_keeps_its_own_provenance(self, collection, regions, picks):
        """The data a row was picked on is the data of its set."""
        loaded = dc.annotations(collection)
        keys = {x.set: (x.acquisition_key, x.data_id) for x in loaded}
        assert keys["hand"] == (regions.attrs.acquisition_key, "decimated")
        assert keys["phasenet"] == (picks.attrs.acquisition_key, "")

    def test_round_trips_through_one_directory(self, collection, tmp_path):
        """A collection saves flat, and what it read is what it reads back."""
        loaded = dc.annotations(collection)
        assert dc.annotations(loaded.io.save(tmp_path / "flat")) == loaded

    def test_an_id_in_two_sets(self, regions, tmp_path):
        """An annotation id is an address into the collection."""
        root = tmp_path / "sets"
        regions.io.save(root / "hand")
        regions.io.save(root / "again")
        with pytest.raises(InvalidAnnotationError, match=r"annotation id r1.*again"):
            dc.annotations(root)

    def test_a_feature_id_in_two_sets(self, tmp_path):
        """Feature ids are unique across a collection, implied ones too."""
        root = tmp_path / "sets"
        frame = pd.DataFrame({"feature_id": ["e1"], "time": [1.0]})
        for name in ("hand", "auto"):
            dc.AnnotationSet(frame, dims=DIMS).io.save(root / name)
        with pytest.raises(InvalidAnnotationError, match=r"feature id e1.*auto and"):
            dc.annotations(root)

    def test_features_and_bases_merge(self, with_features, curve, tmp_path):
        """Each table merges, labeled, and the bases join one mapping."""
        root = tmp_path / "sets"
        with_features.io.save(root / "hand")
        _path_set().io.save(root / "auto")
        loaded = dc.annotations(root)
        assert list(loaded.features["set"]) == ["auto", "hand", "hand"]
        assert loaded.bases == {"arrival": curve}
        assert loaded["p2"].set == "hand"

    def test_child_text_declarations_survive_a_flat_save(self, tmp_path):
        """A flattened collection reads its children's text columns as text."""
        root = tmp_path / "sets"
        for name in ("a", "b"):
            frame = pd.DataFrame(
                {"feature_id": [f"f{name}"], "time": [1.0], "label": ["002"]}
            )
            features = pd.DataFrame({"id": [f"f{name}"], "code": ["001"]})
            dc.AnnotationSet(
                frame,
                features=features,
                dims=("time",),
                annotation_columns={"label": {"dtype": "str"}},
                feature_columns={"code": {"dtype": "str"}},
            ).io.save(root / name)
        loaded = dc.annotations(root)
        flat = dc.annotations(loaded.io.save(tmp_path / "flat"))
        assert list(flat.features["code"]) == ["001", "001"]
        assert list(flat.annotations["label"]) == ["002", "002"]
        assert flat == loaded

    def test_child_text_declarations_survive_a_bare_table(self, tmp_path):
        """A collection written bare reads its children's text columns as text."""
        root = tmp_path / "sets"
        for name, code in (("a", "001"), ("b", "002")):
            frame = pd.DataFrame({"time": [1.0], "station": [code]})
            dc.AnnotationSet(
                frame,
                dims=("time",),
                annotation_columns={"station": {"dtype": "string"}},
            ).io.save(root / name)
        loaded = dc.annotations(root)
        path = tmp_path / "bare.csv"
        loaded.io.to_csv(path)
        bare = dc.annotations(path, dims=loaded.dims, attrs=loaded.attrs)
        assert list(bare.annotations["station"]) == ["001", "002"]

    def test_child_dtype_declarations_survive_a_flat_save(self, tmp_path):
        """Children agreeing on a dtype keep it through a flat save."""
        root = tmp_path / "sets"
        for name, count in (("a", [1, None]), ("b", [2, 3])):
            frame = pd.DataFrame(
                {"time": [1.0, 2.0], "count": pd.array(count, dtype="Int64")}
            )
            dc.AnnotationSet(
                frame,
                dims=("time",),
                annotation_columns={"count": {"dtype": "Int64"}},
            ).io.save(root / name)
        loaded = dc.annotations(root)
        flat = dc.annotations(loaded.io.save(tmp_path / "flat"))
        assert flat.annotations["count"].dtype.name == "Int64"
        assert flat == loaded

    def test_children_disagreeing_on_a_dtype(self, tmp_path):
        """Where children declare different dtypes the column is inferred."""
        root = tmp_path / "sets"
        for name, dtype in (("a", "Int64"), ("b", "float64")):
            frame = pd.DataFrame({"time": [1.0], "count": [1]}).astype({"count": dtype})
            dc.AnnotationSet(
                frame, dims=("time",), annotation_columns={"count": {"dtype": dtype}}
            ).io.save(root / name)
        flat = dc.annotations(dc.annotations(root).io.save(tmp_path / "flat"))
        assert list(flat.annotations["count"]) == [1, 1]

    def test_a_basis_key_naming_two_curves(self, with_features, tmp_path):
        """One key names one curve across the sets loaded together."""
        root = tmp_path / "sets"
        with_features.io.save(root / "hand")
        line = Line(start={"distance": 0.0}, end={"distance": 1.0})
        other = dc.AnnotationSet(
            pd.DataFrame({"distance": [1.0]}), dims=DIMS, bases={"arrival": line}
        )
        other.io.save(root / "auto")
        with pytest.raises(InvalidAnnotationError, match="different curves"):
            dc.annotations(root)

    def test_a_set_which_states_only_attributes(self, collection):
        """A directory of attributes and nothing else is half a set."""
        half = collection / "empty"
        half.mkdir()
        (half / "attrs.json").write_text('{"dims": ["time"]}')
        with pytest.raises(
            InvalidAnnotationError, match="states the attributes of a set but no"
        ):
            dc.annotations(collection)

    def test_a_tree_which_holds_no_set_at_all(self, tmp_path):
        """Nothing is refused until a directory turns out to be a collection."""
        root = tmp_path / "data"
        half = root / "notes"
        half.mkdir(parents=True)
        (half / "attrs.json").write_text('{"dims": ["time"]}')
        with pytest.raises(InvalidAnnotationError, match="holds no annotations table"):
            dc.annotations(root, dims=("time",))

    def test_a_tree_of_collections(self, regions, tmp_path):
        """Sets loaded together are one collection, not a tree of them."""
        root = tmp_path / "sets"
        regions.io.save(root / "outer" / "inner")
        regions.io.save(root / "hand")
        with pytest.raises(InvalidAnnotationError, match="not a tree"):
            dc.annotations(root)

    def test_a_set_which_also_holds_sets(self, regions, picks, tmp_path):
        """A directory stating annotations is the set, whatever sits below it."""
        root = regions.io.save(tmp_path / "sets")
        picks.io.save(root / "hand")
        assert dc.annotations(root) == regions

    def test_a_set_beside_a_folder_of_its_own(self, regions, tmp_path):
        """A folder someone kept beside the tables is not this format's business."""
        root = regions.io.save(tmp_path / "picks")
        (root / "backup").mkdir()
        (root / "backup" / "attrs.json").write_text('{"dims": ["time"]}')
        assert dc.annotations(root) == regions

    def test_a_table_beside_the_sets(self, collection):
        """A table where every set is a directory names no set."""
        (collection / "notes.csv").write_text("note\nnoise\n")
        with pytest.raises(InvalidAnnotationError, match="name no set"):
            dc.annotations(collection)

    def test_dimensions_for_the_sets_which_state_none(self, tmp_path):
        """They reach the children which declare none, and without them none do."""
        root = tmp_path / "sets"
        for name in ("hand", "auto"):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "annotations.csv").write_text(
                f"id,note,time_min,time_max\n{name},noise,1.0,2.0\n"
            )
        assert len(dc.annotations(root, dims=("time",))) == 2
        with pytest.raises(InvalidAnnotationError, match="states no dimensions"):
            dc.annotations(root)

    def test_dimensions_stated_beside_the_sets(self, tmp_path):
        """A collection may declare them, and then refuses a caller restating them."""
        root = tmp_path / "sets"
        directory = root / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.csv").write_text("note,time_min,time_max\nq,1,2\n")
        (root / "attrs.json").write_text('{"dims": ["time"]}')
        assert dc.annotations(root).dims == ("time",)
        with pytest.raises(
            InvalidAnnotationError, match="a directory of sets stating its own"
        ):
            dc.annotations(root, dims=("time",))

    def test_a_dimension_spelled_two_ways(self, tmp_path):
        """A set of values beside a set of ranges merges; rows keep their own."""
        root = tmp_path / "sets"
        ranges = pd.DataFrame({"time_min": [1.0], "time_max": [2.0]})
        dc.AnnotationSet(ranges, dims=("time",)).io.save(root / "ranges")
        values = pd.DataFrame({"time": [5.0]})
        dc.AnnotationSet(values, dims=("time",)).io.save(root / "points")
        bounds = _bounds(dc.annotations(root))
        assert bounds == [{"time": (5.0, 5.0)}, {"time": (1.0, 2.0)}]

    def test_a_set_which_states_a_set_column(self, tmp_path):
        """The set column names the set, so a set may not fill it in."""
        root = tmp_path / "sets"
        directory = root / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.csv").write_text("set,note,time\nother,noise,1.0\n")
        with pytest.raises(InvalidAnnotationError, match="states a set column"):
            dc.annotations(root, dims=("time",))

    def test_sets_stated_twice(self, collection):
        """A collection states each of its sets once."""
        document = {"dims": ["time"], "sets": {"hand": {"dims": ["time"]}}}
        (collection / "attrs.json").write_text(json.dumps(document))
        with pytest.raises(
            InvalidAnnotationError, match="states each of its sets once"
        ):
            dc.annotations(collection)

    def test_hidden_directories_are_not_sets(self, collection, regions):
        """A hidden name beside the sets describes the data, not them."""
        regions.io.save(collection / ".annotations")
        assert sorted(dc.annotations(collection).attrs.sets) == ["hand", "phasenet"]

    def test_a_directory_which_is_no_set(self, collection):
        """A directory participating in no convention here is left alone."""
        (collection / "figures").mkdir()
        assert len(dc.annotations(collection)) == 4

    def test_a_row_keeps_the_acquisition_it_names(self, regions, tmp_path):
        """A row naming its own acquisition outranks its set's, merged or not."""
        root = tmp_path / "sets"
        regions.io.save(root / "hand")
        frame = pd.DataFrame(
            {
                "id": ["m1", "m2"],
                "acquisition_key": ["NET.OTHER.00.das", None],
                "time_min": TIMES[:2],
                "time_max": TIMES[:2] + np.timedelta64(1, "s"),
            }
        )
        other = dc.AnnotationSet(
            frame, dims=("time",), acquisition_key="NET.SET.00.das"
        )
        other.io.save(root / "auto")
        keys = [(x.set, x.acquisition_key) for x in dc.annotations(root)]
        assert keys == [
            ("auto", "NET.OTHER.00.das"),
            ("auto", "NET.SET.00.das"),
            ("hand", regions.attrs.acquisition_key),
            ("hand", regions.attrs.acquisition_key),
        ]

    def test_what_the_collection_states_reaches_the_merged_set(self, tmp_path):
        """A collection may state its own provenance beside its sets."""
        root = tmp_path / "sets"
        directory = root / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.csv").write_text("note,time\nq,1\n")
        document = {
            "dims": ["time"],
            "acquisition_key": "NET.COLL.00.das",
            "data_id": "coll",
        }
        (root / "attrs.json").write_text(json.dumps(document))
        loaded = dc.annotations(root)
        assert loaded.attrs.data_id == "coll"
        (feature,) = list(loaded)
        assert (feature.acquisition_key, feature.data_id) == ("NET.COLL.00.das", "coll")

    def test_dimensions_given_for_a_set_which_states_its_own(self, collection):
        """Dropping the argument silently is the worse failure, here as anywhere."""
        with pytest.raises(InvalidAnnotationError, match="which states its own"):
            dc.annotations(collection, dims=("distance", "time"))

    def test_a_dimension_stated_in_two_kinds(self, tmp_path):
        """One set stating a time in seconds and another in dates cannot merge."""
        root = tmp_path / "sets"
        for name, cell in (("clock", "2020-01-01T00:00:01"), ("numeric", "1.5")):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "annotations.csv").write_text(f"note,time\nq,{cell}\n")
            (directory / "attrs.json").write_text('{"dims": ["time"]}')
        with pytest.raises(InvalidAnnotationError, match="different kinds of value"):
            dc.annotations(root)

    def test_two_kinds_across_spellings(self, tmp_path):
        """Seconds as values in one set and dates as ranges in another is two kinds."""
        root = tmp_path / "sets"
        values = pd.DataFrame({"time": [1.5]})
        dc.AnnotationSet(values, dims=("time",)).io.save(root / "a")
        ranges = pd.DataFrame({"time_min": TIMES[:1], "time_max": TIMES[1:2]})
        dc.AnnotationSet(ranges, dims=("time",)).io.save(root / "b")
        with pytest.raises(InvalidAnnotationError, match="different kinds of value"):
            dc.annotations(root)

    def test_a_dimension_no_row_of_one_set_states(self, picks, tmp_path):
        """A column every row leaves empty states no kind, so it agrees."""
        root = tmp_path / "sets"
        picks.io.save(root / "phasenet")
        blank = root / "quiet"
        blank.mkdir(parents=True)
        (blank / "annotations.csv").write_text(
            "note,time_min,time_max,distance\nq,,,5\n"
        )
        (blank / "attrs.json").write_text('{"dims": ["time", "distance"]}')
        assert len(dc.annotations(root)) == len(picks) + 1

    def test_a_column_which_is_a_dimension_in_one_set_only(self, tmp_path):
        """A column another set dimensions must not silently become a bound."""
        root = tmp_path / "sets"
        notes = root / "notes"
        notes.mkdir(parents=True)
        (notes / "annotations.csv").write_text("note,time,distance\nq,1,shallow\n")
        (notes / "attrs.json").write_text('{"dims": ["time"]}')
        boxes = root / "boxes"
        boxes.mkdir(parents=True)
        (boxes / "annotations.csv").write_text("note,time,distance\nb,2,50\n")
        (boxes / "attrs.json").write_text('{"dims": ["time", "distance"]}')
        with pytest.raises(InvalidAnnotationError, match="without declaring"):
            dc.annotations(root)

    @pytest.mark.skipif(FOLDS_CASE, reason="this filesystem holds one of the two")
    def test_set_names_which_differ_only_in_case(self, regions, picks, tmp_path):
        """A set name is a label, so it must name one set on any filesystem."""
        root = tmp_path / "sets"
        regions.io.save(root / "hand")
        picks.io.save(root / "HAND")
        with pytest.raises(InvalidAnnotationError, match="differ only in case"):
            dc.annotations(root)

    def test_a_hidden_table_beside_the_sets(self, collection):
        """A half-copied file is a companion, not a table which names no set."""
        (collection / ".annotations.csv").write_text("note\nnoise\n")
        assert len(dc.annotations(collection)) == 4

    def test_a_stray_table_whatever_its_case(self, collection):
        """The suffix is matched as the loader matches every other one."""
        (collection / "NOTES.CSV").write_text("note\nnoise\n")
        with pytest.raises(InvalidAnnotationError, match=r"NOTES\.CSV"):
            dc.annotations(collection)

    def test_a_set_named_in_upper_case(self, tmp_path):
        """A set states its table once, in whichever case it spells the suffix."""
        directory = tmp_path / "sets" / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.CSV").write_text("note,time\nq,1\n")
        assert len(dc.annotations(tmp_path / "sets", dims=("time",))) == 1

    def test_a_directory_of_other_things(self, collection):
        """A directory which states no annotations is left alone, empty or not."""
        figures = collection / "figures"
        figures.mkdir()
        (figures / "map.png").write_bytes(b"not an image either")
        (figures / "notes.txt").write_text("nothing to do with the format")
        assert len(dc.annotations(collection)) == 4

    def test_a_collection_saved_flat_is_not_a_member(self, collection, tmp_path):
        """A directory this library wrote is named for what it is."""
        root = tmp_path / "outer"
        dc.annotations(collection).io.save(root / "merged")
        with pytest.raises(InvalidAnnotationError, match="already a collection"):
            dc.annotations(root)

    @pytest.mark.skipif(not DENIES_ACCESS, reason="a mode cannot deny a read here")
    def test_a_directory_which_cannot_be_read(self, collection):
        """A tightened permission is named as an annotation error, not an OSError."""
        locked = collection / "locked"
        locked.mkdir()
        locked.chmod(0o000)
        try:
            with pytest.raises(InvalidAnnotationError, match="Could not read"):
                dc.annotations(collection)
        finally:
            locked.chmod(0o755)

    def test_a_label_naming_no_set(self, collection, tmp_path):
        """A label reaches back to what its set says, so it names one."""
        flat = dc.annotations(collection).io.save(tmp_path / "flat")
        table = flat / "annotations.csv"
        table.write_text(table.read_text().replace(",hand", ",typo"))
        with pytest.raises(InvalidAnnotationError, match="name no set stated here"):
            dc.annotations(flat)

    def test_a_row_with_no_label(self, collection, tmp_path):
        """A row loaded with others says which of them it came from."""
        flat = dc.annotations(collection).io.save(tmp_path / "flat")
        table = flat / "annotations.csv"
        table.write_text(table.read_text().replace(",hand", ",", 1))
        with pytest.raises(InvalidAnnotationError, match="state no set"):
            dc.annotations(flat)

    def test_a_feature_with_no_label(self, tmp_path):
        """The features table of a collection labels its rows too."""
        root = tmp_path / "sets"
        _path_set().io.save(root / "auto")
        flat = dc.annotations(root).io.save(tmp_path / "flat")
        table = flat / "features.csv"
        table.write_text(table.read_text().replace(",auto", ","))
        with pytest.raises(InvalidAnnotationError, match="of the features state no"):
            dc.annotations(flat)

    def test_a_table_with_no_label_column(self, collection, tmp_path):
        """Sets stated with no column to name them leave every row adrift."""
        flat = dc.annotations(collection).io.save(tmp_path / "flat")
        table = flat / "annotations.csv"
        frame = dc.annotations(flat).annotations.drop(columns="set")
        table.write_text(frame.to_csv(index=False))
        with pytest.raises(InvalidAnnotationError, match="no set column"):
            dc.annotations(flat)

    def test_a_collection_of_empty_sets(self, tmp_path):
        """A collection holding no rows at all labels none of them."""
        root = tmp_path / "sets"
        for name in ("hand", "auto"):
            dc.AnnotationSet(None, dims=("time",)).io.save(root / name)
        loaded = dc.annotations(root)
        assert len(loaded) == 0
        assert sorted(loaded.attrs.sets) == ["auto", "hand"]

    def test_a_set_column_on_a_set_of_its_own(self):
        """A set which states no sets is not a collection, so its labels are its own."""
        frame = pd.DataFrame({"set": ["whatever"], "time": [1.0]})
        (feature,) = list(dc.AnnotationSet(frame, dims=("time",)))
        assert feature.set == "whatever"

    def test_a_set_with_no_annotations(self, picks, tmp_path):
        """A set which states nothing is still one of the sets loaded."""
        root = tmp_path / "sets"
        picks.io.save(root / "phasenet")
        dc.AnnotationSet(None, dims=("time",)).io.save(root / "empty")
        loaded = dc.annotations(root)
        assert len(loaded) == len(picks)
        assert sorted(loaded.attrs.sets) == ["empty", "phasenet"]

    def test_paths_from_two_sets(self, with_features, tmp_path):
        """Each path still reads as the shape it was."""
        root = tmp_path / "sets"
        with_features.io.save(root / "hand")
        _path_set().io.save(root / "auto")
        loaded = dc.annotations(root)
        assert loaded.geometry("p1").vertices[0]["distance"] == (10.0, 95.0, 185.0)
        assert loaded.geometry("p9").vertices[0]["distance"] == (1000.0, 1100.0, 1200.0)

    def test_paths_in_different_dimensions(self, with_features, tmp_path):
        """Features may be drawn in different dimensions, from different sets."""
        root = tmp_path / "sets"
        with_features.io.save(root / "hand")
        _path_set("f1", (1.0, 2.0), dims=("distance",)).io.save(root / "flat")
        loaded = dc.annotations(root)
        assert loaded.geometry("f1").dims == ("distance",)
        assert loaded.geometry("p1").dims == ("distance", "time")


class TestDeclaringDimensionsInTheTable:
    """A bare table has no attrs file, so it may declare them above its header."""

    def test_a_bare_table_declares_them(self, tmp_path):
        """The dimensions travel with the file rather than with the call."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: distance, time\nnote,time_min,time_max\nq,1,2\n")
        loaded = dc.annotations(path)
        assert loaded.dims == ("distance", "time")
        assert set(loaded.annotations.columns) == {
            "note",
            "time_min",
            "time_max",
            "feature_id",
        }
        assert _extras(loaded)[0]["note"] == "q"
        assert _bounds(loaded)[0]["time"] == (1.0, 2.0)

    def test_other_comments_are_comments(self, tmp_path):
        """A line above the header which declares nothing says nothing."""
        path = tmp_path / "picks.csv"
        path.write_text("# picked by hand\n#dims:time\nnote,time\nq,1\n")
        loaded = dc.annotations(path)
        assert loaded.dims == ("time",)
        assert _bounds(loaded)[0]["time"] == (1.0, 1.0)

    def test_restating_them_is_allowed(self, tmp_path):
        """Two spellings of one fact agree or they are not one fact."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\nnote,time\nq,1\n")
        assert dc.annotations(path, dims=("time",)).dims == ("time",)

    def test_disagreeing_with_the_caller(self, tmp_path):
        """The table's dimensions and the caller's are not merged."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\nnote,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="where the two agree"):
            dc.annotations(path, dims=("distance",))

    def test_a_header_which_starts_with_the_mark(self, tmp_path):
        """A column may be named `#note`, and that line is the header."""
        path = tmp_path / "picks.csv"
        path.write_text("#note,group,time\nfirst,a,1\nsecond,b,2\n")
        loaded = dc.annotations(path, dims=("time",))
        assert len(loaded) == 2
        assert _extras(loaded)[0]["#note"] == "first"

    def test_a_declaration_commented_out(self, tmp_path):
        """A struck-out declaration declares nothing, so the line is a header."""
        path = tmp_path / "picks.csv"
        path.write_text("## dims: time\nnote,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="cells where its header"):
            dc.annotations(path, dims=("time",))

    def test_a_header_which_reads_as_a_comment(self, tmp_path):
        """A column may be named `# note`, which this library writes unquoted."""
        frame = pd.DataFrame({"# note": ["first", "second"], "time": [1.0, 2.0]})
        picks = dc.AnnotationSet(frame, dims=("time",))
        path = tmp_path / "picks.csv"
        picks.io.to_csv(path)
        loaded = dc.annotations(path, dims=("time",))
        assert len(loaded) == 2
        assert _extras(loaded)[0]["# note"] == "first"

    def test_comments_beside_a_declaration(self, tmp_path):
        """Where a table declares its dimensions, comments ride with it."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\n# picked by hand\nnote,time\nq,1\n")
        loaded = dc.annotations(path)
        assert loaded.dims == ("time",)
        assert set(loaded.annotations.columns) == {"note", "time", "feature_id"}

    def test_the_keyword_is_read_in_any_case(self, tmp_path):
        """A hand-authored line is read as written, whatever case it names."""
        path = tmp_path / "picks.csv"
        path.write_text("# Dims:  time , distance\nnote,time\nq,1\n")
        assert dc.annotations(path).dims == ("time", "distance")

    def test_a_blank_line_above_the_declaration(self, tmp_path):
        """Blank lines above the header are skipped with the comments."""
        path = tmp_path / "picks.csv"
        path.write_text("\n# dims: time\n\nnote,time\nq,1\n")
        assert dc.annotations(path).dims == ("time",)

    def test_a_table_which_cannot_be_decoded(self, tmp_path):
        """A table which cannot be read has no dimensions to be found in it."""
        path = tmp_path / "picks.csv"
        path.write_bytes(b"# dims: time\nnote,time\n\xff\xfe,1\n")
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(path)

    def test_a_comment_holding_a_quote(self, tmp_path):
        """A comment is one line, whatever a csv reader makes of its quotes."""
        path = tmp_path / "picks.csv"
        path.write_text('# dims: time\n# it is ,"odd\nnote,time\nq,1\n')
        loaded = dc.annotations(path)
        assert loaded.dims == ("time",)
        assert set(loaded.annotations.columns) == {"note", "time", "feature_id"}

    def test_disagreeing_with_what_the_attrs_state(self, regions, tmp_path):
        """The message says where the other spelling came from."""
        directory = regions.io.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        table.write_text("# dims: depth\n" + table.read_text())
        with pytest.raises(InvalidAnnotationError, match="is stated in its attributes"):
            dc.annotations(directory)

    def test_a_child_declaring_its_own_above_its_table(self, tmp_path):
        """A pragma is a set stating its dimensions, as its attrs would be."""
        root = tmp_path / "sets"
        for name, dim in (("hand", "time"), ("auto", "distance")):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "annotations.csv").write_text(
                f"# dims: {dim}\nnote,{dim}\nq,1\n"
            )
        (root / "attrs.json").write_text('{"dims": ["time"]}')
        with pytest.raises(InvalidAnnotationError, match="which states its own"):
            dc.annotations(root)

    def test_a_set_directory_may_declare_them(self, tmp_path):
        """A hand-made set directory need not carry an attrs file."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "annotations.csv").write_text("# dims: time\nnote,time\nq,1\n")
        loaded = dc.annotations(directory)
        assert loaded.dims == ("time",)
        assert _bounds(loaded)[0]["time"] == (1.0, 1.0)

    def test_sets_loaded_together_may_declare_them(self, tmp_path):
        """Each set in a collection may state its own, above its own table."""
        root = tmp_path / "sets"
        for name, dim in (("hand", "time"), ("auto", "distance")):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "annotations.csv").write_text(
                f"# dims: {dim}\nnote,{dim}\nq,1\n"
            )
        loaded = dc.annotations(root)
        assert loaded.dims == ("distance", "time")
        assert _bounds(loaded) == [{"distance": (1.0, 1.0)}, {"time": (1.0, 1.0)}]

    def test_declared_twice(self, tmp_path):
        """One table states its dimensions once."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\n# dims: distance\nnote,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="more than once"):
            dc.annotations(path)

    def test_declared_but_named_none(self, tmp_path):
        """A declaration which names nothing declares nothing."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims:\nnote,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="names none"):
            dc.annotations(path)

    def test_features_declare_nothing(self, with_features, tmp_path):
        """Features hold no coordinates, so they declare no dimensions."""
        directory = with_features.io.save(tmp_path / "picks")
        table = directory / "features.csv"
        table.write_text("# dims: time\n" + table.read_text())
        with pytest.raises(InvalidAnnotationError, match="hold no coordinates"):
            dc.annotations(directory)

    def test_features_take_no_preamble(self, with_features, tmp_path):
        """Features declare nothing, so they have nothing to comment beside."""
        directory = with_features.io.save(tmp_path / "picks")
        table = directory / "features.csv"
        table.write_text("# drawn on a screen\n" + table.read_text())
        with pytest.raises(InvalidAnnotationError, match="cells where its header"):
            dc.annotations(directory)


class TestCarriedAnnotations:
    """A directory of data carries what it was annotated with, hidden beside it."""

    @pytest.fixture
    def data(self, tmp_path):
        """A directory of data, with nothing of this format visible in it."""
        directory = tmp_path / "data"
        directory.mkdir()
        (directory / "das_1.h5").write_text("pretend this is data")
        return directory

    def test_a_carried_set(self, data, regions):
        """Loading a directory of data loads the annotations it carries."""
        regions.io.save(data / ".annotations")
        assert dc.annotations(data) == regions

    def test_a_carried_collection(self, data, regions, picks):
        """What is carried may be many named sets, as anywhere else."""
        regions.io.save(data / ".annotations" / "hand")
        picks.io.save(data / ".annotations" / "phasenet")
        assert sorted(dc.annotations(data).attrs.sets) == ["hand", "phasenet"]

    def test_a_carried_table(self, data):
        """The bare table spelling is carried under the same name."""
        (data / ".annotations.csv").write_text("# dims: time\nnote,time\nq,1\n")
        assert dc.annotations(data).dims == ("time",)

    def test_a_carried_table_takes_what_it_does_not_state(self, data):
        """A bare table states no attributes, so the caller may state them."""
        (data / ".annotations.csv").write_text("note,time\nq,1\n")
        loaded = dc.annotations(data, attrs={"dims": ("time",), "data_id": "de"})
        assert loaded.dims == ("time",)
        assert loaded.attrs.data_id == "de"

    def test_a_carried_set_directory_states_its_own(self, data, regions):
        """A carried directory holds its attributes, as any set directory does."""
        regions.io.save(data / ".annotations")
        with pytest.raises(InvalidAnnotationError, match="which states them"):
            dc.annotations(data, attrs={"dims": DIMS})

    def test_carried_twice(self, data, regions):
        """A directory states what it carries once."""
        regions.io.save(data / ".annotations")
        (data / ".annotations.csv").write_text("note,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="more than once"):
            dc.annotations(data)

    def test_the_wrong_kind_of_thing(self, data):
        """Something under the blessed name in a form it does not take."""
        (data / ".annotations").write_text("not a set")
        with pytest.raises(InvalidAnnotationError, match="is a file"):
            dc.annotations(data)

    def test_a_table_name_holding_a_directory(self, data):
        """The csv spelling is a table; a directory is the other one."""
        (data / ".annotations.csv").mkdir()
        with pytest.raises(InvalidAnnotationError, match="is a directory"):
            dc.annotations(data)

    def test_a_visible_set_is_the_set(self, data, regions, picks):
        """A directory stating annotations is a set, not something carrying one."""
        picks.io.save(data / ".annotations")
        regions.io.save(data)
        assert dc.annotations(data) == regions

    def test_carrying_nothing(self, data):
        """A directory with no annotations says so, and names the convention."""
        with pytest.raises(InvalidAnnotationError, match=r"\.annotations"):
            dc.annotations(data, dims=DIMS)

    def test_find_annotations_judges_only_the_name(self, data):
        """The path comes back because of its name, not because it loads."""
        assert find_annotations(data) is None
        table = data / ".annotations.csv"
        table.write_text("this is not a table at all")
        assert find_annotations(data) == table

    def test_the_data_directory_keeps_its_own_attrs(self, data, regions):
        """A data directory's attrs file is about the data, so it is not read."""
        (data / "attrs.json").write_text('{"object_type": "SomethingElse"}')
        regions.io.save(data / ".annotations")
        assert dc.annotations(data) == regions


def _forge(frame: pd.DataFrame, path, documents: str) -> None:
    """Write a parquet file whose document footer this library did not write."""
    table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    kept = {**(table.schema.metadata or {}), DOCUMENT_KEY: documents}
    pyarrow.parquet.write_table(table.replace_schema_metadata(kept), path)


class TestPrivateColumns:
    """A column an author kept for themselves is read by nothing."""

    def test_a_bare_table(self, tmp_path):
        """A note on how something was deployed stays in the file."""
        path = tmp_path / "picks.csv"
        path.write_text(
            "id,note,distance_min,distance_max,_crew\nr1,noise,10.0,60.0,north crew\n"
        )
        loaded = dc.annotations(path, dims=DIMS)
        assert "_crew" not in loaded.annotations.columns
        assert _extras(loaded)[0]["note"] == "noise"

    def test_a_saved_set(self, regions, tmp_path):
        """One added to a written table changes nothing about the set."""
        directory = regions.io.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        header, *rows = table.read_text().splitlines()
        written = [f"{header},_crew", *[f"{row},north crew" for row in rows]]
        table.write_text("\n".join(written) + "\n")
        assert dc.annotations(directory) == regions

    def test_nothing_reads_what_it_holds(self, tmp_path):
        """A declaration a private column cannot meet is not checked."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "annotations.csv").write_text(
            "id,distance,_count\nr1,1.0,not a number\n"
        )
        (directory / "attrs.yaml").write_text(
            yaml.safe_dump(
                {
                    "object_type": "AnnotationSetAttrs",
                    "dims": list(DIMS),
                    "annotation_columns": {"_count": {"dtype": "Int64"}},
                }
            )
        )
        assert len(dc.annotations(directory)) == 1

    @pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
    def test_a_private_document_column(self, tmp_path):
        """Parquet reads one no further than a CSV does."""
        frame = pd.DataFrame({"id": ["r1"], "distance": [1.0], "_crew": ["{oops"]})
        path = tmp_path / "picks.parquet"
        _forge(frame, path, '["_crew"]')
        loaded = dc.annotations(path, dims=DIMS)
        assert "_crew" not in loaded.annotations.columns

    def test_a_table_of_only_private_columns(self, tmp_path):
        """Rows no column of the set states are refused, not lost."""
        path = tmp_path / "picks.csv"
        path.write_text("_crew\nnorth crew\n")
        with pytest.raises(InvalidAnnotationError, match="read by nothing"):
            dc.annotations(path, dims=DIMS)

    def test_features(self, with_features, tmp_path):
        """Features are a table like any other, so they take one too."""
        directory = with_features.io.save(tmp_path / "picks")
        table = directory / "features.csv"
        header, *rows = table.read_text().splitlines()
        written = [f"{header},_source", *[f"{row},drawing 4" for row in rows]]
        table.write_text("\n".join(written) + "\n")
        assert dc.annotations(directory) == with_features


@pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
class TestParquet:
    """The same tables, with their types kept, for a set too big to want text."""

    @pytest.fixture
    def mixed(self) -> dc.AnnotationSet:
        """A set whose columns hold what a CSV would have to spell as text."""
        frame = pd.DataFrame(
            {
                "id": ["r1", "r2"],
                "value": ["car", 3],
                "tags": [["road", "car"], None],
                "time_min": [
                    np.datetime64("2020-01-01T00:00:10"),
                    np.datetime64("2020-01-01T00:00:20"),
                ],
                "time_max": [
                    np.datetime64("2020-01-01T00:00:12"),
                    np.datetime64("2020-01-01T00:00:22"),
                ],
                "score": [0.9, 0.2],
                "checked": [True, False],
                "meta": [{"a": 1}, None],
            }
        )
        return dc.AnnotationSet(frame, dims=DIMS, acquisition_key="NET.ARR.00.das")

    def test_a_bare_table(self, mixed, tmp_path):
        """A set of regions is one file, and reads back as the set it was."""
        loaded = dc.annotations(mixed.io.to_parquet(tmp_path / "picks.parquet"))
        assert loaded.annotations.equals(mixed.annotations)

    def test_the_dimensions_travel_with_the_file(self, mixed, tmp_path):
        """A parquet file states its dimensions where it can: its footer."""
        path = mixed.io.to_parquet(tmp_path / "picks.parquet")
        assert dc.annotations(path).dims == DIMS

    def test_restating_the_dimensions(self, mixed, tmp_path):
        """Agreement is allowed, disagreement is not, as with every spelling."""
        path = mixed.io.to_parquet(tmp_path / "picks.parquet")
        assert dc.annotations(path, dims=DIMS).dims == DIMS
        with pytest.raises(InvalidAnnotationError, match="where the two agree"):
            dc.annotations(path, dims=("depth",))

    def test_kinds_a_csv_would_lose(self, mixed, tmp_path):
        """A column with no one type is written as documents, not as text."""
        loaded = dc.annotations(mixed.io.to_parquet(tmp_path / "picks.parquet"))
        extras = _extras(loaded)
        assert [type(x["value"]).__name__ for x in extras] == ["str", "int"]
        assert extras[0]["meta"] == {"a": 1}
        assert extras[0]["tags"] == ("road", "car")

    def test_text_stays_text(self, tmp_path):
        """A typed format has a boolean, so a cell reading 'true' is the word."""
        frame = pd.DataFrame({"note": ["true"], "time": [1.0]})
        picks = dc.AnnotationSet(frame, dims=("time",))
        loaded = dc.annotations(picks.io.to_parquet(tmp_path / "picks.parquet"))
        assert _extras(loaded)[0]["note"] == "true"

    def test_a_set_of_no_annotations(self, tmp_path):
        """A set which states nothing writes a table which states nothing."""
        empty = dc.AnnotationSet(None, dims=("time",))
        saved = empty.io.save(tmp_path / "picks", format="parquet")
        assert dc.annotations(saved) == empty
        loaded = dc.annotations(empty.io.to_parquet(tmp_path / "picks.parquet"))
        assert len(loaded) == 0

    def test_an_empty_set_beside_a_full_one(self, picks, tmp_path):
        """One set holding nothing does not take its collection down with it."""
        root = tmp_path / "sets"
        picks.io.save(root / "phasenet", format="parquet")
        dc.AnnotationSet(None, dims=("time",)).io.save(root / "empty", format="parquet")
        assert len(dc.annotations(root)) == len(picks)

    def test_numbers_numpy_made(self, tmp_path):
        """A value from numpy is the number it is, not the text str() gives."""
        frame = pd.DataFrame(
            {"value": [np.int64(3), 5, "text"], "time": [1.0, 2.0, 3.0]}
        )
        picks = dc.AnnotationSet(frame, dims=("time",))
        loaded = dc.annotations(picks.io.save(tmp_path / "picks", format="parquet"))
        assert [x["value"] for x in _extras(loaded)] == [3, 5, "text"]

    def test_a_time_inside_a_document(self, tmp_path):
        """A time has no JSON type, so it is spelled as DASCore spells one."""
        frame = pd.DataFrame(
            {
                "meta": [{"when": np.datetime64("2020-01-01T00:00:01")}, "note"],
                "time": [1.0, 2.0],
            }
        )
        picks = dc.AnnotationSet(frame, dims=("time",))
        loaded = dc.annotations(picks.io.save(tmp_path / "picks", format="parquet"))
        assert _extras(loaded)[0]["meta"] == {"when": "2020-01-01T00:00:01.000000000"}

    @pytest.mark.parametrize(
        ("stated", "message"),
        [(json.dumps("note"), "it is a list of names"), ("{oops", "not a JSON")],
    )
    def test_a_footer_this_cannot_read(self, stated, message, tmp_path):
        """A footer another writer left is read, or named where it cannot be."""
        path = tmp_path / "picks.parquet"
        _forge(pd.DataFrame({"note": ["a"], "time": [1.0]}), path, stated)
        with pytest.raises(InvalidAnnotationError, match=message):
            dc.annotations(path, dims=("time",))

    def test_a_directory(self, with_features, curve, tmp_path):
        """Every part a set states is written under its own name."""
        directory = with_features.io.save(tmp_path / "picks", format="parquet")
        assert sorted(x.name for x in directory.iterdir()) == [
            "annotations.parquet",
            "attrs.json",
            "bases.json",
            "features.parquet",
        ]
        loaded = dc.annotations(directory)
        assert loaded == with_features
        assert loaded["p2"].basis == curve

    def test_nested_feature_cells(self, tmp_path):
        """A nested feature cell, held frozen, reads back equal."""
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        features = pd.DataFrame({"id": ["a"], "meta": [{"values": [1, 2]}]})
        out = dc.AnnotationSet(frame, features=features, dims=DIMS)
        loaded = dc.annotations(out.io.save(tmp_path / "picks", format="parquet"))
        assert loaded == out
        assert loaded.features.loc[0, "meta"] == {"values": (1, 2)}

    def test_a_collection(self, regions, picks, tmp_path):
        """A set is a set whichever encoding it is written in."""
        root = tmp_path / "sets"
        regions.io.save(root / "hand", format="parquet")
        picks.io.save(root / "phasenet")
        assert sorted(dc.annotations(root).attrs.sets) == ["hand", "phasenet"]

    def test_carried_beside_data(self, regions, tmp_path):
        """The hidden name takes the parquet spelling too."""
        directory = tmp_path / "data"
        directory.mkdir()
        regions.io.to_parquet(directory / ".annotations.parquet")
        assert dc.annotations(directory).annotations.equals(regions.annotations)

    def test_carried_whatever_case_it_names(self, regions, tmp_path):
        """The carried name is matched as every other table name is."""
        directory = tmp_path / "data"
        directory.mkdir()
        regions.io.to_parquet(directory / ".annotations.PARQUET")
        assert dc.annotations(directory).annotations.equals(regions.annotations)

    def test_a_typed_column_which_cannot_be_a_dimension(self, tmp_path):
        """Stating a type is not stating one a coordinate can be."""
        path = tmp_path / "picks.parquet"
        write_parquet(pd.DataFrame({"note": ["a"], "time": [True]}), path)
        with pytest.raises(InvalidAnnotationError, match="where it states numbers"):
            dc.annotations(path, dims=("time",))

    def test_a_typed_order_which_is_not_a_number(self, tmp_path):
        """A member states its place in the order as a number, typed or not."""
        out = _path_set()
        directory = out.io.save(tmp_path / "picks", format="parquet")
        frame = out.annotations
        frame["seq"] = pd.date_range("2020-01-01", periods=len(frame))
        write_parquet(frame, directory / "annotations.parquet")
        with pytest.raises(InvalidAnnotationError, match="where it states a number"):
            dc.annotations(directory)

    def test_the_other_encoding_is_superseded(self, regions, tmp_path):
        """A set written twice states itself once, not once per encoding."""
        directory = regions.io.save(tmp_path / "picks")
        assert (directory / "annotations.csv").exists()
        regions.io.save(directory, format="parquet")
        assert not (directory / "annotations.csv").exists()
        assert dc.annotations(directory) == regions
        regions.io.save(directory)
        assert not (directory / "annotations.parquet").exists()

    def test_both_encodings_at_once(self, regions, tmp_path):
        """A directory holding both says two things; neither is chosen."""
        directory = regions.io.save(tmp_path / "picks")
        regions.io.to_parquet(directory / "annotations.parquet")
        with pytest.raises(InvalidAnnotationError, match="each of its parts once"):
            dc.annotations(directory)

    def test_a_bare_table_refuses_features(self, with_features, tmp_path):
        """One file holds only annotations, whatever its encoding."""
        with pytest.raises(ParameterError, match="holds features or bases"):
            with_features.io.to_parquet(tmp_path / "picks.parquet")

    def test_an_unknown_encoding(self, regions, tmp_path):
        """A set is written in an encoding it has, and says which it has."""
        with pytest.raises(ParameterError, match="not a table encoding"):
            regions.io.save(tmp_path / "picks", format="feather")

    def test_features_declare_nothing(self, with_features, tmp_path):
        """Features hold no coordinates, whatever their encoding."""
        directory = with_features.io.save(tmp_path / "picks", format="parquet")
        frame = with_features.features
        write_parquet(frame, directory / "features.parquet", {DIMS_KEY: '["time"]'})
        with pytest.raises(InvalidAnnotationError, match="hold no coordinates"):
            dc.annotations(directory)

    def test_dimensions_which_are_not_a_document(self, regions, tmp_path):
        """What the footer states is read, and named where it is not readable."""
        path = tmp_path / "picks.parquet"
        write_parquet(regions.annotations, path, {DIMS_KEY: "time, distance"})
        with pytest.raises(InvalidAnnotationError, match="not a JSON document"):
            dc.annotations(path)

    def test_dimensions_which_name_none(self, regions, tmp_path):
        """A file which declares its dimensions names them."""
        path = tmp_path / "picks.parquet"
        write_parquet(regions.annotations, path, {DIMS_KEY: "[]"})
        with pytest.raises(InvalidAnnotationError, match="names none"):
            dc.annotations(path)

    def test_a_stray_parquet_table(self, regions, tmp_path):
        """A near-miss on the convention is a near-miss in either encoding."""
        directory = regions.io.save(tmp_path / "picks")
        regions.io.to_parquet(directory / "annotation.parquet")
        with pytest.raises(InvalidAnnotationError, match=r"annotation\.parquet"):
            dc.annotations(directory)

    def test_a_file_which_is_not_parquet(self, tmp_path):
        """Whatever pyarrow makes of it, the error names the file."""
        path = tmp_path / "picks.parquet"
        path.write_text("note,time\na,1\n")
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(path, dims=("time",))

    def test_a_table_which_is_neither(self, tmp_path):
        """A bare set is a table, and the message names the encodings it takes."""
        path = tmp_path / "picks.txt"
        path.write_text("note\na\n")
        with pytest.raises(InvalidAnnotationError, match=r"\.csv or \.parquet"):
            dc.annotations(path, dims=("time",))
