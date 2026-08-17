"""Tests for reading and writing stored annotation sets."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pyarrow
except ImportError:
    pyarrow = None

import dascore as dc
from dascore.core.annotation_loader import find_annotations
from dascore.core.annotations import DIMS_KEY, Line, Moveout
from dascore.exceptions import InvalidAnnotationError, ParameterError
from dascore.utils.tables import write_parquet

DIMS = ("distance", "time")


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


@pytest.fixture
def curve() -> Moveout:
    """A moveout a path may be drawn from."""
    return Moveout(
        apex_distance=100.0,
        apex_time=np.datetime64("2020-01-01T00:00:05"),
        velocity=1500.0,
        standoff=30.0,
        distance_start=0.0,
        distance_end=200.0,
    )


@pytest.fixture
def regions() -> dc.AnnotationSet:
    """A set of regions, which a bare table can hold."""
    frame = pd.DataFrame(
        {
            "id": ["r1", "r2"],
            "group": ["noise", "noise"],
            "tags": [("road", "car"), None],
            "distance_start": [120.0, 10.0],
            "distance_end": [340.0, 60.0],
            "time_start": [
                np.datetime64("2020-01-01T00:00:10"),
                np.datetime64("2020-01-01T00:00:20"),
            ],
            "time_end": [
                np.datetime64("2020-01-01T00:00:12"),
                np.datetime64("2020-01-01T00:00:22"),
            ],
            "note": ["traffic", "walker"],
            "score": [0.9, 0.2],
            "checked": [True, False],
        }
    )
    return dc.AnnotationSet(
        frame, dims=DIMS, acquisition_key="NET.ARR.00.das", history=("decimate",)
    )


@pytest.fixture
def with_vertices(curve) -> dc.AnnotationSet:
    """A set holding a hand-drawn path and one drawn from a curve."""
    drawn = curve.vertices(5)
    vertices = pd.DataFrame(
        {
            "id": ["p1", "p1", "p1", *["p2"] * 5],
            "seq": [0, 1, 2, *range(5)],
            "distance": [10.0, 95.0, 185.0, *drawn["distance"]],
            "time": [
                np.datetime64("2020-01-01T00:00:00.1"),
                np.datetime64("2020-01-01T00:00:01"),
                np.datetime64("2020-01-01T00:00:01.9"),
                *drawn["time"],
            ],
        }
    )
    frame = pd.DataFrame(
        {
            "id": ["p1", "p2", "r1"],
            "group": ["picks", "picks", "noise"],
            "geometry": ["path", "path", "region"],
            "basis": [None, curve, None],
            "distance_start": [np.nan, np.nan, 5.0],
            "distance_end": [np.nan, np.nan, 15.0],
        }
    )
    return dc.AnnotationSet(frame, dims=DIMS, vertices=vertices)


@pytest.fixture
def picks() -> dc.AnnotationSet:
    """A set of time ranges made by a picker, on its own acquisition."""
    frame = pd.DataFrame(
        {
            "id": ["m1", "m2"],
            "group": ["arrival", "arrival"],
            "value": ["p", "s"],
            "time_start": [
                np.datetime64("2020-01-01T00:00:01"),
                np.datetime64("2020-01-01T00:00:03"),
            ],
            "time_end": [
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


class TestRoundTrip:
    """A set written out and read back is the set it was."""

    def test_regions_through_a_directory(self, regions, tmp_path):
        """Bounds, extras and provenance all survive a directory."""
        regions.save(tmp_path / "picks")
        assert dc.annotations(tmp_path / "picks") == regions

    def test_regions_through_a_bare_table(self, regions, tmp_path):
        """A set of regions is a table, and its dims are stated again."""
        path = tmp_path / "picks.csv"
        regions.to_csv(path)
        loaded = dc.annotations(path, dims=DIMS)
        assert loaded.to_dataframe().equals(regions.to_dataframe())

    def test_vertices_and_basis(self, with_vertices, curve, tmp_path):
        """Vertices and the curve they were drawn from both survive."""
        with_vertices.save(tmp_path / "picks")
        loaded = dc.annotations(tmp_path / "picks")
        assert loaded == with_vertices
        assert loaded[1].geometry.basis == curve

    def test_extras_keep_their_kind(self, regions, tmp_path):
        """A cell written as a number or a boolean reads back as one."""
        loaded = dc.annotations(regions.save(tmp_path / "picks"))
        assert loaded[0].extra["score"] == 0.9
        assert loaded[0].extra["checked"] is True
        assert loaded[1].extra["checked"] is False

    def test_tags_keep_their_shape(self, regions, tmp_path):
        """Tags are one spelling however they arrive."""
        loaded = dc.annotations(regions.save(tmp_path / "picks"))
        assert loaded[0].tags == ("road", "car")
        assert loaded[1].tags == ()

    def test_times_keep_their_type(self, regions, tmp_path):
        """A time endpoint reads back as a time, not as its text."""
        loaded = dc.annotations(regions.save(tmp_path / "picks"))
        start, _ = loaded[0].region.bounds["time"]
        assert isinstance(start, np.datetime64)

    def test_line_basis(self, tmp_path):
        """A line survives a round trip as the curve it is."""
        line = Line(
            start={"distance": 0.0, "time": np.datetime64("2020-01-01")},
            end={"distance": 50.0, "time": np.datetime64("2020-01-01")},
        )
        vertices = pd.DataFrame(
            {
                "id": ["p1", "p1"],
                "seq": [0, 1],
                "distance": [0.0, 50.0],
                "time": [np.datetime64("2020-01-01")] * 2,
            }
        )
        frame = pd.DataFrame({"id": ["p1"], "geometry": ["path"], "basis": [line]})
        annotations = dc.AnnotationSet(frame, dims=DIMS, vertices=vertices)
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].geometry.basis == line

    def test_an_unstated_bound(self, tmp_path):
        """An empty dimension cell reads back as unconstrained."""
        frame = pd.DataFrame(
            {
                "group": ["a", "b"],
                "distance_start": [1.0, np.nan],
                "distance_end": [2.0, np.nan],
            }
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded == annotations
        assert "distance" not in loaded[1].region.bounds


class TestWhatATableCannotSay:
    """A CSV has no types, and these are the corners where that shows."""

    def test_an_extra_named_seq(self, tmp_path):
        """Only the vertices order by seq; an annotation's is its own."""
        frame = pd.DataFrame(
            {"group": ["a"], "distance": [1.0], "seq": ["third"]},
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].extra["seq"] == "third"

    def test_an_empty_cell_is_unset(self, tmp_path):
        """A table cannot tell an empty cell from an empty string."""
        frame = pd.DataFrame({"group": ["", "b"], "distance": [1.0, 2.0]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        assert dc.annotations(annotations.save(tmp_path / "picks")) == annotations
        assert annotations[0].group == ""

    def test_a_datetime_extra_reads_back_as_text(self, tmp_path):
        """Only a declared dimension is known to hold times, so only it is read
        as one; an extra keeps the text it was written as.
        """
        frame = pd.DataFrame(
            {
                "group": ["a"],
                "distance": [1.0],
                "when": [np.datetime64("2020-01-01T00:00:00")],
            }
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].extra["when"] == "2020-01-01T00:00:00.000000000"

    def test_an_ambiguous_value_is_refused_at_the_write(self, tmp_path):
        """A value column a table would retype could make a group mix kinds,
        so the set refuses to write a store it would not read.
        """
        frame = pd.DataFrame(
            {"group": ["phase"] * 2, "value": ["P", "true"], "distance": [1.0, 2.0]}
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        with pytest.raises(ParameterError, match="read back as a boolean"):
            annotations.save(tmp_path / "picks")

    def test_an_unambiguous_value_still_writes(self, tmp_path):
        """Only text a table would read as another kind is refused."""
        frame = pd.DataFrame(
            {"group": ["phase"] * 2, "value": ["P", "S"], "distance": [1.0, 2.0]}
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        assert dc.annotations(annotations.save(tmp_path / "picks")) == annotations

    def test_a_non_finite_looking_extra_stays_text(self, tmp_path):
        """A cell reading 'nan' is text, not a value which then vanishes."""
        frame = pd.DataFrame({"group": ["a"], "distance": [1.0], "note": ["nan"]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].extra["note"] == "nan"

    def test_an_extra_some_rows_leave_blank(self, tmp_path):
        """A blank cell is unset; the rows which state one still read."""
        frame = pd.DataFrame(
            {"group": ["a", "b"], "distance": [1.0, 2.0], "note": ["seen", None]}
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].extra["note"] == "seen"
        assert "note" not in loaded[1].extra

    def test_a_numeric_looking_extra_reads_as_a_number(self, tmp_path):
        """A cell is read the way its own text states it, as every table is."""
        frame = pd.DataFrame({"group": ["a"], "distance": [1.0], "zip": ["01234"]})
        annotations = dc.AnnotationSet(frame, dims=("distance",))
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].extra["zip"] == 1234


class TestSavingOverASet:
    """Writing states the whole directory, not only the parts it has."""

    def test_a_stale_vertices_table_is_cleared(self, with_vertices, regions, tmp_path):
        """A set without vertices leaves none behind for the next read."""
        directory = tmp_path / "picks"
        with_vertices.save(directory)
        regions.save(directory)
        assert not (directory / "vertices.csv").exists()
        assert dc.annotations(directory) == regions

    @pytest.mark.skipif(yaml is None, reason="pyyaml is not installed")
    def test_a_hand_authored_yaml_is_superseded(self, tmp_path):
        """Saving a set read from YAML does not leave two attrs files."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "attrs.yaml").write_text(yaml.safe_dump({"dims": list(DIMS)}))
        (directory / "annotations.csv").write_text("group,distance\nnoise,1.0\n")
        loaded = dc.annotations(directory)
        loaded.save(directory)
        assert not (directory / "attrs.yaml").exists()
        assert dc.annotations(directory) == loaded

    def test_a_file_owing_this_format_nothing_is_left(self, regions, tmp_path):
        """Only the spellings a set claims are cleared."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.bak").write_text("mine")
        regions.save(directory)
        assert (directory / "attrs.bak").read_text() == "mine"

    def test_a_shouted_suffix_is_read_and_superseded(self, regions, tmp_path):
        """One data model stands behind a suffix however it is spelled."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").rename(directory / "attrs.JSON")
        loaded = dc.annotations(directory)
        assert loaded == regions
        loaded.save(directory)
        # One attrs file, whatever it ends up called: a case-insensitive
        # filesystem holds the shouted name and the written one in the
        # same file, so the spelling on disk is the platform's to decide
        # and only the count is this format's.
        assert (
            len([x for x in directory.iterdir() if x.stem.casefold() == "attrs"]) == 1
        )
        assert dc.annotations(directory) == regions


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

    def test_a_directory_refuses_what_it_states(self, regions, tmp_path):
        """A directory holds its own attributes and vertices."""
        directory = regions.save(tmp_path / "picks")
        with pytest.raises(InvalidAnnotationError, match="which states them"):
            dc.annotations(directory, attrs={"dims": DIMS})
        with pytest.raises(InvalidAnnotationError, match="which states them"):
            dc.annotations(directory, vertices=pd.DataFrame())

    def test_a_dataframe(self):
        """A frame becomes a set, as the constructor makes one."""
        frame = pd.DataFrame({"group": ["a"], "distance": [1.0]})
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
        path.write_text("group\na\n")
        with pytest.raises(InvalidAnnotationError, match="not a table"):
            dc.annotations(path, dims=DIMS)

    def test_errors_are_annotation_errors(self, tmp_path):
        """The neutral errors the table reader raises are named here."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "annotations.csv").write_text("group\nnoise,extra\n")
        with pytest.raises(InvalidAnnotationError, match="states 2 cells"):
            dc.annotations(directory, dims=DIMS)

    def test_a_set_of_none(self, tmp_path):
        """A set of no annotations writes an empty table and reads back."""
        empty = dc.annotations(dims=DIMS)
        assert dc.annotations(empty.save(tmp_path / "picks")) == empty


class TestDeclaringDimensions:
    """Cells cannot be read before the dimensions are known."""

    def test_stated_by_the_attrs(self, regions, tmp_path):
        """A directory states its own dimensions."""
        assert dc.annotations(regions.save(tmp_path / "picks")).dims == DIMS

    def test_stated_by_the_caller(self, regions, tmp_path):
        """A bare table has the caller state them."""
        path = tmp_path / "picks.csv"
        regions.to_csv(path)
        assert dc.annotations(path, dims=DIMS).dims == DIMS

    def test_stated_by_neither(self, regions, tmp_path):
        """A source stating none fails saying how to state them."""
        path = tmp_path / "picks.csv"
        regions.to_csv(path)
        with pytest.raises(InvalidAnnotationError, match="states no dimensions"):
            dc.annotations(path)

    @pytest.mark.parametrize("source", ["file", "caller"])
    def test_stated_as_a_bare_string(self, tmp_path, source):
        """One dimension may be a lone string, as the constructor takes it,
        rather than a sequence of its own letters.
        """
        directory = tmp_path / "picks"
        directory.mkdir()
        stated = '{"dims": "distance"}' if source == "file" else "{}"
        (directory / "attrs.json").write_text(stated)
        (directory / "annotations.csv").write_text("group,distance\na,1.0\n")
        dims = None if source == "file" else "distance"
        loaded = dc.annotations(directory, dims=dims)
        assert loaded.dims == ("distance",)
        # The cells were typed against one dimension, not eight letters.
        assert loaded[0].region.bounds["distance"] == (1.0, 1.0)

    def test_a_document_which_does_not_build(self, tmp_path):
        """A bad stored document is named as a bad file, not as a bad call."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "attrs.json").write_text('{"dims": ["distance"], "n": 1}')
        (directory / "annotations.csv").write_text("group,distance\na,1.0\n")
        with pytest.raises(InvalidAnnotationError, match="Extra inputs"):
            dc.annotations(directory)

    def test_a_directory_which_states_them_refuses_others(self, regions, tmp_path):
        """Reading the cells against other dimensions would type them
        differently and build a set which is not the one stored.
        """
        directory = regions.save(tmp_path / "picks")
        with pytest.raises(InvalidAnnotationError, match="its own dimensions"):
            dc.annotations(directory, dims=("time", "distance"))

    def test_a_directory_which_states_none_takes_them(self, regions, tmp_path):
        """Where a directory states none, the caller's are the only ones."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").unlink()
        assert dc.annotations(directory, dims=("time", "distance")).dims == (
            "time",
            "distance",
        )


class TestTheAttrsFile:
    """What a set directory says about itself."""

    @pytest.mark.skipif(yaml is None, reason="pyyaml is not installed")
    def test_yaml_spelling(self, regions, tmp_path):
        """One data model stands behind both spellings; a set may be authored
        in the more readable one.
        """
        directory = regions.save(tmp_path / "picks")
        document = json.loads((directory / "attrs.json").read_text())
        (directory / "attrs.yaml").write_text(yaml.safe_dump(document))
        (directory / "attrs.json").unlink()
        assert dc.annotations(directory) == regions

    def test_two_spellings(self, regions, tmp_path):
        """A set spells each of its parts once."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.yml").write_text("{}")
        with pytest.raises(InvalidAnnotationError, match="more than once"):
            dc.annotations(directory)

    def test_the_wrong_object(self, regions, tmp_path):
        """A file declaring another model is a misfiled object."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").write_text(
            '{"object_type": "Inventory", "dims": ["time"]}'
        )
        with pytest.raises(InvalidAnnotationError, match="declares 'Inventory'"):
            dc.annotations(directory)

    def test_which_is_not_a_mapping(self, regions, tmp_path):
        """A document stating a list defines no attributes."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").write_text('["distance", "time"]')
        with pytest.raises(InvalidAnnotationError, match="no mapping"):
            dc.annotations(directory)

    @pytest.mark.skipif(yaml is None, reason="pyyaml is not installed")
    def test_which_does_not_parse(self, regions, tmp_path):
        """Unparseable YAML names the file rather than the parser."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").unlink()
        (directory / "attrs.yaml").write_text("dims: [\n")
        with pytest.raises(InvalidAnnotationError, match="Could not parse YAML"):
            dc.annotations(directory)

    def test_bad_json(self, regions, tmp_path):
        """Unparseable JSON names the file too."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").write_text("{")
        with pytest.raises(InvalidAnnotationError, match="Could not parse JSON"):
            dc.annotations(directory)

    def test_which_cannot_be_read(self, regions, tmp_path):
        """A file which does not decode names itself, not the codec."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").write_bytes(b'{"dims": ["\xff\xfe"]}')
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(directory)

    def test_no_attrs_file(self, regions, tmp_path):
        """A directory without one is read on the caller's dimensions."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").unlink()
        assert dc.annotations(directory, dims=DIMS).dims == DIMS


class TestTheTables:
    """What a set directory holds, and what it may not."""

    def test_no_annotations_table(self, tmp_path):
        """A directory without one states no annotations."""
        directory = tmp_path / "picks"
        directory.mkdir()
        with pytest.raises(InvalidAnnotationError, match=r"no annotations\.csv"):
            dc.annotations(directory, dims=DIMS)

    def test_a_table_which_cannot_be_read(self, regions, tmp_path):
        """A table which does not decode is the table reader's to name."""
        directory = regions.save(tmp_path / "picks")
        (directory / "annotations.csv").write_bytes(b"group\n\xff\xfe\n")
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(directory)

    def test_vertices_named_in_another_case(self, with_vertices, tmp_path):
        """Every part of a set is found the same way, so none is skipped."""
        directory = with_vertices.save(tmp_path / "picks")
        table = directory / "vertices.csv"
        table.rename(directory / "vertices.CSV")
        assert dc.annotations(directory) == with_vertices

    @pytest.mark.skipif(FOLDS_CASE, reason="this filesystem holds one of the two")
    def test_vertices_spelled_twice(self, with_vertices, tmp_path):
        """A set spells each of its parts once, vertices included."""
        directory = with_vertices.save(tmp_path / "picks")
        text = (directory / "vertices.csv").read_text()
        (directory / "vertices.CSV").write_text(text)
        with pytest.raises(InvalidAnnotationError, match="states vertices more than"):
            dc.annotations(directory)

    def test_a_stray_table(self, regions, tmp_path):
        """A near-miss on the convention raises rather than being skipped."""
        directory = regions.save(tmp_path / "picks")
        (directory / "vertexes.csv").write_text("id,seq\n")
        with pytest.raises(InvalidAnnotationError, match=r"vertexes\.csv"):
            dc.annotations(directory)

    def test_a_basis_which_is_not_json(self, with_vertices, tmp_path):
        """A stored basis is what its curve dumps."""
        directory = with_vertices.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        table.write_text(table.read_text().replace('"{""object_type', '"{oops'))
        with pytest.raises(InvalidAnnotationError, match="not a JSON document"):
            dc.annotations(directory)

    def test_a_basis_which_is_not_a_curve(self, with_vertices, tmp_path):
        """A document which parses but names no curve is still refused."""
        directory = with_vertices.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        original = table.read_text()
        start = original.index('"{""object_type')
        end = original.index('"', start + 1)
        while original[end : end + 2] == '""':
            end = original.index('"', end + 2)
        table.write_text(original[:start] + '"{}"' + original[end + 1 :])
        with pytest.raises(InvalidAnnotationError, match="as a curve"):
            dc.annotations(directory)

    def test_a_non_numeric_seq(self, with_vertices, tmp_path):
        """A vertex states its place in the order as a number."""
        directory = with_vertices.save(tmp_path / "picks")
        table = directory / "vertices.csv"
        table.write_text(table.read_text().replace("p1,0,", "p1,first,"))
        with pytest.raises(InvalidAnnotationError, match="non-numeric seq"):
            dc.annotations(directory)

    def test_a_dimension_which_is_neither(self, regions, tmp_path):
        """A dimension column holds numbers or times, and says so."""
        directory = regions.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        table.write_text(table.read_text().replace("120.0", "far"))
        with pytest.raises(InvalidAnnotationError, match="neither numbers nor times"):
            dc.annotations(directory)

    def test_a_dimension_no_row_states(self, tmp_path):
        """A column every row leaves empty constrains nothing."""
        path = tmp_path / "picks.csv"
        path.write_text("group,time_start,time_end\nquiet,,\n")
        loaded = dc.annotations(path, dims=("time",))
        assert "time" not in loaded[0].region.bounds

    def test_a_minute_resolution_time(self, tmp_path):
        """Numpy writes only the fields a unit carries, and they all read back."""
        frame = pd.DataFrame(
            {
                "group": ["a"],
                "time_start": [np.datetime64("2020-01-01T12:30")],
                "time_end": [np.datetime64("2020-01-01T12:35")],
            }
        )
        annotations = dc.AnnotationSet(frame, dims=("time",))
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded == annotations
        assert isinstance(loaded[0].region.bounds["time"][0], np.datetime64)

    def test_a_dimension_some_rows_leave_blank(self):
        """Times as text beside empty cells read as times and as unset."""
        frame = pd.DataFrame(
            {
                "group": ["a", "b"],
                "time_start": ["2020-01-01", None],
                "time_end": ["2020-01-02", None],
            }
        )
        out = dc.AnnotationSet(frame, dims=("time",))
        assert out[0].region.bounds["time"][0] == np.datetime64("2020-01-01")
        assert "time" not in out[1].region.bounds

    def test_a_text_dimension_column_agrees_with_its_region(self):
        """The frame and the geometry built from it say the same thing."""
        frame = pd.DataFrame({"time_start": ["2020-01-01"], "time_end": ["2020-01-02"]})
        out = dc.AnnotationSet(frame, dims=("time",))
        held = out.to_dataframe()["time_start"][0]
        assert isinstance(held, pd.Timestamp | np.datetime64)
        assert out[0].region.bounds["time"][0] == np.datetime64("2020-01-01")

    def test_an_id_which_looks_like_a_number(self, tmp_path):
        """An id is the label its vertices name it by, never a number."""
        frame = pd.DataFrame({"id": ["1"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {"id": ["1", "1"], "seq": [0, 1], "distance": [1.0, 2.0]}
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",), vertices=vertices)
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].id == "1"
        # The frame too, not only the model: Annotation.id is typed str, so
        # it would coerce an int back and hide the damage.
        assert loaded.to_dataframe()["id"][0] == "1"
        assert loaded.to_vertices()["id"][0] == "1"
        assert loaded == annotations


class TestWriting:
    """How a set spells itself out."""

    def test_to_csv_returns_text(self, regions):
        """The text comes back whether or not it is written."""
        text = regions.to_csv()
        assert text.splitlines()[0].startswith("id,group,tags")

    def test_to_csv_refuses_vertices(self, with_vertices):
        """A bare table states one grain."""
        with pytest.raises(ParameterError, match="holds vertices"):
            with_vertices.to_csv()

    def test_save_makes_the_directory(self, regions, tmp_path):
        """Saving into a directory which is not there makes it."""
        directory = regions.save(tmp_path / "deep" / "picks")
        assert directory.is_dir()

    def test_save_writes_no_empty_vertices(self, regions, tmp_path):
        """A set without vertices states no vertices table."""
        directory = regions.save(tmp_path / "picks")
        assert not (directory / "vertices.csv").exists()

    def test_save_writes_what_it_holds(self, with_vertices, tmp_path):
        """A set with vertices states all three parts."""
        directory = with_vertices.save(tmp_path / "picks")
        written = {x.name for x in directory.iterdir()}
        assert written == {"attrs.json", "annotations.csv", "vertices.csv"}

    def test_save_over_itself(self, regions, tmp_path):
        """Saving twice into one directory rewrites it."""
        regions.save(tmp_path / "picks")
        assert dc.annotations(regions.save(tmp_path / "picks")) == regions

    def test_times_are_written_unambiguously(self, regions, tmp_path):
        """A time is written the way DASCore writes every datetime."""
        text = regions.to_csv()
        assert "2020-01-01T00:00:10.000000000" in text

    def test_a_nested_extra_is_written_as_its_document(self, tmp_path):
        """A cell a table has no column shape for is written as text."""
        frame = pd.DataFrame({"group": ["a"], "distance": [1.0], "meta": [{"n": 1}]})
        text = dc.AnnotationSet(frame, dims=("distance",)).to_csv()
        assert '{""n"": 1}' in text

    def test_an_extra_json_cannot_spell(self, tmp_path):
        """A nested value with no json type is written as its text rather
        than dying as a circular reference.
        """
        frame = pd.DataFrame({"group": ["a"], "distance": [1.0], "meta": [{"s": {1}}]})
        text = dc.AnnotationSet(frame, dims=("distance",)).to_csv()
        assert '{""s"": ""1""}' in text

    def test_the_attrs_name_their_model(self, regions, tmp_path):
        """The document says what it holds, as every stored object does."""
        directory = regions.save(tmp_path / "picks")
        text = (directory / "attrs.json").read_text()
        assert '"object_type": "AnnotationSetAttrs"' in text


class TestCollections:
    """Sets stored side by side read as one set which says where each row came from."""

    @pytest.fixture
    def collection(self, regions, picks, tmp_path):
        """A directory holding two sets, each stating its own dimensions."""
        root = tmp_path / "sets"
        regions.save(root / "hand")
        picks.save(root / "phasenet")
        return root

    def test_reads_as_one_set(self, collection, regions, picks):
        """Every set is in the table, in the dimensions all of them state."""
        loaded = dc.annotations(collection)
        assert len(loaded) == len(regions) + len(picks)
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

    def test_a_row_keeps_its_own_acquisition(self, collection, regions, picks):
        """The address a row was picked on is the address of its set."""
        loaded = dc.annotations(collection)
        keys = {x.set: x.acquisition_key for x in loaded}
        assert keys["hand"] == regions.attrs.acquisition_key
        assert keys["phasenet"] == picks.attrs.acquisition_key

    def test_round_trips_through_one_directory(self, collection, tmp_path):
        """A collection saves flat, and what it read is what it reads back."""
        loaded = dc.annotations(collection)
        assert dc.annotations(loaded.save(tmp_path / "flat")) == loaded

    def test_an_id_in_two_sets(self, regions, tmp_path):
        """An id is an address into the collection, so it names one row."""
        root = tmp_path / "sets"
        regions.save(root / "hand")
        regions.save(root / "again")
        with pytest.raises(InvalidAnnotationError, match="again and hand"):
            dc.annotations(root)

    def test_a_set_which_states_only_attributes(self, collection, tmp_path):
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
        # The half-set is not what is wrong here: this directory states no
        # annotations of its own, and holds no set to make it a collection.
        with pytest.raises(InvalidAnnotationError, match=r"holds no annotations\.csv"):
            dc.annotations(root, dims=("time",))

    def test_a_tree_of_collections(self, regions, tmp_path):
        """Sets loaded together are one collection, not a tree of them."""
        root = tmp_path / "sets"
        regions.save(root / "outer" / "inner")
        regions.save(root / "hand")
        with pytest.raises(InvalidAnnotationError, match="not a tree"):
            dc.annotations(root)

    def test_a_set_which_also_holds_sets(self, regions, picks, tmp_path):
        """A directory stating annotations is the set, whatever sits below it."""
        root = regions.save(tmp_path / "sets")
        picks.save(root / "hand")
        assert dc.annotations(root) == regions

    def test_a_set_beside_a_folder_of_its_own(self, regions, tmp_path):
        """A folder someone kept beside the tables is not this format's business."""
        root = regions.save(tmp_path / "picks")
        (root / "backup").mkdir()
        (root / "backup" / "attrs.json").write_text('{"dims": ["time"]}')
        assert dc.annotations(root) == regions

    def test_a_table_beside_the_sets(self, collection):
        """A table where every set is a directory names no set."""
        (collection / "notes.csv").write_text("group\nnoise\n")
        with pytest.raises(InvalidAnnotationError, match="name no set"):
            dc.annotations(collection)

    def test_dimensions_for_the_sets_which_state_none(self, regions, tmp_path):
        """They reach the children which declare none, and without them none do."""
        root = tmp_path / "sets"
        for name in ("hand", "auto"):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "annotations.csv").write_text(
                f"id,group,time_start,time_end\n{name},noise,1.0,2.0\n"
            )
        assert len(dc.annotations(root, dims=("time",))) == 2
        with pytest.raises(InvalidAnnotationError, match="states no dimensions"):
            dc.annotations(root)

    def test_dimensions_stated_beside_the_sets(self, tmp_path):
        """A collection may declare them, and then refuses a caller restating them."""
        root = tmp_path / "sets"
        directory = root / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.csv").write_text("group,time_start,time_end\nq,1,2\n")
        (root / "attrs.json").write_text('{"dims": ["time"]}')
        assert dc.annotations(root).dims == ("time",)
        with pytest.raises(
            InvalidAnnotationError, match="a directory of sets stating its own"
        ):
            dc.annotations(root, dims=("time",))

    def test_a_dimension_spelled_two_ways(self, picks, tmp_path):
        """A point is not a range of no width, so neither stands in."""
        root = tmp_path / "sets"
        picks.save(root / "ranges")
        frame = pd.DataFrame({"group": ["a"], "time": [1.0]})
        dc.AnnotationSet(frame, dims=("time",)).save(root / "points")
        with pytest.raises(InvalidAnnotationError, match="spelled as a point") as info:
            dc.annotations(root)
        # The constructor refuses this too; naming the sets is what this adds.
        assert "points" in str(info.value) and "ranges" in str(info.value)

    def test_a_set_which_states_a_set_column(self, tmp_path):
        """The set column names the set, so a set may not fill it in."""
        root = tmp_path / "sets"
        directory = root / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.csv").write_text("set,group,time\nother,noise,1.0\n")
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
        regions.save(collection / ".annotations")
        assert sorted(dc.annotations(collection).attrs.sets) == ["hand", "phasenet"]

    def test_a_directory_which_is_no_set(self, collection):
        """A directory participating in no convention here is left alone."""
        (collection / "figures").mkdir()
        assert len(dc.annotations(collection)) == 4

    def test_a_row_keeps_the_acquisition_it_names(self, regions, tmp_path):
        """A row naming its own acquisition outranks its set's, merged or not."""
        root = tmp_path / "sets"
        regions.save(root / "hand")
        frame = pd.DataFrame(
            {
                "id": ["m1", "m2"],
                "acquisition_key": ["NET.OTHER.00.das", None],
                "time_start": [
                    np.datetime64("2020-01-01T00:00:31"),
                    np.datetime64("2020-01-01T00:00:33"),
                ],
                "time_end": [
                    np.datetime64("2020-01-01T00:00:32"),
                    np.datetime64("2020-01-01T00:00:34"),
                ],
            }
        )
        other = dc.AnnotationSet(
            frame, dims=("time",), acquisition_key="NET.SET.00.das"
        )
        other.save(root / "auto")
        keys = {x.id: x.acquisition_key for x in dc.annotations(root)}
        assert keys["m1"] == "NET.OTHER.00.das"
        assert keys["m2"] == "NET.SET.00.das"
        assert keys["r1"] == regions.attrs.acquisition_key

    def test_what_the_collection_states_reaches_the_merged_set(self, tmp_path):
        """A collection may state its own provenance beside its sets."""
        root = tmp_path / "sets"
        directory = root / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.csv").write_text("group,time\nq,1\n")
        document = {
            "dims": ["time"],
            "acquisition_key": "NET.COLL.00.das",
            "history": ["decimate"],
        }
        (root / "attrs.json").write_text(json.dumps(document))
        loaded = dc.annotations(root)
        assert loaded.attrs.history == ("decimate",)
        # The row's own set states no key, so the collection's is its address.
        assert loaded[0].acquisition_key == "NET.COLL.00.das"

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
            (directory / "annotations.csv").write_text(f"group,time\nq,{cell}\n")
            (directory / "attrs.json").write_text('{"dims": ["time"]}')
        with pytest.raises(InvalidAnnotationError, match="different kinds of value"):
            dc.annotations(root)

    def test_a_dimension_no_row_of_one_set_states(self, picks, tmp_path):
        """A column every row leaves empty states no kind, so it agrees."""
        root = tmp_path / "sets"
        picks.save(root / "phasenet")
        blank = root / "quiet"
        blank.mkdir(parents=True)
        (blank / "annotations.csv").write_text("group,time_start,time_end\nq,,\n")
        (blank / "attrs.json").write_text('{"dims": ["time"]}')
        assert len(dc.annotations(root)) == len(picks) + 1

    def test_a_column_which_is_a_dimension_in_one_set_only(self, tmp_path):
        """A column another set dimensions must not silently become a bound."""
        root = tmp_path / "sets"
        notes = root / "notes"
        notes.mkdir(parents=True)
        (notes / "annotations.csv").write_text("group,time,distance\nq,1,shallow\n")
        (notes / "attrs.json").write_text('{"dims": ["time"]}')
        boxes = root / "boxes"
        boxes.mkdir(parents=True)
        (boxes / "annotations.csv").write_text("group,time,distance\nb,2,50\n")
        (boxes / "attrs.json").write_text('{"dims": ["time", "distance"]}')
        with pytest.raises(InvalidAnnotationError, match="without declaring"):
            dc.annotations(root)

    @pytest.mark.skipif(FOLDS_CASE, reason="this filesystem holds one of the two")
    def test_set_names_which_differ_only_in_case(self, regions, picks, tmp_path):
        """A set name is a label, so it must name one set on any filesystem."""
        root = tmp_path / "sets"
        regions.save(root / "hand")
        picks.save(root / "HAND")
        with pytest.raises(InvalidAnnotationError, match="differ only in case"):
            dc.annotations(root)

    def test_a_hidden_table_beside_the_sets(self, collection):
        """A half-copied file is a companion, not a table which names no set."""
        (collection / ".annotations.csv").write_text("group\nnoise\n")
        assert len(dc.annotations(collection)) == 4

    def test_a_stray_table_whatever_its_case(self, collection):
        """The suffix is matched as the loader matches every other one."""
        (collection / "NOTES.CSV").write_text("group\nnoise\n")
        with pytest.raises(InvalidAnnotationError, match=r"NOTES\.CSV"):
            dc.annotations(collection)

    def test_a_set_named_in_upper_case(self, tmp_path):
        """A set states its table once, in whichever case it spells the suffix."""
        directory = tmp_path / "sets" / "hand"
        directory.mkdir(parents=True)
        (directory / "annotations.CSV").write_text("group,time\nq,1\n")
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
        dc.annotations(collection).save(root / "merged")
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
        flat = dc.annotations(collection).save(tmp_path / "flat")
        table = flat / "annotations.csv"
        table.write_text(table.read_text().replace(",hand", ",typo"))
        with pytest.raises(InvalidAnnotationError, match="name no set stated here"):
            dc.annotations(flat)

    def test_a_row_with_no_label(self, collection, tmp_path):
        """A row loaded with others says which of them it came from."""
        flat = dc.annotations(collection).save(tmp_path / "flat")
        table = flat / "annotations.csv"
        text = table.read_text().replace(",hand", ",", 1)
        table.write_text(text)
        with pytest.raises(InvalidAnnotationError, match="state no set"):
            dc.annotations(flat)

    def test_a_table_with_no_label_column(self, collection, tmp_path):
        """Sets stated with no column to name them leave every row adrift."""
        flat = dc.annotations(collection).save(tmp_path / "flat")
        table = flat / "annotations.csv"
        frame = dc.annotations(flat).to_dataframe().drop(columns="set")
        table.write_text(frame.to_csv(index=False))
        with pytest.raises(InvalidAnnotationError, match="no set column"):
            dc.annotations(flat)

    def test_a_collection_of_empty_sets(self, tmp_path):
        """A collection holding no rows at all labels none of them."""
        root = tmp_path / "sets"
        for name in ("hand", "auto"):
            dc.AnnotationSet(None, dims=("time",)).save(root / name)
        loaded = dc.annotations(root)
        assert len(loaded) == 0
        assert sorted(loaded.attrs.sets) == ["auto", "hand"]

    def test_a_set_column_on_a_set_of_its_own(self):
        """A set which states no sets is not a collection, so its labels are its own."""
        frame = pd.DataFrame({"set": ["whatever"], "group": ["a"], "time": [1.0]})
        assert dc.AnnotationSet(frame, dims=("time",))[0].set == "whatever"

    def test_a_set_with_no_annotations(self, picks, tmp_path):
        """A set which states nothing is still one of the sets loaded."""
        root = tmp_path / "sets"
        picks.save(root / "phasenet")
        dc.AnnotationSet(None, dims=("time",)).save(root / "empty")
        loaded = dc.annotations(root)
        assert len(loaded) == len(picks)
        assert sorted(loaded.attrs.sets) == ["empty", "phasenet"]

    def test_paths_from_two_sets(self, with_vertices, tmp_path):
        """Vertices merge too, and each path still reads as the shape it was."""
        root = tmp_path / "sets"
        with_vertices.save(root / "hand")
        frame = pd.DataFrame({"id": ["p9"], "group": ["auto"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {
                "id": ["p9"] * 3,
                "seq": [0, 1, 2],
                "distance": [1000.0, 1100.0, 1200.0],
                "time": np.array(
                    [
                        "2020-01-01T00:00:05",
                        "2020-01-01T00:00:06",
                        "2020-01-01T00:00:07",
                    ],
                    dtype="datetime64[ns]",
                ),
            }
        )
        dc.AnnotationSet(frame, dims=DIMS, vertices=vertices).save(root / "auto")
        drawn = {x.id: x.geometry for x in dc.annotations(root) if x.id.startswith("p")}
        assert drawn["p1"].vertices["distance"] == (10.0, 95.0, 185.0)
        assert drawn["p9"].vertices["distance"] == (1000.0, 1100.0, 1200.0)
        assert drawn["p9"].region.bounds["distance"] == (1000.0, 1200.0)

    def test_vertices_in_different_dimensions(self, with_vertices, tmp_path):
        """A vertex states every dimension its table names, so these cannot merge."""
        root = tmp_path / "sets"
        with_vertices.save(root / "hand")
        frame = pd.DataFrame({"id": ["f1"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {"id": ["f1"] * 2, "seq": [0, 1], "distance": [1.0, 2.0]}
        )
        dc.AnnotationSet(frame, dims=("distance",), vertices=vertices).save(
            root / "flat"
        )
        with pytest.raises(
            InvalidAnnotationError, match="different dimensions"
        ) as info:
            dc.annotations(root)
        assert "hand" in str(info.value) and "flat" in str(info.value)


class TestDeclaringDimensionsInTheTable:
    """A bare table has no attrs file, so it may declare them above its header."""

    def test_a_bare_table_declares_them(self, tmp_path):
        """The dimensions travel with the file rather than with the call."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: distance, time\ngroup,time_start,time_end\nq,1,2\n")
        loaded = dc.annotations(path)
        assert loaded.dims == ("distance", "time")
        # The header is the one below the pragma, not the pragma itself.
        assert set(loaded.to_dataframe().columns) == {"group", "time_start", "time_end"}
        assert loaded[0].group == "q"
        assert loaded[0].region.bounds["time"] == (1.0, 2.0)

    def test_other_comments_are_comments(self, tmp_path):
        """A line above the header which declares nothing says nothing."""
        path = tmp_path / "picks.csv"
        path.write_text("# picked by hand\n#dims:time\ngroup,time\nq,1\n")
        loaded = dc.annotations(path)
        assert loaded.dims == ("time",)
        assert loaded[0].region.bounds["time"] == (1.0, 1.0)

    def test_restating_them_is_allowed(self, tmp_path):
        """Two spellings of one fact agree or they are not one fact."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\ngroup,time\nq,1\n")
        loaded = dc.annotations(path, dims=("time",))
        assert loaded.dims == ("time",)
        assert loaded[0].region.bounds["time"] == (1.0, 1.0)

    def test_disagreeing_with_the_caller(self, tmp_path):
        """The table's dimensions and the caller's are not merged."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\ngroup,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="where the two agree"):
            dc.annotations(path, dims=("distance",))

    def test_a_header_which_starts_with_the_mark(self, tmp_path):
        """A column may be named `#note`, and that line is the header."""
        path = tmp_path / "picks.csv"
        path.write_text("#note,group,time\nfirst,a,1\nsecond,b,2\n")
        loaded = dc.annotations(path, dims=("time",))
        assert len(loaded) == 2
        assert loaded[0].extra["#note"] == "first"

    def test_a_declaration_commented_out(self, tmp_path):
        """A struck-out declaration declares nothing, so the line is a header."""
        path = tmp_path / "picks.csv"
        path.write_text("## dims: time\ngroup,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="cells where its header"):
            dc.annotations(path, dims=("time",))

    def test_a_header_which_reads_as_a_comment(self, tmp_path):
        """A column may be named `# note`, which this library writes unquoted."""
        frame = pd.DataFrame(
            {"# note": ["first", "second"], "group": ["a", "b"], "time": [1.0, 2.0]}
        )
        picks = dc.AnnotationSet(frame, dims=("time",))
        path = tmp_path / "picks.csv"
        picks.to_csv(path)
        loaded = dc.annotations(path, dims=("time",))
        assert len(loaded) == 2
        assert loaded[0].extra["# note"] == "first"

    def test_comments_beside_a_declaration(self, tmp_path):
        """Where a table declares its dimensions, comments ride with it."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\n# picked by hand\ngroup,time\nq,1\n")
        loaded = dc.annotations(path)
        assert loaded.dims == ("time",)
        assert set(loaded.to_dataframe().columns) == {"group", "time"}

    def test_the_keyword_is_read_in_any_case(self, tmp_path):
        """A hand-authored line is read as written, whatever case it names."""
        path = tmp_path / "picks.csv"
        path.write_text("# Dims:  time , distance\ngroup,time\nq,1\n")
        assert dc.annotations(path).dims == ("time", "distance")

    def test_a_blank_line_above_the_declaration(self, tmp_path):
        """Blank lines above the header are skipped with the comments."""
        path = tmp_path / "picks.csv"
        path.write_text("\n# dims: time\n\ngroup,time\nq,1\n")
        assert dc.annotations(path).dims == ("time",)

    def test_a_table_which_cannot_be_decoded(self, tmp_path):
        """A table which cannot be read has no dimensions to be found in it."""
        path = tmp_path / "picks.csv"
        path.write_bytes(b"# dims: time\ngroup,time\n\xff\xfe,1\n")
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(path)

    def test_a_comment_holding_a_quote(self, tmp_path):
        """A comment is one line, whatever a csv reader makes of its quotes."""
        path = tmp_path / "picks.csv"
        path.write_text('# dims: time\n# it is ,"odd\ngroup,time\nq,1\n')
        loaded = dc.annotations(path)
        assert loaded.dims == ("time",)
        assert set(loaded.to_dataframe().columns) == {"group", "time"}

    def test_disagreeing_with_what_the_attrs_state(self, regions, tmp_path):
        """The message says where the other spelling came from."""
        directory = regions.save(tmp_path / "picks")
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
                f"# dims: {dim}\ngroup,{dim}\nq,1\n"
            )
        (root / "attrs.json").write_text('{"dims": ["time"]}')
        with pytest.raises(InvalidAnnotationError, match="which states its own"):
            dc.annotations(root)

    def test_disagreeing_with_the_attrs(self, regions, tmp_path):
        """A set directory states them once, wherever it states them."""
        directory = regions.save(tmp_path / "picks")
        table = directory / "annotations.csv"
        table.write_text("# dims: depth\n" + table.read_text())
        with pytest.raises(InvalidAnnotationError, match="where the two agree"):
            dc.annotations(directory)

    def test_a_set_directory_may_declare_them(self, tmp_path):
        """A hand-made set directory need not carry an attrs file."""
        directory = tmp_path / "picks"
        directory.mkdir()
        (directory / "annotations.csv").write_text("# dims: time\ngroup,time\nq,1\n")
        loaded = dc.annotations(directory)
        assert loaded.dims == ("time",)
        assert loaded[0].region.bounds["time"] == (1.0, 1.0)

    def test_sets_loaded_together_may_declare_them(self, tmp_path):
        """Each set in a collection may state its own, above its own table."""
        root = tmp_path / "sets"
        for name, dim in (("hand", "time"), ("auto", "distance")):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "annotations.csv").write_text(
                f"# dims: {dim}\ngroup,{dim}\nq,1\n"
            )
        loaded = dc.annotations(root)
        assert loaded.dims == ("distance", "time")
        bounds = sorted(tuple(x.region.bounds.items()) for x in loaded)
        assert bounds == [
            (("distance", (1.0, 1.0)),),
            (("time", (1.0, 1.0)),),
        ]

    def test_declared_twice(self, tmp_path):
        """One table states its dimensions once."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims: time\n# dims: distance\ngroup,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="more than once"):
            dc.annotations(path)

    def test_declared_but_named_none(self, tmp_path):
        """A declaration which names nothing declares nothing."""
        path = tmp_path / "picks.csv"
        path.write_text("# dims:\ngroup,time\nq,1\n")
        with pytest.raises(InvalidAnnotationError, match="names none"):
            dc.annotations(path)

    def test_vertices_declare_nothing(self, with_vertices, tmp_path):
        """Vertices are read in the dimensions of the set they belong to."""
        directory = with_vertices.save(tmp_path / "picks")
        table = directory / "vertices.csv"
        table.write_text("# dims: time\n" + table.read_text())
        with pytest.raises(InvalidAnnotationError, match="states them once"):
            dc.annotations(directory)

    def test_vertices_take_no_preamble(self, with_vertices, tmp_path):
        """Vertices declare nothing, so they have nothing to comment beside."""
        directory = with_vertices.save(tmp_path / "picks")
        table = directory / "vertices.csv"
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
        regions.save(data / ".annotations")
        assert dc.annotations(data) == regions

    def test_a_carried_collection(self, data, regions, picks):
        """What is carried may be many named sets, as anywhere else."""
        regions.save(data / ".annotations" / "hand")
        picks.save(data / ".annotations" / "phasenet")
        assert sorted(dc.annotations(data).attrs.sets) == ["hand", "phasenet"]

    def test_a_carried_table(self, data):
        """The bare table spelling is carried under the same name."""
        (data / ".annotations.csv").write_text("# dims: time\ngroup,time\nq,1\n")
        assert dc.annotations(data).dims == ("time",)

    def test_a_carried_table_takes_what_it_does_not_state(self, data):
        """A bare table states no attributes, so the caller may state them."""
        (data / ".annotations.csv").write_text("group,time\nq,1\n")
        loaded = dc.annotations(data, attrs={"dims": ("time",), "history": ("de",)})
        assert loaded.dims == ("time",)
        assert loaded.attrs.history == ("de",)

    def test_a_carried_set_directory_states_its_own(self, data, regions):
        """A carried directory holds its attributes, as any set directory does."""
        regions.save(data / ".annotations")
        with pytest.raises(InvalidAnnotationError, match="which states them"):
            dc.annotations(data, attrs={"dims": DIMS})

    def test_carried_twice(self, data, regions):
        """A directory states what it carries once."""
        regions.save(data / ".annotations")
        (data / ".annotations.csv").write_text("group,time\nq,1\n")
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
        picks.save(data / ".annotations")
        regions.save(data)
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
        regions.save(data / ".annotations")
        assert dc.annotations(data) == regions


@pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")
class TestParquet:
    """The same tables, with their types kept, for a set too big to want text."""

    @pytest.fixture
    def mixed(self) -> dc.AnnotationSet:
        """A set whose columns hold what a CSV would have to spell as text."""
        frame = pd.DataFrame(
            {
                "id": ["r1", "r2"],
                "group": ["noise", "quiet"],
                "value": ["car", True],
                "tags": [("road", "car"), None],
                "time_start": [
                    np.datetime64("2020-01-01T00:00:10"),
                    np.datetime64("2020-01-01T00:00:20"),
                ],
                "time_end": [
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
        loaded = dc.annotations(mixed.to_parquet(tmp_path / "picks.parquet"))
        assert loaded.to_dataframe().equals(mixed.to_dataframe())

    def test_the_dimensions_travel_with_the_file(self, mixed, tmp_path):
        """A parquet file states its dimensions where it can: its footer."""
        path = mixed.to_parquet(tmp_path / "picks.parquet")
        assert dc.annotations(path).dims == DIMS

    def test_restating_the_dimensions(self, mixed, tmp_path):
        """Agreement is allowed, disagreement is not, as with every spelling."""
        path = mixed.to_parquet(tmp_path / "picks.parquet")
        assert dc.annotations(path, dims=DIMS).dims == DIMS
        with pytest.raises(InvalidAnnotationError, match="where the two agree"):
            dc.annotations(path, dims=("depth",))

    def test_kinds_a_csv_would_lose(self, mixed, tmp_path):
        """A column with no one type is written as documents, not as text."""
        loaded = dc.annotations(mixed.to_parquet(tmp_path / "picks.parquet"))
        assert [type(x).__name__ for x in loaded.to_dataframe()["value"]] == [
            "str",
            "bool",
        ]
        assert loaded[0].extra["meta"] == {"a": 1}
        assert loaded[0].tags == ("road", "car")

    def test_text_stays_text(self, tmp_path):
        """A typed format has a boolean, so a cell reading 'true' is the word."""
        frame = pd.DataFrame({"group": ["a"], "note": ["true"], "time": [1.0]})
        picks = dc.AnnotationSet(frame, dims=("time",))
        loaded = dc.annotations(picks.to_parquet(tmp_path / "picks.parquet"))
        assert loaded[0].extra["note"] == "true"

    def test_a_value_a_csv_would_refuse(self, tmp_path):
        """Text a table would read back as a boolean is safe where types are kept."""
        frame = pd.DataFrame({"group": ["a"], "value": ["true"], "time": [1.0]})
        picks = dc.AnnotationSet(frame, dims=("time",))
        with pytest.raises(ParameterError, match="a table would read back"):
            picks.to_csv()
        loaded = dc.annotations(picks.to_parquet(tmp_path / "picks.parquet"))
        assert loaded[0].value == "true"

    def test_a_directory(self, with_vertices, curve, tmp_path):
        """Every part a set states is written under its own name."""
        directory = with_vertices.save(tmp_path / "picks", format="parquet")
        assert sorted(x.name for x in directory.iterdir()) == [
            "annotations.parquet",
            "attrs.json",
            "vertices.parquet",
        ]
        loaded = dc.annotations(directory)
        assert loaded == with_vertices
        assert loaded[1].geometry.basis == curve

    def test_a_collection(self, regions, picks, tmp_path):
        """A set is a set whichever encoding it is written in."""
        root = tmp_path / "sets"
        regions.save(root / "hand", format="parquet")
        picks.save(root / "phasenet")
        assert sorted(dc.annotations(root).attrs.sets) == ["hand", "phasenet"]

    def test_carried_beside_data(self, regions, tmp_path):
        """The hidden name takes the parquet spelling too."""
        directory = tmp_path / "data"
        directory.mkdir()
        regions.to_parquet(directory / ".annotations.parquet")
        assert dc.annotations(directory).to_dataframe().equals(regions.to_dataframe())

    def test_the_other_encoding_is_superseded(self, regions, tmp_path):
        """A set written twice states itself once, not once per encoding."""
        directory = regions.save(tmp_path / "picks")
        assert (directory / "annotations.csv").exists()
        regions.save(directory, format="parquet")
        assert not (directory / "annotations.csv").exists()
        assert dc.annotations(directory) == regions
        regions.save(directory)
        assert not (directory / "annotations.parquet").exists()

    def test_both_encodings_at_once(self, regions, tmp_path):
        """A directory holding both says two things; neither is chosen."""
        directory = regions.save(tmp_path / "picks")
        regions.to_parquet(directory / "annotations.parquet")
        with pytest.raises(InvalidAnnotationError, match="each of its parts once"):
            dc.annotations(directory)

    def test_a_bare_table_refuses_vertices(self, with_vertices, tmp_path):
        """One file states one grain, whatever its encoding."""
        with pytest.raises(ParameterError, match="a bare table has no row for"):
            with_vertices.to_parquet(tmp_path / "picks.parquet")

    def test_an_unknown_encoding(self, regions, tmp_path):
        """A set is written in an encoding it has, and says which it has."""
        with pytest.raises(ParameterError, match="not a table encoding"):
            regions.save(tmp_path / "picks", format="feather")

    def test_vertices_declare_nothing(self, with_vertices, tmp_path):
        """Vertices are read in the dimensions of the set they belong to."""
        directory = with_vertices.save(tmp_path / "picks", format="parquet")
        frame = with_vertices.to_vertices()
        write_parquet(frame, directory / "vertices.parquet", {DIMS_KEY: '["time"]'})
        with pytest.raises(InvalidAnnotationError, match="states them once"):
            dc.annotations(directory)

    def test_dimensions_which_are_not_a_document(self, regions, tmp_path):
        """What the footer states is read, and named where it is not readable."""
        path = tmp_path / "picks.parquet"
        write_parquet(regions.to_dataframe(), path, {DIMS_KEY: "time, distance"})
        with pytest.raises(InvalidAnnotationError, match="not a JSON document"):
            dc.annotations(path)

    def test_dimensions_which_name_none(self, regions, tmp_path):
        """A file which declares its dimensions names them."""
        path = tmp_path / "picks.parquet"
        write_parquet(regions.to_dataframe(), path, {DIMS_KEY: "[]"})
        with pytest.raises(InvalidAnnotationError, match="names none"):
            dc.annotations(path)

    def test_a_stray_parquet_table(self, regions, tmp_path):
        """A near-miss on the convention is a near-miss in either encoding."""
        directory = regions.save(tmp_path / "picks")
        regions.to_parquet(directory / "annotation.parquet")
        with pytest.raises(InvalidAnnotationError, match=r"annotation\.parquet"):
            dc.annotations(directory)

    def test_a_file_which_is_not_parquet(self, tmp_path):
        """Whatever pyarrow makes of it, the error names the file."""
        path = tmp_path / "picks.parquet"
        path.write_text("group,time\na,1\n")
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(path, dims=("time",))

    def test_a_table_which_is_neither(self, tmp_path):
        """A bare set is a table, and the message names the encodings it takes."""
        path = tmp_path / "picks.txt"
        path.write_text("group\na\n")
        with pytest.raises(InvalidAnnotationError, match=r"\.csv or \.parquet"):
            dc.annotations(path, dims=("time",))
