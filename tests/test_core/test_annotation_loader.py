"""Tests for reading and writing stored annotation sets."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

try:
    import yaml
except ImportError:
    yaml = None

import dascore as dc
from dascore.core.annotations import Line, Moveout
from dascore.exceptions import InvalidAnnotationError, ParameterError

DIMS = ("distance", "time")


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
        with pytest.raises(InvalidAnnotationError, match="a set directory"):
            dc.annotations(directory, attrs={"dims": DIMS})
        with pytest.raises(InvalidAnnotationError, match="a set directory"):
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
