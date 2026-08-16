"""Tests for reading and writing stored annotation sets."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

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


class TestTheDoor:
    """Everything a set may be loaded from goes through one function."""

    def test_a_set_is_itself(self, regions):
        """Loading a set which is already loaded hands it back."""
        assert dc.annotations(regions) is regions

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

    def test_the_caller_wins(self, regions, tmp_path):
        """A caller stating dimensions states them for the whole read."""
        directory = regions.save(tmp_path / "picks")
        assert dc.annotations(directory, dims=("time", "distance")).dims == (
            "time",
            "distance",
        )


class TestTheAttrsFile:
    """What a set directory says about itself."""

    def test_json_spelling(self, regions, tmp_path):
        """One data model stands behind both spellings."""
        directory = regions.save(tmp_path / "picks")
        document = json.loads(
            regions.attrs.model_dump_json(exclude_defaults=True),
        )
        (directory / "attrs.json").write_text(json.dumps(document))
        (directory / "attrs.yaml").unlink()
        assert dc.annotations(directory) == regions

    def test_two_spellings(self, regions, tmp_path):
        """A set spells each of its parts once."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.json").write_text("{}")
        with pytest.raises(InvalidAnnotationError, match="more than once"):
            dc.annotations(directory)

    def test_the_wrong_object(self, regions, tmp_path):
        """A file declaring another model is a misfiled object."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.yaml").write_text("object_type: Inventory\ndims: [time]\n")
        with pytest.raises(InvalidAnnotationError, match="declares 'Inventory'"):
            dc.annotations(directory)

    def test_which_is_not_a_mapping(self, regions, tmp_path):
        """A document stating a list defines no attributes."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.yaml").write_text("- distance\n- time\n")
        with pytest.raises(InvalidAnnotationError, match="no mapping"):
            dc.annotations(directory)

    def test_which_does_not_parse(self, regions, tmp_path):
        """Unparseable YAML names the file rather than the parser."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.yaml").write_text("dims: [\n")
        with pytest.raises(InvalidAnnotationError, match="Could not parse YAML"):
            dc.annotations(directory)

    def test_bad_json(self, regions, tmp_path):
        """Unparseable JSON names the file too."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.yaml").unlink()
        (directory / "attrs.json").write_text("{")
        with pytest.raises(InvalidAnnotationError, match="Could not parse JSON"):
            dc.annotations(directory)

    def test_which_cannot_be_read(self, regions, tmp_path):
        """A file which does not decode names itself, not the codec."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.yaml").write_bytes(b"dims: [\xff\xfe]\n")
        with pytest.raises(InvalidAnnotationError, match="Could not read"):
            dc.annotations(directory)

    def test_no_attrs_file(self, regions, tmp_path):
        """A directory without one is read on the caller's dimensions."""
        directory = regions.save(tmp_path / "picks")
        (directory / "attrs.yaml").unlink()
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

    def test_an_id_which_looks_like_a_number(self, tmp_path):
        """An id is the label its vertices name it by, never a number."""
        frame = pd.DataFrame({"id": ["1"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {"id": ["1", "1"], "seq": [0, 1], "distance": [1.0, 2.0]}
        )
        annotations = dc.AnnotationSet(frame, dims=("distance",), vertices=vertices)
        loaded = dc.annotations(annotations.save(tmp_path / "picks"))
        assert loaded[0].id == "1"


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
        assert written == {"attrs.yaml", "annotations.csv", "vertices.csv"}

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

    def test_the_attrs_name_their_model(self, regions, tmp_path):
        """The document says what it holds, as every stored object does."""
        directory = regions.save(tmp_path / "picks")
        text = (directory / "attrs.yaml").read_text()
        assert "object_type: AnnotationSetAttrs" in text
