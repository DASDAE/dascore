"""Tests for annotation sets and the models they hand out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core.annotations import (
    Annotation,
    AnnotationBasis,
    AnnotationSet,
    AnnotationSetAttrs,
    Line,
    Moveout,
    Path,
    Polygon,
    Region,
)
from dascore.exceptions import ParameterError

DIMS = ("time", "distance")

TIMES = np.array(
    ["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"],
    dtype="datetime64[ns]",
)


@pytest.fixture(scope="module")
def region_set():
    """A set of plain regions, the common case."""
    frame = pd.DataFrame(
        {
            "group": ["event", "event", "noisy"],
            "value": ["car", "truck", True],
            "distance_start": [10.0, 30.0, 0.0],
            "distance_end": [80.0, 90.0, 100.0],
        }
    )
    return AnnotationSet(frame, dims=DIMS)


@pytest.fixture(scope="module")
def path_set():
    """A set holding one path beside one region."""
    frame = pd.DataFrame(
        {
            "id": ["p1", "e1"],
            "group": ["pick", "pick"],
            "value": ["car", "truck"],
            "geometry": ["path", "region"],
            "distance_start": [np.nan, 10.0],
            "distance_end": [np.nan, 80.0],
        }
    )
    vertices = pd.DataFrame(
        {"id": ["p1"] * 3, "seq": [0, 1, 2], "time": TIMES, "distance": [1.0, 5.0, 9.0]}
    )
    return AnnotationSet(frame, dims=DIMS, vertices=vertices)


def _polygon_set(count: int = 3):
    """Build a set holding one polygon of ``count`` vertices."""
    frame = pd.DataFrame({"id": ["g1"], "geometry": ["polygon"]})
    vertices = pd.DataFrame(
        {
            "id": ["g1"] * count,
            "seq": list(range(count)),
            "distance": [float(x) for x in range(count)],
        }
    )
    return AnnotationSet(frame, dims=DIMS, vertices=vertices)


class TestConstruction:
    """A set is built from a frame and the dimensions it is stated in."""

    def test_len_and_iteration(self, region_set):
        """Every row is one annotation."""
        assert len(region_set) == 3
        assert [x.group for x in region_set] == ["event", "event", "noisy"]

    def test_rows_are_annotations(self, region_set):
        """Indexing builds the row's model."""
        assert isinstance(region_set[0], Annotation)

    def test_dims_required(self):
        """A set which names no dimension states nothing about where."""
        with pytest.raises(ParameterError, match="states its dimensions"):
            AnnotationSet(pd.DataFrame({"group": ["a"]}))

    def test_dims_from_attrs(self):
        """Dimensions may come from an attrs object instead of the keyword."""
        attrs = AnnotationSetAttrs(dims=DIMS)
        assert AnnotationSet(pd.DataFrame({"group": ["a"]}), attrs=attrs).dims == DIMS

    def test_keyword_overrides_attrs(self):
        """An explicit keyword wins over what attrs states."""
        attrs = AnnotationSetAttrs(dims=DIMS, acquisition_key="OLD.A.L.ACQ")
        out = AnnotationSet(None, attrs=attrs, acquisition_key="NEW.A.L.ACQ")
        assert out.attrs.acquisition_key == "NEW.A.L.ACQ"

    def test_empty_set(self):
        """A set with no rows is a set, not an error."""
        out = AnnotationSet(None, dims=DIMS)
        assert len(out) == 0
        assert list(out) == []

    def test_records(self):
        """Anything a frame can be built from works."""
        out = AnnotationSet([{"group": "a"}, {"group": "b"}], dims=DIMS)
        assert len(out) == 2

    def test_unreadable_data(self):
        """Something which is not tabular says so."""
        with pytest.raises(ParameterError, match="as a dataframe"):
            AnnotationSet(object(), dims=DIMS)

    def test_duplicate_columns_refused(self):
        """Pandas allows a repeated name; every reader here expects one column."""
        frame = pd.DataFrame([["g", "g", 1]], columns=["group", "group", "value"])
        with pytest.raises(ParameterError, match="more than once"):
            AnnotationSet(frame, dims=DIMS)

    def test_repr_names_contents(self, region_set):
        """The repr says how many annotations and over which dimensions."""
        assert "3 annotations" in repr(region_set)
        assert "time" in repr(region_set)

    def test_equality(self, region_set):
        """Two sets built the same way are equal."""
        same = AnnotationSet(region_set.to_dataframe(), attrs=region_set.attrs)
        assert same == region_set
        assert region_set != AnnotationSet(None, dims=DIMS)

    def test_not_equal_to_other_types(self, region_set):
        """A set is not equal to something which is not one."""
        assert region_set != "not a set"


class TestDimensionSpelling:
    """A dimension is a point, a range, or unconstrained."""

    def test_range_columns(self, region_set):
        """A start/end pair is a half-open range."""
        assert region_set[0].region.bounds["distance"] == (10.0, 80.0)

    def test_bare_column_is_a_point(self):
        """A bare dimension column states a point, which is a zero-width range."""
        out = AnnotationSet(pd.DataFrame({"distance": [5.0]}), dims=DIMS)
        assert out[0].region.bounds["distance"] == (5.0, 5.0)
        assert out[0].region.is_point("distance")

    def test_region_names_its_dims(self, region_set):
        """A region says which dimensions it constrains."""
        assert region_set[0].region.dims == ("distance",)

    def test_timedelta_endpoints_keep_their_type(self):
        """A bound on a lag dimension is a duration, not the integer behind it."""
        lags = np.array([1, 5], dtype="timedelta64[s]")
        frame = pd.DataFrame({"time_start": lags[:1], "time_end": lags[1:]})
        start, end = AnnotationSet(frame, dims=DIMS)[0].region.bounds["time"]
        assert isinstance(start, np.timedelta64)
        assert isinstance(end, np.timedelta64)

    def test_unconstrained_dim_absent(self, region_set):
        """A dimension no column names does not appear in the bounds."""
        assert "time" not in region_set[0].region.bounds

    def test_unstated_cell_is_unconstrained(self):
        """A blank cell constrains nothing, even where the column exists."""
        frame = pd.DataFrame({"distance_start": [np.nan], "distance_end": [np.nan]})
        assert AnnotationSet(frame, dims=DIMS)[0].region.bounds == {}

    def test_both_spellings_refused(self):
        """One dimension is spelled one way."""
        frame = pd.DataFrame({"time": [1], "time_start": [1], "time_end": [2]})
        with pytest.raises(ParameterError, match="both as a point"):
            AnnotationSet(frame, dims=DIMS)

    def test_half_a_range_refused(self):
        """A start with no end does not bound anything."""
        with pytest.raises(ParameterError, match="half a range"):
            AnnotationSet(pd.DataFrame({"time_start": [1]}), dims=DIMS)

    def test_reversed_range_refused(self):
        """An impossible range is structural, so the set does not load."""
        frame = pd.DataFrame({"distance_start": [9.0], "distance_end": [1.0]})
        with pytest.raises(ParameterError, match="ends before it starts"):
            AnnotationSet(frame, dims=DIMS)

    def test_reversed_range_names_its_row(self):
        """The refused row is named, so a long set says which one."""
        frame = pd.DataFrame({"distance_start": [0.0, 9.0], "distance_end": [1.0, 1.0]})
        with pytest.raises(ParameterError, match="Row 1"):
            AnnotationSet(frame, dims=DIMS)

    @pytest.mark.parametrize("side", ["distance_start", "distance_end"])
    def test_half_a_range_cell_refused(self, side):
        """A row states both ends or neither; one end bounds nothing."""
        frame = pd.DataFrame({"distance_start": [np.nan], "distance_end": [np.nan]})
        frame[side] = [1.0]
        with pytest.raises(ParameterError, match="half a distance range"):
            AnnotationSet(frame, dims=DIMS)

    def test_incomparable_range_refused(self):
        """A range whose ends cannot be compared says so, not TypeError."""
        frame = pd.DataFrame({"distance_start": [1.0, "a"], "distance_end": ["b", 2.0]})
        with pytest.raises(ParameterError, match="cannot be compared"):
            AnnotationSet(frame, dims=DIMS)

    def test_datetime_endpoints_keep_their_type(self):
        """A time bound is a time, not the integer behind it."""
        frame = pd.DataFrame({"time_start": TIMES[:1], "time_end": TIMES[2:]})
        start, end = AnnotationSet(frame, dims=DIMS)[0].region.bounds["time"]
        assert isinstance(start, np.datetime64)
        assert isinstance(end, np.datetime64)


class TestColumns:
    """Unknown columns carry; near-misses do not."""

    def test_unknown_column_is_an_extra(self):
        """A column the set does not model rides along with its row."""
        out = AnnotationSet(pd.DataFrame({"score": [0.9]}), dims=DIMS)
        assert out[0].extra["score"] == 0.9

    def test_unstated_extra_dropped(self):
        """A blank extra states nothing, so the row does not carry it."""
        out = AnnotationSet(pd.DataFrame({"score": [np.nan]}), dims=DIMS)
        assert "score" not in out[0].extra

    def test_undeclared_range_pair_refused(self):
        """A range naming no declared dimension is a forgotten dimension."""
        frame = pd.DataFrame({"depth_start": [1], "depth_end": [2]})
        with pytest.raises(ParameterError, match="name no declared dimension"):
            AnnotationSet(frame, dims=DIMS)

    def test_lone_range_column_is_an_extra(self):
        """One half of a range names no dimension, so it is just a column."""
        out = AnnotationSet(pd.DataFrame({"depth_start": [1]}), dims=DIMS)
        assert out[0].extra["depth_start"] == 1

    def test_declared_column_documents_only(self):
        """Documenting a column does not gate any other one."""
        out = AnnotationSet(
            pd.DataFrame({"score": [0.9], "other": [1]}),
            dims=DIMS,
            columns={"score": {"description": "Confidence", "units": "dimensionless"}},
        )
        assert out.attrs.columns["score"].description == "Confidence"
        assert "other" in out[0].extra

    def test_stated_dtype_checked(self):
        """A column which says what it holds must hold it."""
        with pytest.raises(ParameterError, match="states dtype"):
            AnnotationSet(
                pd.DataFrame({"score": ["high"]}),
                dims=DIMS,
                columns={"score": {"dtype": "float64"}},
            )

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [("str", ["a", "b"]), ("category", ["a", "b"]), ("Int64", [1, 2])],
    )
    def test_extension_dtypes_declarable(self, dtype, values):
        """Pandas gives plain text a `str` dtype, which numpy cannot name."""
        frame = pd.DataFrame({"note": values}).astype(dtype)
        out = AnnotationSet(frame, dims=DIMS, columns={"note": {"dtype": dtype}})
        assert len(out) == 2

    def test_category_needs_no_categories(self):
        """A column documented as categorical says so, not which categories."""
        frame = pd.DataFrame({"note": ["a", "b"]}).astype("category")
        assert (
            len(
                AnnotationSet(frame, dims=DIMS, columns={"note": {"dtype": "category"}})
            )
            == 2
        )

    def test_datetime_unit_still_distinguished(self):
        """Comparing by name keeps a nanosecond column from passing as microsecond."""
        frame = pd.DataFrame({"when": TIMES[:1]})
        with pytest.raises(ParameterError, match="states dtype"):
            AnnotationSet(
                frame, dims=DIMS, columns={"when": {"dtype": "datetime64[us]"}}
            )

    def test_unreadable_dtype_refused(self):
        """A dtype naming nothing says so, rather than raising numpy's error."""
        with pytest.raises(ParameterError, match="declares the dtype"):
            AnnotationSet(
                pd.DataFrame({"score": [1.0]}),
                dims=DIMS,
                columns={"score": {"dtype": "not-a-dtype"}},
            )

    def test_stated_dtype_absent_column(self):
        """Documenting a column the frame lacks is not an error."""
        out = AnnotationSet(None, dims=DIMS, columns={"score": {"dtype": "float64"}})
        assert "score" in out.attrs.columns


class TestValues:
    """A group holds one kind of value."""

    def test_mixed_kinds_refused(self):
        """A boolean and a number in one group are two variables."""
        frame = pd.DataFrame({"group": ["g", "g"], "value": [True, 3]})
        with pytest.raises(ParameterError, match="mixes"):
            AnnotationSet(frame, dims=DIMS)

    def test_unnamed_group_is_still_a_group(self):
        """Pandas drops a null grouping key; the unnamed group is checked anyway."""
        frame = pd.DataFrame({"group": [None, None], "value": [True, 1]})
        with pytest.raises(ParameterError, match="mixes"):
            AnnotationSet(frame, dims=DIMS)

    def test_int_and_float_are_one_kind(self):
        """Numbers are numbers; the model reads them alike."""
        frame = pd.DataFrame({"group": ["g", "g"], "value": [1.5, 3]})
        assert len(AnnotationSet(frame, dims=DIMS)) == 2

    def test_groups_are_independent(self):
        """Two groups may hold different kinds."""
        frame = pd.DataFrame({"group": ["a", "b"], "value": [True, "car"]})
        assert len(AnnotationSet(frame, dims=DIMS)) == 2

    def test_overlap_is_not_checked(self):
        """Non-overlap only means anything at projection, so it is deferred."""
        frame = pd.DataFrame(
            {
                "group": ["g", "g"],
                "value": ["a", "b"],
                "distance_start": [0.0, 5.0],
                "distance_end": [10.0, 15.0],
            }
        )
        assert len(AnnotationSet(frame, dims=DIMS)) == 2

    def test_value_defaults_to_true(self):
        """An annotation with no value is a bare flag."""
        assert AnnotationSet(pd.DataFrame({"group": ["a"]}), dims=DIMS)[0].value is True

    def test_numpy_bool_stays_a_flag(self):
        """A mask element is membership, not the number one."""
        frame = pd.DataFrame({"group": ["a"], "value": [np.bool_(True)]})
        assert AnnotationSet(frame, dims=DIMS)[0].value is True

    def test_non_finite_value_refused(self):
        """A value which cannot survive a round trip is not a value."""
        frame = pd.DataFrame({"group": ["a"], "value": [np.inf]})
        with pytest.raises(ParameterError, match="must be finite"):
            AnnotationSet(frame, dims=DIMS)


class TestTags:
    """Tags are multi-membership labels."""

    def test_comma_separated_text(self):
        """A tag cell written as text splits on commas."""
        out = AnnotationSet(pd.DataFrame({"tags": ["a, b"]}), dims=DIMS)
        assert out[0].tags == ("a", "b")

    def test_sequence(self):
        """A cell holding a sequence is taken as it is."""
        out = AnnotationSet(pd.DataFrame({"tags": [["a", "b"]]}), dims=DIMS)
        assert out[0].tags == ("a", "b")

    def test_scalar_is_one_tag(self):
        """A label which happens to be a number is still one label."""
        assert AnnotationSet(pd.DataFrame({"tags": [5]}), dims=DIMS)[0].tags == ("5",)

    def test_absent(self):
        """No tags is an empty tuple, not None."""
        assert AnnotationSet(pd.DataFrame({"group": ["a"]}), dims=DIMS)[0].tags == ()


class TestIdentity:
    """Ids are the producer's, and nothing here invents one."""

    def test_absent_id_is_blank(self, region_set):
        """A set without ids is fine; identity-needing operations are not."""
        assert region_set[0].id == ""

    def test_duplicate_ids_refused(self):
        """An id names one row."""
        with pytest.raises(ParameterError, match="more than one row"):
            AnnotationSet(pd.DataFrame({"id": ["a", "a"]}), dims=DIMS)

    def test_blank_ids_may_repeat(self):
        """Unstated identity is not a clash."""
        frame = pd.DataFrame({"id": [None, None], "group": ["a", "b"]})
        assert len(AnnotationSet(frame, dims=DIMS)) == 2

    def test_parent_must_exist(self):
        """A parent names an annotation of this set."""
        frame = pd.DataFrame({"id": ["a"], "parent": ["z"]})
        with pytest.raises(ParameterError, match="name no annotation"):
            AnnotationSet(frame, dims=DIMS)

    def test_parent_resolves(self):
        """A pick group points at its parent by id."""
        frame = pd.DataFrame({"id": ["a", "b"], "parent": ["", "a"]})
        assert AnnotationSet(frame, dims=DIMS)[1].parent == "a"


class TestAcquisitionKeyColumn:
    """A set may span acquisitions, so a row may name its own."""

    def test_row_takes_the_set_key(self):
        """A row naming none is addressed by the set."""
        out = AnnotationSet(
            pd.DataFrame({"group": ["a"]}), dims=DIMS, acquisition_key="N.A.L.ACQ"
        )
        assert out[0].acquisition_key == "N.A.L.ACQ"

    def test_row_overrides_the_set_key(self):
        """A row naming one overrides the set-level address."""
        frame = pd.DataFrame({"acquisition_key": ["N.A.L.OTHER"]})
        out = AnnotationSet(frame, dims=DIMS, acquisition_key="N.A.L.ACQ")
        assert out[0].acquisition_key == "N.A.L.OTHER"

    def test_blank_row_key_falls_back(self):
        """An empty cell states nothing, so the set's key stands."""
        frame = pd.DataFrame({"acquisition_key": [None]})
        out = AnnotationSet(frame, dims=DIMS, acquisition_key="N.A.L.ACQ")
        assert out[0].acquisition_key == "N.A.L.ACQ"

    def test_row_key_validated(self):
        """A row's key is checked like the set's."""
        frame = pd.DataFrame({"acquisition_key": ["nope"]})
        with pytest.raises(ValidationError, match="Invalid acquisition_key"):
            AnnotationSet(frame, dims=DIMS)[0]

    def test_key_column_is_not_an_extra(self):
        """The column is modelled, so it does not also ride along as an extra."""
        frame = pd.DataFrame({"acquisition_key": ["N.A.L.ACQ"]})
        assert "acquisition_key" not in AnnotationSet(frame, dims=DIMS)[0].extra


class TestGeometryKinds:
    """A row states which geometry it is."""

    def test_default_is_region(self, region_set):
        """A set naming no geometry is a set of regions."""
        assert isinstance(region_set[0].geometry, Region)

    def test_unknown_kind_refused(self):
        """A geometry this format has no meaning for is refused."""
        with pytest.raises(ParameterError, match="is not one of"):
            AnnotationSet(pd.DataFrame({"geometry": ["blob"]}), dims=DIMS)

    def test_path_requires_id(self):
        """Vertices are grouped by id, so a path without one binds to nothing."""
        with pytest.raises(ParameterError, match="no id"):
            AnnotationSet(pd.DataFrame({"geometry": ["path"]}), dims=DIMS)

    def test_path_geometry(self, path_set):
        """A path row builds a Path holding its vertices."""
        geometry = path_set[0].geometry
        assert isinstance(geometry, Path)
        assert len(geometry) == 3
        assert geometry.vertices["distance"] == (1.0, 5.0, 9.0)

    def test_polygon_geometry(self):
        """A polygon row builds a Polygon."""
        assert isinstance(_polygon_set()[0].geometry, Polygon)

    def test_region_beside_a_path(self, path_set):
        """A set may mix geometries; each row is read as what it says."""
        assert isinstance(path_set[1].geometry, Region)

    def test_vertex_dims(self, path_set):
        """The vertices name the dimensions they are stated in."""
        assert set(path_set[0].geometry.dims) == set(DIMS)

    def test_datetime_vertices_keep_their_type(self, path_set):
        """A vertex on the time dimension is a time."""
        assert isinstance(path_set[0].geometry.vertices["time"][0], np.datetime64)

    def test_region_property_for_every_geometry(self, path_set):
        """Every annotation has a bounding region, whatever its geometry."""
        assert isinstance(path_set[0].region, Region)
        assert isinstance(path_set[1].region, Region)


class TestVertices:
    """The vertices frame is checked against the rows which need it."""

    def test_bounds_derived_from_vertices(self, path_set):
        """A path's bounding region is the box its vertices fill."""
        assert path_set[0].region.bounds["distance"] == (1.0, 9.0)

    def test_derived_bounds_reach_the_frame(self, path_set):
        """The derived box is a real column, so table operations see it."""
        row = path_set.to_dataframe().iloc[0]
        assert (row["distance_start"], row["distance_end"]) == (1.0, 9.0)

    def test_stated_bounds_must_agree(self):
        """The vertices are the shape; a box which disagrees is refused."""
        frame = pd.DataFrame(
            {
                "id": ["x"],
                "geometry": ["path"],
                "distance_start": [99.0],
                "distance_end": [100.0],
            }
        )
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, 1], "distance": [1.0, 2.0]}
        )
        with pytest.raises(ParameterError, match="disagrees with their vertices"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_path_spanning_a_point_dimension_refused(self):
        """A set spelling a dimension as a point holds no geometry spanning it."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"], "distance": [np.nan]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, 1], "distance": [1.0, 2.0]}
        )
        with pytest.raises(ParameterError, match="spells as a point"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_degenerate_path_on_a_point_dimension(self):
        """Vertices which do not move along a point dimension fill it in."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"], "distance": [np.nan]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, 1], "distance": [2.0, 2.0]}
        )
        out = AnnotationSet(frame, dims=DIMS, vertices=vertices)
        assert out[0].region.bounds["distance"] == (2.0, 2.0)

    def test_stated_point_must_agree(self):
        """A stated point which the vertices contradict is refused."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"], "distance": [99.0]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, 1], "distance": [2.0, 2.0]}
        )
        with pytest.raises(ParameterError, match="disagrees with their vertices"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_missing_vertices_refused(self):
        """A path with no vertices states no shape."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        with pytest.raises(ParameterError, match="state no vertices"):
            AnnotationSet(frame, dims=DIMS)

    def test_one_path_without_vertices_refused(self):
        """Every path needs vertices, not just one of them."""
        frame = pd.DataFrame({"id": ["x", "y"], "geometry": ["path", "path"]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, 1], "distance": [1.0, 2.0]}
        )
        with pytest.raises(ParameterError, match="y state no vertices"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_stray_vertices_refused(self):
        """Vertices belong to a path or polygon of this set."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["region"]})
        vertices = pd.DataFrame({"id": ["y"], "seq": [0], "distance": [1.0]})
        with pytest.raises(ParameterError, match="name no path or polygon"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_scaffolding_columns_required(self):
        """Vertices are grouped by id and ordered by seq."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        vertices = pd.DataFrame({"id": ["x", "x"], "distance": [1.0, 2.0]})
        with pytest.raises(ParameterError, match="state no seq"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_undeclared_vertex_dim_refused(self):
        """A vertex column names a declared dimension."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        vertices = pd.DataFrame({"id": ["x", "x"], "seq": [0, 1], "depth": [1.0, 2.0]})
        with pytest.raises(ParameterError, match="name no declared dimension"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_vertices_place_something(self):
        """Vertices with no dimension column place nothing."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        vertices = pd.DataFrame({"id": ["x", "x"], "seq": [0, 1]})
        with pytest.raises(ParameterError, match="no dimension column"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_blank_seq_refused(self):
        """A vertex with no seq has no place in the order."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, None], "distance": [1.0, 2.0]}
        )
        with pytest.raises(ParameterError, match="state no seq"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_blank_vertex_dimension_refused(self):
        """A vertex leaving a dimension empty places nothing there."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, 1], "distance": [np.nan, 1.0]}
        )
        with pytest.raises(ParameterError, match="leave a dimension empty"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_repeated_seq_refused(self):
        """Two vertices in one place do not say which comes first."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [0, 0], "distance": [1.0, 2.0]}
        )
        with pytest.raises(ParameterError, match="repeat a seq"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_vertices_ordered_by_seq(self):
        """Vertices are read in seq order, not the order they were written."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        vertices = pd.DataFrame(
            {"id": ["x", "x"], "seq": [1, 0], "distance": [9.0, 1.0]}
        )
        out = AnnotationSet(frame, dims=DIMS, vertices=vertices)
        assert out[0].geometry.vertices["distance"] == (1.0, 9.0)

    @pytest.mark.parametrize(("kind", "least"), [("path", 2), ("polygon", 3)])
    def test_too_few_vertices_refused(self, kind, least):
        """A shape needs enough points to be one."""
        frame = pd.DataFrame({"id": ["x"], "geometry": [kind]})
        count = least - 1
        vertices = pd.DataFrame(
            {
                "id": ["x"] * count,
                "seq": list(range(count)),
                "distance": [float(x) for x in range(count)],
            }
        )
        with pytest.raises(ParameterError, match="at least"):
            AnnotationSet(frame, dims=DIMS, vertices=vertices)

    def test_to_vertices_round_trips(self, path_set):
        """The vertices come back out whole: order, coordinates and types."""
        out = path_set.to_vertices()
        assert list(out["id"]) == ["p1"] * 3
        assert list(out["seq"]) == [0, 1, 2]
        assert list(out["distance"]) == [1.0, 5.0, 9.0]
        assert out["time"].dtype == np.dtype("datetime64[ns]")
        assert list(out["time"]) == list(pd.Series(TIMES))

    def test_empty_vertices_frame(self, region_set):
        """A set with no vertex geometry has an empty vertices frame."""
        assert region_set.to_vertices().empty


class TestFrames:
    """The frames the set hands back are copies of its own."""

    def test_to_dataframe_is_a_copy(self, region_set):
        """Mutating what a set handed out does not reach the set."""
        frame = region_set.to_dataframe()
        frame.loc[0, "group"] = "changed"
        assert region_set[0].group == "event"

    def test_to_vertices_is_a_copy(self, path_set):
        """The same holds for the vertices."""
        vertices = path_set.to_vertices()
        vertices.loc[0, "distance"] = 999.0
        assert path_set[0].geometry.vertices["distance"][0] == 1.0

    def test_extras_are_frozen(self):
        """A mutable cell cannot be edited through the annotation holding it."""
        out = AnnotationSet(pd.DataFrame({"note": [[1, 2]]}), dims=DIMS)
        assert out[0].extra["note"] == (1, 2)
        with pytest.raises(AttributeError):
            out[0].extra["note"].append(3)

    def test_nested_extras_are_frozen(self):
        """Freezing reaches inside a mapping cell too."""
        out = AnnotationSet(pd.DataFrame({"note": [{"a": [1]}]}), dims=DIMS)
        with pytest.raises(TypeError):
            out[0].extra["note"]["a"] = 2

    def test_round_trip(self, region_set):
        """A set rebuilt from its own frame holds the same annotations."""
        rebuilt = AnnotationSet(region_set.to_dataframe(), attrs=region_set.attrs)
        assert [x.group for x in rebuilt] == [x.group for x in region_set]
        assert rebuilt == region_set


class TestAttrs:
    """What a set says about itself."""

    def test_dims_required(self):
        """A set states at least one dimension."""
        with pytest.raises(ValidationError, match="at least one dimension"):
            AnnotationSetAttrs(dims=())

    def test_dims_unique(self):
        """A dimension named twice is one dimension."""
        with pytest.raises(ValidationError, match="must be unique"):
            AnnotationSetAttrs(dims=("time", "time"))

    def test_dim_may_not_alias_another_dims_range(self):
        """`distance_start` would be both a point and the start of `distance`."""
        with pytest.raises(ValidationError, match="spelled like the range column"):
            AnnotationSetAttrs(dims=("distance", "distance_start"))

    def test_dim_may_not_shadow_a_reserved_column(self):
        """A dimension named `group` would collide with the group column."""
        with pytest.raises(ValidationError, match="reserved column"):
            AnnotationSetAttrs(dims=("group", "time"))

    def test_creation_info_default(self):
        """A set carries provenance even when nothing was said."""
        assert AnnotationSetAttrs(dims=DIMS).creation_info.author == ""

    def test_acquisition_key_carries(self):
        """Provenance is an acquisition key, not a file pointer."""
        attrs = AnnotationSetAttrs(dims=DIMS, acquisition_key="NET.ARRAY.LOC.ACQ")
        assert attrs.acquisition_key == "NET.ARRAY.LOC.ACQ"

    def test_acquisition_key_validated(self):
        """The key is spelled as PatchAttrs spells it, and checked alike."""
        with pytest.raises(ValidationError, match="Invalid acquisition_key"):
            AnnotationSetAttrs(dims=DIMS, acquisition_key="nope")

    def test_provenance_defaults_to_unset(self):
        """Phase 2 cannot know a patch, so the producer supplies these."""
        attrs = AnnotationSetAttrs(dims=DIMS)
        assert attrs.acquisition_key == ""
        assert attrs.history == ()

    def test_history_keeps_the_lineage(self):
        """Picks made on decimated data only mean anything against that."""
        attrs = AnnotationSetAttrs(dims=DIMS, history=("decimate(8)", "pass_filter"))
        assert attrs.history == ("decimate(8)", "pass_filter")

    def test_lone_history_entry(self):
        """PatchAttrs.history may be one string; that is a history of one."""
        attrs = AnnotationSetAttrs(dims=DIMS, history="decimate(8)")
        assert attrs.history == ("decimate(8)",)

    def test_creation_info_identifies_the_producer(self):
        """A picker names itself the way the inventory names any process."""
        attrs = AnnotationSetAttrs(
            dims=DIMS,
            creation_info={"author": "phasenet", "version": "2.1"},
        )
        assert attrs.creation_info.author == "phasenet"
        assert attrs.creation_info.version == "2.1"

    def test_attrs_are_frozen(self):
        """Attributes are immutable, like every DASCore model."""
        with pytest.raises(ValidationError):
            AnnotationSetAttrs(dims=DIMS).dims = ("other",)


class TestBasis:
    """Curves regenerate vertices; they are not geometries themselves."""

    def test_line_walks_between_its_ends(self):
        """A line is sampled evenly from one end to the other."""
        out = Line(start={"distance": 0.0}, end={"distance": 10.0})
        assert list(out.vertices(3)["distance"]) == [0.0, 5.0, 10.0]

    def test_line_carries_real_coordinates(self):
        """A time endpoint is a time, so the curve needs no separate origin."""
        out = Line(
            start={"distance": 0.0, "time": TIMES[0]},
            end={"distance": 100.0, "time": TIMES[2]},
        )
        drawn = out.vertices(3)
        assert drawn["time"][0] == TIMES[0]
        assert drawn["time"][-1] == TIMES[2]
        assert drawn["time"].dtype == np.dtype("datetime64[ns]")

    def test_line_can_be_an_instant(self):
        """One time across all distance -- a shot, a trigger -- is a line."""
        out = Line(
            start={"distance": 0.0, "time": TIMES[0]},
            end={"distance": 100.0, "time": TIMES[0]},
        )
        assert set(out.vertices(4)["time"]) == {TIMES[0]}

    def test_line_names_its_dims(self):
        """The endpoints name the dimensions, so nothing states them twice."""
        out = Line(start={"distance": 0.0}, end={"distance": 1.0})
        assert out.dims == ("distance",)

    def test_line_ends_must_place_the_same_dims(self):
        """Two ends in different frames draw no line between them."""
        with pytest.raises(ValidationError, match="different dimensions"):
            Line(start={"distance": 0.0}, end={"time": TIMES[0]})

    def test_line_states_somewhere(self):
        """An endpoint naming no dimension is nowhere."""
        with pytest.raises(ValidationError, match="states no dimension"):
            Line(start={}, end={})

    def test_line_of_no_length(self):
        """A line beginning where it ends is a point."""
        with pytest.raises(ValidationError, match="no length"):
            Line(start={"distance": 1.0}, end={"distance": 1.0})

    def test_moveout_apex_is_the_earliest_arrival(self):
        """The apex anchors the curve, and nothing arrives before it."""
        out = Moveout(
            apex_distance=50.0,
            apex_time=TIMES[0],
            velocity=3000.0,
            distance_start=0.0,
            distance_end=100.0,
        )
        drawn = out.vertices(11)
        assert drawn["time"].min() == TIMES[0]
        assert drawn["time"][5] == TIMES[0]

    def test_moveout_on_the_cable_is_straight(self):
        """A source with no standoff runs both ways at its velocity."""
        out = Moveout(
            apex_distance=50.0,
            apex_time=TIMES[0],
            velocity=3000.0,
            distance_start=0.0,
            distance_end=100.0,
        )
        seconds = (out.vertices(3)["time"] - TIMES[0]) / np.timedelta64(1, "s")
        assert np.allclose(seconds, [50 / 3000, 0.0, 50 / 3000])

    def test_standoff_flattens_the_apex(self):
        """A source off the cable arrives sooner away from the apex."""
        shared = {
            "apex_distance": 50.0,
            "apex_time": TIMES[0],
            "velocity": 3000.0,
            "distance_start": 0.0,
            "distance_end": 100.0,
        }
        straight = Moveout(**shared).vertices(3)["time"]
        curved = Moveout(**shared, standoff=40.0).vertices(3)["time"]
        assert curved[0] < straight[0]
        assert curved[1] == straight[1] == TIMES[0]

    def test_moveout_times_are_times(self):
        """The curve draws in the dimension's own coordinates."""
        out = Moveout(
            apex_distance=0.0,
            apex_time=TIMES[0],
            velocity=1000.0,
            distance_start=0.0,
            distance_end=10.0,
        )
        assert out.vertices(2)["time"].dtype == np.dtype("datetime64[ns]")

    def test_moveout_is_pinned_to_its_dims(self):
        """A moveout is physics, so it relates fiber distance to arrival time."""
        out = Moveout(
            apex_distance=0.0,
            apex_time=TIMES[0],
            velocity=1000.0,
            distance_start=0.0,
            distance_end=1.0,
        )
        assert out.dims == ("distance", "time")

    def test_moveout_velocity_positive(self):
        """A wavefront which does not move has no moveout."""
        with pytest.raises(ValidationError):
            Moveout(
                apex_distance=0.0,
                apex_time=TIMES[0],
                velocity=0.0,
                distance_start=0.0,
                distance_end=1.0,
            )

    def test_moveout_standoff_not_negative(self):
        """A source is off the cable or on it, never behind it."""
        with pytest.raises(ValidationError):
            Moveout(
                apex_distance=0.0,
                apex_time=TIMES[0],
                velocity=1.0,
                standoff=-1.0,
                distance_start=0.0,
                distance_end=1.0,
            )

    def test_moveout_span_must_be_positive(self):
        """A curve which ends where it starts draws nothing."""
        with pytest.raises(ValidationError, match="must exceed"):
            Moveout(
                apex_distance=0.0,
                apex_time=TIMES[0],
                velocity=1.0,
                distance_start=1.0,
                distance_end=1.0,
            )

    @pytest.mark.parametrize("count", [0, 1])
    def test_too_few_points(self, count):
        """A curve is drawn from at least two points."""
        out = Line(start={"distance": 0.0}, end={"distance": 1.0})
        with pytest.raises(ParameterError, match="at least 2 points"):
            out.vertices(count)

    def test_base_is_abstract(self):
        """The base class states the interface and implements none of it."""
        with pytest.raises(NotImplementedError):
            AnnotationBasis().vertices()

    def test_base_names_no_dims(self):
        """A curve says which dimensions it is stated in; the base cannot."""
        with pytest.raises(NotImplementedError):
            AnnotationBasis().dims

    def test_carried_by_a_path(self):
        """A path may keep the curve its vertices came from."""
        basis = Line(start={"distance": 0.0}, end={"distance": 1.0})
        out = Path(
            region=Region(bounds={"distance": (0.0, 1.0)}),
            vertices={"distance": (0.0, 1.0)},
            basis=basis,
        )
        assert out.basis == basis

    def test_regenerates_what_the_frame_holds(self):
        """The point of a basis: its output is vertices, not numbers to convert."""
        basis = Moveout(
            apex_distance=50.0,
            apex_time=TIMES[0],
            velocity=3000.0,
            standoff=40.0,
            distance_start=0.0,
            distance_end=100.0,
        )
        drawn = basis.vertices(5)
        vertices = pd.DataFrame({"id": ["m"] * 5, "seq": range(5), **drawn})
        frame = pd.DataFrame({"id": ["m"], "geometry": ["path"], "basis": [basis]})
        out = AnnotationSet(frame, dims=DIMS, vertices=vertices)
        assert out[0].geometry.vertices["time"] == tuple(drawn["time"])


class TestGeometryModels:
    """A geometry built straight from a document checks itself."""

    def test_no_dimension_refused(self):
        """Vertices naming no dimension place the geometry nowhere."""
        with pytest.raises(ValidationError, match="states no dimension"):
            Path(region=Region(bounds={}), vertices={})

    def test_ragged_vertices_refused(self):
        """Every dimension states every point, or they pair up wrongly."""
        with pytest.raises(ValidationError, match="differ in length"):
            Path(
                region=Region(bounds={}),
                vertices={"time": (1, 2, 3), "distance": (1,)},
            )

    @pytest.mark.parametrize(("model", "least"), [(Path, 2), (Polygon, 3)])
    def test_too_few_vertices_refused(self, model, least):
        """A shape needs enough points to be one, however it was built."""
        with pytest.raises(ValidationError, match="at least"):
            model(
                region=Region(bounds={}),
                vertices={"distance": tuple(range(least - 1))},
            )

    def test_length_is_the_vertex_count(self):
        """Every dimension is the same length, so any of them is the count."""
        out = Path(region=Region(bounds={}), vertices={"distance": (0.0, 1.0, 2.0)})
        assert len(out) == 3


class TestBasisColumn:
    """A set carries the curve its vertices were drawn from."""

    @staticmethod
    def _vertices():
        """Two vertices for the path every test here builds."""
        return pd.DataFrame({"id": ["x", "x"], "seq": [0, 1], "distance": [0.0, 1.0]})

    def test_basis_as_a_document(self):
        """A cell holding the curve's document reads back as the model."""
        document = {
            "object_type": "Line",
            "start": {"distance": 0.0},
            "end": {"distance": 1.0},
        }
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"], "basis": [document]})
        out = AnnotationSet(frame, dims=DIMS, vertices=self._vertices())
        assert isinstance(out[0].geometry.basis, Line)

    def test_basis_as_a_model(self):
        """A cell holding the model itself is taken as it is."""
        basis = Moveout(
            apex_distance=0.0,
            apex_time=TIMES[0],
            velocity=2.0,
            distance_start=0.0,
            distance_end=1.0,
        )
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"], "basis": [basis]})
        out = AnnotationSet(frame, dims=DIMS, vertices=self._vertices())
        assert out[0].geometry.basis == basis

    def test_no_basis(self):
        """A path without a curve is simply vertices."""
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"]})
        out = AnnotationSet(frame, dims=DIMS, vertices=self._vertices())
        assert out[0].geometry.basis is None

    def test_basis_needs_vertices(self):
        """A basis regenerates vertices, so a region has no use for one."""
        frame = pd.DataFrame(
            {"id": ["x"], "geometry": ["region"], "basis": [{"object_type": "Line"}]}
        )
        with pytest.raises(ParameterError, match="no path or polygon"):
            AnnotationSet(frame, dims=DIMS)

    def test_unreadable_basis(self):
        """A cell naming no curve says so, when the set loads."""
        frame = pd.DataFrame(
            {"id": ["x"], "geometry": ["path"], "basis": [{"object_type": "Nope"}]}
        )
        with pytest.raises(ParameterError, match="as a curve"):
            AnnotationSet(frame, dims=DIMS, vertices=self._vertices())

    def test_basis_dims_must_be_declared(self):
        """A curve in an unrelated frame regenerates unrelated vertices."""
        document = {
            "object_type": "Line",
            "start": {"depth": 0.0, "other": 1.0},
            "end": {"depth": 1.0, "other": 2.0},
        }
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"], "basis": [document]})
        with pytest.raises(ParameterError, match="does not declare"):
            AnnotationSet(frame, dims=DIMS, vertices=self._vertices())

    def test_basis_without_a_geometry_column(self):
        """A frame naming no geometry is all regions, which carry no curve."""
        frame = pd.DataFrame({"id": ["x"], "basis": [{"object_type": "Line"}]})
        with pytest.raises(ParameterError, match="no path or polygon"):
            AnnotationSet(frame, dims=DIMS)


class TestSerialization:
    """The models are documents, like every other DASCore model."""

    @pytest.mark.parametrize(
        "basis",
        [
            Line(start={"distance": 0.0}, end={"distance": 1.0}),
            Moveout(
                apex_distance=0.0,
                apex_time=TIMES[0],
                velocity=1.0,
                distance_start=0.0,
                distance_end=1.0,
            ),
        ],
    )
    def test_basis_names_its_class(self, basis):
        """A document says which curve it holds, so the union can dispatch."""
        assert basis.model_dump(mode="json")["object_type"] == type(basis).__name__

    def test_basis_round_trip(self):
        """A path rebuilds its basis as the class which wrote it."""
        basis = Moveout(
            apex_distance=0.0,
            apex_time=TIMES[0],
            velocity=2.0,
            distance_start=0.0,
            distance_end=1.0,
        )
        path = Path(
            region=Region(bounds={"distance": (0.0, 1.0)}),
            vertices={"distance": (0.0, 1.0)},
            basis=basis,
        )
        rebuilt = Path(**path.model_dump(mode="json"))
        assert isinstance(rebuilt.basis, Moveout)
        assert rebuilt.basis.velocity == 2.0

    def test_region_round_trip(self):
        """A region survives a document."""
        region = Region(bounds={"distance": (0.0, 1.0)})
        assert Region(**region.model_dump(mode="json")) == region

    def test_datetime_bounds_write_a_document(self):
        """A time bound has no json type, so it is written as DASCore spells it."""
        region = Region(bounds={"time": (TIMES[0], TIMES[2])})
        written = region.model_dump(mode="json")["bounds"]["time"]
        assert written == [str(TIMES[0]), str(TIMES[2])]
        assert all(isinstance(x, str) for x in written)

    def test_python_dump_keeps_the_time(self):
        """A python dump is what equality compares, so it keeps the value."""
        region = Region(bounds={"time": (TIMES[0], TIMES[2])})
        kept = region.model_dump()["bounds"]["time"][0]
        assert isinstance(kept, np.datetime64) and kept == TIMES[0]

    def test_numpy_numbers_write_plainly(self):
        """A numpy number is written as the number it is."""
        region = Region(bounds={"distance": (np.float64(1.5), np.float64(2.5))})
        assert region.model_dump(mode="json")["bounds"]["distance"] == [1.5, 2.5]

    def test_vertices_write_a_document(self):
        """The same holds for the vertices of a path."""
        path = Path(region=Region(bounds={}), vertices={"time": (TIMES[0], TIMES[1])})
        written = path.model_dump(mode="json")["vertices"]["time"]
        assert written == [str(TIMES[0]), str(TIMES[1])]

    def test_geometry_kinds_are_distinct(self):
        """A polygon is not a path which happens to close."""
        assert not isinstance(
            Polygon(
                region=Region(bounds={}),
                vertices={"distance": (0.0, 1.0, 2.0)},
            ),
            Path,
        )


class TestTopLevel:
    """The set is reachable from the top-level namespace."""

    def test_dc_annotation_set(self):
        """`dc.AnnotationSet` is the in-memory door."""
        assert dc.AnnotationSet is AnnotationSet
