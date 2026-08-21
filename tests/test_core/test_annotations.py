"""Tests for annotation sets and the models they hand out."""

from __future__ import annotations

import datetime
from contextlib import suppress

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
from dascore.utils.namespace import AnnotationNameSpace

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
            "value": ["car", "truck", None],
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
        same = AnnotationSet(region_set.io.to_dataframe(), attrs=region_set.attrs)
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
        frame = pd.DataFrame({"time_start": [1.0, 2.0], "time_end": TIMES[:2]})
        with pytest.raises(ParameterError, match="cannot be compared"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_dimension_of_text_is_refused(self):
        """A coordinate is a number, a time or a duration; a label is none."""
        frame = pd.DataFrame({"distance_start": ["alpha"], "distance_end": ["omega"]})
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_dimension_of_text_is_not_quietly_dropped(self):
        """Text a time parser reads as NaT is refused, not deleted."""
        frame = pd.DataFrame({"time_start": ["alpha", "beta"], "time_end": ["c", "d"]})
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(frame, dims=DIMS)

    def test_numbers_written_as_text_are_numbers(self):
        """A range of numbers compares as numbers, not by its spelling."""
        frame = pd.DataFrame({"distance_start": ["9"], "distance_end": ["10"]})
        assert AnnotationSet(frame, dims=DIMS)[0].region.bounds["distance"] == (9, 10)

    def test_a_backwards_range_of_text_numbers_is_refused(self):
        """'10' before '9' is backwards as numbers, whatever text sorts as."""
        frame = pd.DataFrame({"distance_start": ["10"], "distance_end": ["9"]})
        with pytest.raises(ParameterError, match="ends before it starts"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_blank_beside_a_time_keeps_the_column_a_time(self):
        """A blank cell states nothing, so it does not make the column text."""
        frame = pd.DataFrame(
            {"time_start": ["2020-01-01T00:00:00", ""], "time_end": [TIMES[1], None]}
        )
        out = AnnotationSet(frame, dims=DIMS)
        assert out.io.to_dataframe()["time_start"].dtype == np.dtype("datetime64[ns]")
        assert out[1].region.bounds == {}

    @pytest.mark.parametrize(
        "column",
        [
            pytest.param(pd.Series([True, False]), id="boolean"),
            pytest.param(pd.Series([1 + 2j]), id="complex"),
            pytest.param(pd.Series([True], dtype=object), id="boolean-object"),
        ],
    )
    def test_a_dimension_which_states_no_coordinate(self, column):
        """A true is not a second past the epoch, and is refused as neither."""
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(pd.DataFrame({"distance": column}), dims=DIMS)

    def test_a_dimension_mixing_numbers_and_times(self):
        """A number already read as one is not re-read as an epoch."""
        cells = pd.Series([1, np.datetime64("2020-01-01")], dtype=object)
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(pd.DataFrame({"time": cells}), dims=DIMS)

    def test_a_dimension_no_coordinate_could_hold(self):
        """A year no coordinate holds is refused, never raised past the set.

        Whether the conversion overflows or quietly wraps is numpy's to
        decide, and its versions decide differently; what is pinned here is
        that neither reaches the caller as an implementation error.
        """
        frame = pd.DataFrame({"time": ["1000", "2020-01-01"]})
        with suppress(ParameterError):
            AnnotationSet(frame, dims=DIMS)

    @pytest.mark.parametrize(
        "cell",
        [
            pytest.param(np.datetime64("2020-01-01"), id="time"),
            pytest.param(np.timedelta64(1, "s"), id="duration"),
        ],
    )
    def test_a_dimension_of_objects_still_reads(self, cell):
        """A frame may hold coordinates in an object column; they are read."""
        frame = pd.DataFrame({"offset": pd.Series([cell], dtype=object)})
        held = AnnotationSet(frame, dims=("offset",)).io.to_dataframe()["offset"]
        assert held.dtype.kind in "Mm"

    def test_a_dimension_of_python_durations(self):
        """A frame may hold the stdlib's own duration; it is read as one."""
        cells = pd.Series([datetime.timedelta(seconds=1)], dtype=object)
        held = AnnotationSet(
            pd.DataFrame({"offset": cells}), dims=("offset",)
        ).io.to_dataframe()["offset"]
        assert held.dtype == np.dtype("timedelta64[ns]")

    def test_a_duration_dimension_is_a_coordinate(self):
        """A dimension may be an offset from something, which is a duration."""
        spans = np.array([1, 3], dtype="timedelta64[s]")
        frame = pd.DataFrame({"offset_start": spans[:1], "offset_end": spans[1:]})
        out = AnnotationSet(frame, dims=("offset",))
        assert out.io.to_dataframe()["offset_start"].dtype == "timedelta64[ns]"

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

    def test_the_set_column_is_a_label(self):
        """A row read with others says which set it came from."""
        out = AnnotationSet(pd.DataFrame({"set": ["picks"]}), dims=DIMS)
        assert out[0].set == "picks"
        assert "set" not in out[0].extra

    def test_no_set_column_is_no_label(self):
        """A set read on its own is not in a collection, so it names none."""
        out = AnnotationSet(pd.DataFrame({"group": ["a"]}), dims=DIMS)
        assert out[0].set == ""

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

    def test_datetime_unit_is_always_nanoseconds(self):
        """A set holds times at nanoseconds, so another unit names no column
        it could hold, and the error says so rather than blaming the data.
        """
        frame = pd.DataFrame({"when": np.array(["2020-01-01"], dtype="datetime64[us]")})
        with pytest.raises(ParameterError, match="every time at nanoseconds"):
            AnnotationSet(
                frame, dims=DIMS, columns={"when": {"dtype": "datetime64[us]"}}
            )

    def test_a_declared_nanosecond_column(self):
        """The unit a set does hold is the one which may be declared."""
        frame = pd.DataFrame({"when": np.array(["2020-01-01"], dtype="datetime64[us]")})
        out = AnnotationSet(
            frame, dims=DIMS, columns={"when": {"dtype": "datetime64[ns]"}}
        )
        assert out.io.to_dataframe()["when"].dtype == np.dtype("datetime64[ns]")

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
        """A membership and a number in one group are two variables."""
        frame = pd.DataFrame({"group": ["g", "g"], "value": [None, 3]})
        with pytest.raises(ParameterError, match="mixes"):
            AnnotationSet(frame, dims=DIMS)

    def test_unnamed_group_is_still_a_group(self):
        """Pandas drops a null grouping key; the unnamed group is checked anyway."""
        frame = pd.DataFrame({"group": [None, None], "value": [None, 1]})
        with pytest.raises(ParameterError, match="mixes"):
            AnnotationSet(frame, dims=DIMS)

    def test_int_and_float_are_one_kind(self):
        """Numbers are numbers; the model reads them alike."""
        frame = pd.DataFrame({"group": ["g", "g"], "value": [1.5, 3]})
        assert len(AnnotationSet(frame, dims=DIMS)) == 2

    def test_groups_are_independent(self):
        """Two groups may hold different kinds."""
        frame = pd.DataFrame({"group": ["a", "b"], "value": [None, "car"]})
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

    def test_no_value_states_membership(self):
        """An annotation with no value states membership, and the value stays unset."""
        assert AnnotationSet(pd.DataFrame({"group": ["a"]}), dims=DIMS)[0].value is None

    @pytest.mark.parametrize("value", [True, False, np.bool_(True)])
    def test_boolean_refused(self, value):
        """Membership has a spelling already, so true and false state nothing."""
        frame = pd.DataFrame({"group": ["a"], "value": [value]})
        with pytest.raises(ParameterError, match=r"group 'a'.*true and false are not"):
            AnnotationSet(frame, dims=DIMS)

    @pytest.mark.parametrize("blank", [None, np.nan])
    def test_a_blank_cell_in_a_valued_group_is_refused(self, blank):
        """A blank cell states membership, which a valued group cannot hold."""
        frame = pd.DataFrame({"group": ["amp", "amp"], "value": [3.0, blank]})
        with pytest.raises(ParameterError, match="blank cell states membership"):
            AnnotationSet(frame, dims=DIMS)

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

    def test_a_padded_tag_is_held_stripped(self):
        """Tags are held as they read back, so padding does not survive."""
        out = AnnotationSet(pd.DataFrame({"tags": [(" a", "b ")]}), dims=DIMS)
        assert out.io.to_dataframe()["tags"][0] == ("a", "b")

    def test_an_empty_tag_is_no_tag(self):
        """A tag holding nothing cannot be written down, so it is not held."""
        out = AnnotationSet(pd.DataFrame({"tags": [("", "b")]}), dims=DIMS)
        assert out[0].tags == ("b",)

    def test_a_tag_holding_a_comma(self):
        """A comma separates tags, so a tag holding one would become two."""
        frame = pd.DataFrame({"tags": [("a,b", "c")]})
        with pytest.raises(ParameterError, match="hold a comma"):
            AnnotationSet(frame, dims=DIMS)


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

    def test_whole_number_ids_are_their_own_text(self):
        """An id of 1 is named `1`, not the `1.0` a blank beside it makes."""
        frame = pd.DataFrame({"id": [1, 2], "parent": [None, 1]})
        out = AnnotationSet(frame, dims=DIMS)
        assert [x.id for x in out] == ["1", "2"]
        assert out[1].parent == "1"

    def test_a_whole_number_is_named_the_same_in_every_column(self):
        """A blank in one identity column does not rename what another means."""
        frame = pd.DataFrame({"id": [1.0, 2.5], "parent": [None, 1.0]})
        out = AnnotationSet(frame, dims=DIMS)
        assert [x.id for x in out] == ["1", "2.5"]
        assert out[1].parent == "1"

    def test_a_vertex_names_the_row_it_belongs_to(self):
        """The vertex frame's ids are spelled as the annotations' are."""
        frame = pd.DataFrame({"id": [1.0, None], "geometry": ["path", "region"]})
        vertices = pd.DataFrame({"id": [1, 1], "seq": [0, 1], "distance": [0.0, 1.0]})
        out = AnnotationSet(frame, dims=DIMS, vertices=vertices)
        assert out[0].geometry.vertices["distance"] == (0.0, 1.0)

    def test_an_id_beyond_where_a_float_counts_by_ones(self):
        """A float that large names no one integer, so it keeps its own text."""
        frame = pd.DataFrame({"id": [1e20, None]})
        assert AnnotationSet(frame, dims=DIMS)[0].id == "1e+20"

    def test_an_id_is_not_read_from_a_row(self):
        """A row of a frame holds one dtype; an id does not take the float
        bounds beside it.
        """
        frame = pd.DataFrame(
            {"id": [1], "value": [1], "distance_start": [0.0], "distance_end": [1.0]}
        )
        out = AnnotationSet(frame, dims=DIMS)[0]
        assert out.id == "1"
        assert out.value == 1 and isinstance(out.value, int)


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
        row = path_set.io.to_dataframe().iloc[0]
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
        out = path_set.io.to_vertices()
        assert list(out["id"]) == ["p1"] * 3
        assert list(out["seq"]) == [0, 1, 2]
        assert list(out["distance"]) == [1.0, 5.0, 9.0]
        assert out["time"].dtype == np.dtype("datetime64[ns]")
        assert list(out["time"]) == list(pd.Series(TIMES))

    def test_empty_vertices_frame(self, region_set):
        """A set with no vertex geometry has an empty vertices frame."""
        assert region_set.io.to_vertices().empty


class TestReadingOne:
    """An annotation is reached by its position."""

    def test_negative_position(self, region_set):
        """Counting from the end reads the last annotation."""
        assert region_set[-1].group == region_set[len(region_set) - 1].group

    @pytest.mark.parametrize("position", [slice(0, 2), "a", 1.0])
    def test_what_is_not_a_position(self, region_set, position):
        """Anything else says so, rather than building a nonsense row."""
        with pytest.raises(TypeError, match="read by its position"):
            region_set[position]


class TestVertexOrder:
    """A vertex states its place in the order as a number."""

    @staticmethod
    def _frame():
        """One path, whose vertices are stated below."""
        return pd.DataFrame({"id": ["p"], "geometry": ["path"]})

    def test_text_order_reads_as_numbers(self):
        """A seq written as text orders as the number it says."""
        vertices = pd.DataFrame(
            {
                "id": ["p"] * 4,
                "seq": ["0", "1", "2", "10"],
                "distance": [0.0, 1.0, 2.0, 10.0],
            }
        )
        out = AnnotationSet(self._frame(), dims=DIMS, vertices=vertices)
        assert out[0].geometry.vertices["distance"] == (0.0, 1.0, 2.0, 10.0)

    @pytest.mark.parametrize(
        "order",
        [
            pytest.param([True, False], id="boolean"),
            pytest.param([1 + 2j, 3 + 4j], id="complex"),
        ],
    )
    def test_an_order_which_does_not_count(self, order):
        """A true and an imaginary number place nothing in a sequence."""
        vertices = pd.DataFrame({"id": ["p"] * 2, "seq": order, "distance": [0.0, 1.0]})
        with pytest.raises(ParameterError, match="does not count"):
            AnnotationSet(self._frame(), dims=DIMS, vertices=vertices)

    def test_order_which_is_not_a_number_refused(self):
        """A shape ordered by a label has no order a reader can keep."""
        vertices = pd.DataFrame(
            {"id": ["p"] * 2, "seq": ["first", "second"], "distance": [0.0, 1.0]}
        )
        with pytest.raises(ParameterError, match="not a number"):
            AnnotationSet(self._frame(), dims=DIMS, vertices=vertices)

    def test_a_blank_vertex_cell_says_so(self):
        """An empty cell leaves a dimension unplaced, and is named as that."""
        vertices = pd.DataFrame({"id": ["p"] * 2, "seq": [0, 1], "distance": ["", 1.0]})
        with pytest.raises(ParameterError, match="leave a dimension empty"):
            AnnotationSet(self._frame(), dims=DIMS, vertices=vertices)


class TestFrames:
    """The frames the set hands back are copies of its own."""

    def test_to_dataframe_is_a_copy(self, region_set):
        """Mutating what a set handed out does not reach the set."""
        frame = region_set.io.to_dataframe()
        frame.loc[0, "group"] = "changed"
        assert region_set[0].group == "event"

    def test_to_vertices_is_a_copy(self, path_set):
        """The same holds for the vertices."""
        vertices = path_set.io.to_vertices()
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

    def test_column_order_is_not_what_a_set_says(self, region_set):
        """Two frames stating the same thing in a different order are one set."""
        frame = region_set.io.to_dataframe()
        shuffled = frame[list(reversed(frame.columns))]
        assert AnnotationSet(shuffled, attrs=region_set.attrs) == region_set

    def test_a_column_stating_nothing_has_one_dtype(self):
        """A column no row states arrives as whatever each reader inferred."""
        frame = pd.DataFrame({"group": ["a"], "note": [None]})
        held = AnnotationSet(frame, dims=DIMS).io.to_dataframe()
        assert held["note"].dtype == object

    def test_a_declared_dtype_is_not_overruled(self):
        """A column saying what it holds is not canonicalized out of it."""
        frame = pd.DataFrame({"group": ["a"], "note": [np.nan]})
        out = AnnotationSet(frame, dims=DIMS, columns={"note": {"dtype": "float64"}})
        assert out.io.to_dataframe()["note"].dtype == np.dtype("float64")

    def test_a_categorical_column_carries(self):
        """A category column is text, blanks and all."""
        frame = pd.DataFrame({"group": pd.Categorical(["a", ""])})
        out = AnnotationSet(frame, dims=DIMS)
        assert out[0].group == "a" and out[1].group == ""

    def test_a_categorical_column_is_held_as_text(self):
        """Only a frame has a category; every table reads the text back."""
        frame = pd.DataFrame({"group": pd.Categorical(["a", "b"])})
        held = AnnotationSet(frame, dims=DIMS).io.to_dataframe()["group"]
        assert held.dtype == object

    def test_round_trip(self, region_set):
        """A set rebuilt from its own frame holds the same annotations."""
        rebuilt = AnnotationSet(region_set.io.to_dataframe(), attrs=region_set.attrs)
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

    def test_sets_are_one_level_deep(self):
        """Sets loaded together are one collection, not a tree of them."""
        child = AnnotationSetAttrs(dims=("time",), sets={"deeper": {"dims": ("time",)}})
        with pytest.raises(ValidationError, match="not a tree"):
            AnnotationSetAttrs(dims=DIMS, sets={"picks": child})

    def test_a_child_dimension_nothing_holds(self):
        """A set states the dimensions the sets loaded with it are read in."""
        with pytest.raises(ValidationError, match="which the sets loaded with it"):
            AnnotationSetAttrs(dims=("time",), sets={"picks": {"dims": ("depth",)}})


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

    def test_basis_as_the_text_a_table_holds(self):
        """A cell holding the curve's JSON is the curve, as a table states it."""
        document = (
            '{"object_type": "Line", "start": {"distance": 0.0}, '
            '"end": {"distance": 1.0}}'
        )
        frame = pd.DataFrame({"id": ["x"], "geometry": ["path"], "basis": [document]})
        out = AnnotationSet(frame, dims=DIMS, vertices=self._vertices())
        assert isinstance(out[0].geometry.basis, Line)

    def test_a_curve_over_an_offset_dimension(self):
        """A duration endpoint reads back as the duration it was dumped from."""
        line = Line(
            start={"offset": np.timedelta64(1, "s")},
            end={"offset": np.timedelta64(3, "s")},
        )
        assert Line.model_validate(line.model_dump(mode="json")) == line

    def test_basis_which_is_not_a_document(self):
        """Text which is no document says that, not what pydantic made of it."""
        frame = pd.DataFrame(
            {"id": ["x"], "geometry": ["path"], "basis": ["not a curve"]}
        )
        with pytest.raises(ParameterError, match="not a JSON document"):
            AnnotationSet(frame, dims=DIMS, vertices=self._vertices())

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

    def test_datetime_bounds_read_back_as_times(self):
        """A time written as text is a time again, not the text."""
        region = Region(bounds={"time": (TIMES[0], TIMES[2])})
        assert Region(**region.model_dump(mode="json")) == region

    def test_datetime_vertices_read_back_as_times(self):
        """The same holds for a path's vertices and a line's endpoints."""
        line = Line(start={"time": TIMES[0]}, end={"time": TIMES[2]})
        path = Path(
            region=Region(bounds={}),
            vertices={"time": (TIMES[0], TIMES[1])},
            basis=line,
        )
        assert Path(**path.model_dump(mode="json")) == path

    def test_a_label_is_not_a_time(self):
        """Only the spelling DASCore writes a datetime with is read as one."""
        region = Region(bounds={"stage": ("2020-13-45", "before")})
        assert region.bounds["stage"] == ("2020-13-45", "before")

    @pytest.mark.parametrize("model", [Region, Line])
    def test_coordinates_which_are_not_a_mapping(self, model):
        """A coordinate map which is not a map is pydantic's to refuse."""
        with pytest.raises(ValidationError, match=r"valid dictionary|Extra inputs"):
            model(bounds="everywhere", start="here", end="there")

    @pytest.mark.parametrize(
        "spelling",
        ["2020-01-01", "2020-01-01T12", "2020-01-01T12:30", "2020-01-01T12:30:45"],
    )
    def test_every_resolution_reads_back_as_a_time(self, spelling):
        """Numpy writes only the fields a unit carries, and all of them read
        back: an hour- or minute-resolution pick is an ordinary one.
        """
        time = np.datetime64(spelling)
        region = Region(bounds={"time": (time, time)})
        assert Region(**region.model_dump(mode="json")) == region
        assert isinstance(region.bounds["time"][0], np.datetime64)

    @pytest.mark.parametrize("label", ["2020", "2020-01", "spring", "12:30"])
    def test_a_partial_date_is_not_a_time(self, label):
        """A label which is not a whole date stays the label it was; nothing
        distinguishes a bare year from a string spelled like one.
        """
        region = Region(bounds={"stage": (label, label)})
        assert region.bounds["stage"] == (label, label)

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


class TestAnnotationNamespaces:
    """A set hosts method namespaces, as a patch and a spool do."""

    def test_io_namespace(self, region_set):
        """The io namespace DASCore registers is reachable."""
        frame = region_set.io.to_dataframe()
        assert len(frame) == len(region_set)
        assert list(frame["group"]) == [x.group for x in region_set]

    def test_local_namespace_attaches(self, region_set):
        """A namespace defined without an entry point still attaches."""

        class _Local(AnnotationNameSpace):
            name = "some_local_namespace"

            def group_count(annotations) -> int:  # noqa: N805
                """Return how many distinct groups the set holds."""
                return annotations.io.to_dataframe()["group"].nunique()

        assert region_set.some_local_namespace.group_count() == 2

    def test_unknown_attr_raises(self, region_set):
        """A name no namespace claims raises DASCore's message."""
        msg = "AnnotationSet has no attribute 'nope'"
        with pytest.raises(AttributeError, match=msg):
            region_set.nope
