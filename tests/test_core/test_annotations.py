"""Tests for annotation sets and the models they hand out."""

from __future__ import annotations

import datetime
from contextlib import suppress

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from rich.text import Text

import dascore as dc
from dascore.core.annotations import (
    AnnotationBasis,
    AnnotationSet,
    AnnotationSetAttrs,
    Feature,
    Group,
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


def _bounds(annotations: AnnotationSet) -> list[dict]:
    """Return the bounds of every lone row, in row order."""
    return [dict(x.geometry.bounds) for x in annotations if not x.id]


def _first(annotations: AnnotationSet) -> Feature:
    """Return the first feature a set iterates."""
    return next(iter(annotations))


def _moveout(**kwargs) -> Moveout:
    """A moveout over distance and time."""
    stated = {
        "apex_distance": 50.0,
        "apex_time": TIMES[0],
        "velocity": 3000.0,
        "distance_min": 0.0,
        "distance_max": 100.0,
    }
    return Moveout(**{**stated, **kwargs})


@pytest.fixture(scope="module")
def picks():
    """Time-only picks, which span the whole fiber."""
    frame = pd.DataFrame(
        {
            "time": TIMES[:2],
            "phase": ["P", "S"],
            "confidence": [0.9, 0.4],
        }
    )
    return AnnotationSet(frame, dims=DIMS)


@pytest.fixture(scope="module")
def boxes():
    """Plain regions, one row each."""
    frame = pd.DataFrame(
        {
            "note": ["car", "truck", None],
            "distance_min": [10.0, 30.0, 0.0],
            "distance_max": [80.0, 90.0, 100.0],
        }
    )
    return AnnotationSet(frame, dims=DIMS)


@pytest.fixture(scope="module")
def tracks():
    """A path, a group of picks, and a lone box in one set."""
    frame = pd.DataFrame(
        {
            "feature_id": ["t1", "t1", "t1", "e1", "e1", None],
            "time": [*TIMES, TIMES[0], TIMES[1], pd.NaT],
            "distance": [1.0, 5.0, 9.0, np.nan, np.nan, np.nan],
            "distance_min": [np.nan] * 5 + [10.0],
            "distance_max": [np.nan] * 5 + [80.0],
            "velocity": [3.0, 4.0, 5.0, np.nan, np.nan, np.nan],
        }
    )
    features = pd.DataFrame(
        {"id": ["t1"], "geometry": ["path"], "vehicle_type": ["train"]}
    )
    return AnnotationSet(frame, features=features, dims=DIMS)


def _polygon(rings: dict, feature: str = "g1", **kwargs) -> AnnotationSet:
    """Build a set holding one polygon from ``{(part, ring): [(t, d), ...]}``."""
    rows = [
        {"feature_id": feature, "part": p, "ring": r, "time": t, "distance": d}
        for (p, r), points in rings.items()
        for t, d in points
    ]
    features = pd.DataFrame({"id": [feature], "geometry": ["polygon"]})
    return AnnotationSet(pd.DataFrame(rows), features=features, dims=DIMS, **kwargs)


TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


class TestConstruction:
    """A set is built from a frame and the dimensions it is stated in."""

    def test_picks_are_one_frame(self, picks):
        """A picker's frame is the whole set: each row is its own feature."""
        assert len(picks) == 2
        assert [x.extra["phase"] for x in picks] == ["P", "S"]
        assert picks.features.empty

    def test_time_only_picks_broadcast(self, picks):
        """A row stating only time spans the whole fiber."""
        assert _bounds(picks)[0] == {"time": (TIMES[0], TIMES[0])}

    def test_dims_required(self):
        """A set which names no dimension states nothing about where."""
        with pytest.raises(ParameterError, match="states its dimensions"):
            AnnotationSet(pd.DataFrame({"time": [1.0]}))

    def test_dims_from_attrs(self):
        """Dimensions may come from an attrs object instead of the keyword."""
        attrs = AnnotationSetAttrs(dims=DIMS)
        assert AnnotationSet(pd.DataFrame({"time": [1.0]}), attrs=attrs).dims == DIMS

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
        assert list(out.annotations.columns) == ["feature_id"]

    def test_records(self):
        """Anything a frame can be built from works."""
        out = AnnotationSet([{"time": 1.0}, {"time": 2.0}], dims=DIMS)
        assert len(out) == 2

    def test_unreadable_data(self):
        """Something which is not tabular says so."""
        with pytest.raises(ParameterError, match="as a dataframe"):
            AnnotationSet(object(), dims=DIMS)

    def test_duplicate_columns_refused(self):
        """Pandas allows a repeated name; every reader here expects one column."""
        frame = pd.DataFrame([["g", "g", 1.0]], columns=["note", "note", "time"])
        with pytest.raises(ParameterError, match="more than once"):
            AnnotationSet(frame, dims=DIMS)

    def test_equality(self, tracks):
        """Two sets built from the same tables are equal."""
        same = AnnotationSet(
            tracks.annotations, features=tracks.features, attrs=tracks.attrs
        )
        assert same == tracks
        assert tracks != AnnotationSet(None, dims=DIMS)

    def test_equality_sees_features(self, tracks):
        """A features column is part of what a set says."""
        features = tracks.features.assign(vehicle_type=["car", None])
        other = AnnotationSet(tracks.annotations, features=features, attrs=tracks.attrs)
        assert other != tracks

    def test_equality_sees_bases(self):
        """Bases are part of what a set says, used or not."""
        frame = pd.DataFrame({"time": [1.0]})
        plain = AnnotationSet(frame, dims=DIMS)
        based = AnnotationSet(frame, dims=DIMS, bases={"m": _moveout()})
        assert plain != based

    def test_not_equal_to_other_types(self, boxes):
        """A set is not equal to something which is not one."""
        assert boxes != "not a set"


class TestRepr:
    """What a set says about itself."""

    def test_counts(self, tracks):
        """Annotations, features by kind, and bases are counted."""
        out = repr(tracks)
        assert "annotations: 6" in out
        assert "features: 3 (groups: 2, paths: 1, polygons: 0)" in out
        assert "bases: 0" in out

    def test_columns_of_each_table(self, tracks):
        """Both tables name their columns."""
        out = repr(tracks)
        assert "annotation columns:" in out and "velocity" in out
        assert "feature columns:" in out and "vehicle_type" in out

    def test_dimension_extent(self, boxes):
        """A dimension the set states is shown by what it spans."""
        out = repr(boxes)
        assert "min: 0.000 max: 100.000" in out
        assert "unstated" in out  # time is spelled by no column

    def test_extent_spans_values_and_ranges(self, tracks):
        """Both spellings of a dimension count toward its extent."""
        out = repr(tracks)
        assert "min: 1.000 max: 80.000" in out
        assert "(value, range)" in out

    def test_empty_set(self):
        """An empty set still has a repr, and claims no contents."""
        out = repr(AnnotationSet(None, dims=DIMS))
        assert "AnnotationSet" in out
        assert "Contents" not in out

    def test_attributes(self):
        """What the set says of itself is shown, and defaults are not."""
        out = repr(AnnotationSet(None, dims=DIMS, acquisition_key="XT.TUN1.00.DAS"))
        assert "acquisition_key: XT.TUN1.00.DAS" in out
        assert "data_id" not in out

    def test_rich(self, boxes):
        """Annotation sets have a rich representation."""
        assert isinstance(boxes.__rich__(), Text)

    def test_dimension_values_not_styled_as_keys(self, boxes):
        """A dimension's extent is a value, not the label in front of it."""
        text = boxes.__rich__()
        assert text.style == ""
        start = text.plain.index("100.000")
        assert not [x for x in text.spans if x.start <= start < x.end]

    def test_feature_repr(self, tracks):
        """One feature names its class and what it states."""
        assert repr(tracks["t1"]).startswith("Feature(")


class TestDimensionSpelling:
    """A dimension is a value, a range, or unconstrained."""

    def test_range_columns(self, boxes):
        """A min/max pair is a half-open range."""
        assert _bounds(boxes)[0]["distance"] == (10.0, 80.0)

    def test_bare_column_is_a_point(self):
        """A bare dimension column states a value, a zero-width range."""
        region = _first(AnnotationSet(pd.DataFrame({"distance": [5.0]}), dims=DIMS))
        assert region.geometry.bounds["distance"] == (5.0, 5.0)
        assert region.geometry.is_point("distance")

    def test_values_and_ranges_in_one_frame(self):
        """One dimension may be a value on one row and a range on the next."""
        frame = pd.DataFrame(
            {
                "time": [1.0, np.nan],
                "time_min": [np.nan, 2.0],
                "time_max": [np.nan, 3.0],
            }
        )
        out = _bounds(AnnotationSet(frame, dims=DIMS))
        assert out == [{"time": (1.0, 1.0)}, {"time": (2.0, 3.0)}]

    def test_a_row_stating_both_refused(self):
        """One row states a dimension one way."""
        frame = pd.DataFrame({"time": [1.0], "time_min": [1.0], "time_max": [2.0]})
        with pytest.raises(ParameterError, match="as a value and as a range"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_row_stating_nothing_refused(self):
        """Every annotation is somewhere along at least one dimension."""
        frame = pd.DataFrame({"time": [1.0, np.nan], "note": ["a", "b"]})
        with pytest.raises(ParameterError, match=r"Row\(s\) 1 state no dimension"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_frame_of_no_dimension_refused(self):
        """A column of extras locates nothing."""
        with pytest.raises(ParameterError, match="state no dimension"):
            AnnotationSet(pd.DataFrame({"score": [0.9]}), dims=DIMS)

    def test_one_kind_per_dimension(self):
        """A value column of numbers beside a range of times is two kinds."""
        frame = pd.DataFrame(
            {
                "time": [1.0, np.nan],
                "time_min": [pd.NaT, TIMES[0]],
                "time_max": [pd.NaT, TIMES[1]],
            }
        )
        with pytest.raises(ParameterError, match="one kind of coordinate"):
            AnnotationSet(frame, dims=DIMS)

    def test_region_names_its_dims(self, boxes):
        """A region says which dimensions it constrains."""
        assert _first(boxes).geometry.dims == ("distance",)

    def test_timedelta_endpoints_keep_their_type(self):
        """A bound on a lag dimension is a duration, not the integer behind it."""
        lags = np.array([1, 5], dtype="timedelta64[s]")
        frame = pd.DataFrame({"time_min": lags[:1], "time_max": lags[1:]})
        start, end = _bounds(AnnotationSet(frame, dims=DIMS))[0]["time"]
        assert isinstance(start, np.timedelta64)
        assert isinstance(end, np.timedelta64)

    def test_unconstrained_dim_absent(self, boxes):
        """A dimension no column names does not appear in the bounds."""
        assert "time" not in _bounds(boxes)[0]

    def test_half_a_range_refused(self):
        """A start with no end does not bound anything."""
        with pytest.raises(ParameterError, match="half a range"):
            AnnotationSet(pd.DataFrame({"time_min": [1]}), dims=DIMS)

    def test_reversed_range_refused(self):
        """An impossible range is structural, so the set does not load."""
        frame = pd.DataFrame({"distance_min": [9.0], "distance_max": [1.0]})
        with pytest.raises(ParameterError, match="ends before it starts"):
            AnnotationSet(frame, dims=DIMS)

    def test_reversed_range_names_its_row(self):
        """The refused row is named, so a long set says which one."""
        frame = pd.DataFrame({"distance_min": [0.0, 9.0], "distance_max": [1.0, 1.0]})
        with pytest.raises(ParameterError, match="Row 1"):
            AnnotationSet(frame, dims=DIMS)

    @pytest.mark.parametrize("side", ["distance_min", "distance_max"])
    def test_half_a_range_cell_refused(self, side):
        """A row states both ends or neither; one end bounds nothing."""
        frame = pd.DataFrame(
            {"time": [1.0], "distance_min": [np.nan], "distance_max": [np.nan]}
        )
        frame[side] = [1.0]
        with pytest.raises(ParameterError, match="half a distance range"):
            AnnotationSet(frame, dims=DIMS)

    def test_incomparable_range_refused(self):
        """A range whose ends cannot be compared says so, not TypeError."""
        when = np.datetime64("2020-01-01", "ns")
        frame = pd.DataFrame({"distance_min": [1.0], "distance_max": [when]})
        with pytest.raises(ParameterError, match="cannot be compared"):
            AnnotationSet(frame, dims=DIMS)

    @pytest.mark.parametrize("spelling", ["distance", "distance_min"])
    def test_text_in_a_dimension_refused(self, spelling):
        """A dimension is a coordinate, so a word is no place on it."""
        frame = pd.DataFrame({spelling: ["alpha"]})
        if spelling != "distance":
            frame["distance_max"] = ["omega"]
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(frame, dims=DIMS)

    def test_numeric_text_in_a_dimension_is_read_as_numbers(self):
        """Read as a stored table reads it, so the two cannot disagree."""
        frame = pd.DataFrame(
            {
                "time": [np.nan, 1.0],
                "distance_min": ["1.5", None],
                "distance_max": ["2", None],
            }
        )
        out = _bounds(AnnotationSet(frame, dims=DIMS))
        assert out[0]["distance"] == (1.5, 2.0)
        assert "distance" not in out[1]

    @pytest.mark.parametrize(
        "text", ["2020-01-01 10:00:00", "2020-01-01T10:00:00Z", "2020-01-01T10:00:00"]
    )
    def test_time_text_in_any_spelling_is_a_time(self, text):
        """What `to_csv` writes and `read_csv` hands back is a time here too."""
        out = AnnotationSet(pd.DataFrame({"time": [text]}), dims=DIMS)
        assert _bounds(out)[0]["time"][0] == np.datetime64("2020-01-01T10:00:00")

    def test_a_malformed_date_is_refused_not_a_value_error(self):
        """Shaped like a date without being one is text, and said to be."""
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(pd.DataFrame({"time": ["2020-13-45"]}), dims=DIMS)

    @pytest.mark.parametrize(
        "values",
        [[True], pd.array([True, None], dtype="boolean"), [np.True_, None]],
    )
    def test_a_boolean_dimension_refused(self, values):
        """A truth value is no place on an axis, numpy's and a nullable one too."""
        frame = pd.DataFrame({"distance": pd.Series(values, dtype=object)})
        with pytest.raises(ParameterError, match="numbers or times"):
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
        held = AnnotationSet(frame, dims=("offset",)).annotations["offset"]
        assert held.dtype.kind in "Mm"

    def test_a_dimension_of_python_durations(self):
        """A frame may hold the stdlib's own duration; it is read as one."""
        cells = pd.Series([datetime.timedelta(seconds=1)], dtype=object)
        held = AnnotationSet(
            pd.DataFrame({"offset": cells}), dims=("offset",)
        ).annotations["offset"]
        assert held.dtype == np.dtype("timedelta64[ns]")

    def test_a_duration_dimension_is_a_coordinate(self):
        """A dimension may be an offset from something, which is a duration."""
        spans = np.array([1, 3], dtype="timedelta64[s]")
        frame = pd.DataFrame({"offset_min": spans[:1], "offset_max": spans[1:]})
        out = AnnotationSet(frame, dims=("offset",))
        assert out.annotations["offset_min"].dtype == "timedelta64[ns]"

    def test_a_dimension_mixing_numbers_and_times(self):
        """A number already read as one is not re-read as an epoch."""
        cells = pd.Series([1, np.datetime64("2020-01-01")], dtype=object)
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(pd.DataFrame({"time": cells}), dims=DIMS)

    def test_a_dimension_no_coordinate_could_hold(self):
        """A year no coordinate holds is refused, never raised past the set."""
        frame = pd.DataFrame({"time": ["1000", "2020-01-01"]})
        with suppress(ParameterError):
            AnnotationSet(frame, dims=DIMS)

    def test_a_complex_dimension_refused(self):
        """`to_numeric` hands a complex column back; it is no coordinate."""
        frame = pd.DataFrame({"distance": [1 + 2j]})
        with pytest.raises(ParameterError, match="neither numbers, times"):
            AnnotationSet(frame, dims=DIMS)

    def test_datetime_endpoints_keep_their_type(self):
        """A time bound is a time, not the integer behind it."""
        frame = pd.DataFrame({"time_min": TIMES[:1], "time_max": TIMES[2:]})
        start, end = _bounds(AnnotationSet(frame, dims=DIMS))[0]["time"]
        assert isinstance(start, np.datetime64)
        assert isinstance(end, np.datetime64)


class TestColumns:
    """Unknown columns carry; near-misses do not."""

    def test_arbitrary_columns_round_trip(self):
        """Any column and dtype rides along untouched."""
        frame = pd.DataFrame(
            {
                "time": [1.0, 2.0],
                "phase": ["P", "S"],
                "confidence": [0.5, 0.75],
                "count": [1, 2],
                "checked": [True, False],
                "group": ["a", "b"],
                "value": [3, None],
                "tags": ["x, y", None],
                "parent": ["e1", None],
            }
        )
        out = AnnotationSet(frame, dims=DIMS).annotations
        pd.testing.assert_frame_equal(out[frame.columns[:5]], frame[frame.columns[:5]])
        assert _first(AnnotationSet(frame, dims=DIMS)).extra["group"] == "a"

    def test_unstated_extra_dropped(self):
        """A blank extra states nothing, so the feature does not carry it."""
        out = AnnotationSet(pd.DataFrame({"time": [1.0], "score": [np.nan]}), dims=DIMS)
        assert "score" not in _first(out).extra

    @pytest.mark.parametrize("column", ["geometry", "basis"])
    def test_a_feature_column_on_annotations_refused(self, column):
        """Geometry and basis are what a feature states, not a row."""
        frame = pd.DataFrame({"time": [1.0], column: ["path"]})
        with pytest.raises(ParameterError, match="features table"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_retired_range_spelling_says_what_to_write(self):
        """A set written before the rename is told its columns' new names."""
        frame = pd.DataFrame({"distance": [1.0], "time_start": [0], "time_end": [1]})
        with pytest.raises(ParameterError, match="now spells _min/_max"):
            AnnotationSet(frame, dims=DIMS)

    def test_undeclared_range_pair_refused(self):
        """A range naming no declared dimension is a forgotten dimension."""
        frame = pd.DataFrame({"time": [1.0], "depth_min": [1], "depth_max": [2]})
        with pytest.raises(ParameterError, match="name no declared dimension"):
            AnnotationSet(frame, dims=DIMS)

    def test_lone_range_column_is_an_extra(self):
        """One half of a range names no dimension, so it is just a column."""
        out = AnnotationSet(pd.DataFrame({"time": [1.0], "depth_min": [1]}), dims=DIMS)
        assert _first(out).extra["depth_min"] == 1

    def test_the_set_column_is_a_label(self):
        """A row read with others says which set it came from."""
        frame = pd.DataFrame({"time": [1.0], "set": ["p"]})
        attrs = {"dims": DIMS, "sets": {"p": {"dims": ("time",)}}}
        out = AnnotationSet(frame, attrs=attrs)
        assert _first(out).set == "p"
        assert "set" not in _first(out).extra

    def test_a_label_without_sets_refused(self):
        """A label names a set loaded together, so a plain set holds none."""
        frame = pd.DataFrame({"time": [2.0], "id": ["p"], "set": ["x"]})
        with pytest.raises(ParameterError, match="states the set 'x'"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_feature_label_without_sets_refused(self):
        """The features table's labels are held to the same rule."""
        frame = pd.DataFrame({"time": [2.0], "feature_id": ["f"]})
        features = pd.DataFrame({"id": ["f"], "set": ["x"]})
        with pytest.raises(ParameterError, match="states the set 'x'"):
            AnnotationSet(frame, features=features, dims=DIMS)

    def test_blank_labels_allowed(self):
        """A blank label is no label, in a plain set too."""
        frame = pd.DataFrame({"time": [2.0], "set": [None]})
        assert len(AnnotationSet(frame, dims=DIMS)) == 1

    def test_private_column_is_not_an_extra(self):
        """An underscore says the column is the author's, not the set's."""
        frame = pd.DataFrame({"time": [1.0], "_crew": ["north crew"]})
        out = AnnotationSet(frame, dims=DIMS)
        assert "_crew" not in _first(out).extra
        assert "_crew" not in out.annotations.columns

    def test_a_private_column_states_no_dimension(self):
        """Underscoring a range column makes it nothing, not a bound."""
        frame = pd.DataFrame(
            {"time": [1.0], "_distance_min": [1.0], "_distance_max": [2.0]}
        )
        assert "distance" not in _bounds(AnnotationSet(frame, dims=DIMS))[0]

    def test_a_private_dimension(self):
        """A dimension is a column, and no set reads a private one."""
        frame = pd.DataFrame({"_distance": [1.0]})
        with pytest.raises(ValidationError, match="begin with an underscore"):
            AnnotationSet(frame, dims=("_distance", "time"))

    def test_rows_no_column_can_hold(self):
        """Rows a table cannot write are refused where they can be named."""
        frame = pd.DataFrame({"_crew": ["north crew", "south crew"]})
        with pytest.raises(ParameterError, match="none is theirs"):
            AnnotationSet(frame, dims=DIMS)

    def test_rows_which_state_nothing_at_all(self):
        """The same, for a frame which never had a column to lose."""
        with pytest.raises(ParameterError, match="no column to hold them"):
            AnnotationSet(pd.DataFrame(index=range(2)), dims=DIMS)

    def test_declared_column_documents_only(self):
        """Documenting a column does not gate any other one."""
        out = AnnotationSet(
            pd.DataFrame({"time": [1.0], "score": [0.9], "other": [1]}),
            dims=DIMS,
            annotation_columns={"score": {"description": "Confidence"}},
        )
        assert out.attrs.annotation_columns["score"].description == "Confidence"
        assert "other" in _first(out).extra

    def test_stated_dtype_checked(self):
        """A column which says what it holds must hold it."""
        with pytest.raises(ParameterError, match="states dtype"):
            AnnotationSet(
                pd.DataFrame({"time": [1.0], "score": ["high"]}),
                dims=DIMS,
                annotation_columns={"score": {"dtype": "float64"}},
            )

    def test_feature_dtype_checked(self):
        """The features table checks its own declarations."""
        with pytest.raises(ParameterError, match="states dtype"):
            AnnotationSet(
                pd.DataFrame({"time": [1.0], "feature_id": ["e"]}),
                features=pd.DataFrame({"id": ["e"], "magnitude": ["big"]}),
                dims=DIMS,
                feature_columns={"magnitude": {"dtype": "float64"}},
            )

    def test_an_implied_feature_must_fit_a_declared_dtype(self):
        """An implied feature leaves a column blank, which int64 cannot hold."""
        frame = pd.DataFrame({"feature_id": ["a", "b"], "time": [1.0, 2.0]})
        features = pd.DataFrame({"id": ["a"], "rank": [1]})
        with pytest.raises(ParameterError, match="which feature_id implies"):
            AnnotationSet(
                frame,
                features=features,
                dims=DIMS,
                feature_columns={"rank": {"dtype": "int64"}},
            )
        nullable = features.astype({"rank": "Int64"})
        out = AnnotationSet(
            frame,
            features=nullable,
            dims=DIMS,
            feature_columns={"rank": {"dtype": "Int64"}},
        )
        assert out.features["rank"].dtype.name == "Int64"

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [("str", ["a", "b"]), ("category", ["a", "b"]), ("Int64", [1, 2])],
    )
    def test_extension_dtypes_declarable(self, dtype, values):
        """Pandas gives plain text a `str` dtype, which numpy cannot name."""
        frame = pd.DataFrame({"note": values}).astype(dtype)
        frame["time"] = [1.0, 2.0]
        columns = {"note": {"dtype": dtype}}
        out = AnnotationSet(frame, dims=DIMS, annotation_columns=columns)
        assert len(out) == 2

    @pytest.mark.parametrize("declared", ["str", "string", "object"])
    def test_a_text_dtype_matches_any_text_spelling(self, declared):
        """A declaration of any text spelling is a declaration of text."""
        columns = {"note": {"dtype": declared}}
        for frame in (
            pd.DataFrame({"time": [1.0], "note": pd.Series(["a"], dtype=object)}),
            pd.DataFrame({"time": [1.0], "note": pd.Series(["a"], dtype="string")}),
        ):
            out = AnnotationSet(frame, dims=DIMS, annotation_columns=columns)
            assert len(out) == 1
        with pytest.raises(ParameterError, match="states dtype"):
            AnnotationSet(
                pd.DataFrame({"time": [1.0], "note": [1.0]}),
                dims=DIMS,
                annotation_columns=columns,
            )

    def test_an_object_column_is_text_only_if_its_cells_are(self):
        """`object` may hold anything, so a text declaration reads its cells."""
        frame = pd.DataFrame(
            {"time": [1.0, 2.0], "note": pd.Series(["a", {"x": 1}], dtype=object)}
        )
        with pytest.raises(ParameterError, match="states dtype"):
            AnnotationSet(
                frame, dims=DIMS, annotation_columns={"note": {"dtype": "string"}}
            )
        columns = {"note": {"dtype": "object"}}
        assert len(AnnotationSet(frame, dims=DIMS, annotation_columns=columns)) == 2

    def test_an_extra_holding_timedeltas_is_held_at_nanoseconds(self):
        """Every time is held at DASCore's resolution, an extra's too."""
        lags = np.array([1, 5], dtype="timedelta64[s]")
        frame = pd.DataFrame({"time": [1.0, 2.0], "lag": lags})
        held = AnnotationSet(frame, dims=DIMS).annotations["lag"]
        assert held.dtype == np.dtype("timedelta64[ns]")
        assert held.iloc[0] == np.timedelta64(1, "s")

    def test_a_categorical_column_builds(self):
        """A categorical extra is carried like any other, blank cells and all."""
        frame = pd.DataFrame(
            {"time": [1.0, 2.0], "note": pd.Series(["a", ""], dtype="category")}
        )
        extras = [x.extra for x in AnnotationSet(frame, dims=DIMS)]
        assert extras[0]["note"] == "a"
        assert "note" not in extras[1]

    def test_a_column_named_by_something_other_than_a_string(self):
        """A table names a column by a string, so a set does too."""
        frame = pd.DataFrame({"time": [1.0], 1: ["x"]})
        with pytest.raises(ParameterError, match="other than a string"):
            AnnotationSet(frame, dims=DIMS)

    def test_datetime_unit_is_always_nanoseconds(self):
        """Another unit names no column a set could hold, and says so."""
        when = np.array(["2020-01-01"], dtype="datetime64[us]")
        with pytest.raises(ParameterError, match="every time at nanoseconds"):
            AnnotationSet(
                pd.DataFrame({"time": [1.0], "when": when}),
                dims=DIMS,
                annotation_columns={"when": {"dtype": "datetime64[us]"}},
            )

    def test_unreadable_dtype_refused(self):
        """A dtype naming nothing says so, rather than raising numpy's error."""
        with pytest.raises(ParameterError, match="declares the dtype"):
            AnnotationSet(
                pd.DataFrame({"time": [1.0], "score": [1.0]}),
                dims=DIMS,
                annotation_columns={"score": {"dtype": "not-a-dtype"}},
            )

    def test_stated_dtype_absent_column(self):
        """Documenting a column the frame lacks is not an error."""
        columns = {"score": {"dtype": "float64"}}
        out = AnnotationSet(None, dims=DIMS, annotation_columns=columns)
        assert "score" in out.attrs.annotation_columns


class TestIdentity:
    """Annotation ids are optional; feature ids are required."""

    def test_annotation_ids_are_optional(self, boxes):
        """A set without ids is fine, and nothing invents one."""
        assert "id" not in boxes.annotations.columns

    def test_duplicate_annotation_ids_refused(self):
        """A stated annotation id names one row."""
        frame = pd.DataFrame({"id": ["a", "a"], "time": [1.0, 2.0]})
        with pytest.raises(ParameterError, match=r"annotation id.*more than one row"):
            AnnotationSet(frame, dims=DIMS)

    def test_blank_annotation_ids_may_repeat(self):
        """Unstated identity is not a clash."""
        frame = pd.DataFrame({"id": [None, None], "time": [1.0, 2.0]})
        assert len(AnnotationSet(frame, dims=DIMS)) == 2

    def test_whole_number_ids_are_their_own_text(self):
        """An id of 1 is named `1`, not the `1.0` a blank beside it makes."""
        frame = pd.DataFrame(
            {"id": [1.0, 2.0], "feature_id": [7.0, np.nan], "time": [1.0, 2.0]}
        )
        out = AnnotationSet(frame, dims=DIMS)
        assert list(out.annotations["id"]) == ["1", "2"]
        assert list(out.features["id"]) == ["7"]

    def test_an_id_beyond_where_a_float_counts_by_ones(self):
        """A float that large names no one integer, so it keeps its own text."""
        frame = pd.DataFrame({"id": [1e20, None], "time": [1.0, 2.0]})
        assert AnnotationSet(frame, dims=DIMS).annotations["id"][0] == "1e+20"

    def test_feature_ids_required(self):
        """Annotations reference a feature by id, so each has one."""
        features = pd.DataFrame({"id": ["a", None], "geometry": [None, None]})
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        with pytest.raises(ParameterError, match="state no id"):
            AnnotationSet(frame, features=features, dims=DIMS)

    def test_features_without_an_id_column_refused(self):
        """A features table names what it holds."""
        features = pd.DataFrame({"name": ["train"]})
        with pytest.raises(ParameterError, match="no id column"):
            AnnotationSet(None, features=features, dims=DIMS)

    def test_duplicate_feature_ids_refused(self):
        """A feature id names one feature."""
        features = pd.DataFrame({"id": ["a", "a"]})
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        with pytest.raises(ParameterError, match=r"feature id.*more than one row"):
            AnnotationSet(frame, features=features, dims=DIMS)


class TestFeatures:
    """Every annotation belongs to one feature."""

    def test_feature_id_creates_a_group(self):
        """A feature_id naming no features row creates one, stated explicitly."""
        frame = pd.DataFrame({"feature_id": ["e1", "e1", "e2"], "time": [1.0, 2, 3]})
        out = AnnotationSet(frame, dims=DIMS)
        assert list(out.features["id"]) == ["e1", "e2"]
        assert out.features["geometry"].isna().all()
        assert len(out) == 2
        assert [len(x.geometry.regions) for x in out] == [2, 1]

    def test_explicit_features_frame(self):
        """Feature columns ride on the features table and reach the view."""
        frame = pd.DataFrame({"feature_id": ["e1", "e1"], "time": [1.0, 2.0]})
        features = pd.DataFrame({"id": ["e1"], "name": ["quake"], "magnitude": [2.5]})
        out = AnnotationSet(frame, features=features, dims=DIMS)["e1"]
        assert out.name == "quake"
        assert out.extra == {"magnitude": 2.5}
        assert out.kind == "group" and isinstance(out.geometry, Group)

    def test_iteration_order(self):
        """Features in table order, then each lone row in row order."""
        frame = pd.DataFrame(
            {"feature_id": [None, "b", "a", None], "time": [1.0, 2.0, 3.0, 4.0]}
        )
        features = pd.DataFrame({"id": ["a"]})
        out = AnnotationSet(frame, features=features, dims=DIMS)
        assert [x.id for x in out] == ["a", "b", "", ""]
        assert [x.geometry.bounds["time"][0] for x in out if not x.id] == [1.0, 4.0]

    def test_a_lone_row_is_a_region(self, boxes):
        """A row naming no feature is its own group, located by its region."""
        feature = _first(boxes)
        assert (feature.id, feature.kind) == ("", "group")
        assert isinstance(feature.geometry, Region)
        assert feature.extra == {"note": "car"}

    def test_lookup_by_id(self, tracks):
        """A feature is reached by its id."""
        assert tracks["t1"].kind == "path"
        with pytest.raises(KeyError, match="No feature"):
            tracks["nope"]

    def test_a_lone_row_has_no_id(self, boxes):
        """The empty id names no feature."""
        with pytest.raises(KeyError):
            boxes[""]

    @pytest.mark.parametrize("spelling", ["region", "group", ""])
    def test_group_spellings(self, spelling):
        """A group may be spelled blank, group or region."""
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        features = pd.DataFrame({"id": ["a"], "geometry": [spelling]})
        out = AnnotationSet(frame, features=features, dims=DIMS)
        assert out.features["geometry"].isna().all()

    def test_unknown_geometry_refused(self):
        """A geometry this format has no meaning for is refused."""
        features = pd.DataFrame({"id": ["a"], "geometry": ["blob"]})
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        with pytest.raises(ParameterError, match="blob is not one of"):
            AnnotationSet(frame, features=features, dims=DIMS)

    @pytest.mark.parametrize("column", ["time", "distance_min"])
    def test_coordinates_on_features_refused(self, column):
        """Features hold no coordinates; their members do."""
        features = pd.DataFrame({"id": ["a"], column: [1.0]})
        if column == "distance_min":
            features["distance_max"] = [2.0]
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        with pytest.raises(ParameterError, match="coordinate column"):
            AnnotationSet(frame, features=features, dims=DIMS)

    @pytest.mark.parametrize("column", ["feature_id", "seq"])
    def test_row_columns_on_features_refused(self, column):
        """What names and orders a row within a feature is not a feature's."""
        features = pd.DataFrame({"id": ["a"], column: ["x"]})
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        with pytest.raises(ParameterError, match="which an annotation states"):
            AnnotationSet(frame, features=features, dims=DIMS)

    def test_a_feature_locating_nothing_refused(self):
        """No members and no basis is nowhere."""
        features = pd.DataFrame({"id": ["a"]})
        with pytest.raises(ParameterError, match="nothing locates it"):
            AnnotationSet(None, features=features, dims=DIMS)

    def test_mixed_dimension_subsets(self):
        """Features in one set may be drawn in different dimensions."""
        frame = pd.DataFrame(
            {
                "feature_id": ["a", "a", "b", "b"],
                "time": [0.0, 1.0, np.nan, np.nan],
                "distance": [np.nan, np.nan, 0.0, 5.0],
            }
        )
        features = pd.DataFrame({"id": ["a", "b"], "geometry": ["path", "path"]})
        out = AnnotationSet(frame, features=features, dims=DIMS)
        assert out.geometry("a").dims == ("time",)
        assert out.geometry("b").dims == ("distance",)

    def test_members_may_mix_spellings(self):
        """A group's members may be values, ranges and broadcasts together."""
        frame = pd.DataFrame(
            {
                "feature_id": ["e"] * 3,
                "time": [1.0, np.nan, 2.0],
                "distance_min": [np.nan, 0.0, np.nan],
                "distance_max": [np.nan, 5.0, np.nan],
            }
        )
        regions = AnnotationSet(frame, dims=DIMS)["e"].geometry.regions
        bounds = [x.bounds for x in regions]
        assert bounds == [
            {"time": (1.0, 1.0)},
            {"distance": (0.0, 5.0)},
            {"time": (2.0, 2.0)},
        ]


class TestPaths:
    """A path is ordered by seq within part."""

    def test_vertices_follow_seq(self):
        """Members are read in seq order, not row order."""
        frame = pd.DataFrame(
            {"feature_id": ["p"] * 3, "seq": [2, 0, 1], "distance": [9.0, 1.0, 5.0]}
        )
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"]})
        path = AnnotationSet(frame, features=features, dims=DIMS).geometry("p")
        assert isinstance(path, Path)
        assert path.vertices[0]["distance"] == (1.0, 5.0, 9.0)

    def test_blank_seq_is_row_order(self, tracks):
        """A path stating no seq takes the order its rows are in."""
        assert list(tracks.annotations["seq"][:3]) == [0, 1, 2]
        assert tracks.geometry("t1").vertices[0]["distance"] == (1.0, 5.0, 9.0)

    def test_datetime_vertices_keep_their_type(self, tracks):
        """A vertex on the time dimension is a time."""
        assert isinstance(tracks.geometry("t1").vertices[0]["time"][0], np.datetime64)

    def test_multipart(self):
        """Disconnected parts are numbered by part."""
        frame = pd.DataFrame(
            {
                "feature_id": ["p"] * 4,
                "part": [1, 1, 0, 0],
                "distance": [5.0, 6.0, 0.0, 1.0],
            }
        )
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"]})
        path = AnnotationSet(frame, features=features, dims=DIMS).geometry("p")
        assert [x["distance"] for x in path.vertices] == [(0.0, 1.0), (5.0, 6.0)]

    def test_large_ordinals_stay_exact(self):
        """Seq values past a float's integer precision are neither rounded
        together nor taken for a repeat.
        """
        big = [2**53 + 2, 2**53 + 1]
        out = self._path(distance=[2.0, 1.0], seq=big)
        assert list(out.annotations["seq"]) == big
        assert out.geometry("p").vertices[0]["distance"] == (1.0, 2.0)

    def test_text_ordinals_sort_as_numbers(self):
        """A seq written as text orders as the number it says."""
        frame = pd.DataFrame(
            {"feature_id": ["p"] * 3, "seq": ["10", "2", "1"], "distance": [3.0, 2, 1]}
        )
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"]})
        out = AnnotationSet(frame, features=features, dims=DIMS)
        assert out.geometry("p").vertices[0]["distance"] == (1.0, 2.0, 3.0)
        assert out.annotations["seq"].dtype == "Int64"

    @staticmethod
    def _path(**columns):
        """Build a one-path set from annotation columns."""
        count = len(next(iter(columns.values())))
        frame = pd.DataFrame({"feature_id": ["p"] * count, **columns})
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"]})
        return AnnotationSet(frame, features=features, dims=DIMS)

    def test_a_range_member_refused(self):
        """A path is drawn through values."""
        with pytest.raises(ParameterError, match="drawn through values"):
            self._path(
                time=[1.0, 2.0], distance_min=[0.0, 1.0], distance_max=[1.0, 2.0]
            )

    def test_members_in_different_dims_refused(self):
        """Every vertex states the same dimensions."""
        with pytest.raises(ParameterError, match="different dimensions"):
            self._path(time=[1.0, 2.0], distance=[1.0, np.nan])

    def test_one_member_refused(self):
        """Each part needs two vertices."""
        frame = pd.DataFrame({"feature_id": ["p"], "distance": [1.0]})
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"]})
        with pytest.raises(ParameterError, match="at least 2"):
            AnnotationSet(frame, features=features, dims=DIMS)

    def test_a_part_of_one_refused(self):
        """The count is per part, not per path."""
        with pytest.raises(ParameterError, match="part 1"):
            self._path(distance=[1.0, 2.0, 3.0], part=[0, 0, 1])

    def test_a_ring_refused(self):
        """Only a polygon has rings."""
        with pytest.raises(ParameterError, match="only a polygon has rings"):
            self._path(distance=[1.0, 2.0], ring=[1, 1])

    def test_a_repeated_seq_refused(self):
        """Two members in one place do not say which comes first."""
        with pytest.raises(ParameterError, match="repeats a seq"):
            self._path(distance=[1.0, 2.0], seq=[0, 0])

    def test_seq_on_some_members_refused(self):
        """Seq is stated on every member of a part, or on none."""
        with pytest.raises(ParameterError, match="some members and not others"):
            self._path(distance=[1.0, 2.0], seq=[0, None])

    @pytest.mark.parametrize("seq", [[-1, 0], [0.5, 1]])
    def test_an_ordinal_is_a_whole_number(self, seq):
        """An order is a non-negative whole number."""
        with pytest.raises(ParameterError, match="non-negative whole number"):
            self._path(distance=[1.0, 2.0], seq=seq)

    @pytest.mark.parametrize(
        "seq",
        [["first", "second"], [True, False], [np.True_, np.False_], [1 + 2j, 3j]],
    )
    def test_non_numeric_seq_refused(self, seq):
        """A truth value is not counted as 1 or 0, nor a word as anything."""
        with pytest.raises(ParameterError, match="is not numeric"):
            self._path(distance=[1.0, 2.0], seq=pd.Series(seq, dtype=object))

    def test_a_blank_seq_cell_is_no_seq(self):
        """The empty string is how a table spells an unset cell, here too."""
        out = self._path(distance=[2.0, 1.0], seq=["", ""])
        assert out.geometry("p").vertices[0]["distance"] == (2.0, 1.0)


class TestPolygons:
    """A polygon is ordered by seq within (part, ring)."""

    def test_a_polygon(self):
        """Rings of three or more members close an area."""
        polygon = _polygon({(0, 0): TRIANGLE}).geometry("g1")
        assert isinstance(polygon, Polygon)
        assert polygon.vertices[0][0]["time"] == (0.0, 1.0, 0.0)

    def test_a_hole(self):
        """Ring 0 is the outer boundary; a later ring is a hole."""
        inner = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
        polygon = _polygon({(0, 0): SQUARE, (0, 1): inner}).geometry("g1")
        assert len(polygon.vertices) == 1
        assert [len(x["time"]) for x in polygon.vertices[0]] == [4, 3]

    def test_multipolygon(self):
        """Parts are disconnected polygons, each with its own rings."""
        far = [(20.0, 20.0), (21.0, 20.0), (20.0, 21.0)]
        polygon = _polygon({(0, 0): SQUARE, (1, 0): far}).geometry("g1")
        assert len(polygon.vertices) == 2

    def test_too_few_distinct_vertices_refused(self):
        """A ring of repeated points bounds no area."""
        with pytest.raises(ParameterError, match="3 distinct"):
            _polygon({(0, 0): [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0)]})

    def test_a_closing_repeat_refused(self):
        """Closure is implied, so the first vertex is not written again."""
        with pytest.raises(ParameterError, match="closure is implied"):
            _polygon({(0, 0): [*TRIANGLE, TRIANGLE[0]]})

    def test_a_part_without_an_outer_ring_refused(self):
        """A hole is a hole in something."""
        with pytest.raises(ParameterError, match="no ring 0"):
            _polygon({(0, 0): SQUARE, (1, 1): TRIANGLE})

    def test_one_dimension_refused(self):
        """An area needs two dimensions."""
        frame = pd.DataFrame({"feature_id": ["g"] * 3, "distance": [0.0, 1.0, 2.0]})
        features = pd.DataFrame({"id": ["g"], "geometry": ["polygon"]})
        with pytest.raises(ParameterError, match="an area needs two"):
            AnnotationSet(frame, features=features, dims=DIMS)


class TestOrderColumns:
    """seq, part and ring belong to ordered features only."""

    def test_absent_without_an_ordered_feature(self, picks):
        """A picker's frame never sees them."""
        assert set(picks.annotations.columns) == {
            "time",
            "phase",
            "confidence",
            "feature_id",
        }

    def test_blank_columns_are_dropped(self):
        """A column of blanks states nothing, so a picks set holds none."""
        frame = pd.DataFrame({"time": [1.0], "seq": [None], "part": [np.nan]})
        assert list(AnnotationSet(frame, dims=DIMS).annotations.columns) == [
            "time",
            "feature_id",
        ]

    def test_present_once_a_path_exists(self, tracks):
        """Blank on everything but the path, and zero-filled there."""
        frame = tracks.annotations
        assert list(frame["part"]) == [0, 0, 0, pd.NA, pd.NA, pd.NA]
        assert list(frame["ring"][:3]) == [0, 0, 0]

    def test_present_for_a_path_with_no_members(self):
        """A basis-only path is still a path, so the columns are there."""
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"], "basis": ["m"]})
        frame = pd.DataFrame({"time": [1.0]})
        out = AnnotationSet(
            frame, features=features, bases={"m": _moveout()}, dims=DIMS
        ).annotations
        for name in ("seq", "part", "ring"):
            assert out[name].dtype.name == "Int64"
            assert out[name].isna().all()

    @pytest.mark.parametrize("name", ["seq", "part", "ring"])
    def test_on_a_group_member_refused(self, name):
        """Ordering a member of an unordered feature says nothing."""
        frame = pd.DataFrame({"time": [1.0], name: [0]})
        with pytest.raises(ParameterError, match="orders the members of a path"):
            AnnotationSet(frame, dims=DIMS)


class TestBases:
    """Keyed curves a path may be drawn from."""

    @staticmethod
    def _path(basis, key="m", members=True, **kwargs):
        """Build a set with one path naming ``key``."""
        frame = None
        if members:
            frame = pd.DataFrame(
                {
                    "feature_id": ["p", "p"],
                    "distance": [0.0, 100.0],
                    "time": TIMES[:2],
                }
            )
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"], "basis": [key]})
        return AnnotationSet(
            frame, features=features, bases={"m": basis}, dims=DIMS, **kwargs
        )

    def test_a_model(self):
        """A basis may be the model itself."""
        curve = _moveout()
        assert self._path(curve)["p"].basis == curve

    def test_a_document(self):
        """A basis may be its plain dict, normalized to the model."""
        curve = _moveout()
        out = self._path(curve.model_dump(mode="json"))
        assert out.bases["m"] == curve

    def test_an_unknown_key_refused(self):
        """A key names one of the bases."""
        with pytest.raises(ParameterError, match="not among the bases"):
            self._path(_moveout(), key="other")

    def test_an_unused_key_is_allowed(self):
        """A basis nothing names is still a basis."""
        frame = pd.DataFrame({"time": [1.0]})
        out = AnnotationSet(frame, dims=DIMS, bases={"spare": _moveout()})
        assert list(out.bases) == ["spare"]

    @pytest.mark.parametrize("kind", [None, "polygon"])
    def test_only_a_path_names_one(self, kind):
        """A group or polygon is not drawn from a curve."""
        frame = pd.DataFrame(
            {"feature_id": ["g"] * 3, "time": [0.0, 1, 0], "distance": [0.0, 0, 1]}
        )
        features = pd.DataFrame({"id": ["g"], "geometry": [kind], "basis": ["m"]})
        with pytest.raises(ParameterError, match="only a path"):
            AnnotationSet(frame, features=features, bases={"m": _moveout()}, dims=DIMS)

    def test_dims_must_match_the_members(self):
        """A curve in distance alone does not draw a path in distance and time."""
        line = Line(start={"distance": 0.0}, end={"distance": 1.0})
        with pytest.raises(ParameterError, match="its basis in"):
            self._path(line)

    def test_basis_over_more_dims_than_members_refused(self):
        """A curve over distance and time does not draw a path in distance."""
        frame = pd.DataFrame({"feature_id": ["p", "p"], "distance": [0.0, 100.0]})
        features = pd.DataFrame({"id": ["p"], "geometry": ["path"], "basis": ["m"]})
        with pytest.raises(ParameterError, match="its basis in"):
            AnnotationSet(frame, features=features, bases={"m": _moveout()}, dims=DIMS)

    def test_basis_only_path_samples(self):
        """A path with a curve and no members is drawn on demand."""
        out = self._path(_moveout(), members=False)
        assert out.annotations.empty
        path = out.geometry("p", count=5)
        assert len(path.vertices[0]["distance"]) == 5
        assert path.basis == _moveout()
        assert len(out["p"].geometry.vertices[0]["time"]) == 64

    def test_members_are_authoritative(self):
        """With both, the members are the shape and the curve rides along."""
        path = self._path(_moveout()).geometry("p", count=5)
        assert path.vertices[0]["distance"] == (0.0, 100.0)

    def test_bases_are_immutable(self):
        """The mapping handed out cannot change the set."""
        out = self._path(_moveout())
        with pytest.raises(TypeError):
            out.bases["m"] = None

    def test_unreadable_basis(self):
        """An entry naming no curve says so."""
        with pytest.raises(ParameterError, match="Could not read the basis 'm'"):
            AnnotationSet(None, dims=DIMS, bases={"m": {"object_type": "Nope"}})

    def test_basis_dims_must_be_declared(self):
        """A curve in an unrelated frame draws nothing here."""
        line = Line(start={"depth": 0.0}, end={"depth": 1.0})
        with pytest.raises(ParameterError, match="does not declare"):
            AnnotationSet(None, dims=DIMS, bases={"m": line})

    def test_keys_colliding_as_text_refused(self):
        """A key of 1 and a key of "1" name one basis, so both are refused."""
        line = Line(start={"distance": 0.0}, end={"distance": 1.0})
        other = Line(start={"distance": 0.0}, end={"distance": 2.0})
        with pytest.raises(ParameterError, match="keyed '1'"):
            AnnotationSet(None, dims=DIMS, bases={1: line, "1": other})

    def test_a_blank_key_refused(self):
        """A key is what a feature names a basis by, so it says something."""
        with pytest.raises(ParameterError, match="nonblank key"):
            AnnotationSet(None, dims=DIMS, bases={"": _moveout()})

    def test_bases_are_a_mapping(self):
        """A list of curves has no keys to name them by."""
        with pytest.raises(ParameterError, match="mapping of key to curve"):
            AnnotationSet(None, dims=DIMS, bases=[_moveout()])


class TestFrames:
    """What a set hands back is a copy of its own."""

    def test_annotations_are_a_copy(self, boxes):
        """Mutating what a set handed out does not reach the set."""
        frame = boxes.annotations
        frame.loc[0, "note"] = "changed"
        assert boxes.annotations.loc[0, "note"] == "car"

    def test_features_are_a_copy(self, tracks):
        """The same holds for the features."""
        frame = tracks.features
        frame.loc[0, "vehicle_type"] = "changed"
        assert tracks.features.loc[0, "vehicle_type"] == "train"

    def test_a_zero_dimensional_array_cell(self):
        """A 0-d array is one value, not a sequence."""
        frame = pd.DataFrame({"time": [1.0], "m": [np.array(5)]})
        assert AnnotationSet(frame, dims=DIMS).annotations["m"][0] == 5

    def test_nested_annotation_cells_are_frozen(self):
        """A nested cell handed out cannot change the set."""
        frame = pd.DataFrame({"time": [1.0], "meta": [{"values": [1]}]})
        out = AnnotationSet(frame, dims=DIMS)
        with pytest.raises(AttributeError):
            out.annotations.loc[0, "meta"]["values"].append(2)
        with pytest.raises(TypeError):
            out.annotations.loc[0, "meta"]["values"] = None
        assert out.annotations.loc[0, "meta"] == {"values": (1,)}

    def test_nested_feature_cells_are_frozen(self):
        """The same holds for the features table."""
        frame = pd.DataFrame({"feature_id": ["a"], "time": [1.0]})
        features = pd.DataFrame({"id": ["a"], "meta": [{"values": [1]}]})
        out = AnnotationSet(frame, features=features, dims=DIMS)
        with pytest.raises(AttributeError):
            out.features.loc[0, "meta"]["values"].append(2)
        assert out.features.loc[0, "meta"] == {"values": (1,)}

    def test_extras_are_frozen(self):
        """A mutable cell cannot be edited through the feature holding it."""
        frame = pd.DataFrame({"time": [1.0], "n": [[1]]})
        out = _first(AnnotationSet(frame, dims=DIMS))
        assert out.extra["n"] == (1,)
        with pytest.raises(AttributeError):
            out.extra["n"].append(3)

    def test_nested_extras_are_frozen(self):
        """Freezing reaches inside a mapping cell too."""
        frame = pd.DataFrame({"time": [1.0], "n": [{"a": [1]}]})
        with pytest.raises(TypeError):
            _first(AnnotationSet(frame, dims=DIMS)).extra["n"]["a"] = 2

    def test_column_order_is_not_what_a_set_says(self, tracks):
        """Two tables stating the same thing in a different order are one set."""
        frame = tracks.annotations
        shuffled = frame[list(reversed(frame.columns))]
        rebuilt = AnnotationSet(shuffled, features=tracks.features, attrs=tracks.attrs)
        assert rebuilt == tracks

    def test_a_column_stating_nothing_has_one_dtype(self):
        """A column no row states arrives as whatever each reader inferred."""
        frame = pd.DataFrame({"time": [1.0], "note": [None]})
        assert AnnotationSet(frame, dims=DIMS).annotations["note"].dtype == object

    def test_a_declared_dtype_is_not_overruled(self):
        """A column saying what it holds is not canonicalized out of it."""
        frame = pd.DataFrame({"time": [1.0], "note": [np.nan]})
        columns = {"note": {"dtype": "float64"}}
        out = AnnotationSet(frame, dims=DIMS, annotation_columns=columns)
        assert out.annotations["note"].dtype == np.dtype("float64")

    def test_a_categorical_column_is_held_as_text(self):
        """Only a frame has a category; every table reads the text back."""
        frame = pd.DataFrame({"time": [1.0, 2.0], "kind": pd.Categorical(["a", "b"])})
        assert AnnotationSet(frame, dims=DIMS).annotations["kind"].dtype == object


class TestProvenance:
    """A feature names its own provenance, else its set's, else the collection's."""

    @pytest.fixture
    def collected(self):
        """Two sets loaded together, rows and features stating some of it."""
        frame = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0],
                "set": ["a", "a", "b", "b"],
                "feature_id": [None, None, None, "f"],
                "acquisition_key": ["N.ROW.00.das", None, None, None],
                "data_id": ["row", None, None, None],
            }
        )
        features = pd.DataFrame({"id": ["f"], "set": ["b"], "data_id": ["feat"]})
        sets = {
            "a": {"dims": ("time",), "acquisition_key": "N.SETA.00.das"},
            "b": {"dims": ("time",), "data_id": "set-b"},
        }
        return AnnotationSet(
            frame,
            features=features,
            dims=DIMS,
            acquisition_key="N.COLL.00.das",
            data_id="coll",
            attrs={"dims": DIMS, "sets": sets},
        )

    def test_rows(self, collected):
        """A lone row falls back row, then its set, then the collection."""
        lone = [x for x in collected if not x.id]
        keys = [x.acquisition_key for x in lone]
        assert keys == ["N.ROW.00.das", "N.SETA.00.das", "N.COLL.00.das"]
        assert [x.data_id for x in lone] == ["row", "coll", "set-b"]

    def test_features(self, collected):
        """A feature's own row states its provenance first."""
        feature = collected["f"]
        assert feature.data_id == "feat"
        assert feature.acquisition_key == "N.COLL.00.das"
        assert feature.set == "b"

    def test_an_implied_feature_takes_its_rows_set(self):
        """A feature a collection's rows imply is labeled as their set."""
        frame = pd.DataFrame({"time": [1.0], "set": ["a"], "feature_id": ["e"]})
        attrs = {"dims": DIMS, "sets": {"a": {"dims": ("time",)}}}
        out = AnnotationSet(frame, attrs=attrs)
        assert list(out.features["set"]) == ["a"]

    @pytest.mark.parametrize("field", ["acquisition_key", "data_id"])
    def test_a_member_states_no_provenance(self, field):
        """A member inherits from its feature, so it may not state its own."""
        frame = pd.DataFrame(
            {"time": [1.0, 2.0], "feature_id": [None, "f"], field: [None, "N.A.L.ACQ"]}
        )
        with pytest.raises(ParameterError, match=r"Row 1.*belongs on the feature"):
            AnnotationSet(frame, dims=DIMS)

    def test_a_member_inherits_its_feature(self, collected):
        """A member's provenance is its feature's, which falls back to its set."""
        members = collected.select(feature_id="f")
        assert len(members.select(data_id="feat").annotations) == 1

    def test_a_row_key_is_validated(self):
        """A row's key is checked like the set's, when the set is built."""
        frame = pd.DataFrame({"time": [1.0], "acquisition_key": ["nope"]})
        with pytest.raises(ParameterError, match="Invalid acquisition_key"):
            AnnotationSet(frame, dims=DIMS)

    def test_the_key_column_is_not_an_extra(self):
        """The column is modelled, so it does not also ride along as an extra."""
        frame = pd.DataFrame({"time": [1.0], "acquisition_key": ["N.A.L.ACQ"]})
        assert "acquisition_key" not in _first(AnnotationSet(frame, dims=DIMS)).extra

    def test_a_feature_label_naming_no_set(self):
        """The features table's labels reach back to a set too."""
        frame = pd.DataFrame({"time": [1.0], "set": ["a"], "feature_id": ["f"]})
        features = pd.DataFrame({"id": ["f"], "set": ["zz"]})
        attrs = {"dims": DIMS, "sets": {"a": {"dims": ("time",)}}}
        with pytest.raises(ParameterError, match="name no set stated here"):
            AnnotationSet(frame, features=features, attrs=attrs)


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
        """`distance_min` would be both a point and the start of `distance`."""
        with pytest.raises(ValidationError, match="spelled like the range column"):
            AnnotationSetAttrs(dims=("distance", "distance_min"))

    @pytest.mark.parametrize("dim", ["feature_id", "seq", "geometry"])
    def test_dim_may_not_shadow_a_reserved_column(self, dim):
        """A dimension may not take a column either table reserves."""
        with pytest.raises(ValidationError, match="reserved column"):
            AnnotationSetAttrs(dims=(dim, "time"))

    def test_creation_info_default(self):
        """A set carries provenance even when nothing was said."""
        assert AnnotationSetAttrs(dims=DIMS).creation_info.author == ""

    def test_acquisition_key_validated(self):
        """The key is spelled as PatchAttrs spells it, and checked alike."""
        with pytest.raises(ValidationError, match="Invalid acquisition_key"):
            AnnotationSetAttrs(dims=DIMS, acquisition_key="nope")

    def test_data_id_replaces_history(self):
        """The data a set was made on is named by its id, not its lineage."""
        attrs = AnnotationSetAttrs(dims=DIMS, data_id="abc123")
        assert attrs.data_id == "abc123"
        with pytest.raises(ValidationError, match="Extra inputs"):
            AnnotationSetAttrs(dims=DIMS, history=("decimate",))

    @pytest.mark.parametrize(
        ("field", "column", "dtype"),
        [
            ("annotation_columns", "seq", "int64"),
            ("feature_columns", "geometry", "category"),
            ("annotation_columns", "time_min", "float64"),
        ],
    )
    def test_a_fixed_column_takes_no_dtype(self, field, column, dtype):
        """A reserved or dimension column's dtype is the set's, not declared."""
        with pytest.raises(ValidationError, match="dtype is fixed"):
            AnnotationSetAttrs(dims=DIMS, **{field: {column: {"dtype": dtype}}})

    def test_a_fixed_column_may_be_documented(self):
        """A description of a reserved column is still welcome."""
        columns = {"geometry": {"description": "what the members draw"}}
        attrs = AnnotationSetAttrs(dims=DIMS, feature_columns=columns)
        assert attrs.feature_columns["geometry"].description

    def test_columns_per_table(self):
        """Each table documents its own columns."""
        attrs = AnnotationSetAttrs(
            dims=DIMS,
            annotation_columns={"phase": {"description": "P or S"}},
            feature_columns={"magnitude": {"units": "dimensionless"}},
        )
        assert attrs.annotation_columns["phase"].description == "P or S"
        assert "magnitude" in attrs.feature_columns

    def test_creation_info_identifies_the_producer(self):
        """A picker names itself the way the inventory names any process."""
        attrs = AnnotationSetAttrs(
            dims=DIMS, creation_info={"author": "phasenet", "version": "2.1"}
        )
        assert attrs.creation_info.author == "phasenet"

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


class TestBasisModels:
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

    def test_a_time_beyond_nanoseconds(self):
        """A time nanoseconds cannot hold is a validation error naming it."""
        with pytest.raises(ValidationError, match="3000"):
            Line(
                start={"time": np.datetime64("3000-01-01")},
                end={"time": np.datetime64("3001-01-01")},
            )

    def test_line_of_no_length(self):
        """A line beginning where it ends is a point."""
        with pytest.raises(ValidationError, match="no length"):
            Line(start={"distance": 1.0}, end={"distance": 1.0})

    def test_moveout_apex_is_the_earliest_arrival(self):
        """The apex anchors the curve, and nothing arrives before it."""
        drawn = _moveout().vertices(11)
        assert drawn["time"].min() == TIMES[0]
        assert drawn["time"][5] == TIMES[0]

    def test_moveout_on_the_cable_is_straight(self):
        """A source with no standoff runs both ways at its velocity."""
        seconds = (_moveout().vertices(3)["time"] - TIMES[0]) / np.timedelta64(1, "s")
        assert np.allclose(seconds, [50 / 3000, 0.0, 50 / 3000])

    def test_standoff_flattens_the_apex(self):
        """A source off the cable arrives sooner away from the apex."""
        straight = _moveout().vertices(3)["time"]
        curved = _moveout(standoff=40.0).vertices(3)["time"]
        assert curved[0] < straight[0]
        assert curved[1] == straight[1] == TIMES[0]

    def test_moveout_is_pinned_to_its_dims(self):
        """A moveout is physics, so it relates fiber distance to arrival time."""
        assert _moveout().dims == ("distance", "time")

    @pytest.mark.parametrize(
        "kwargs",
        [{"velocity": 0.0}, {"standoff": -1.0}, {"distance_min": 100.0}],
    )
    def test_moveout_refusals(self, kwargs):
        """No speed, a source behind the cable, or no span draws nothing."""
        with pytest.raises(ValidationError):
            _moveout(**kwargs)

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
        with pytest.raises(NotImplementedError):
            AnnotationBasis().extent()
        with pytest.raises(NotImplementedError):
            AnnotationBasis().dims

    def test_a_curve_over_an_offset_dimension(self):
        """A duration endpoint reads back as the duration it was dumped from."""
        line = Line(
            start={"offset": np.timedelta64(1, "s")},
            end={"offset": np.timedelta64(3, "s")},
        )
        assert Line.model_validate(line.model_dump(mode="json")) == line


class TestGeometryModels:
    """A geometry built straight from a document checks itself."""

    def test_no_dimension_refused(self):
        """Vertices naming no dimension place the geometry nowhere."""
        with pytest.raises(ValidationError, match="states no dimension"):
            Path(vertices=({},))

    def test_ragged_vertices_refused(self):
        """Every dimension states every point, or they pair up wrongly."""
        with pytest.raises(ValidationError, match="differ in length"):
            Path(vertices=({"time": (1, 2, 3), "distance": (1,)},))

    @pytest.mark.parametrize(
        ("build", "least"),
        [
            (lambda v: Path(vertices=(v,)), 2),
            (lambda v: Polygon(vertices=((v,),)), 3),
        ],
    )
    def test_too_few_vertices_refused(self, build, least):
        """A shape needs enough points to be one, however it was built."""
        with pytest.raises(ValidationError, match="at least"):
            build({"distance": tuple(range(least - 1))})

    def test_parts_share_their_dims(self):
        """Parts of one path are drawn in one frame."""
        with pytest.raises(ValidationError, match="different dimensions"):
            Path(vertices=({"time": (0, 1)}, {"distance": (0, 1)}))

    def test_polygon_parts_and_rings(self):
        """A part states a ring, and every ring is drawn in one frame."""
        ring = {"distance": (0.0, 1.0, 2.0), "time": (0.0, 1.0, 0.0)}
        assert Polygon(vertices=((ring,),)).dims == ("distance", "time")
        with pytest.raises(ValidationError, match="states no ring"):
            Polygon(vertices=((),))
        other = {"distance": (0.0, 1.0, 2.0), "depth": (0.0, 1.0, 0.0)}
        with pytest.raises(ValidationError, match="different dimensions"):
            Polygon(vertices=((ring, other),))

    def test_a_group_holds_a_region(self):
        """An empty group locates nothing."""
        with pytest.raises(ValidationError):
            Group(regions=())

    def test_geometry_kinds_are_distinct(self):
        """A polygon is not a path which happens to close."""
        ring = {"distance": (0.0, 1.0, 2.0), "time": (0.0, 1.0, 0.0)}
        assert not isinstance(Polygon(vertices=((ring,),)), Path)


class TestSerialization:
    """The models are documents, like every other DASCore model."""

    @pytest.mark.parametrize(
        "basis",
        [Line(start={"distance": 0.0}, end={"distance": 1.0}), _moveout()],
    )
    def test_basis_names_its_class(self, basis):
        """A document says which curve it holds, so the union can dispatch."""
        assert basis.model_dump(mode="json")["object_type"] == type(basis).__name__

    def test_path_round_trip(self):
        """A path rebuilds its basis as the class which wrote it."""
        path = Path(vertices=({"distance": (0.0, 1.0)},), basis=_moveout())
        rebuilt = Path(**path.model_dump(mode="json"))
        assert rebuilt == path
        assert isinstance(rebuilt.basis, Moveout)

    def test_feature_round_trip(self, tracks):
        """A feature view is a document too."""
        feature = tracks["t1"]
        assert Feature(**feature.model_dump(mode="json")) == feature

    def test_region_round_trip(self):
        """A region survives a document."""
        region = Region(bounds={"distance": (0.0, 1.0)})
        assert Region(**region.model_dump(mode="json")) == region

    def test_datetime_bounds_write_a_document(self):
        """A time bound has no json type, so it is written as DASCore spells it."""
        region = Region(bounds={"time": (TIMES[0], TIMES[2])})
        written = region.model_dump(mode="json")["bounds"]["time"]
        assert written == [str(TIMES[0]), str(TIMES[2])]
        assert Region(**region.model_dump(mode="json")) == region

    def test_python_dump_keeps_the_time(self):
        """A python dump is what equality compares, so it keeps the value."""
        region = Region(bounds={"time": (TIMES[0], TIMES[2])})
        kept = region.model_dump()["bounds"]["time"][0]
        assert isinstance(kept, np.datetime64) and kept == TIMES[0]

    def test_numpy_numbers_write_plainly(self):
        """A numpy number is written as the number it is."""
        region = Region(bounds={"distance": (np.float64(1.5), np.float64(2.5))})
        assert region.model_dump(mode="json")["bounds"]["distance"] == [1.5, 2.5]

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
        """Numpy writes only the fields a unit carries, and all of them read back."""
        time = np.datetime64(spelling)
        region = Region(bounds={"time": (time, time)})
        assert Region(**region.model_dump(mode="json")) == region

    @pytest.mark.parametrize("label", ["2020", "2020-01", "spring", "12:30"])
    def test_a_partial_date_is_not_a_time(self, label):
        """A label which is not a whole date stays the label it was."""
        region = Region(bounds={"stage": (label, label)})
        assert region.bounds["stage"] == (label, label)


class TestTopLevel:
    """The set is reachable from the top-level namespace."""

    def test_dc_annotation_set(self):
        """`dc.AnnotationSet` is the in-memory door."""
        assert dc.AnnotationSet is AnnotationSet


class TestAnnotationNamespaces:
    """A set hosts method namespaces, as a patch and a spool do."""

    def test_io_namespace(self, boxes):
        """The io namespace DASCore registers is reachable."""
        assert "distance_min" in boxes.io.to_csv()

    def test_removed_io_functions(self, boxes):
        """The frames are properties now, not io functions."""
        with pytest.raises(AttributeError):
            boxes.io.to_dataframe
        with pytest.raises(AttributeError):
            boxes.io.to_vertices

    def test_local_namespace_attaches(self, boxes):
        """A namespace defined without an entry point still attaches."""

        class _Local(AnnotationNameSpace):
            name = "some_local_namespace"

            def note_count(annotations) -> int:  # noqa: N805
                """Return how many distinct notes the set holds."""
                return annotations.annotations["note"].nunique()

        assert boxes.some_local_namespace.note_count() == 2

    def test_unknown_attr_raises(self, boxes):
        """A name no namespace claims raises DASCore's message."""
        msg = "AnnotationSet has no attribute 'nope'"
        with pytest.raises(AttributeError, match=msg):
            boxes.nope
