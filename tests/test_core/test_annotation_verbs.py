"""Tests for the verbs which build, read, select and edit annotation sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.annotations import AnnotationSet, Line, Moveout
from dascore.exceptions import InvalidAnnotationError, ParameterError

DIMS = ("time", "distance")

TIMES = np.array(
    ["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"],
    dtype="datetime64[ns]",
)


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


def _round_trip(annotations: AnnotationSet, path) -> AnnotationSet:
    """Save a set as a directory and read it back."""
    return dc.annotations(annotations.io.save(path))


@pytest.fixture(scope="module")
def picks():
    """Lone picks with ids, time only."""
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "time": [1.0, 2.0, 3.0],
            "phase": ["P", "S", "P"],
            "confidence": [0.9, 0.95, 0.3],
        }
    )
    return AnnotationSet(frame, dims=DIMS, data_id="patch-1")


@pytest.fixture(scope="module")
def tracks():
    """A path, a group of picks, and a lone box in one set."""
    frame = pd.DataFrame(
        {
            "id": ["v0", "v1", "v2", "p1", "p2", "box"],
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


@pytest.fixture(scope="module")
def based():
    """A path drawn through three members from a basis, and a basis-only path."""
    frame = pd.DataFrame(
        {
            "id": ["m0", "m1", "m2"],
            "feature_id": ["near", "near", "near"],
            "time": TIMES,
            "distance": [0.0, 25.0, 50.0],
            "q": [1, 2, 3],
        }
    )
    features = pd.DataFrame(
        {
            "id": ["near", "far"],
            "geometry": ["path", "path"],
            "basis": ["curve", "curve"],
        }
    )
    return AnnotationSet(
        frame, features=features, bases={"curve": _moveout()}, dims=DIMS
    )


@pytest.fixture(scope="module")
def square():
    """A polygon: a square with a triangular hole."""
    outer = {"time": [0.0, 10.0, 10.0, 0.0], "distance": [0.0, 0.0, 10.0, 10.0]}
    hole = {"time": [2.0, 4.0, 2.0], "distance": [2.0, 2.0, 4.0]}
    empty = AnnotationSet(dims=DIMS)
    return empty.add_polygon("sq", rings=[outer, hole], note="noise")


def _collection(root, names, data_ids=None) -> AnnotationSet:
    """Save one small set per name under root and load them together."""
    for number, name in enumerate(names):
        frame = pd.DataFrame(
            {
                "id": [f"{name}0", f"{name}1"],
                "time": [1.0 + 2 * number, 2.0 + 2 * number],
                "feature_id": [f"ev_{name}", None],
            }
        )
        columns = {"note": {"description": f"remark by {name}"}}
        stated = AnnotationSet(
            frame, dims=DIMS, data_id=name, annotation_columns=columns
        )
        stated.io.save(root / name)
    return dc.annotations(root)


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    """Two sets, hand and auto, loaded together."""
    return _collection(tmp_path_factory.mktemp("sets"), ("hand", "auto"))


class TestFromPatch:
    """A set is stamped with the patch it was made on."""

    def test_stamps_all_three(self):
        """Dimensions, acquisition key and data id come from the patch."""
        patch = dc.get_example_patch().update_attrs(acquisition_key="N.F.00.das")
        out = AnnotationSet.from_patch(patch, pd.DataFrame({"distance": [1.0]}))
        assert out.dims == patch.dims
        assert out.attrs.acquisition_key == "N.F.00.das"
        assert out.attrs.data_id == patch.attrs.data_id

    def test_override_wins(self):
        """An attribute given explicitly beats the patch's."""
        patch = dc.get_example_patch()
        out = AnnotationSet.from_patch(patch, data_id="mine", dims=("time",))
        assert out.attrs.data_id == "mine"
        assert out.dims == ("time",)

    def test_blank_key_does_not_override_attrs(self):
        """A patch with no acquisition key leaves a given attrs' key alone."""
        patch = dc.get_example_patch()
        attrs = {"dims": patch.dims, "acquisition_key": "N.F.00.das"}
        out = AnnotationSet.from_patch(patch, attrs=attrs)
        assert out.attrs.acquisition_key == "N.F.00.das"

    def test_round_trip(self, tmp_path):
        """The stamped set saves and reads back equal."""
        patch = dc.get_example_patch()
        out = AnnotationSet.from_patch(patch, pd.DataFrame({"distance": [1.0]}))
        assert _round_trip(out, tmp_path / "set") == out


class TestAdd:
    """Tables are appended to a set."""

    def test_rows(self, picks):
        """Rows are appended, lone rows becoming features."""
        out = picks.add(pd.DataFrame({"time": [4.0], "phase": ["S"]}))
        assert len(out) == 4
        assert list(out.annotations["phase"]) == ["P", "S", "P", "S"]

    def test_rows_join_existing_feature(self, tracks):
        """A row naming an existing group adds a member to it."""
        out = tracks.add(pd.DataFrame({"feature_id": ["e1"], "time": [TIMES[2]]}))
        assert len(out["e1"].geometry.regions) == 3

    def test_features_and_bases(self, picks):
        """A feature and a basis arrive together, the feature drawn from it."""
        features = pd.DataFrame({"id": ["m"], "geometry": ["path"], "basis": ["c"]})
        out = picks.add(features=features, bases={"c": _moveout().model_dump()})
        assert out["m"].basis == _moveout()

    def test_implicit_feature(self, picks):
        """A new feature_id creates a group, as the constructor does."""
        out = picks.add(pd.DataFrame({"feature_id": ["ev"], "time": [5.0]}))
        assert list(out.features["id"]) == ["ev"]

    def test_text_times_read(self, tracks):
        """Added rows are read as the set reads its own: text times are times."""
        out = tracks.add(pd.DataFrame({"time": ["2020-01-01T00:00:05"]}))
        assert out.annotations["time"].dtype == np.dtype("datetime64[ns]")

    def test_feature_id_collision(self, tracks):
        """A feature id already in the set is refused, naming it."""
        with pytest.raises(ParameterError, match="feature id e1"):
            tracks.add(features=pd.DataFrame({"id": ["e1"]}))

    def test_annotation_id_collision(self, picks):
        """An annotation id already in the set is refused, naming it."""
        with pytest.raises(ParameterError, match="annotation id b"):
            picks.add(pd.DataFrame({"id": ["b"], "time": [9.0]}))

    def test_integer_id_collision(self):
        """An id is compared as the text it is named by."""
        frame = pd.DataFrame({"id": ["1"], "time": [1.0]})
        with pytest.raises(ParameterError, match="annotation id 1"):
            AnnotationSet(frame, dims=DIMS).add(pd.DataFrame({"id": [1], "time": [2]}))

    def test_basis_clash(self, based):
        """A key already held must name the same curve."""
        with pytest.raises(ParameterError, match="different curves"):
            based.add(bases={"curve": _moveout(velocity=10.0)})

    def test_same_basis_accepted(self, based):
        """Restating a held curve under its key is not a clash."""
        assert based.add(bases={"curve": _moveout()}) == based

    def test_mixed_kinds(self, tracks):
        """A dimension stated in another kind is refused, naming both."""
        with pytest.raises(ParameterError, match="different kinds of value"):
            tracks.add(pd.DataFrame({"time": [1.0]}))

    def test_round_trip(self, tracks, tmp_path):
        """The result saves and reads back equal."""
        out = tracks.add(pd.DataFrame({"time": [TIMES[2]], "note": ["late"]}))
        assert _round_trip(out, tmp_path / "set") == out


class TestAddFeature:
    """The generic builder splits arrays from scalars and adopts rows."""

    def test_array_scalar_split(self):
        """Arrays are member columns; scalars are feature columns."""
        empty = AnnotationSet(dims=DIMS)
        out = empty.add_feature(
            "ev", time=np.array([1.0, 2.0]), phase=("P", "S"), magnitude=2.0
        )
        assert list(out.annotations["phase"]) == ["P", "S"]
        assert list(out.annotations["feature_id"]) == ["ev", "ev"]
        assert out.features.loc[0, "magnitude"] == 2.0
        assert "magnitude" not in out.annotations.columns

    def test_series_is_array(self):
        """A Series is taken by value, not aligned on its index."""
        empty = AnnotationSet(dims=DIMS)
        out = empty.add_feature("ev", time=pd.Series([1.0, 2.0], index=[5, 9]))
        assert list(out.annotations["time"]) == [1.0, 2.0]

    def test_part_arrays(self):
        """Part arrays make a multipart path, seq taken per part."""
        empty = AnnotationSet(dims=DIMS)
        out = empty.add_feature(
            "p",
            "path",
            time=[0.0, 1.0, 5.0, 6.0],
            distance=[0.0, 1.0, 5.0, 6.0],
            part=[0, 0, 1, 1],
        )
        assert len(out["p"].geometry.vertices) == 2
        assert list(out.annotations["seq"]) == [0, 1, 0, 1]

    def test_ring_arrays(self):
        """Ring arrays make a polygon with a hole."""
        out = AnnotationSet(dims=DIMS).add_feature(
            "p",
            "polygon",
            time=[0.0, 9.0, 9.0, 0.0, 2.0, 4.0, 2.0],
            distance=[0.0, 0.0, 9.0, 9.0, 2.0, 2.0, 4.0],
            ring=[0, 0, 0, 0, 1, 1, 1],
        )
        assert [len(x["time"]) for x in out["p"].geometry.vertices[0]] == [4, 3]

    def test_adopt_by_mask(self, picks):
        """A mask adopts rows, in row order."""
        mask = np.array([True, False, True])
        out = picks.add_feature("ev", members=mask, event_type="quake")
        assert list(out.annotations["feature_id"]) == ["ev", None, "ev"]
        assert out["ev"].extra == {"event_type": "quake"}

    def test_adopt_by_ids_orders_path(self, picks):
        """Ids adopt rows in the order given, which is a path's order."""
        frame = picks.annotations.assign(distance=[0.0, 1.0, 2.0])
        spread = AnnotationSet(frame, dims=DIMS)
        out = spread.add_path("p", members=["c", "a"])
        assert out["p"].geometry.vertices[0]["time"] == (3.0, 1.0)

    def test_adopt_taken_row(self, tracks):
        """A row already in a feature is refused."""
        with pytest.raises(ParameterError, match="belongs to one feature"):
            tracks.add_feature("ev", members=["p1"])

    def test_adopt_unknown_id(self, picks):
        """An id naming no row is refused."""
        with pytest.raises(KeyError, match="No annotation"):
            picks.add_feature("ev", members=["zz"])

    def test_adopt_twice(self, picks):
        """One row cannot be adopted twice."""
        with pytest.raises(ParameterError, match="twice"):
            picks.add_feature("ev", members=["a", "a"])

    def test_mask_length(self, picks):
        """A mask the wrong length is refused."""
        with pytest.raises(ParameterError, match="one truth value per"):
            picks.add_feature("ev", members=[True])

    def test_members_and_arrays(self, picks):
        """Adopting and creating rows at once is refused."""
        with pytest.raises(ParameterError, match="one or the other"):
            picks.add_feature("ev", members=["a"], time=[1.0])

    def test_ragged_arrays(self):
        """Member columns are one length."""
        with pytest.raises(ParameterError, match="differ in length"):
            AnnotationSet(dims=DIMS).add_feature("ev", time=[1.0], phase=["P", "S"])

    def test_unknown_kind(self, picks):
        """Only the three kinds exist."""
        with pytest.raises(ParameterError, match="got 'line'"):
            picks.add_feature("ev", "line", members=["a"])

    def test_blank_id(self, picks):
        """A feature is named."""
        with pytest.raises(ParameterError, match="nonblank id"):
            picks.add_feature("", members=["a"])

    @pytest.mark.parametrize("name", ["geometry", "id"])
    def test_reserved_scalar(self, picks, name):
        """What id and kind state is not given again as a column."""
        with pytest.raises(ParameterError, match="given by id and kind"):
            picks.add_feature("ev", members=["a"], **{name: "path"})

    def test_feature_id_array(self):
        """New rows belong to the feature being added."""
        with pytest.raises(ParameterError, match="feature_id is not given"):
            AnnotationSet(dims=DIMS).add_feature("ev", time=[1.0], feature_id=["x"])

    def test_member_ids(self):
        """Annotation ids for new rows are given as an id array."""
        out = AnnotationSet(dims=DIMS).add_feature("ev", time=[1.0], id=["x"])
        assert list(out.annotations["id"]) == ["x"]

    def test_duplicate_feature_id(self, tracks):
        """A feature id already held is refused."""
        with pytest.raises(ParameterError, match="feature id t1"):
            tracks.add_feature("t1", time=[TIMES[0]])

    def test_validated_as_constructor(self):
        """A path with one vertex is refused by the ordinary checks."""
        with pytest.raises(ParameterError, match="at least 2"):
            AnnotationSet(dims=DIMS).add_path("p", time=[1.0], distance=[1.0])

    def test_round_trip(self, picks, tmp_path):
        """The result saves and reads back equal."""
        out = picks.add_feature("ev", members=["a", "b"], magnitude=1.5)
        assert _round_trip(out, tmp_path / "set") == out


class TestAddPath:
    """A path is a feature drawn through ordered members or from a curve."""

    def test_members(self):
        """Seq is taken from position."""
        out = AnnotationSet(dims=DIMS).add_path(
            "t", time=[0.0, 1.0, 2.0], distance=[0.0, 5.0, 9.0], vehicle="train"
        )
        assert list(out.annotations["seq"]) == [0, 1, 2]
        assert out["t"].kind == "path"
        assert out["t"].extra == {"vehicle": "train"}

    def test_basis_only(self):
        """A curve alone is a path, stored under the feature's id."""
        out = AnnotationSet(dims=DIMS).add_path("m", basis=_moveout().model_dump())
        assert out.bases == {"m": _moveout()}
        assert out["m"].basis == _moveout()

    def test_existing_key(self, based):
        """A string names a curve the set already holds."""
        out = based.add_path("again", basis="curve")
        assert out.features.set_index("id").loc["again", "basis"] == "curve"

    def test_unknown_key(self, picks):
        """A string naming no held curve is refused."""
        with pytest.raises(ParameterError, match="not among the bases"):
            picks.add_path("m", basis="nope")

    def test_basis_on_group(self, picks):
        """Only a path has a basis."""
        with pytest.raises(ParameterError, match="only a path"):
            picks.add_feature("g", "group", basis=_moveout(), members=["a"])

    def test_basis_key_clash(self, based):
        """A basis stored under an id holding another curve is refused."""
        with pytest.raises(ParameterError, match="different curves"):
            based.add_path("curve", basis=_moveout(velocity=1.0))

    def test_round_trip(self, tmp_path):
        """A basis-only path saves and reads back equal."""
        out = AnnotationSet(dims=DIMS).add_path("m", basis=_moveout())
        assert _round_trip(out, tmp_path / "set") == out


class TestAddPolygon:
    """Rings are frames, the outer first."""

    def test_rings(self, square):
        """Each ring's rows get their ring number and seq by position."""
        frame = square.annotations
        assert list(frame["ring"]) == [0] * 4 + [1] * 3
        assert list(frame["seq"]) == [0, 1, 2, 3, 0, 1, 2]
        assert (frame["part"] == 0).all()
        assert len(square["sq"].geometry.vertices[0]) == 2
        assert square["sq"].extra == {"note": "noise"}

    def test_array_column_refused(self):
        """Columns of the vertices belong in the rings."""
        ring = {"time": [0.0, 1.0, 0.0], "distance": [0.0, 0.0, 1.0]}
        with pytest.raises(ParameterError, match="put annotation columns"):
            AnnotationSet(dims=DIMS).add_polygon("p", rings=[ring], note=["x"])

    def test_one_frame_refused(self):
        """A bare frame is not a sequence of rings."""
        ring = pd.DataFrame({"time": [0.0, 1.0, 0.0], "distance": [0.0, 0.0, 1.0]})
        with pytest.raises(ParameterError, match="sequence of frames"):
            AnnotationSet(dims=DIMS).add_polygon("p", rings=ring)

    def test_round_trip(self, square, tmp_path):
        """The result saves and reads back equal."""
        assert _round_trip(square, tmp_path / "set") == square


class TestBounds:
    """Bounds are derived per feature, never stored."""

    def test_order_and_columns(self, tracks):
        """Features first, then lone rows; one min and max per dimension."""
        out = tracks.bounds()
        assert list(out.columns) == [
            "feature_id",
            "annotation",
            "kind",
            "time_min",
            "time_max",
            "distance_min",
            "distance_max",
        ]
        assert list(out["feature_id"]) == ["t1", "e1", None]
        assert list(out["kind"]) == ["path", "group", "group"]
        assert out["annotation"].isna().tolist() == [True, True, False]
        assert out["annotation"].iloc[2] == 5

    def test_lone_row(self, tracks):
        """A lone box's bounds are its range; an unstated dim spans."""
        row = tracks.bounds().iloc[2]
        assert (row["distance_min"], row["distance_max"]) == (10.0, 80.0)
        assert pd.isna(row["time_min"]) and pd.isna(row["time_max"])

    def test_group_spans(self, tracks):
        """A group of time picks envelopes time and spans distance."""
        row = tracks.bounds().iloc[1]
        assert (row["time_min"], row["time_max"]) == (TIMES[0], TIMES[1])
        assert pd.isna(row["distance_min"])

    def test_group_member_spanning(self):
        """One member spanning a dimension makes the group span it."""
        frame = pd.DataFrame(
            {"feature_id": ["g", "g"], "time": [1.0, 2.0], "distance": [5.0, None]}
        )
        row = AnnotationSet(frame, dims=DIMS).bounds().iloc[0]
        assert pd.isna(row["distance_min"])
        assert (row["time_min"], row["time_max"]) == (1.0, 2.0)

    def test_group_of_ranges_and_values(self):
        """A group envelopes ranges and values together."""
        frame = pd.DataFrame(
            {
                "feature_id": ["g", "g"],
                "time": [None, 20.0],
                "time_min": [1.0, None],
                "time_max": [4.0, None],
            }
        )
        row = AnnotationSet(frame, dims=DIMS).bounds().iloc[0]
        assert (row["time_min"], row["time_max"]) == (1.0, 20.0)

    def test_path(self, tracks):
        """A path's bounds envelope its vertices."""
        row = tracks.bounds().iloc[0]
        assert (row["distance_min"], row["distance_max"]) == (1.0, 9.0)
        assert (row["time_min"], row["time_max"]) == (TIMES[0], TIMES[2])

    def test_polygon(self, square):
        """A polygon's bounds envelope every ring."""
        row = square.bounds().iloc[0]
        assert (row["time_min"], row["time_max"]) == (0.0, 10.0)
        assert row["kind"] == "polygon"

    def test_basis_only_path(self, based):
        """A path drawn only from its curve is bounded by the curve."""
        row = based.bounds().set_index("feature_id").loc["far"]
        drawn = _moveout().vertices(64)
        assert row["distance_min"] == 0.0 and row["distance_max"] == 100.0
        assert row["time_max"] == drawn["time"].max()

    def test_members_beat_basis(self, based):
        """Where a path has members, they bound it rather than its curve."""
        row = based.bounds().set_index("feature_id").loc["near"]
        assert (row["distance_min"], row["distance_max"]) == (0.0, 50.0)

    def test_moveout_apex(self):
        """A curve's bounds are exact, so its apex time is its earliest."""
        out = AnnotationSet(dims=DIMS).add_path("m", basis=_moveout())
        assert out.bounds().iloc[0]["time_min"] == TIMES[0]
        assert len(out.overlapping(time=TIMES[0])) == 1

    def test_moveout_apex_outside(self):
        """With its apex off the span, a curve is earliest at the nearer end."""
        curve = _moveout(apex_distance=-100.0)
        out = AnnotationSet(dims=DIMS).add_path("m", basis=curve)
        row = out.bounds().iloc[0]
        assert row["time_min"] == curve.vertices(2)["time"][0]
        assert row["time_max"] == curve.vertices(2)["time"][1]

    def test_nullable_integer_dim(self):
        """A nullable integer dimension bounds as floats, blanks spanning."""
        frame = pd.DataFrame({"time": [1.0, 2.0]})
        frame["distance"] = pd.array([10, None], "Int64")
        out = AnnotationSet(frame, dims=DIMS)
        assert out.bounds()["distance_min"].dtype == np.float64
        assert len(out.overlapping(distance=(0, 20))) == 2
        assert len(out.overlapping(distance=(30, 40))) == 1

    def test_value_bound_inclusive(self):
        """A value bound is a point: min equals max."""
        out = AnnotationSet(pd.DataFrame({"time": [3.0]}), dims=DIMS).bounds()
        assert out.iloc[0]["time_min"] == out.iloc[0]["time_max"] == 3.0

    def test_empty(self):
        """An empty set has no bounds rows."""
        assert AnnotationSet(dims=DIMS).bounds().empty

    def test_integer_values(self):
        """Integer coordinates beside a spanning row stay readable."""
        frame = pd.DataFrame({"time": [1, 2], "feature_id": [None, "g"]})
        out = AnnotationSet(frame, dims=DIMS).bounds()
        assert list(out["time_min"]) == [2, 1]


class TestSelect:
    """Column filters over both tables."""

    def test_annotation_filter(self, picks):
        """A value and a range with an open end."""
        out = picks.select(phase="P", confidence=(0.8, None))
        assert list(out.annotations["id"]) == ["a"]

    def test_min_spelling(self, picks):
        """A <name>_min filter bounds one end."""
        assert len(picks.select(confidence_min=0.9)) == 2

    def test_glob_and_membership(self, picks):
        """A string is a glob; a list is membership."""
        assert len(picks.select(phase="[S]")) == 1
        assert len(picks.select(confidence=[0.9, 0.3])) == 2

    def test_glob_skips_blank(self):
        """A blank cell does not match a glob."""
        frame = pd.DataFrame({"time": [1.0, 2.0], "note": ["car", None]})
        assert len(AnnotationSet(frame, dims=DIMS).select(note="*")) == 1

    def test_feature_filter(self, tracks):
        """A feature filter drops failing features with their members."""
        out = tracks.select(vehicle_type="train")
        assert list(out.features["id"]) == ["t1"]
        assert set(out.annotations["feature_id"]) == {"t1"}

    def test_both_tables(self, tracks):
        """A feature and an annotation filter in one call."""
        out = tracks.select(vehicle_type="train", velocity=(3.5, None))
        assert list(out.annotations["id"]) == ["v1", "v2"]

    def test_drops_emptied_feature(self, tracks):
        """A group whose members all fail is dropped."""
        out = tracks.select(velocity_min=0)
        assert list(out.features["id"]) == ["t1"]
        assert len(out) == 1

    def test_under_minimum_path(self, tracks):
        """A path left with one vertex is refused, naming it."""
        with pytest.raises(ParameterError, match=r"'t1'.*at least 2"):
            tracks.select(velocity=5.0)

    def test_basis_only_untouched(self, based):
        """A path with no members is not an annotation filter's to drop."""
        out = based.select(feature_id="nothing")
        assert list(out.features["id"]) == ["far"]

    def test_bases_kept(self, based):
        """Bases are kept whatever is selected."""
        assert based.select(feature_id="nothing").bases == based.bases

    @pytest.mark.parametrize(
        "query", [{"k": 1}, {"k": (0, 5)}, {"k_min": 0}, {"k_max": 5}]
    )
    def test_nullable_column(self, query):
        """A blank in a nullable column fails a filter rather than raising."""
        frame = pd.DataFrame({"time": [1.0, 2.0]})
        frame["k"] = pd.array([1, None], "Int64")
        assert len(AnnotationSet(frame, dims=DIMS).select(**query)) == 1

    def test_seq_range(self, tracks):
        """Seq is blank off paths, and a range over it still filters."""
        out = tracks.select(seq=(0, 1))
        assert list(out.annotations["id"]) == ["v0", "v1"]

    def test_dimension_refused(self, picks):
        """A dimension is overlapping's to select."""
        with pytest.raises(ParameterError, match="overlapping"):
            picks.select(time=(1.0, 2.0))

    def test_ambiguous(self, tracks):
        """A column on both tables is refused."""
        with pytest.raises(ParameterError, match="ambiguous"):
            tracks.update(feature="t1", name="a").update(
                annotation="v0", name="b"
            ).select(name="x")

    def test_id_refused(self, picks):
        """Id is ambiguous even with no features, and says what to use."""
        with pytest.raises(ParameterError, match="feature_id="):
            picks.select(id="a")

    def test_range_spelling_refused(self, picks):
        """A dimension's range spelling is overlapping's too."""
        with pytest.raises(ParameterError, match="overlapping"):
            picks.select(time_min=1.0)

    def test_text_tuple_refused(self, picks):
        """A tuple on a text column is not a range; a list is membership."""
        with pytest.raises(ParameterError, match="list"):
            picks.select(phase=("P", "S"))

    def test_resolved_provenance(self, picks):
        """A provenance filter matches rows inheriting the set's value."""
        assert len(picks.select(data_id="patch-1")) == 3
        assert len(picks.select(data_id="other")) == 0

    @pytest.mark.parametrize("name", ["geometry", "basis"])
    def test_feature_columns_without_features(self, picks, name):
        """Geometry and basis filter features, even where there are none."""
        assert len(picks.select(**{name: "path"})) == 0

    def test_geometry(self, tracks):
        """Geometry selects features by kind."""
        assert list(tracks.select(geometry="path").features["id"]) == ["t1"]

    def test_basis_filter(self, based):
        """A basis key selects the paths drawn from it."""
        assert set(based.select(basis="curve").features["id"]) == {"near", "far"}

    def test_group_kind(self, tracks):
        """A group's kind is spelled group, though stored blank."""
        assert list(tracks.select(geometry="group").features["id"]) == ["e1"]

    def test_set_label(self, collection):
        """Set filters both tables by label."""
        out = collection.select(set="hand")
        assert set(out.annotations["set"]) == {"hand"}
        assert set(out.features["set"]) == {"hand"}

    def test_clears_basis_of_trimmed_path(self, based):
        """Dropping some of a basis path's members clears its basis."""
        out = based.select(q_min=2)
        assert out["near"].basis is None
        assert out["far"].basis == _moveout()

    def test_unknown(self, picks):
        """A column on neither table is refused."""
        with pytest.raises(ParameterError, match="neither"):
            picks.select(nope=1)

    def test_round_trip(self, tracks, tmp_path):
        """The result saves and reads back equal."""
        out = tracks.select(vehicle_type="train")
        assert _round_trip(out, tmp_path / "set") == out


class TestOverlapping:
    """Features whose derived bounds meet a query, kept whole."""

    def test_range(self, tracks):
        """A distance range keeps what meets it and what spans distance."""
        out = tracks.overlapping(distance=(20.0, 30.0))
        assert list(out.features["id"]) == ["e1"]
        assert list(out.annotations["id"]) == ["p1", "p2", "box"]

    def test_open_end(self, tracks):
        """An open end is unbounded; a value at the top of a path is inside."""
        out = tracks.overlapping(time=(TIMES[2], None))
        assert set(out.features["id"]) == {"t1"}
        assert "box" in set(out.annotations["id"])

    def test_spanning_dim(self, tracks):
        """A feature spanning the dimension always overlaps."""
        out = tracks.overlapping(time=(TIMES[0] - np.timedelta64(9, "s"), TIMES[0]))
        assert list(out.annotations["id"]) == ["box"]

    def test_half_open_range(self):
        """A range row ending where the query starts does not overlap."""
        frame = pd.DataFrame({"time_min": [0.0, 5.0], "time_max": [5.0, 9.0]})
        boxes = AnnotationSet(frame, dims=DIMS)
        assert len(boxes.overlapping(time=(5.0, 6.0))) == 1

    def test_point_query(self):
        """A single value meets a range containing it, and a point equal to it."""
        frame = pd.DataFrame({"time_min": [0.0, None], "time_max": [5.0, None]})
        frame["time"] = [None, 5.0]
        out = AnnotationSet(frame, dims=DIMS).overlapping(time=5.0)
        assert out.annotations["time"].tolist() == [5.0]

    def test_text_time(self, tracks):
        """A time query may be written as text."""
        out = tracks.overlapping(time=("2020-01-01T00:00:02", None))
        assert "t1" in set(out.features["id"])

    def test_no_trim(self, tracks):
        """Members follow their feature, none trimmed."""
        out = tracks.overlapping(distance=(0.0, 2.0))
        assert list(out.annotations["id"]) == ["v0", "v1", "v2", "p1", "p2"]

    def test_basis_only_path(self, based):
        """A curve-only path is placed by its curve."""
        out = based.overlapping(distance=(60.0, 70.0))
        assert list(out.features["id"]) == ["far"]

    def test_dim_nobody_states(self, picks):
        """Where every feature spans the dimension, every feature overlaps."""
        assert picks.overlapping(distance=(0.0, 1.0)) == picks

    def test_duration_dim(self):
        """A duration dimension is queried in durations."""
        offsets = np.array([1, 5], dtype="timedelta64[s]")
        frame = pd.DataFrame({"offset": offsets})
        out = AnnotationSet(frame, dims=("offset",)).overlapping(
            offset=(np.timedelta64(3, "s"), None)
        )
        assert out.annotations["offset"].tolist() == [pd.Timedelta(5, "s")]

    def test_half_open_upper_end(self):
        """A range row starting where the query ends does not overlap."""
        frame = pd.DataFrame({"time_min": [6.0], "time_max": [9.0]})
        assert len(AnnotationSet(frame, dims=DIMS).overlapping(time=(5.0, 6.0))) == 0

    def test_duration_nanoseconds(self):
        """A duration keeps its nanoseconds through bounds and queries."""
        frame = pd.DataFrame({"time": [pd.Timedelta(2500, "ns")]})
        out = AnnotationSet(frame, dims=DIMS)
        assert out.bounds().iloc[0]["time_min"] == pd.Timedelta(2500, "ns")
        assert len(out.overlapping(time=(pd.Timedelta(2500, "ns"), None))) == 1

    def test_unknown_dim(self, tracks):
        """Only declared dimensions are queried."""
        with pytest.raises(ParameterError, match="velocity"):
            tracks.overlapping(velocity=(1, 2))

    def test_round_trip(self, tracks, tmp_path):
        """The result saves and reads back equal."""
        out = tracks.overlapping(distance=(20.0, 30.0))
        assert _round_trip(out, tmp_path / "set") == out


class TestUpdate:
    """One row of one table changes."""

    def test_feature_column(self, tracks):
        """A feature's column is set, a new one blank elsewhere."""
        out = tracks.update(feature="e1", magnitude=2.5)
        assert out["e1"].extra["magnitude"] == 2.5
        assert "magnitude" not in out["t1"].extra

    def test_annotation_column(self, tracks):
        """An annotation's column is set, widening its dtype when needed."""
        out = tracks.update(annotation="v1", velocity="fast")
        assert out.annotations.set_index("id").loc["v1", "velocity"] == "fast"

    def test_coordinate(self, tracks):
        """An annotation's coordinate may change."""
        out = tracks.update(annotation="box", distance_max=90.0)
        assert out.bounds().iloc[2]["distance_max"] == 90.0

    def test_clears_basis(self, based):
        """Moving a member of a basis-backed path clears its basis."""
        out = based.update(annotation="m1", distance=60.0)
        assert out["near"].basis is None
        assert out["far"].basis == _moveout()
        assert "curve" in out.bases

    def test_other_column_keeps_basis(self, based):
        """Changing a non-coordinate column leaves the basis."""
        out = based.update(annotation="m1", note="checked")
        assert out["near"].basis == _moveout()

    def test_same_kind_accepted(self, tracks):
        """Restating a feature's kind is not a change."""
        assert tracks.update(feature="e1", geometry="") == tracks

    def test_kind_refused(self, tracks):
        """A feature's kind does not change."""
        with pytest.raises(ParameterError, match="kind"):
            tracks.update(feature="t1", geometry="polygon")

    def test_id_refused(self, tracks):
        """An id does not change."""
        with pytest.raises(ParameterError, match="cannot change"):
            tracks.update(feature="t1", id="t2")

    @pytest.mark.parametrize("selectors", [{}, {"feature": "t1", "annotation": "v0"}])
    def test_one_selector(self, tracks, selectors):
        """Exactly one of feature and annotation."""
        with pytest.raises(ParameterError, match="exactly one"):
            tracks.update(**selectors, note="x")

    def test_unknown_feature(self, tracks):
        """An unknown feature id is a missing key."""
        with pytest.raises(KeyError, match="No feature"):
            tracks.update(feature="zz", note="x")

    @pytest.mark.parametrize("verb", ["update", "remove"])
    def test_unknown_annotation(self, picks, verb):
        """An annotation id naming no row is a missing key."""
        with pytest.raises(KeyError, match="No annotation"):
            getattr(picks, verb)(annotation="zz")

    def test_integer_column_widens(self, tmp_path):
        """A float set into an integer column makes it float, not object."""
        frame = pd.DataFrame({"id": ["a", "b"], "time": [1.0, 2.0], "n": [1, 2]})
        out = AnnotationSet(frame, dims=DIMS).update(annotation="b", n=2.5)
        assert out.annotations["n"].dtype == np.float64
        assert _round_trip(out, tmp_path / "set") == out

    def test_row_without_id(self):
        """A row with no id cannot be named."""
        out = AnnotationSet(pd.DataFrame({"time": [1.0]}), dims=DIMS)
        with pytest.raises(KeyError, match="No annotation"):
            out.update(annotation="", note="x")

    def test_validated(self, tracks):
        """The edited set is checked like any other."""
        with pytest.raises(ParameterError, match="ends before it starts"):
            tracks.update(annotation="box", distance_max=1.0)

    def test_round_trip(self, tracks, tmp_path):
        """The result saves and reads back equal."""
        out = tracks.update(feature="e1", magnitude=2.5)
        assert _round_trip(out, tmp_path / "set") == out


class TestRemove:
    """A feature or an annotation leaves the set."""

    def test_feature_cascades(self, tracks):
        """Removing a feature removes its members."""
        out = tracks.remove(feature="t1")
        assert list(out.features["id"]) == ["e1"]
        assert "t1" not in set(out.annotations["feature_id"])
        assert len(out.annotations) == 3

    def test_annotation(self, tracks):
        """Removing a member leaves its group."""
        out = tracks.remove(annotation="p1")
        assert len(out["e1"].geometry.regions) == 1

    def test_emptied_group(self, tracks):
        """A group left with no members is removed."""
        out = tracks.remove(annotation="p1").remove(annotation="p2")
        assert "e1" not in set(out.features["id"])

    def test_path_under_minimum(self, tracks):
        """A path left with one vertex is refused."""
        out = tracks.remove(annotation="v0")
        with pytest.raises(ParameterError, match=r"'t1'.*at least 2"):
            out.remove(annotation="v1")

    def test_lone_row(self, tracks):
        """A lone row is removed by its id."""
        assert "box" not in set(tracks.remove(annotation="box").annotations["id"])

    def test_one_selector(self, tracks):
        """Exactly one of feature and annotation."""
        with pytest.raises(ParameterError, match="exactly one"):
            tracks.remove()

    def test_unknown(self, tracks):
        """An unknown id is a missing key."""
        with pytest.raises(KeyError):
            tracks.remove(feature="zz")

    def test_round_trip(self, tracks, tmp_path):
        """The result saves and reads back equal."""
        out = tracks.remove(feature="t1")
        assert _round_trip(out, tmp_path / "set") == out


class TestMerge:
    """Sets combine table by table."""

    def test_tables_join(self, picks, tracks):
        """Rows and features join; columns union."""
        times = pd.DataFrame({"time": TIMES[:1], "phase": ["P"], "id": ["q"]})
        out = tracks.merge(AnnotationSet(times, dims=DIMS))
        assert len(out) == len(tracks) + 1
        assert "phase" in out.annotations.columns

    def test_provenance_written(self, picks):
        """Another set's differing data_id lands in its rows."""
        other = AnnotationSet(
            pd.DataFrame({"time": [9.0]}), dims=DIMS, data_id="patch-2"
        )
        out = picks.merge(other)
        assert out.attrs.data_id == ""
        assert [x.data_id for x in out] == ["patch-1"] * 3 + ["patch-2"]
        assert out.annotations["data_id"].tolist() == ["patch-1"] * 3 + ["patch-2"]

    def test_provenance_order_independent(self, picks):
        """Either order gives the same rows, and no value is invented."""
        other = AnnotationSet(pd.DataFrame({"time": [9.0]}), dims=DIMS)
        ahead, behind = picks.merge(other), other.merge(picks)
        for out in (ahead, behind):
            assert out.attrs.data_id == ""
            # Every row is lone here, so features iterate in row order.
            stamped = dict(zip(out.annotations["time"], out, strict=True))
            assert {k: v.data_id for k, v in stamped.items()} == {
                1.0: "patch-1",
                2.0: "patch-1",
                3.0: "patch-1",
                9.0: "",
            }

    def test_other_unchanged(self, picks):
        """The sets merged in are not changed."""
        other = AnnotationSet(pd.DataFrame({"time": [9.0]}), dims=DIMS, data_id="x")
        copy = AnnotationSet(other.annotations, attrs=other.attrs)
        picks.merge(other)
        assert other == copy

    def test_same_provenance_not_written(self, picks):
        """Where provenance agrees, nothing is written."""
        other = AnnotationSet(
            pd.DataFrame({"time": [9.0]}), dims=DIMS, data_id="patch-1"
        )
        assert "data_id" not in picks.merge(other).annotations.columns

    def test_row_provenance_kept(self, picks):
        """A row stating its own provenance keeps it."""
        frame = pd.DataFrame({"time": [9.0], "acquisition_key": ["N.R.00.das"]})
        other = AnnotationSet(frame, dims=DIMS, acquisition_key="N.S.00.das")
        out = picks.merge(other)
        assert [x.acquisition_key for x in out][-1] == "N.R.00.das"

    def test_feature_provenance(self, tracks):
        """A features row takes its set's provenance too."""
        other = AnnotationSet(
            pd.DataFrame({"feature_id": ["g"], "time": TIMES[:1]}),
            dims=DIMS,
            data_id="other",
        )
        assert tracks.merge(other)["g"].data_id == "other"

    def test_shared_feature_id(self, tracks):
        """A feature id in both sets is refused, naming it."""
        with pytest.raises(ParameterError, match="feature id t1"):
            tracks.merge(
                AnnotationSet(dims=DIMS).add_path(
                    "t1", time=TIMES[:2], distance=[0.0, 1.0]
                )
            )

    def test_shared_annotation_id(self, picks):
        """An annotation id in both sets is refused, naming it."""
        with pytest.raises(ParameterError, match="annotation id a"):
            picks.merge(picks)

    def test_members_never_stamped(self):
        """Stamping writes features and lone rows, never a member."""
        frame = pd.DataFrame(
            {"time": [1.0, 2.0, 3.0], "feature_id": ["ev", "ev", None]}
        )
        grouped = AnnotationSet(frame, dims=DIMS, data_id="aaa")
        other = AnnotationSet(pd.DataFrame({"time": [9.0]}), dims=DIMS)
        for out in (grouped.merge(other), other.merge(grouped)):
            assert out.attrs.data_id == ""
            rows = out.annotations.set_index("time")
            assert pd.isna(rows.loc[1.0, "data_id"]) and pd.isna(
                rows.loc[2.0, "data_id"]
            )
            assert rows.loc[3.0, "data_id"] == "aaa" and pd.isna(
                rows.loc[9.0, "data_id"]
            )
            assert out["ev"].data_id == "aaa"
            assert len(out.select(data_id="aaa").annotations) == 3

    def test_agreeing_provenance_kept(self, picks):
        """A value every set states stays set-level."""
        other = AnnotationSet(
            pd.DataFrame({"time": [9.0]}), dims=DIMS, data_id="patch-1"
        )
        assert picks.merge(other).attrs.data_id == "patch-1"

    def test_dims_differ(self, picks):
        """Sets merge in the same dimensions."""
        other = AnnotationSet(pd.DataFrame({"time": [1.0]}), dims=("time",))
        with pytest.raises(ParameterError, match="same dimensions"):
            picks.merge(other)

    def test_dim_order_ignored(self, picks):
        """Dimension order is not a difference; this set's order is kept."""
        other = AnnotationSet(pd.DataFrame({"time": [9.0]}), dims=DIMS[::-1])
        assert picks.merge(other).dims == DIMS

    def test_not_a_set(self, picks):
        """Only sets merge."""
        with pytest.raises(ParameterError, match="Only annotation sets"):
            picks.merge(pd.DataFrame({"time": [1.0]}))

    def test_bases(self, based):
        """Bases join; a shared key names one curve."""
        other = AnnotationSet(dims=DIMS).add_path("m", basis=_moveout(velocity=9.0))
        assert set(based.merge(other).bases) == {"curve", "m"}
        clash = AnnotationSet(dims=DIMS, bases={"curve": _moveout(velocity=9.0)})
        with pytest.raises(ParameterError, match="different curves"):
            based.merge(clash)

    def test_columns_join(self, picks):
        """Column documentation joins; a clashing dtype drops the declaration."""
        declared = {"score": {"dtype": "float64", "description": "how sure"}}
        frame = pd.DataFrame({"time": [9.0], "score": [0.5]})
        other = AnnotationSet(frame, dims=DIMS, annotation_columns=declared)
        out = picks.merge(other)
        assert out.attrs.annotation_columns["score"].description == "how sure"
        clash = {"score": {"dtype": "int64"}}
        mine = AnnotationSet(
            pd.DataFrame({"time": [1.0], "score": [1]}),
            dims=DIMS,
            annotation_columns=clash,
        )
        assert "score" not in mine.merge(other).attrs.annotation_columns
        assert "score" not in other.merge(mine).attrs.annotation_columns

    def test_text_dtypes_agree(self):
        """Two spellings of text do not clash."""
        frames = [pd.DataFrame({"time": [x], "tag": ["a"]}) for x in (1.0, 2.0)]
        sets = [
            AnnotationSet(f, dims=DIMS, annotation_columns={"tag": {"dtype": d}})
            for f, d in zip(frames, ("object", "str"), strict=True)
        ]
        assert len(sets[0].merge(sets[1])) == 2

    def test_empty_set_keeps_dtypes(self):
        """Merging or adding nothing widens no column."""
        frame = pd.DataFrame({"time": [1.0], "n": [1], "flag": [True]})
        plain = AnnotationSet(frame, dims=DIMS)
        for out in (plain.merge(AnnotationSet(dims=DIMS)), plain.add(pd.DataFrame())):
            assert out == plain
            assert out.annotations["n"].dtype == np.int64
            assert out.annotations["flag"].dtype == bool

    def test_empty_frame_columns_kept(self):
        """A column only an empty frame names is kept, blank."""
        plain = AnnotationSet(pd.DataFrame({"time": [1.0], "n": [1]}), dims=DIMS)
        out = plain.add(pd.DataFrame(columns=["n", "note"]))
        assert out.annotations["n"].dtype == np.int64
        assert out.annotations["note"].isna().all()

    def test_nothing_to_merge(self, picks):
        """Merging nothing is the same set."""
        assert picks.merge() == picks

    def test_round_trip(self, picks, tracks, tmp_path):
        """The result saves and reads back equal."""
        times = pd.DataFrame({"time": TIMES[:1], "id": ["q"]})
        other = AnnotationSet(times, dims=DIMS, data_id="x")
        out = tracks.merge(other)
        assert _round_trip(out, tmp_path / "set") == out


class TestBasisClearing:
    """Any edit to a basis path's members clears its basis."""

    def test_remove_member(self, based):
        """Removing a member clears it."""
        assert based.remove(annotation="m1")["near"].basis is None

    def test_update_seq(self, based):
        """Reordering a member clears it."""
        assert based.update(annotation="m2", seq=7)["near"].basis is None

    def test_move_member_out(self, based):
        """Moving a member out clears it, and the row is lone again."""
        out = based.update(annotation="m2", feature_id=None)
        assert out["near"].basis is None
        assert pd.isna(out.annotations.set_index("id").loc["m2", "seq"])

    def test_move_row_in(self, based):
        """Moving a row in appends it to part 0 and clears the basis."""
        later = TIMES[2] + np.timedelta64(1, "s")
        frame = pd.DataFrame({"id": ["x"], "time": [later], "distance": [75.0]})
        added = based.add(frame)
        assert added["near"].basis == _moveout()
        out = added.update(annotation="x", feature_id="near")
        assert out["near"].basis is None
        assert out.annotations.set_index("id").loc["x", "seq"] == 3

    def test_add_member(self, based):
        """Adding a row to the path clears it."""
        later = TIMES[2] + np.timedelta64(1, "s")
        frame = pd.DataFrame(
            {"feature_id": ["near"], "time": [later], "distance": [75.0], "seq": [3]}
        )
        assert based.add(frame)["near"].basis is None

    def test_whole_path_kept(self, based):
        """Overlapping keeps a path whole, so its basis stays."""
        assert based.overlapping(distance=(0.0, 10.0))["near"].basis == _moveout()

    def test_untouched_path_keeps_across_new_columns(self, based):
        """A time range column arriving elsewhere leaves the path's basis."""
        frame = pd.DataFrame({"time_min": [TIMES[0]], "time_max": [TIMES[1]]})
        assert based.add(frame)["near"].basis == _moveout()
        other = AnnotationSet(frame, dims=DIMS)
        assert based.merge(other)["near"].basis == _moveout()

    def test_restating_feature_is_no_move(self, based):
        """Setting a row's feature_id to its own feature changes nothing."""
        assert based.update(annotation="m0", feature_id="near") == based

    def test_feature_edit_keeps(self, based):
        """Editing the feature's own row leaves its members, and its basis."""
        assert based.update(feature="near", note="x")["near"].basis == _moveout()


@pytest.fixture(scope="module")
def stamped():
    """Lone rows stating their own data_id beside a group whose feature states one."""
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "m0", "m1"],
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "data_id": ["x", "x", "y", None, None],
            "feature_id": [None, None, None, "ev", "ev"],
        }
    )
    features = pd.DataFrame({"id": ["ev"], "data_id": ["x"]})
    return AnnotationSet(frame, features=features, dims=DIMS)


class TestProvenanceLifting:
    """Grouping rows lifts their provenance onto the feature, never inventing it."""

    def test_adopt_agreeing(self, stamped):
        """Rows which agree give the feature their value, and blank their own."""
        out = stamped.add_feature("g", members=["a", "b"])
        assert out["g"].data_id == "x"
        assert out.annotations.set_index("id").loc[["a", "b"], "data_id"].isna().all()

    def test_adopt_disagreeing(self, stamped):
        """Rows from two acquisitions do not make one feature."""
        with pytest.raises(ParameterError, match=r"'x' and 'y'.*one acquisition"):
            stamped.add_feature("g", members=["a", "c"])

    def test_adopt_against_given(self, stamped):
        """A stated value the rows contradict is refused."""
        with pytest.raises(ParameterError, match="one acquisition"):
            stamped.add_feature("g", members=["a", "b"], data_id="z")

    def test_adopt_with_given(self, stamped):
        """A stated value the rows agree with is the feature's."""
        out = stamped.add_path("p", members=["a", "b"], data_id="x")
        assert out["p"].data_id == "x"

    def test_move_into_agreeing(self, stamped):
        """A row moving into a feature it agrees with gives up its own cell."""
        out = stamped.update(annotation="a", feature_id="ev")
        assert pd.isna(out.annotations.set_index("id").loc["a", "data_id"])
        assert out["ev"].data_id == "x"

    def test_move_into_disagreeing(self, stamped):
        """A row from another acquisition may not join a feature."""
        with pytest.raises(ParameterError, match=r"'y' and 'x'.*one acquisition"):
            stamped.update(annotation="c", feature_id="ev")

    def test_move_out_keeps_value(self, stamped):
        """A member moving out takes its feature's value as its own."""
        out = stamped.update(annotation="m0", feature_id=None)
        assert out.annotations.set_index("id").loc["m0", "data_id"] == "x"

    def test_move_out_to_same_fallback(self):
        """A member whose value the set already gives needs no cell of its own."""
        frame = pd.DataFrame(
            {
                "id": ["a", "m0"],
                "time": [1.0, 2.0],
                "data_id": ["own", None],
                "feature_id": [None, "ev"],
            }
        )
        base = AnnotationSet(frame, dims=DIMS, data_id="set")
        out = base.update(annotation="m0", feature_id=None)
        assert pd.isna(out.annotations.set_index("id").loc["m0", "data_id"])
        assert [x.data_id for x in out] == ["own", "set"]

    def test_after_merge(self):
        """Rows a merge stamped regroup under a feature holding their value."""
        first = AnnotationSet(
            pd.DataFrame({"id": ["a", "b"], "time": [1.0, 2.0]}),
            dims=DIMS,
            data_id="d-a",
        )
        second = AnnotationSet(
            pd.DataFrame({"id": ["z"], "time": [3.0]}), dims=DIMS, data_id="d-b"
        )
        out = first.merge(second).add_feature("g", members=["a", "b"])
        assert out["g"].data_id == "d-a"
        assert [x.data_id for x in out] == ["d-a", "d-b"]


# Two child sets of different provenance, for label checks.
TWO_CHILDREN = {
    "dims": DIMS,
    "sets": {
        "a": {"dims": ("time",), "data_id": "d-a"},
        "b": {"dims": ("time",), "data_id": "d-b"},
    },
}


class TestOneBlankRule:
    """Adopting and moving follow one rule: a blank agrees, a value lands."""

    @pytest.fixture
    def rows(self):
        """A stated row, a blank one, and one stating another value."""
        frame = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "time": [1.0, 2.0, 3.0],
                "data_id": ["x", None, "y"],
            }
        )
        return AnnotationSet(frame, dims=DIMS)

    def test_either_order(self, rows):
        """Growing a feature row by row equals adopting the rows at once."""
        together = rows.add_feature("g", members=["a", "b"])
        stated_first = rows.add_feature("g", members=["a"]).update(
            annotation="b", feature_id="g"
        )
        blank_first = rows.add_feature("g", members=["b"]).update(
            annotation="a", feature_id="g"
        )
        assert stated_first == together
        assert blank_first == together
        assert together["g"].data_id == "x"

    def test_two_values_refused(self, rows):
        """A stated row moving into a feature of another value is refused."""
        grouped = rows.add_feature("g", members=["a"])
        with pytest.raises(ParameterError, match="one acquisition"):
            grouped.update(annotation="c", feature_id="g")

    def test_move_into_implied_feature(self):
        """A merged row moving into a new feature lifts its value onto it."""
        first = AnnotationSet(
            pd.DataFrame({"id": ["a", "b"], "time": [1.0, 2.0]}),
            dims=DIMS,
            data_id="d-a",
        )
        second = AnnotationSet(
            pd.DataFrame({"id": ["z"], "time": [3.0]}), dims=DIMS, data_id="d-b"
        )
        merged = first.merge(second)
        out = merged.update(annotation="a", feature_id="g").update(
            annotation="b", feature_id="g"
        )
        assert out["g"].data_id == "d-a"
        assert out.annotations.set_index("id").loc[["a", "b"], "data_id"].isna().all()
        assert out == merged.add_feature("g", members=["a", "b"])


class TestMemberLabels:
    """A member's own set label may not disagree with its feature."""

    def test_disagreeing_label_refused(self):
        """Members labeled with two children of different provenance."""
        frame = pd.DataFrame(
            {
                "id": ["r1", "r2"],
                "time": [1.0, 2.0],
                "set": ["a", "b"],
                "feature_id": "e",
            }
        )
        with pytest.raises(ParameterError, match=r"Row 1.*'b'.*'e'"):
            AnnotationSet(frame, attrs=TWO_CHILDREN)

    def test_blank_label_accepted(self):
        """A member with no label takes its feature's provenance."""
        frame = pd.DataFrame(
            {
                "id": ["r1", "r2"],
                "time": [1.0, 2.0],
                "set": ["a", None],
                "feature_id": "e",
            }
        )
        out = AnnotationSet(frame, attrs=TWO_CHILDREN)
        assert out["e"].data_id == "d-a"

    def test_add_refused(self):
        """Add is held to the same rule."""
        frame = pd.DataFrame(
            {"id": ["r1"], "time": [1.0], "set": ["a"], "feature_id": "e"}
        )
        base = AnnotationSet(frame, attrs=TWO_CHILDREN)
        extra = pd.DataFrame({"time": [2.0], "set": ["b"], "feature_id": ["e"]})
        with pytest.raises(ParameterError, match="'e'"):
            base.add(extra)

    @staticmethod
    def _overridden(feature_of_r=None, features=None) -> AnnotationSet:
        """Lone r and member m in child b; f in b overrides its data_id."""
        frame = pd.DataFrame(
            {
                "id": ["r", "m"],
                "time": [1.0, 2.0],
                "set": ["b", "b"],
                "feature_id": [feature_of_r, "f"],
            }
        )
        stated = pd.DataFrame({"id": ["f"], "set": ["b"], "data_id": ["feat"]})
        tables = stated if features is None else pd.concat([stated, features])
        return AnnotationSet(frame, features=tables, attrs=TWO_CHILDREN)

    def test_same_label_joins_override(self):
        """A row of the feature's own child joins it, whatever it overrides."""
        out = self._overridden().update(annotation="r", feature_id="f")
        assert out == self._overridden("f")
        assert out["f"].data_id == "feat"

    def test_same_label_adopted_under_override(self):
        """Adopting a row of the new feature's child takes the override too."""
        out = self._overridden().add_feature(
            "g", members=["r"], set="b", data_id="feat"
        )
        extra = pd.DataFrame({"id": ["g"], "set": ["b"], "data_id": ["feat"]})
        assert out == self._overridden("g", extra)

    def test_other_label_still_refused(self):
        """A row of another child does not join the override."""
        base = self._overridden().update(annotation="r", set="a")
        with pytest.raises(ParameterError, match="one acquisition"):
            base.update(annotation="r", feature_id="f")

    @pytest.fixture
    def labeled(self):
        """r1 a member of e in child a; r2 lone in child b."""
        frame = pd.DataFrame(
            {
                "id": ["r1", "r2"],
                "time": [1.0, 2.0],
                "set": ["a", "b"],
                "feature_id": ["e", None],
            }
        )
        return AnnotationSet(frame, attrs=TWO_CHILDREN)

    def test_relabel_while_moving_in(self, labeled):
        """A row relabeled to agree with the feature it joins is accepted."""
        out = labeled.update(annotation="r2", feature_id="e", set="a")
        assert len(out["e"].geometry.regions) == 2

    def test_relabel_while_moving_out(self, labeled):
        """A row leaving under another label keeps the value it had."""
        out = labeled.update(annotation="r1", feature_id=None, set="b")
        rows = out.annotations.set_index("id")
        assert rows.loc["r1", "data_id"] == "d-a"
        assert [x.data_id for x in out] == ["d-a", "d-b"]


class TestMoves:
    """Moving a row between features behaves as removing and adding it."""

    def test_emptied_group_dropped(self):
        """A group whose last member moves out is dropped."""
        frame = pd.DataFrame(
            {"id": ["h0", "z"], "feature_id": ["h", None], "time": [1.0, 2.0]}
        )
        out = AnnotationSet(frame, dims=DIMS).update(annotation="h0", feature_id="")
        assert out.features.empty
        assert len(out) == 2

    def test_into_group(self, tracks):
        """A row moved into a group needs no order."""
        out = tracks.update(annotation="box", feature_id="e1")
        assert len(out["e1"].geometry.regions) == 3


class TestCollections:
    """Verbs on a set loaded from several, whose rows carry set labels."""

    def test_add(self, collection, tmp_path):
        """A new row belongs to the collection, and saves flat."""
        out = collection.add(pd.DataFrame({"time": [9.0]}))
        assert pd.isna(out.annotations["set"].iloc[-1])
        assert [x.data_id for x in out][-1] == ""
        assert _round_trip(out, tmp_path / "flat") == out

    def test_add_feature(self, collection, tmp_path):
        """A new feature belongs to the collection."""
        out = collection.add_feature("ev", time=[9.0], magnitude=1.0)
        assert out["ev"].set == ""
        assert _round_trip(out, tmp_path / "flat") == out

    def test_merge_plain(self, collection, tmp_path):
        """A plain set merges into a collection."""
        plain = AnnotationSet(pd.DataFrame({"time": [9.0]}), dims=DIMS)
        out = collection.merge(plain)
        assert out.attrs.sets == collection.attrs.sets
        assert _round_trip(out, tmp_path / "flat") == out

    def test_plain_merges_collection(self, collection):
        """Merging a collection into a plain set keeps every child."""
        plain = AnnotationSet(pd.DataFrame({"time": [9.0]}), dims=DIMS)
        out = plain.merge(collection)
        assert out.attrs.sets == collection.attrs.sets
        hand = out.attrs.sets["hand"].annotation_columns["note"]
        assert hand.description == "remark by hand"
        assert set(out.select(data_id="hand").annotations["time"]) == {1.0, 2.0}
        assert set(out.select(data_id="").annotations["time"]) == {9.0}

    def test_two_collections(self, collection, tmp_path):
        """Two collections merge into one holding all four children."""
        other = _collection(tmp_path / "other", ("x", "y"))
        out = collection.merge(other)
        assert set(out.attrs.sets) == {"hand", "auto", "x", "y"}
        flipped = other.merge(collection)
        assert out.attrs == flipped.attrs
        assert len(out.annotations) == len(flipped.annotations) == 8

    def test_select_skips_blank_labels(self, collection):
        """A collection-level row matches no child label, only a blank one."""
        added = collection.add(pd.DataFrame({"id": ["new"], "time": [9.0]}))
        assert "new" not in set(added.select(set="hand").annotations["id"])
        assert list(added.select(set="").annotations["id"]) == ["new"]

    def test_flat_save_writes_no_member_provenance(self, collection, tmp_path):
        """A flat save leaves member rows' provenance to their features."""
        stamped = collection.merge(
            AnnotationSet(pd.DataFrame({"time": [9.0]}), dims=DIMS, data_id="z")
        )
        reloaded = _round_trip(stamped, tmp_path / "flat")
        assert reloaded == stamped
        members = reloaded.annotations["feature_id"].notna()
        assert reloaded.annotations.loc[members, "data_id"].isna().all()
        assert reloaded["ev_hand"].data_id == "hand"

    def test_legacy_member_provenance_refused(self, collection, tmp_path):
        """A flat file whose member rows carry provenance is refused."""
        flat = collection.io.save(tmp_path / "flat")
        table = flat / "annotations.csv"
        frame = pd.read_csv(table)
        frame["data_id"] = frame["set"]
        frame.to_csv(table, index=False)
        with pytest.raises(InvalidAnnotationError, match="belongs on the feature"):
            dc.annotations(flat)

    def test_colliding_labels(self, collection, tmp_path):
        """A child label in both collections is refused."""
        other = _collection(tmp_path / "other", ("hand2", "auto"))
        with pytest.raises(ParameterError, match="set label 'auto'"):
            collection.merge(other.remove(annotation="auto0"))

    def test_header_only_child_keeps_columns(self, tmp_path):
        """A child holding only a header still contributes its columns."""
        root = tmp_path / "sets"
        empty = pd.DataFrame(
            {"time": pd.Series([], dtype=float), "phase": pd.Series([], dtype=str)}
        )
        AnnotationSet(empty, dims=DIMS).io.save(root / "empty")
        AnnotationSet(pd.DataFrame({"time": [1.0]}), dims=DIMS).io.save(root / "full")
        assert "phase" in dc.annotations(root).annotations.columns

    def test_basis_only_children_keep_label(self, tmp_path):
        """Children holding only curves still label the annotations table."""
        root = tmp_path / "sets"
        for name in ("a", "b"):
            AnnotationSet(dims=DIMS).add_path(f"m_{name}", basis=_moveout()).io.save(
                root / name
            )
        loaded = dc.annotations(root)
        assert "set" in loaded.annotations.columns


# One call of each verb on the tracks fixture.
VERBS = {
    "add": lambda x: x.add(pd.DataFrame({"time": [TIMES[2]]})),
    "add_feature": lambda x: x.add_feature("new", time=[TIMES[0]]),
    "adopt": lambda x: x.add_feature("new", members=["box"]),
    "add_path": lambda x: x.add_path("new", time=TIMES[:2], distance=[0.0, 1.0]),
    "add_polygon": lambda x: x.add_polygon(
        "new", rings=[{"time": TIMES, "distance": [0.0, 1.0, 0.0]}]
    ),
    "select": lambda x: x.select(velocity=(3.5, None)),
    "overlapping": lambda x: x.overlapping(distance=(20.0, 30.0)),
    "update": lambda x: x.update(annotation="v0", distance=2.0),
    "update_feature": lambda x: x.update(feature="t1", vehicle_type="car"),
    "remove": lambda x: x.remove(feature="t1"),
    "remove_row": lambda x: x.remove(annotation="p1"),
    "merge": lambda x: x.merge(
        AnnotationSet(dims=DIMS).add_feature("new", time=[TIMES[0]])
    ),
    "bounds": lambda x: x.bounds(),
}


class TestImmutability:
    """No verb changes the set it is called on."""

    @pytest.mark.parametrize("verb", sorted(VERBS))
    def test_input_unchanged(self, tracks, verb):
        """The input equals a fresh copy of itself after the verb."""
        before = AnnotationSet(
            tracks.annotations, features=tracks.features, attrs=tracks.attrs
        )
        frame, features = tracks.annotations, tracks.features
        VERBS[verb](tracks)
        assert tracks == before
        assert tracks.annotations.equals(frame)
        assert tracks.features.equals(features)

    def test_basis_set_unchanged(self, based):
        """Clearing a basis on the result leaves the input's."""
        based.update(annotation="m0", distance=1.0)
        assert based["near"].basis == _moveout()

    def test_line_basis(self):
        """A Line basis works as a path's curve too."""
        line = Line(start={"distance": 0.0}, end={"distance": 10.0})
        out = AnnotationSet(dims=DIMS).add_path("l", basis=line)
        row = out.bounds().iloc[0]
        assert (row["distance_min"], row["distance_max"]) == (0.0, 10.0)
