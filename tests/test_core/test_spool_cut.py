"""Tests for cutting a spool by an annotation set."""

from __future__ import annotations

import re
import time
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.io.core as io_core
import dascore.utils.chunk_plan as chunk_plan
from dascore.core.spool import Spool
from dascore.exceptions import CoordMergeError, ParameterError
from dascore.io.index.catalog import FileResolver
from dascore.io.index.planned import PlanResolver
from dascore.units import m
from dascore.warnings import DASCoreWarning

DIMS = ("distance", "time")
SECOND = pd.Timedelta("1s")


@pytest.fixture(scope="module")
def spool():
    """Three time-adjacent 8 s patches, 300 channels."""
    return dc.get_example_spool()


@pytest.fixture(scope="module")
def start(spool):
    """The spool's first time."""
    return spool.get_contents()["time_min"].min()


def _span(patch, dim="time"):
    """A patch's first and last coordinate along dim."""
    return [patch.coords.min(dim), patch.coords.max(dim)]


def _seconds(start, *offsets):
    """Absolute times offsets seconds after start."""
    return [np.datetime64(start + x * SECOND) for x in offsets]


def _tiles(spool, count):
    """The spool's patches split into count distance tiles each."""
    tiles = [(x[0], x[-1]) for x in np.array_split(np.arange(300), count)]
    return dc.spool([x.select(distance=d) for x in spool for d in tiles])


def _boxes(start):
    """A lone time range, then two boxes bounded in distance too."""
    ann = _ranges(start, (5, 12))
    ann = ann.add_path("box", time=_seconds(start, 5, 12), distance=[10.0, 250.0])
    return ann.add_path("b2", time=_seconds(start, 5, 12), distance=[95.0, 105.0])


def _line(time, distance=(0.0, 1.0), **attrs):
    """A patch of ones on the given float time and distance labels."""
    coords = {"distance": np.asarray(distance), "time": np.asarray(time)}
    shape = tuple(len(x) for x in coords.values())
    return dc.Patch(data=np.ones(shape), coords=coords, dims=DIMS, attrs=attrs)


def _ranges(start, *pairs, dims=DIMS):
    """A set of lone time ranges, given as (low, high) seconds after start."""
    lows, highs = zip(*pairs, strict=True)
    frame = {"time_min": _seconds(start, *lows), "time_max": _seconds(start, *highs)}
    return dc.AnnotationSet(pd.DataFrame(frame), dims=dims)


class TestCut:
    """Tests for Spool.cut."""

    def test_range(self, spool, start):
        """A range cuts to both ends, the half-open end's sample included."""
        cut = spool.cut(_ranges(start, (1, 2)), time=("-1s", "1s"))
        (patch,) = cut  # a pad leaves range ends alone
        assert _span(patch) == _seconds(start, 1, 2)
        assert patch.attrs.feature_id == "" and patch.attrs.annotation == 0
        contents = cut.get_contents()
        assert contents["feature_id"].tolist() == [""]
        assert contents["feature_id"].dtype == "str"

    def test_value_with_pad(self, spool, start):
        """A value is cut around by its pad, in any time spelling."""
        frame = pd.DataFrame({"time": [start + 10 * SECOND]})
        ann = dc.AnnotationSet(frame, dims=DIMS)
        for pad in [("-1s", "2s"), (-1, 2), (-SECOND, np.timedelta64(2, "s"))]:
            (patch,) = spool.cut(ann, time=pad)
            assert _span(patch) == _seconds(start, 9, 12)

    def test_numeric_value_with_pad(self, spool, start):
        """A numeric value takes numeric offsets."""
        ann = dc.AnnotationSet(pd.DataFrame({"distance": [10.0]}), dims=DIMS)
        out = spool.cut(ann, distance=(-2, 3))
        (patch,) = out  # merged along the spanned time too
        assert _span(patch, "distance") == [8, 13]
        assert _span(patch) == _seconds(start, 0, 23.996)

    def test_value_without_pad(self, spool, start):
        """A value with no pad is refused, naming it and the keyword, first."""
        ends = {"time_min": [start, None], "time_max": [start + SECOND, None]}
        frame = pd.DataFrame({**ends, "time": [None, start]})
        match = r"annotation 1 is a single time.*time="
        with mock.patch.object(Spool, "_build_chunk_plan") as plan:
            with pytest.raises(ParameterError, match=match):
                spool.cut(dc.AnnotationSet(frame, dims=DIMS))
        assert not plan.call_count

    def test_crossing_patches(self, spool, start):
        """A window crossing a file boundary is one patch."""
        (patch,) = spool.cut(_ranges(start, (5, 12)))
        assert _span(patch) == _seconds(start, 5, 12)

    def test_wide_seam_splits(self, spool, start):
        """A window joins as chunk does, so a seam 1.3 steps wide splits it."""
        first, second = spool[0], spool[1]
        shift = second.get_coord("time").step * 3 // 10
        late = second.update_coords(time_min=second.coords.min("time") + shift)
        pieces = dc.spool([first, late]).cut(_ranges(start, (5, 12)))
        spans = [_span(x) for x in pieces]
        assert spans == [
            _span(first.select(time=(start + 5 * SECOND, None))),
            _span(late.select(time=(None, start + 12 * SECOND))),
        ]

    def test_tiled(self, spool, start):
        """Patches split along two dims merge along both, in feature order."""
        cut, whole = (x.cut(ann := _boxes(start)) for x in (_tiles(spool, 2), spool))
        assert all(x.equals(y) for x, y in zip(cut, whole, strict=True))
        for tiled in cut, _tiles(spool, 3).cut(ann):  # 2x3 and 3x3 tiles
            assert list(tiled.get_contents()["feature_id"]) == ["box", "b2", ""]
        ends = {"distance_min": [100.0, 10.0], "distance_max": [200.0, 20.0]}
        cut = _tiles(spool, 2).cut(dc.AnnotationSet(pd.DataFrame(ends), dims=DIMS))
        assert cut.get_contents()["annotation"].tolist() == [0, 1]
        for pairs in [(100, 101)], [(1.001, 1.003)]:  # no data, no samples
            assert not len(_tiles(spool, 2).cut(_ranges(start, *pairs)))

    def test_mixed_dims_kept(self, spool, start):
        """A patch lacking the merged dim is cut on its own dims, not lost."""
        lone = spool[0].select(time=0, samples=True).squeeze("time")
        frame = pd.DataFrame({"distance_min": [10.0], "distance_max": [50.0]})
        cut = dc.spool([*spool, lone]).cut(dc.AnnotationSet(frame, dims=DIMS))
        assert sorted(x.shape for x in cut) == [(41,), (41, 6000)]
        lone = spool[0].select(distance=0, samples=True).squeeze("distance")
        cut = dc.spool([*_tiles(spool, 2), lone]).cut(_ranges(start, (1, 2)))
        assert sorted(x.shape for x in cut) == [(251,), (300, 251)]
        # a feature bounding distance skips the patch without it
        ann = _ranges(start, (1, 2)).add_path("d", distance=[10.0, 50.0])
        cut = dc.spool([*spool, lone]).cut(ann)
        stamps = [(x.attrs.feature_id, x.attrs.annotation, x.shape) for x in cut]
        assert stamps == [("d", None, (41, 6000)), ("", 0, (300, 251)), ("", 0, (251,))]

    def test_spanned_dim(self, spool, start):
        """A dimension the feature spans keeps its full extent."""
        (patch,) = spool.cut(_ranges(start, (0, 1)))
        assert patch.shape[patch.dims.index("distance")] == 300

    def test_group(self, spool, start):
        """A group is cut to its members' envelope, which a pad extends."""
        frame = pd.DataFrame({"time": _seconds(start, 7, 9), "feature_id": ["ev"] * 2})
        (patch,) = spool.cut(dc.AnnotationSet(frame, dims=DIMS))
        assert _span(patch) == _seconds(start, 7, 9)
        (patch,) = spool.cut(dc.AnnotationSet(frame, dims=DIMS), time=("-1s", "1s"))
        assert _span(patch) == _seconds(start, 6, 10)

    def test_path(self, spool, start):
        """A path is cut to its vertices' envelope."""
        ann = _ranges(start, (99, 100))
        ann = ann.add_path("p", time=_seconds(start, 2, 4), distance=[10.0, 50.0])
        (patch,) = spool.cut(ann, time=("-1s", "1s"))
        assert _span(patch) == _seconds(start, 1, 5)
        assert _span(patch, "distance") == [10, 50]

    def test_lone_row_and_feature(self, spool, start):
        """Outputs state feature_id, and a lone row its annotation label."""
        frame = pd.DataFrame(
            {
                "time_min": [None, *_seconds(start, 3)],
                "time_max": [None, *_seconds(start, 4)],
                "time": [*_seconds(start, 1), None],
                "feature_id": ["ev", None],
            }
        )
        cut = spool.cut(dc.AnnotationSet(frame, dims=DIMS), time=("-1s", "1s"))
        contents = cut.get_contents()
        stamped = contents[["feature_id", "annotation"]]
        assert list(contents["feature_id"]) == ["ev", ""]
        assert contents["annotation"].dtype == "Int64"
        assert contents["annotation"].isna().tolist() == [True, False]
        feature, lone = cut
        assert feature.attrs.feature_id == "ev" and feature.attrs.annotation is None
        assert isinstance(lone.attrs.annotation, int) and lone.attrs.annotation == 1
        # a cut of a cut states only the new stamps
        again = cut.cut(_ranges(start, (0.5, 1.5)))
        stamps = again.get_contents()[["feature_id", "annotation"]]
        assert stamps.values.tolist() == [["", 0]]
        assert again[0].attrs.feature_id == "" and again[0].attrs.annotation == 0
        ann = dc.AnnotationSet(frame.assign(feature_id="new"), dims=DIMS)
        (patch,) = spool.cut(_ranges(start, (3, 4))).cut(ann)
        assert patch.attrs.feature_id == "new" and patch.attrs.annotation is None
        assert _span(lone) == _seconds(start, 3, 4)
        # a plan over a cut keeps its stamps
        chunked = cut.chunk(time=None)
        assert chunked.get_contents()[["feature_id", "annotation"]].equals(stamped)
        assert [(x.attrs.feature_id, x.attrs.annotation) for x in chunked] == [
            ("ev", None),
            ("", 1),
        ]

    def test_union(self, spool, start):
        """A union keeps stamps only when every member has the same ones."""
        cut = spool.cut(_ranges(start, (3, 4)))
        assert (cut + cut).get_contents()["annotation"].dtype == "Int64"
        other = dc.spool([x.update_attrs(annotation="foo") for x in spool])
        contents = (cut + other).get_contents()
        assert contents["annotation"].tolist() == [0, "foo", "foo", "foo"]

    def test_overlapping_features(self, spool, start):
        """Two features with one window give two patches."""
        assert len(spool.cut(_ranges(start, (0, 1), (0, 1)))) == 2

    def test_nothing(self, spool, start):
        """No data, no samples, an empty range, or an empty set give nothing."""
        contents = spool.cut(_ranges(start, (100, 200))).get_contents()
        dtypes = contents[["feature_id", "annotation"]].dtypes
        assert not len(contents) and list(dtypes) == ["str", "Int64"]
        assert not len(spool.cut(_ranges(start, (1.001, 1.003), (7.997, 7.999))))
        assert not len(spool.cut(_ranges(start, (2, 2)), time=("-1s", "1s")))
        frame = pd.DataFrame({"distance_min": [10.2], "distance_max": [10.7]})
        assert not len(spool.cut(dc.AnnotationSet(frame, dims=DIMS)))
        assert not len(spool.cut(_ranges(start, (0, 1)).select(feature_id="none")))
        # nor does a window between the samples of a coordinate with no step
        irregular = dc.spool(_line([0.0, 1.0, 3.0, 7.0]))
        frame = pd.DataFrame({"time_min": [4.0], "time_max": [6.0]})
        assert not len(irregular.cut(dc.AnnotationSet(frame, dims=DIMS)))

    @pytest.mark.parametrize(
        "pads",
        [
            {"time": "1s"},
            {"time": ("-1", "1s")},
            {"distance": ("1m", 2)},
            {"time": ("1s", "-1s")},
            {"time": (None, 1)},
            {"time": ("-1x", "1s")},
            {"time": (np.datetime64("2020-01-01"), np.datetime64("2020-01-02"))},
            {"distance": (-np.timedelta64(1, "s"), np.timedelta64(1, "s"))},
        ],
    )
    def test_bad_pad(self, spool, start, pads):
        """A malformed pad is refused, naming its keyword, before planning."""
        ann = _ranges(start, (0, 1))
        with mock.patch.object(Spool, "_build_chunk_plan") as plan:
            with pytest.raises(ParameterError, match=f"{next(iter(pads))}="):
                spool.cut(ann, **pads)
        assert not plan.call_count

    def test_dim_spool_lacks(self, spool, start):
        """A dimension the spool lacks must be spanned, and takes no pad."""
        spanned = _ranges(start, (0, 1), dims=("depth", "time"))
        assert len(spool.cut(spanned)) == 1
        with pytest.raises(ParameterError, match="depth= must be"):
            spool.cut(spanned, depth=(0, 1))
        frame = spanned.annotations.drop(columns="feature_id").assign(depth=5.0)
        stated = dc.AnnotationSet(frame, dims=("depth", "time"))
        with pytest.raises(ParameterError, match="the spool lacks"):
            spool.cut(stated)
        # a coordinate which is no presented patch's dimension
        lat = dc.spool(dc.get_example_patch("random_patch_with_lat_lon"))
        frame = pd.DataFrame({"latitude_min": [0.0], "latitude_max": [1.0]})
        with pytest.raises(ParameterError, match="the spool lacks"):
            lat.cut(dc.AnnotationSet(frame, dims=("latitude", "time")))

    def test_box(self, spool, start):
        """A feature bounded in time and distance is the box select gives."""
        ann = _ranges(start, (5, 12))
        ann = ann.add_path("box", time=_seconds(start, 5, 12), distance=[10.0, 50.0])
        patch, _ = spool.cut(ann)  # features come before lone rows
        times = tuple(_seconds(start, 5, 12))
        expected = spool.select(time=times, distance=(10, 50)).chunk(time=None)[0]
        assert patch.equals(expected.update_attrs(feature_id="box", annotation=None))

    def test_many_in_one_plan(self, spool, start):
        """Many features take one plan, not one per feature, in order."""
        times = [start + x * SECOND / 10 for x in range(10, 210)]
        ann = dc.AnnotationSet(pd.DataFrame({"time": times}), dims=DIMS)
        wrapped = mock.patch(
            "dascore.core.spool.build_chunk_plan", wraps=chunk_plan.build_chunk_plan
        )
        planner = mock.patch(
            "dascore.core.spool.window_plan", wraps=chunk_plan.window_plan
        )
        with wrapped as joins, planner as plans:
            contents = spool.cut(ann, time=("-1s", "1s")).get_contents()
        # one plan of every feature, then one join, along time, of those
        # crossing files; none along distance, where each is already whole
        assert plans.call_count == joins.call_count == 1
        assert contents["annotation"].tolist() == list(range(200))

    def test_stamps_after_dropped(self, spool, start):
        """Features holding no data are dropped, the rest keep their stamps."""
        cut = spool.cut(_ranges(start, (100, 101), (1, 2), (1.001, 1.003), (3, 4)))
        stamps = cut.get_contents()["annotation"].tolist()
        assert stamps == [1, 3] and [x.attrs.annotation for x in cut] == stamps

    def test_offset_grids(self):
        """A window past one grid keeps the offset grid's sample in it."""
        spool = dc.spool([_line(np.arange(0.0, 11)), _line(np.arange(0.5, 11))])
        frame = pd.DataFrame({"time_min": [10.1], "time_max": [10.6]})
        (patch,) = spool.cut(dc.AnnotationSet(frame, dims=DIMS))
        (windowed,) = spool.cut(time=np.array([[10.1, 10.6]]))
        assert _span(patch) == _span(windowed) == [10.5, 10.5]

    def test_trim_makes_alike(self):
        """Patches whose trims agree along the other dim merge into one."""
        first = _line(np.arange(10.0), range(101))
        spool = dc.spool([first, _line(np.arange(10.0, 20), range(201))])
        frame = pd.DataFrame({"distance_min": [10.0], "distance_max": [50.0]})
        (patch,) = spool.cut(dc.AnnotationSet(frame, dims=DIMS))
        assert patch.shape == (41, 20)

    def test_nested_conflict(self):
        """A source nested in another must still agree with it."""
        inner = _line(np.arange(2.0, 5), instrument_id="Z")
        spool = dc.spool([_line(np.arange(10.0), instrument_id="A"), inner])
        frame = pd.DataFrame({"time_min": [1.0], "time_max": [5.0]})
        with pytest.raises(CoordMergeError, match="instrument_id"):
            spool.cut(dc.AnnotationSet(frame, dims=DIMS))

    def test_piece_order(self, spool, start):
        """A feature's pieces from unlike patches come in coordinate order."""
        for middle in spool[1].decimate(time=2), spool[1].transpose():
            cut = dc.spool([spool[0], middle, spool[2]]).cut(_ranges(start, (5, 20)))
            starts = cut.get_contents()["time_min"]
            assert len(starts) > 1 and starts.is_monotonic_increasing

    def test_no_deprecation(self, spool, start):
        """Cutting time boxes warns of nothing NumPy will remove."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert len(spool.cut(_boxes(start)))

    def test_hole_in_one_patch(self, holed_patch):
        """A hole inside one patch splits a window across it; inside it, none."""
        frame = pd.DataFrame({"time_min": [50.0, 70.0], "time_max": [110.0, 90.0]})
        cut = dc.spool(holed_patch).cut(dc.AnnotationSet(frame, dims=DIMS))
        assert [_span(x) for x in cut] == [[50, 59], [100, 110]]
        # a feature bounding the other dim too, or only it, is still split
        box = frame.iloc[:1].assign(distance_min=-1.0, distance_max=1.0)
        cut = dc.spool(holed_patch).cut(dc.AnnotationSet(box, dims=DIMS))
        assert [_span(x) for x in cut] == [[50, 59], [100, 110]]
        box = box.drop(columns=["time_min", "time_max"])
        cut = dc.spool(holed_patch).cut(dc.AnnotationSet(box, dims=DIMS))
        assert [_span(x) for x in cut] == [[0, 59], [100, 159]]

    @pytest.mark.parametrize(
        ("time", "samples", "window", "span"),
        [
            # a pending sample selection still bounds what a cut keeps
            (np.arange(40.0), (2, -3), [0.0, 5.0], [2.0, 5.0]),
            # a row with no step names no grid, so a window over it keeps it
            (np.array([0.0, 1.0, 2.5, 7.0, 8.0]), None, [2.0, 7.5], [2.5, 7.0]),
        ],
    )
    def test_window_on_float_rows(self, time, samples, window, span):
        """A cut keeps the samples a window holds, whatever the row's grid."""
        coords = {"distance": [0.0], "time": time}
        patch = dc.Patch(data=np.ones((1, len(time))), coords=coords, dims=DIMS)
        view = dc.spool(patch)
        view = view if samples is None else view.select(time=samples, samples=True)
        frame = pd.DataFrame({"time_min": window[:1], "time_max": window[1:]})
        (cut,) = view.cut(dc.AnnotationSet(frame, dims=DIMS))
        assert _span(cut) == span

    def test_lazy(self, spool, start, tmp_path):
        """Cutting and reading contents load nothing; reading the data does."""
        dc.examples.spool_to_directory(spool, path=tmp_path)
        directory = dc.spool(tmp_path).update()
        plan = mock.patch.object(
            PlanResolver, "resolve", autospec=True, side_effect=PlanResolver.resolve
        )
        file = mock.patch.object(io_core, "_array_reader", wraps=io_core._array_reader)
        source = mock.patch.object(
            FileResolver, "resolve", autospec=True, side_effect=FileResolver.resolve
        )
        with plan as calls, file as reads, source as sources:
            cut = directory.cut(_ranges(start, (5, 12)))
            cut.get_contents()
            assert calls.call_count == reads.call_count == sources.call_count == 0
            (patch,) = cut
            assert calls.call_count and patch.data.size and reads.call_count
        assert _span(patch) == _seconds(start, 5, 12)


def _late(patch, steps):
    """A patch moved `steps` of its own steps later, with no history."""
    time = patch.get_coord("time")
    shift = (time.step * round(steps * 1000)) // 1000
    moved = patch.update_coords(time_min=time.min() + shift)
    return moved.update(attrs=moved.attrs.update(history=[]))


def _windows(start, *pairs):
    """An (n, 2) array of absolute times, pairs of seconds after start."""
    return np.array([_seconds(start, *x) for x in pairs])


class TestChunkRefusesWindows:
    """Explicit windows have left chunk for cut (and select)."""

    @pytest.mark.parametrize("method", ["chunk", "chunk_plan"])
    @pytest.mark.parametrize("nested", [False, True])
    def test_windows_refused(self, spool, start, method, nested):
        """An (n, 2) window array is refused, naming cut and select."""
        windows = _windows(start, (1, 2))
        windows = windows.tolist() if nested else windows
        match = r"Spool\.cut.*Spool\.select"
        with pytest.raises(ParameterError, match=match):
            getattr(spool, method)(time=windows)

    @pytest.mark.parametrize("method", ["chunk", "chunk_plan"])
    def test_no_on_incomplete(self, spool, method):
        """on_incomplete is refused as removed, pointing to cut."""
        match = "on_incomplete was removed.*Spool.cut"
        with pytest.raises(ParameterError, match=match):
            getattr(spool, method)(time=4, on_incomplete="ignore")


class TestCutWindows:
    """Spool.cut by explicit (n, 2) windows."""

    def test_one_output_per_window(self, spool, start):
        """Each window joins across files into one output, in window order."""
        cut = spool.cut(time=_windows(start, (5, 12), (1, 2)))
        assert [_span(x) for x in cut] == [
            _seconds(start, 5, 12),
            _seconds(start, 1, 2),
        ]
        assert cut.get_contents()["window"].tolist() == [0, 1]
        assert cut.get_contents()["window"].dtype == "Int64"
        assert [x.attrs.window for x in cut] == [0, 1]
        bounds = tuple(_seconds(start, 5, 12))
        expected = spool.select(time=bounds).chunk(time=None)[0]
        assert cut[0].equals(expected.update_attrs(window=0))

    def test_split_window_is_labelled(self, spool, start):
        """A window split by a gap gives fragments labelled by their window."""
        first, late = spool[0], _late(spool[1], 0.3)
        cut = dc.spool([first, late]).cut(time=_windows(start, (1, 2), (5, 12)))
        assert cut.get_contents()["window"].tolist() == [0, 1, 1]
        assert [_span(x) for x in cut][1:] == [
            _span(first.select(time=(start + 5 * SECOND, None))),
            _span(late.select(time=(None, start + 12 * SECOND))),
        ]

    def test_tolerance(self, spool, start):
        """Tolerance joins a seam 1.3 steps wide, as chunk's does."""
        source = dc.spool([spool[0], _late(spool[1], 0.3)])
        windows = _windows(start, (5, 12))
        assert len(source.cut(time=windows)) == 2
        assert len(source.cut(time=windows, tolerance=1.25)) == 2
        (patch,) = source.cut(time=windows, tolerance=1.35)
        assert patch.get_coord("time").evenly_sampled
        late = _ranges(start, (5, 12))
        assert len(source.cut(late, tolerance=1.25)) == 2
        assert len(source.cut(late, tolerance=1.35)) == 1

    def test_max_shift(self, spool, start):
        """max_shift bounds the seam, as chunk's does, on either side of 0.15."""
        source = dc.spool([spool[0], _late(spool[1], 0.15)])
        windows = _windows(start, (5, 12))
        assert len(source.cut(time=windows)) == 1
        assert len(source.cut(time=windows, max_shift=0.14)) == 2
        assert len(source.cut(time=windows, max_shift=0.16)) == 1
        assert len(source.cut(_ranges(start, (5, 12)), max_shift=0.14)) == 2

    def test_irregular_samples(self):
        """Windows over labels with no step keep the samples inside them."""
        spool = dc.spool(_line([0.0, 1.0, 3.0, 6.0, 9.0]))
        cut = spool.cut(time=np.array([[2.0, 7.0], [4.0, 5.0], [-1.0, 0.5]]))
        assert [_span(x) for x in cut] == [[3.0, 6.0], [0.0, 0.0]]
        assert cut.get_contents()["window"].tolist() == [0, 2]

    def test_refusals(self, spool, start):
        """Bad window arguments are refused before planning."""
        cases = [
            ({"time": 4}, "(n, 2)"),
            ({"time": _windows(start, (1, 2)), "distance": np.ones((2, 2))}, "same"),
            ({"distance": (2, 4)}, "[[start, stop]]"),
            ({"depth": np.array([[0, 1]])}, "dimensions only"),
            ({}, "got no windows"),
            ({"tolerance": np.ones((1, 2))}, "got no windows"),
        ]
        for kwargs, match in cases:
            with mock.patch("dascore.core.spool.window_plan") as plan:
                with pytest.raises(ParameterError, match=re.escape(match)):
                    spool.cut(**kwargs)
            assert not plan.call_count
        with pytest.raises(ParameterError, match="cut pad time="):
            spool.cut(_ranges(start, (1, 2)), time=_windows(start, (1, 2)))
        for positional in (_windows(start, (1, 2)), [[1, 2]]):
            with pytest.raises(ParameterError, match="windows are keywords"):
                spool.cut(positional)

    def test_many_windows_planned_at_once(self):
        """2,000 windows over 200 patches plan in one pass, reading no data."""
        source = dc.examples.random_spool(length=200, shape=(2, 1000))
        contents = source.get_contents()
        low, high = contents["time_min"].min(), contents["time_max"].max()
        starts = low + (high - low) * np.linspace(0, 0.98, 2000)
        windows = np.stack([starts, starts + (high - low) / 300], axis=1)
        planner = mock.patch(
            "dascore.core.spool.window_plan", wraps=chunk_plan.window_plan
        )
        # windows pair with sources in one array step, not one per window,
        # and each source's labels are searched once, not once per window
        calls, original = [], chunk_plan._window_edges

        def edges(*args):
            *rest, bounds = original(*args)
            return *rest, lambda *x: (calls.append(1), bounds(*x))[1]

        pairing = mock.patch.object(chunk_plan, "_window_edges", side_effect=edges)
        search = mock.patch.object(
            chunk_plan, "_sample_bounds", wraps=chunk_plan._sample_bounds
        )
        loads = mock.patch.object(
            PlanResolver, "resolve", autospec=True, side_effect=PlanResolver.resolve
        )
        tic = time.perf_counter()
        with planner as plans, pairing, search as searches, loads as resolved:
            cut = source.cut(time=windows)
            assert len(cut.get_contents()) >= 2000
        # per dim (time, and distance left open), one pairing and one clip
        assert plans.call_count == 1 and len(calls) == 4 and not resolved.call_count
        assert searches.call_count <= len(source)
        assert time.perf_counter() - tic < 60
        assert sorted(set(cut.get_contents()["window"])) == list(range(2000))

    def test_mixed_units(self):
        """Pieces in metres and feet load, as contents say, in chunk's units."""
        patch = dc.get_example_patch().set_units(distance="m")
        time = patch.get_coord("time")
        later = patch.update_coords(time_min=time.max() + time.step)
        spool = dc.spool([patch, later.convert_units(distance="ft")])
        windows = np.array([[10 * m, 20 * m]], dtype=object)
        cut = spool.cut(distance=windows)
        rows = cut.get_contents().to_dict("records")
        assert len(cut) == len(rows) == 2
        for row, got in zip(rows, cut, strict=True):
            ends = (row["distance_min"], row["distance_max"])
            assert list(ends) == _span(got, "distance")
        metres, feet = cut
        assert str(metres.get_coord("distance").units) == "1 m"
        assert metres.get_coord("distance").values.tolist() == list(range(10, 21))
        # each window held in one piece keeps its source's labels and units
        assert str(feet.get_coord("distance").units) == "1 ft"
        values = feet.get_coord("distance").values * 0.3048
        np.testing.assert_allclose(values, np.arange(10, 21))

    def test_window_attr_overwritten(self, spool, start, tmp_path):
        """A cut replaces any window label, lazy, loaded or read from files."""
        named = dc.spool([spool[0].update_attrs(window="mine"), *spool[1:]])
        windows = _windows(start, (1, 2), (7, 9))
        assert [x.attrs.window for x in named.cut(time=windows)] == [0, 1]
        loaded = dc.spool(list(spool.cut(time=windows)))
        assert loaded.cut(time=windows).get_contents()["window"].tolist() == [0, 1]
        for num, patch in enumerate(spool.cut(time=windows)):
            patch.io.write(tmp_path / f"{num}.h5", "DASDAE")
        files = dc.spool(tmp_path).update(progress=None)
        assert [x.attrs.window for x in files.cut(time=windows)] == [0, 1]
        # a cut's own label is replaced by a later cut's
        again = spool.cut(time=_windows(start, (1, 3))).cut(
            time=_windows(start, (5, 6), (1, 2))
        )
        assert again.get_contents()["window"].tolist() == [1]

    def test_rejoin_needs_conflict(self, spool, start):
        """Cut outputs differ in window, so rejoining them needs conflict=."""
        cut = spool.cut(time=_windows(start, (1, 4), (4.004, 9)))
        with pytest.raises(CoordMergeError, match="window"):
            cut.chunk(time=None)
        (patch,) = cut.chunk(time=None, conflict="drop")
        assert _span(patch) == _seconds(start, 1, 9)

    def test_second_join_along_the_first_dim(self):
        """Pieces lining up only after joining another dim become one patch."""
        patch = dc.get_example_patch()
        time = patch.get_coord("time")
        middle = time.values[len(time) // 2]
        first = patch.select(distance=(0, 149), time=(None, middle - time.step))
        second = patch.select(distance=(0, 149), time=(middle, None))
        rest = patch.select(distance=(150, 299))
        spool = dc.spool([first, second, rest])
        windows = {
            "time": np.array([[time.min(), time.max()]]),
            "distance": np.array([[0, 299]]),
        }
        (out,) = spool.cut(**windows)
        assert out.equals(patch.update_attrs(window=0))

    def test_joins_only_bounded_dims(self):
        """Tolerance and max_shift reach only the dimensions windows bound."""
        patch = dc.get_example_patch()
        time = patch.get_coord("time")
        spool = dc.spool(
            [patch.select(distance=(0, 149)), patch.select(distance=(151, 299))]
        )
        windows = np.array([[time.min(), time.values[100]]])
        assert len(spool.cut(time=windows, tolerance=3)) == 2
        both = dict(distance=np.array([[0, 299]]), tolerance=3)
        with pytest.warns(UserWarning, match="tolerance"):
            (patch,) = spool.cut(time=windows, **both)
        assert patch.shape == (299, 101)

    def test_hole_on_second_dim(self, holed_patch):
        """A window on one dim splits at a hole on another."""
        spool = dc.spool(holed_patch)
        cut = spool.cut(distance=np.array([[-1.0, 1.0]]))
        assert [_span(x) for x in cut] == [[0, 59], [100, 159]]
        boxed = spool.cut(
            time=np.array([[50.0, 110.0]]), distance=np.array([[None, 1]])
        )
        assert [_span(x) for x in boxed] == [[50, 59], [100, 110]]
        assert boxed.get_contents()["window"].tolist() == [0, 0]

    def test_other_kind_of_cut_relabels(self, spool, start):
        """A cut of a cut blanks the other kind's labels, in either order."""
        overlapping = _windows(start, (0, 20), (3, 5))
        ann = _ranges(start, (4, 12))
        first = spool.cut(time=overlapping).cut(ann)
        stamps = first.get_contents()[["feature_id", "annotation", "window"]]
        # the two windows' parts overlap and join into the annotation's one patch
        assert stamps.astype(object).values.tolist() == [["", 0, pd.NA]]
        assert [x.attrs.window for x in first] == [None]
        second = spool.cut(_ranges(start, (0, 20), (3, 5))).cut(
            time=_windows(start, (1, 13))
        )
        stamps = second.get_contents()[["feature_id", "annotation", "window"]]
        assert stamps.astype(object).values.tolist() == [["", pd.NA, 0]]
        assert [x.attrs.annotation for x in second] == [None]

    def test_conflict(self, spool, start):
        """Conflict has chunk's meaning, here for attrs each file states."""
        own = dc.spool(
            [x.update_attrs(instrument_id=str(n)) for n, x in enumerate(spool)]
        )
        windows = _windows(start, (5, 12))
        with pytest.raises(CoordMergeError, match="instrument_id"):
            own.cut(time=windows)
        (patch,) = own.cut(time=windows, conflict="keep_first")
        assert patch.attrs.instrument_id == "0"
        assert _span(patch) == _seconds(start, 5, 12)

    def test_unbounded_dim_keeps_sources_apart(self):
        """Sources unlike on a dim the windows leave open stay apart, as in chunk."""
        first = dc.Patch(
            np.ones((5, 4)),
            coords={"x": np.arange(5.0), "y": [0, 1, 3, 5]},
            dims=("x", "y"),
        )
        second = dc.Patch(
            np.ones((5, 4)),
            coords={"x": np.arange(5.0, 10), "y": [0, 2, 4, 5]},
            dims=("x", "y"),
        )
        spool = dc.spool([first, second])
        cut = spool.cut(x=np.array([[0, 9]]))
        assert len(cut) == len(spool.chunk(x=None)) == 2
        assert [x.get_coord("y").values.tolist() for x in cut] == [
            [0, 1, 3, 5],
            [0, 2, 4, 5],
        ]

    def test_integer_tiles_rejoin(self):
        """Integer pieces joined along one dim still join a passed-through one."""
        data = np.arange(4 * 5 * 12, dtype=float).reshape(4, 5, 12)
        coords = {x: np.arange(n, dtype=np.int64) for x, n in zip("abc", (4, 5, 12))}
        whole = dc.Patch(data=data, coords=coords, dims=("a", "b", "c"))
        parts = [
            whole.select(c=(0, 5)),
            whole.select(b=(0, 2), c=(6, 11)),
            whole.select(b=(3, 4), c=(6, 11)),
        ]
        (out,) = dc.spool(parts).cut(c=np.array([[1, 10]]), b=np.array([[1, 4]]))
        expected = whole.select(c=(1, 10), b=(1, 4))
        assert out.get_coord("b").values.dtype == np.int64
        np.testing.assert_array_equal(out.data, expected.data)
        joined = dc.spool(parts).chunk(b=None)
        assert set(joined._df["_b_coord_dtype"]) == {"int64"}

    def test_join_along_open_dim(self):
        """Tiles split on a dim the windows leave open still join, as chunk does."""
        patch = dc.get_example_patch()
        time = patch.get_coord("time")
        middle = time.values[len(time) // 2]
        parts = [
            patch.select(time=(None, middle - time.step)),
            patch.select(time=(middle, None), distance=(0, 149)),
            patch.select(time=(middle, None), distance=(150, 299)),
        ]
        spool = dc.spool(parts)
        low, high = time.values[100], time.values[-100]
        (out,) = spool.cut(time=np.array([[low, high]]))
        expected = spool.chunk(distance=None).chunk(time=None)[0]
        assert out.equals(expected.select(time=(low, high)).update_attrs(window=0))

    def test_unit_tolerance_on_its_own_dims(self):
        """A tolerance with units applies only along dims of its kind."""
        patch = dc.get_example_patch()
        time = patch.get_coord("time")
        middle = time.values[len(time) // 2]
        quads = [
            patch.select(time=t, distance=d)
            for t in ((None, middle - time.step), (middle, None))
            for d in ((0, 149), (150, 299))
        ]
        spool = dc.spool(quads)
        windows = dict(time=np.array([[time.values[10], time.values[-10]]]))
        windows["distance"] = np.array([[10.0, 200.0]])
        for tolerance in (np.timedelta64(10, "ms"), 2 * m):
            (out,) = spool.cut(**windows, tolerance=tolerance)
            assert out.shape == (191, len(time) - 19)

    def test_float_pieces_one_sample_apart(self):
        """Pieces starting a sample apart keep their own first labels."""
        first = _line([0.0], np.arange(5.0))
        second = _line([7.0], np.arange(1.0, 6.0))
        cut = dc.spool([first, second]).cut(distance=np.array([[0, 5]]))
        rows = cut.get_contents()[["distance_min", "distance_max"]]
        assert rows.to_numpy().tolist() == [_span(x, "distance") for x in cut]
        assert rows.to_numpy().tolist() == [[0, 4], [1, 5]]

    def test_non_dimensional_coordinate_refused(self):
        """Cut windows dimensions only, saying select takes other coordinates."""
        spool = dc.spool(dc.get_example_patch("random_patch_with_lat_lon"))
        windows = np.array([[-200, -109.7]])
        assert len(spool.select(latitude=windows)) == 1
        with pytest.raises(ParameterError, match=r"dimensions only.*Spool\.select"):
            spool.cut(latitude=windows)

    def test_earlier_source_owns_overlap(self):
        """The earlier source wins an overlap, whatever the window (rule 6)."""

        def patch(x, value):
            data = np.full(len(x), value, dtype=np.float64)
            return dc.Patch(data=data, coords={"x": x}, dims=("x",))

        first = patch(np.arange(10.0), 1)
        later = patch(0.08 + 0.96 * np.arange(10), 2)
        spool = dc.spool([first, later])
        for window, labels in (([1.01, 4], [2, 3, 4]), ([0, 4], [0, 1, 2, 3, 4])):
            (out,) = spool.cut(x=np.array([window]))
            assert out.get_coord("x").values.tolist() == labels
            assert (out.data == 1).all()

    def test_bare_bounds_in_each_patch_units(self):
        """Bare numbers mean each patch's own units, as in select."""

        def patch(x, units):
            coords = {
                "x": dc.get_coord(data=np.asarray(x, dtype=np.float64), units=units)
            }
            return dc.Patch(data=np.ones(len(x)), coords=coords, dims=("x",))

        spool = dc.spool([patch(range(5), "m"), patch(range(10, 15), "ft")])
        windows = np.array([[10, 12]])
        (out,) = spool.cut(x=windows)
        (picked,) = spool.select(x=windows)
        assert out.get_coord("x").values.tolist() == [10, 11, 12]
        assert str(out.get_coord("x").units) == "1 ft"
        assert out.get_coord("x") == picked.get_coord("x")
        # each patch's samples chosen in its own units, then joined as chunk
        # joins them, in one unit
        spool = dc.spool(
            [patch(np.arange(100) / 0.3048, "ft"), patch(range(100, 200), "m")]
        )
        cut = spool.cut(x=np.array([[50, 150]]))
        joined = spool.select(x=np.array([[50, 150]])).chunk(x=None)
        for got, expected in zip(cut, joined, strict=True):  # converted on load
            got, expected = got.get_coord("x"), expected.get_coord("x")
            assert str(got.units) == str(expected.units)
            np.testing.assert_allclose(got.values, expected.values, rtol=1e-12)
        metres = [x.convert_units(x="m").get_coord("x") for x in cut]
        np.testing.assert_allclose(
            [(x.min(), x.max()) for x in metres], [[16, 45], [100, 150]]
        )

    @pytest.mark.parametrize("conflict", ["keep_first", "drop"])
    def test_conflicting_coordinates_follow_conflict(
        self, conflict, channel_conflict_pair
    ):
        """A window across sources whose channels differ follows `conflict`."""
        first, second = channel_conflict_pair
        spool = dc.spool([first, second])
        time = first.get_coord("time")
        windows = np.array([[time.values[-10], second.get_coord("time").values[10]]])
        with pytest.raises(CoordMergeError, match="channel"):
            list(spool.cut(time=windows))
        cut = spool.cut(time=windows, conflict=conflict)
        with pytest.warns(DASCoreWarning):  # loading settles the channels
            (out,) = cut
        if conflict == "keep_first":
            np.testing.assert_array_equal(
                out.get_coord("channel").values, first.get_coord("channel").values
            )
        else:
            assert "channel" not in out.coords.coord_map

    def test_crossing_window_reads_arrays(self, spool, start, tmp_path):
        """A window across files loads each file's part with array reads."""
        dc.examples.spool_to_directory(spool, path=tmp_path)
        directory = dc.spool(tmp_path).update()
        cut = directory.cut(time=_windows(start, (5, 12)))
        arrays = mock.patch.object(
            io_core, "_array_reader", wraps=io_core._array_reader
        )
        with arrays as reads, mock.patch.object(dc, "read", wraps=dc.read) as patches:
            (patch,) = cut
        assert reads.call_count == 2 and not patches.call_count  # one per file
        assert _span(patch) == _seconds(start, 5, 12)
