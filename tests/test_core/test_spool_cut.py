"""Tests for cutting a spool by an annotation set."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.io.index.planned import PlanResolver

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


def _ranges(start, *pairs, dims=DIMS):
    """A set of lone time ranges, given as (low, high) seconds after start."""
    lows, highs = zip(*pairs, strict=True)
    frame = {"time_min": _seconds(start, *lows), "time_max": _seconds(start, *highs)}
    return dc.AnnotationSet(pd.DataFrame(frame), dims=dims)


class TestCut:
    """Tests for Spool.cut."""

    def test_range(self, spool, start):
        """A range cuts to both ends, the half-open end's sample included."""
        (patch,) = spool.cut(_ranges(start, (1, 2)))
        assert _span(patch) == _seconds(start, 1, 2)

    def test_value_with_pad(self, spool, start):
        """A value is cut around by its pad, in any time spelling."""
        frame = pd.DataFrame({"time": [start + 10 * SECOND]})
        ann = dc.AnnotationSet(frame, dims=DIMS)
        for pad in [("-1s", "2s"), (-1, 2), (-SECOND, np.timedelta64(2, "s"))]:
            (patch,) = spool.cut(ann, time=pad)
            assert _span(patch) == _seconds(start, 9, 12)

    def test_numeric_value_with_pad(self, spool):
        """A numeric value takes numeric offsets."""
        ann = dc.AnnotationSet(pd.DataFrame({"distance": [10.0]}), dims=DIMS)
        out = spool.cut(ann, distance=(-2, 3))
        assert len(out) == 3  # time is spanned, so nothing merges along it
        assert all(_span(x, "distance") == [8, 13] for x in out)

    def test_value_without_pad(self, spool, start):
        """A value with no pad is refused, naming the feature and keyword."""
        frame = pd.DataFrame({"time": [start], "feature_id": ["ev"]})
        with pytest.raises(ParameterError, match=r"ev is a single time.*time="):
            spool.cut(dc.AnnotationSet(frame, dims=DIMS))

    def test_crossing_patches(self, spool, start):
        """A window crossing a file boundary is one patch."""
        (patch,) = spool.cut(_ranges(start, (5, 12)))
        assert _span(patch) == _seconds(start, 5, 12)

    def test_spanned_dim(self, spool, start):
        """A dimension the feature spans keeps its full extent."""
        (patch,) = spool.cut(_ranges(start, (0, 1)))
        assert patch.shape[patch.dims.index("distance")] == 300

    def test_group(self, spool, start):
        """A group is cut to its members' envelope."""
        frame = pd.DataFrame({"time": _seconds(start, 7, 9), "feature_id": ["ev"] * 2})
        (patch,) = spool.cut(dc.AnnotationSet(frame, dims=DIMS))
        assert _span(patch) == _seconds(start, 7, 9)

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
        assert list(contents["feature_id"]) == ["ev", ""]
        assert contents["annotation"].isna().tolist() == [True, False]
        assert contents["annotation"].iloc[1] == 1
        feature, lone = cut
        assert feature.attrs.feature_id == "ev"
        assert "annotation" not in feature.attrs.model_dump()
        assert lone.attrs.annotation == 1
        assert _span(lone) == _seconds(start, 3, 4)

    def test_overlapping_features(self, spool, start):
        """Two features with one window give two patches."""
        assert len(spool.cut(_ranges(start, (0, 1), (0, 1)))) == 2

    def test_nothing(self, spool, start):
        """A feature overlapping no data, or an empty set, gives nothing."""
        assert not len(spool.cut(_ranges(start, (100, 200))))
        assert not len(spool.cut(_ranges(start, (0, 1)).select(feature_id="none")))

    def test_dim_spool_lacks(self, spool, start):
        """A dimension the spool lacks must be spanned, and takes no pad."""
        spanned = _ranges(start, (0, 1), dims=("depth", "time"))
        assert len(spool.cut(spanned)) == 1
        with pytest.raises(ParameterError, match="not dimensions of both"):
            spool.cut(spanned, depth=(0, 1))
        frame = spanned.annotations.drop(columns="feature_id").assign(depth=5.0)
        stated = dc.AnnotationSet(frame, dims=("depth", "time"))
        with pytest.raises(ParameterError, match="the spool lacks"):
            spool.cut(stated)

    def test_lazy(self, spool, start, tmp_path):
        """Cutting and reading contents load nothing; iterating does."""
        dc.examples.spool_to_directory(spool, path=tmp_path)
        directory = dc.spool(tmp_path).update()
        resolve = PlanResolver.resolve
        with mock.patch.object(
            PlanResolver, "resolve", autospec=True, side_effect=resolve
        ) as calls:
            cut = directory.cut(_ranges(start, (5, 12)))
            cut.get_contents()
            assert calls.call_count == 0
            (patch,) = cut
            assert calls.call_count
        assert _span(patch) == _seconds(start, 5, 12)
