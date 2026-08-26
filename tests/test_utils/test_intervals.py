"""Tests for half-open interval helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from dascore.core import annotations, inventory
from dascore.core.annotations import AnnotationSet
from dascore.exceptions import ParameterError
from dascore.utils.intervals import (
    clip_intervals,
    interval_masks,
    intervals_overlap,
    normalize_value,
    value_kind,
)


class _Span(BaseModel):
    """A minimal interval item with the default field names."""

    distance_min: float
    distance_max: float
    label: str = ""


class _Window(BaseModel):
    """An interval item naming its bounds something else."""

    time_start: float
    time_end: float


class TestIntervalMasks:
    """Coverage is half-open apart from the end of a run."""

    def test_half_open(self):
        """The start is covered and everything past the end is not."""
        (mask,) = interval_masks([-1, 0, 1, 3], [(0, 2)])
        assert list(mask) == [False, True, True, False]

    def test_run_end_included(self):
        """The last value of a coverage run belongs to the interval ending there."""
        first, second = interval_masks([0, 1, 2, 3], [(0, 2), (2, 3)])
        assert list(first) == [True, True, False, False]
        assert list(second) == [False, False, True, True]

    def test_claimed_end_not_shared(self):
        """A value another interval already claims stays with that interval."""
        first, second = interval_masks([0, 1, 2], [(0, 2), (2, 4)])
        assert list(first) == [True, True, False]
        assert list(second) == [False, False, True]

    def test_point_marker_covers_nothing(self):
        """Equal start and end cover no values at all."""
        (mask,) = interval_masks([0, 1, 2], [(1, 1)])
        assert not mask.any()

    def test_returns_one_mask_per_interval(self):
        """Every interval gets a mask, in the order it was given."""
        masks = interval_masks([0, 1], [(0, 1), (5, 6), (1, 1)])
        assert len(masks) == 3
        assert all(len(x) == 2 for x in masks)


class TestIntervalsOverlap:
    """The first overlapping pair, or None."""

    def test_touching_do_not_overlap(self):
        """Half-open intervals sharing an endpoint are disjoint."""
        assert intervals_overlap([(0, 2), (2, 4)]) is None

    def test_overlap_found(self):
        """An overlapping pair comes back in sorted order."""
        assert intervals_overlap([(2, 4), (0, 3)]) == ((0, 3), (2, 4))

    def test_point_markers_ignored(self):
        """A point marker covers nothing so it cannot overlap."""
        assert intervals_overlap([(0, 2), (1, 1)]) is None

    def test_empty(self):
        """Nothing to compare means no overlap."""
        assert intervals_overlap([]) is None


class TestClipIntervals:
    """Clipping keeps coverage, drops what falls outside, and keeps points."""

    def test_clipped_to_bounds(self):
        """An interval straddling the clip is trimmed to it."""
        (out,) = clip_intervals([_Span(distance_min=0, distance_max=10)], 2, 6)
        assert (out.distance_min, out.distance_max) == (2, 6)

    def test_outside_dropped(self):
        """An interval left with no coverage is dropped."""
        assert clip_intervals([_Span(distance_min=8, distance_max=10)], 2, 6) == []

    def test_other_fields_kept(self):
        """Only the bounds change; the item itself is left alone."""
        span = _Span(distance_min=0, distance_max=10, label="rail")
        (out,) = clip_intervals([span], 2, 6)
        assert out.label == "rail"
        assert (span.distance_min, span.distance_max) == (0, 10)

    def test_point_inside_survives(self):
        """A point marker inside the clip is kept as it was."""
        point = _Span(distance_min=3, distance_max=3)
        assert clip_intervals([point], 2, 6) == [point]

    def test_point_on_outer_end_survives(self):
        """A point on the outermost included endpoint is not lost."""
        point = _Span(distance_min=6, distance_max=6)
        assert clip_intervals([point], 2, 6) == []
        assert clip_intervals([point], 2, 6, outer=6) == [point]

    def test_field_names(self):
        """Any pair of start/end fields works, not just the inventory's."""
        window = _Window(time_start=0, time_end=10)
        (out,) = clip_intervals(
            [window], 2, 6, min_field="time_start", max_field="time_end"
        )
        assert (out.time_start, out.time_end) == (2, 6)


class TestValueKind:
    """The kind decides the shape of the group a value belongs to."""

    def test_none_is_membership(self):
        """No value at all is how membership is stated."""
        assert value_kind(None) == "membership"

    def test_string(self):
        """Text is a string kind."""
        assert value_kind("car") == "string"

    @pytest.mark.parametrize("value", [1, 1.5, -3])
    def test_numeric(self, value):
        """Ints and floats share one kind."""
        assert value_kind(value) == "numeric"


class TestNormalizeValue:
    """Values keep their python type and must be finite."""

    @pytest.mark.parametrize("value", [True, False, np.bool_(True)])
    def test_bool_refused(self, value):
        """Membership is stated by having no value, so a boolean is not one."""
        with pytest.raises(ParameterError, match="true and false are not values"):
            normalize_value(value)

    def test_numpy_int_unwrapped(self):
        """A numpy int becomes a python int, not a float."""
        out = normalize_value(np.int64(5))
        assert isinstance(out, int) and not isinstance(out, bool)

    def test_python_value_untouched(self):
        """A plain value comes back as it went in."""
        assert normalize_value("car") == "car"

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_non_finite_refused(self, value):
        """A non-finite number cannot survive a JSON round trip."""
        with pytest.raises(ParameterError, match="must be finite"):
            normalize_value(value)

    def test_error_class(self):
        """The caller's format decides which exception it raises."""

        class _MyError(ValueError):
            """A format-specific error."""

        with pytest.raises(_MyError, match="must be finite"):
            normalize_value(np.nan, error=_MyError)


class TestIntervalValueType:
    """One value type, refused in each subsystem's own vocabulary."""

    @pytest.mark.parametrize("annotated", ["LabelValue", "AnnotationValue"])
    def test_the_refusal_reaches_the_caller(self, annotated):
        """Both subsystems refuse a boolean, and say why.

        The type each raises is not observable here: pydantic wraps
        whatever a validator raises, keeping the message. The message is
        therefore what the caller actually gets, so it is what is pinned.
        """
        module = inventory if annotated == "LabelValue" else annotations
        adapter = TypeAdapter(getattr(module, annotated))
        with pytest.raises(ValidationError, match="true and false are not values"):
            adapter.validate_python(True)

    def test_a_direct_call_raises_the_stated_error(self):
        """Where the value is checked outside pydantic, the type survives.

        `AnnotationSet` checks its own frame, so this is the path on which
        the factory's `error` argument is the difference it claims to be.
        """
        frame = pd.DataFrame(
            {"group": ["g"], "value": [True], "time_min": [0], "time_max": [1]}
        )
        with pytest.raises(ParameterError, match="true and false are not values"):
            AnnotationSet(frame, dims=("time",))
