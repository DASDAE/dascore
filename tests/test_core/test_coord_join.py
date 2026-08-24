"""Tests for predicting a join from coordinate summaries."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coord_join import join_summaries
from dascore.core.coords import concat_coords, get_coord


def _range(start, stop, step=1.0, **kwargs):
    """A range coordinate, for brevity."""
    return get_coord(start=start, stop=stop, step=step, **kwargs)


def _joined(coords, tolerance=None):
    """What the real join gives, as a summary."""
    joined = concat_coords(*coords)
    if tolerance:
        step = joined.step
        if step is None:
            step = max(abs(x.step) for x in coords if x.step is not None)
        joined = joined.simplify(tolerance * np.abs(step))
    return joined.to_summary()


class TestAgreesWithTheRealJoin:
    """The prediction is the real join's answer, or nothing."""

    @pytest.mark.parametrize(
        "coords",
        [
            [_range(0.0, 10.0), _range(10.0, 20.0)],  # contiguous
            [_range(0.0, 10.0), _range(15.0, 25.0)],  # gapped
            [_range(10.0, 20.0), _range(0.0, 10.0)],  # supplied out of order
            [_range(0.0, 10.0), _range(10.0, 20.0), _range(20.0, 30.0)],
            [_range(10.0, 0.0, -1.0), _range(0.0, -10.0, -1.0)],  # descending
            [_range(0.0, 10.0, units="m"), _range(10.0, 20.0, units="m")],
        ],
    )
    def test_matches(self, coords):
        """Every field, fingerprint included, is what the join produces."""
        predicted = join_summaries([x.to_summary() for x in coords])
        assert predicted == _joined(coords)

    def test_time_coords_match(self):
        """Datetime members join the same way."""
        t0 = np.datetime64("2020-01-01", "ns")
        step = np.timedelta64(4_000_000, "ns")
        coords = [
            get_coord(start=t0, stop=t0 + 100 * step, step=step),
            get_coord(start=t0 + 100 * step, stop=t0 + 200 * step, step=step),
        ]
        assert join_summaries([x.to_summary() for x in coords]) == _joined(coords)

    def test_patch_coords_match(self):
        """Coordinates as the index stores them predict exactly."""
        patch = dc.get_example_patch()
        time = patch.get_coord("time")
        half = len(time) // 2
        first, second = time[:half], time[half:]
        predicted = join_summaries([first.to_summary(), second.to_summary()])
        assert predicted == _joined([first, second])
        assert predicted.fingerprint == time.fingerprint()

    def test_a_coarser_step_is_not_vouched_for(self):
        """
        A step spelled in coarser units leaves the join unidentified.

        The index stores nanoseconds, so a coordinate written at another
        precision does not rebuild into itself; the envelope still spans
        the members, but nothing is claimed about how they sample it.
        """
        t0 = np.datetime64("2020-01-01", "ms")
        step = np.timedelta64(4, "ms")
        coords = [
            get_coord(start=t0, stop=t0 + 100 * step, step=step),
            get_coord(start=t0 + 100 * step, stop=t0 + 200 * step, step=step),
        ]
        predicted = join_summaries([x.to_summary() for x in coords])
        assert predicted.fingerprint is None
        assert predicted.step is None
        assert predicted.len is None
        # the envelope is still the one the join produces
        expected = _joined(coords)
        assert predicted.min == expected.min
        assert predicted.max == expected.max


class TestClaimsNothingWhenItCannotTell:
    """Where summaries are not enough, the answer is None."""

    def test_no_summaries(self):
        """Nothing in, nothing claimed."""
        assert join_summaries([]) is None

    def test_one_summary_passes_through(self):
        """A lone member is its own answer, untouched."""
        summary = _range(0.0, 10.0).to_summary()
        assert join_summaries([summary]) is summary

    def test_member_without_a_step(self):
        """An array member's values are not in its summary."""
        array = get_coord(values=np.array([0.0, 1.0, 3.0]))
        summaries = [array.to_summary(), _range(10.0, 20.0).to_summary()]
        assert join_summaries(summaries) is None

    def test_value_less_member(self):
        """A coordinate which states no values cannot be joined."""
        blank = dc.get_example_patch().mean("time").get_coord("time")
        summaries = [blank.to_summary(), blank.to_summary()]
        assert join_summaries(summaries) is None

    def test_members_spelled_two_ways(self):
        """Which spelling wins is decided on the values, not here."""
        coords = [_range(0.0, 10.0, units="m"), _range(10.0, 20.0, units="cm")]
        assert join_summaries([x.to_summary() for x in coords]) is None

    def test_overlapping_members(self):
        """Overlapping members will raise at load; the row claims nothing."""
        coords = [_range(0.0, 10.0), _range(5.0, 15.0)]
        assert join_summaries([x.to_summary() for x in coords]) is None

    def test_string_members(self):
        """Label coordinates carry no step, so they take the same path."""
        labels = get_coord(values=np.array(["a", "b", "c"]))
        summaries = [labels.to_summary(), labels.to_summary()]
        assert join_summaries(summaries) is None
