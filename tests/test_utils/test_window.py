"""Tests for resolving a window along a patch's dimensions."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import CoordError, ParameterError
from dascore.units import percent
from dascore.utils.misc import suppress_warnings
from dascore.utils.patch import get_patch_window_size, get_window_axis_step
from dascore.utils.window import Window, resolve_window
from dascore.workflow.meta import PatchMeta


@pytest.fixture()
def simple_patch():
    """The example patch resampled so that windows in seconds are reasonable."""
    patch = dc.get_example_patch()
    return patch.update_coords(time_step=0.2)


class TestWindowSize:
    """Windows given per dimension, in units or samples."""

    def test_basic_window_size(self, simple_patch):
        """A window along one dimension is one everywhere else."""
        window = resolve_window(simple_patch, {"time": 0.6})
        assert window.dims == ("time",)
        assert window.axes == (simple_patch.get_axis("time"),)
        assert window.size[0] > 1
        assert window.overlap is None
        size = window.full_size()
        assert len(size) == simple_patch.data.ndim
        assert size[simple_patch.get_axis("distance")] == 1

    def test_multiple_dimensions(self, simple_patch):
        """Two dimensions, in the order they were named."""
        window = resolve_window(simple_patch, {"time": 0.6, "distance": 3.0})
        assert window.dims == ("time", "distance")
        assert all(size > 1 for size in window.size)

    def test_samples_true(self, simple_patch):
        """A bare number is a sample count when the call says so."""
        window = resolve_window(simple_patch, {"time": 5}, samples=True)
        assert window.size == (5,)

    def test_quantity_overrides_samples(self, simple_patch):
        """A quantity carries its units whatever the call says."""
        by_units = resolve_window(simple_patch, {"time": 1.0 * dc.units.s})
        under_samples = resolve_window(
            simple_patch, {"time": 1.0 * dc.units.s}, samples=True
        )
        assert by_units.size == under_samples.size == (5,)

    def test_require_odd_adjusts_units(self, simple_patch):
        """Given in units, an even count is rounded up to odd."""
        step = simple_patch.get_coord("time").step
        window = resolve_window(simple_patch, {"time": step * 4}, require_odd=True)
        assert window.size == (5,)

    def test_require_odd_refuses_even_samples(self, simple_patch):
        """Given in samples, an even count is the caller's to fix."""
        with pytest.raises(ParameterError, match="windows must be odd"):
            resolve_window(simple_patch, {"time": 4}, samples=True, require_odd=True)

    def test_require_odd_passes_odd_samples(self, simple_patch):
        """An odd count is left alone."""
        window = resolve_window(
            simple_patch, {"time": 5}, samples=True, require_odd=True
        )
        assert window.size == (5,)

    def test_min_samples(self, simple_patch):
        """A window below the floor is refused."""
        with pytest.raises(ParameterError, match="at least 3 samples"):
            resolve_window(simple_patch, {"time": 2}, samples=True, min_samples=3)

    def test_min_samples_message_suggests_samples(self, simple_patch):
        """A too-small window in coordinate units should mention samples (#1046)."""
        with pytest.raises(ParameterError, match="samples=True"):
            resolve_window(simple_patch, {"time": 0.2}, min_samples=3)
        # When the value is already in samples there is nothing to suggest.
        with pytest.raises(ParameterError, match="Try increasing"):
            resolve_window(simple_patch, {"time": 2}, samples=True, min_samples=3)

    def test_warn_above(self, simple_patch):
        """A large window warns."""
        with pytest.warns(UserWarning, match="Large window size.*may result in slow"):
            resolve_window(simple_patch, {"time": 15}, samples=True, warn_above=10)

    def test_warn_above_uses_total_window(self, simple_patch):
        """The threshold applies to the window's area, not each dimension."""
        kwargs = {"time": 5, "distance": 5}
        with pytest.warns(UserWarning, match="Large window size \\(25 samples\\)"):
            resolve_window(simple_patch, kwargs, samples=True, warn_above=10)

    def test_no_warning_under_threshold(self, simple_patch):
        """A small window says nothing."""
        with suppress_warnings(action="error"):
            window = resolve_window(
                simple_patch, {"time": 5}, samples=True, warn_above=10
            )
        assert window.size == (5,)

    def test_empty_kwargs_refused(self, simple_patch):
        """A windowed function wants a window."""
        with pytest.raises(ParameterError, match="at least one dimension"):
            resolve_window(simple_patch, {})

    def test_too_many_dimensions_refused(self, simple_patch):
        """A function windowing one dimension refuses two."""
        with pytest.raises(ParameterError, match="exactly one dimension"):
            resolve_window(
                simple_patch,
                {"time": 5, "distance": 5},
                samples=True,
                allow_multiple=False,
            )

    def test_invalid_dimension_raises(self, simple_patch):
        """A dimension the patch lacks is refused."""
        with pytest.raises(ParameterError):
            resolve_window(simple_patch, {"invalid_dim": 5})

    def test_non_evenly_sampled_raises(self, simple_patch):
        """A window in units needs a step to convert with."""
        time_size = simple_patch.shape[simple_patch.get_axis("time")]
        time_vals = np.concatenate(
            [[0.0, 0.1, 0.3, 0.7, 1.5], np.linspace(2.0, 10.0, time_size - 5)]
        )
        irregular = simple_patch.update_coords(time=time_vals)
        with pytest.raises(CoordError, match="not evenly sampled"):
            resolve_window(irregular, {"time": 0.5})

    def test_enforce_lt_coord(self, simple_patch):
        """A window longer than its coordinate is refused when asked."""
        with pytest.raises(ParameterError, match="results in a window"):
            resolve_window(
                simple_patch, {"time": 10_000}, samples=True, enforce_lt_coord=True
            )
        # Not by default: the dense filters hand one on to scipy.
        window = resolve_window(simple_patch, {"time": 10_000}, samples=True)
        assert window.size == (10_000,)

    def test_from_meta(self, simple_patch):
        """A patch's metadata is enough; no data need be in reach."""
        meta = PatchMeta.from_patch(simple_patch)
        from_meta = resolve_window(meta, {"time": 0.6, "distance": 3.0})
        from_patch = resolve_window(simple_patch, {"time": 0.6, "distance": 3.0})
        assert from_meta == from_patch


class TestOverlap:
    """Overlap and step between windows."""

    window = 16
    step = 8

    def test_apply_with_overlap(self, random_patch):
        """An overlap in units is converted to a stride."""
        coord = random_patch.get_coord("distance")
        window = self.window * coord.step
        overlap = (self.window - self.step) * coord.step
        out = resolve_window(random_patch, {"distance": window}, overlap=overlap)
        assert out.size == (self.window,)
        assert out.stride == (self.step,)
        assert out.axes == (random_patch.get_axis("distance"),)

    def test_apply_with_percent_overlap(self, random_patch):
        """A percent is a fraction of the window."""
        coord = random_patch.get_coord("distance")
        window = self.window * coord.step
        out = resolve_window(random_patch, {"distance": window}, overlap=50 * percent)
        assert out.stride == (self.step,)

    def test_apply_with_percent_overlap_and_samples(self, random_patch):
        """Whatever `samples` says."""
        out = resolve_window(
            random_patch, {"distance": self.window}, overlap=50 * percent, samples=True
        )
        assert out.stride == (self.step,)

    @pytest.mark.parametrize("window,overlap", [(5, 2), (7, 4), (9, 4)])
    def test_percent_rounds_half_to_even(self, random_patch, window, overlap):
        """50% of an odd window rounds as numpy rounds, which stft relied on."""
        out = resolve_window(
            random_patch, {"distance": window}, overlap=50 * percent, samples=True
        )
        assert out.overlap == (overlap,)

    def test_step(self, random_patch):
        """A step is the stride said directly."""
        out = resolve_window(random_patch, {"distance": 16}, step=4, samples=True)
        assert out.stride == (4,)
        assert out.overlap == (12,)

    def test_negative_overlap_raises(self, random_patch):
        """An overlap cannot retreat."""
        step = random_patch.get_coord("distance").step
        with pytest.raises(ParameterError, match="overlap must be non-negative"):
            resolve_window(
                random_patch, {"distance": self.window * step}, overlap=-step
            )

    def test_invalid_percent_overlap_raises(self, random_patch):
        """A percent is between 0 and 100."""
        with pytest.raises(
            ParameterError, match="Percentage must be between 0 and 100"
        ):
            resolve_window(
                random_patch,
                {"distance": self.window},
                overlap=101 * percent,
                samples=True,
            )

    def test_complete_overlap_raises(self, random_patch):
        """A window which never advances is refused."""
        with pytest.raises(
            ParameterError, match="Window step must be greater than zero"
        ):
            resolve_window(
                random_patch,
                {"distance": self.window},
                overlap=100 * percent,
                samples=True,
            )

    def test_overlap_larger_than_window_raises(self, random_patch):
        """As is one which retreats past the start."""
        with pytest.raises(
            ParameterError, match="Window step must be greater than zero"
        ):
            resolve_window(
                random_patch,
                {"distance": self.window},
                overlap=self.window + 1,
                samples=True,
            )

    def test_step_and_overlap_raises(self, random_patch):
        """One hop, one spelling."""
        step = random_patch.get_coord("distance").step
        with pytest.raises(
            ParameterError, match="step and overlap are mutually exclusive"
        ):
            resolve_window(
                random_patch,
                {"distance": self.window * step},
                step=self.step * step,
                overlap=50 * percent,
            )

    def test_none_overlap_is_none(self, random_patch):
        """Nothing given and no default leaves the overlap unsaid."""
        step = random_patch.get_coord("distance").step
        out = resolve_window(random_patch, {"distance": self.window * step})
        assert out.overlap is None
        assert out.stride is None

    def test_default_overlap_as_a_count(self, random_patch):
        """A function may say what an unsaid overlap is."""
        out = resolve_window(
            random_patch, {"distance": 16}, samples=True, default_overlap=3
        )
        assert out.overlap == (3,)

    def test_default_overlap_as_a_rule(self, random_patch):
        """Or how to make one from the window."""
        out = resolve_window(
            random_patch,
            {"distance": 16, "time": 8},
            samples=True,
            default_overlap=lambda size: size // 2 - 1,
        )
        assert out.overlap == (7, 3)

    def test_default_is_in_samples_whatever_the_units(self, random_patch):
        """A default is a sample count even when the windows are in units."""
        coord = random_patch.get_coord("distance")
        out = resolve_window(
            random_patch, {"distance": 16 * coord.step}, default_overlap=7
        )
        assert out.overlap == (7,)

    def test_mapping_per_dimension(self, random_patch):
        """A mapping gives each dimension its own overlap."""
        out = resolve_window(
            random_patch,
            {"distance": 16, "time": 8},
            samples=True,
            overlap={"distance": 4, "time": 50 * percent},
        )
        assert out.overlap == (4, 4)

    def test_mapping_fills_missing_from_default(self, random_patch):
        """A dimension the mapping leaves out takes the default."""
        out = resolve_window(
            random_patch,
            {"distance": 16, "time": 8},
            samples=True,
            overlap={"time": 2},
            default_overlap=lambda size: size // 2 - 1,
        )
        assert out.overlap == (7, 2)

    def test_mapping_missing_without_default_refused(self, random_patch):
        """Without one there is nothing to fill it with."""
        with pytest.raises(ParameterError, match="not \\['time'\\]"):
            resolve_window(
                random_patch,
                {"distance": 16, "time": 8},
                samples=True,
                overlap={"distance": 2},
            )

    def test_mapping_unknown_dimension_refused(self, random_patch):
        """An overlap for a dimension not windowed is a mistake."""
        with pytest.raises(ParameterError, match="not being windowed"):
            resolve_window(
                random_patch, {"distance": 16}, samples=True, overlap={"time": 2}
            )

    def test_scalar_applies_to_every_dimension(self, random_patch):
        """One value, every dimension."""
        out = resolve_window(
            random_patch, {"distance": 16, "time": 8}, samples=True, overlap=2
        )
        assert out.overlap == (2, 2)


class TestPolicies:
    """The knobs a function turns."""

    def test_min_samples_none_is_no_floor(self, simple_patch):
        """A zero window passes when a function sets no floor."""
        window = resolve_window(
            simple_patch, {"time": 0}, samples=True, min_samples=None
        )
        assert window.size == (0,)

    def test_uneven_coordinate_with_sample_counts(self):
        """Without the even-sampling requirement, sample counts skip the coordinate."""
        wacky = dc.get_example_patch("wacky_dim_coords_patch")
        window = resolve_window(
            wacky, {"time": 16}, samples=True, require_evenly_sampled=False
        )
        assert window.size == (16,)
        # Units still need a step to convert with.
        with pytest.raises(CoordError):
            resolve_window(wacky, {"time": 1.0}, require_evenly_sampled=False)

    def test_quantity_under_samples_adjusts_to_odd(self, simple_patch):
        """A quantity is in units, so an even count is rounded up, not refused."""
        window = resolve_window(
            simple_patch, {"time": 0.8 * dc.units.s}, samples=True, require_odd=True
        )
        assert window.size == (5,)

    def test_exclusivity_is_checked_first(self, simple_patch):
        """Both hop spellings at once is the first thing said, before any size."""
        with pytest.raises(ParameterError, match="mutually exclusive"):
            resolve_window(
                simple_patch,
                {"time": 10_000},
                samples=True,
                step=2,
                overlap=2,
                enforce_lt_coord=True,
            )

    def test_default_overlap_is_checked(self, random_patch):
        """A default which leaves no advance is refused like a given one."""
        with pytest.raises(ParameterError, match="greater than zero"):
            resolve_window(
                random_patch, {"distance": 5}, samples=True, default_overlap=5
            )
        with pytest.raises(ParameterError, match="non-negative"):
            resolve_window(
                random_patch, {"distance": 5}, samples=True, default_overlap=-1
            )

    def test_explicit_none_in_mapping_refused(self, random_patch):
        """None in a mapping is not the same as leaving the dimension out."""
        with pytest.raises(ParameterError, match="leave it out"):
            resolve_window(
                random_patch,
                {"distance": 16, "time": 8},
                samples=True,
                overlap={"time": None},
                default_overlap=2,
            )


class TestDeprecatedResolvers:
    """The two public utilities this replaces still answer, with a warning."""

    def test_get_patch_window_size(self, simple_patch):
        """The full-size tuple, as before."""
        with pytest.warns(DeprecationWarning, match="resolve_window"):
            size = get_patch_window_size(simple_patch, {"time": 5}, samples=True)
        assert (
            size == resolve_window(simple_patch, {"time": 5}, samples=True).full_size()
        )

    def test_get_patch_window_size_with_no_window(self, simple_patch):
        """As before, no dimension gives one along every axis."""
        with pytest.warns(DeprecationWarning, match="resolve_window"):
            size = get_patch_window_size(simple_patch, {})
        assert size == (1,) * simple_patch.data.ndim

    def test_get_window_axis_step(self, random_patch):
        """Window, axis, and step, as before."""
        with pytest.warns(DeprecationWarning, match="resolve_window"):
            out = get_window_axis_step(
                random_patch, distance=16, overlap=50 * percent, samples=True
            )
        assert out == (16, random_patch.get_axis("distance"), 8)


class TestWindow:
    """The resolved object itself."""

    def test_tiles(self, random_patch):
        """A window with a stride plans tiles over the axes it selects."""
        window = resolve_window(
            random_patch, {"distance": 16, "time": 8}, samples=True, overlap=2
        )
        plan = window.tiles(random_patch.shape)
        assert plan.size == (16, 8)
        assert plan.stride == (14, 6)
        assert plan.shape == tuple(random_patch.shape[axis] for axis in window.axes)

    def test_tiles_need_a_stride(self, random_patch):
        """With no overlap or step given there is no stride to tile at."""
        window = resolve_window(random_patch, {"time": 8}, samples=True)
        with pytest.raises(ParameterError, match="does not tile"):
            window.tiles(random_patch.shape)

    def test_full_size_fill(self):
        """Unselected axes take the fill."""
        window = Window(("time",), (1,), (5,), None, 3)
        assert window.full_size() == (1, 5, 1)
        assert window.full_size(fill=0) == (0, 5, 0)

    def test_overlap_is_size_less_stride(self):
        """Overlap is derived, so it and the stride can never disagree."""
        window = Window(("distance", "time"), (0, 1), (16, 8), (9, 5), 2)
        assert window.overlap == (7, 3)

    def test_a_gap_is_a_negative_overlap(self, random_patch):
        """A stride longer than the window is allowed, and says so."""
        window = resolve_window(random_patch, {"distance": 5}, samples=True, step=8)
        assert window.stride == (8,)
        assert window.overlap == (-3,)

    def test_default_overlap_takes_a_numpy_integer(self, random_patch):
        """A sample count from numpy is a sample count."""
        window = resolve_window(
            random_patch, {"distance": 16}, samples=True, default_overlap=np.int64(2)
        )
        assert window.overlap == (2,)
