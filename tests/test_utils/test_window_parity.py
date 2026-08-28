"""
What every windowed patch function does today, pinned before the resolvers merge.

Each windowed function used to turn "a window of X" into sample counts its own
way, and the ways differed at the edges: whether an even window is adjusted or
refused, whether a window longer than the coordinate is refused, what
`overlap=None` means, how a percent rounds. The tests here pin those edges
with literal values taken from the functions before the resolvers were made
one, so the merge can be checked against them rather than trusted.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.signal.windows import hann

import dascore as dc
from dascore.exceptions import CoordError, ParameterError
from dascore.units import percent, s
from dascore.utils.signal import get_taper


@pytest.fixture(scope="module")
def patch():
    """The example patch: 300 channels by 2000 samples at 4 ms."""
    return dc.get_example_patch()


@pytest.fixture(scope="module")
def wacky():
    """A patch whose coordinates are not evenly sampled."""
    return dc.get_example_patch("wacky_dim_coords_patch")


class TestOddWindows:
    """An even window is adjusted, refused, or accepted, depending on who asks."""

    def test_hampel_adjusts_units_up_to_odd(self, patch):
        """16 ms is 4 samples; hampel makes it 5 without saying so."""
        by_units = patch.hampel_filter(time=0.016)
        by_samples = patch.hampel_filter(time=5, samples=True)
        assert np.allclose(by_units.data, by_samples.data)

    def test_hampel_refuses_even_samples(self, patch):
        """Given in samples, an even window is the caller's mistake."""
        with pytest.raises(ParameterError, match="must be odd"):
            patch.hampel_filter(time=4, samples=True)

    def test_median_accepts_even_samples(self, patch):
        """The median filter has never asked for an odd window."""
        out = patch.median_filter(time=4, samples=True)
        assert out.shape == patch.shape


class TestOversizeWindows:
    """A window longer than the coordinate is refused by some functions only."""

    @pytest.mark.parametrize("name", ["rolling", "stft"])
    def test_refused(self, patch, name):
        """Rolling and stft check the window against the coordinate."""
        func = getattr(patch, name)
        with pytest.raises(ParameterError, match="results in a window"):
            func(time=5000, samples=True)

    @pytest.mark.parametrize("name", ["median_filter", "gaussian_filter"])
    def test_accepted(self, patch, name):
        """The dense filters hand an oversize window to scipy, which copes."""
        out = getattr(patch, name)(time=5000, samples=True)
        assert out.shape == patch.shape


class TestOverlapNone:
    """`overlap=None` means three different things."""

    def test_rolling_means_a_stride_of_one(self, patch):
        """Every sample gets a window."""
        out = patch.rolling(time=5, samples=True).mean()
        assert out.shape == patch.shape

    def test_stft_omitted_means_half(self, patch):
        """Stft's default is not None but 50%: windows of 8 hop by 4."""
        out = patch.stft(time=8, samples=True)
        assert out.attrs["_tile_stride_time"] == 4

    def test_stft_none_means_no_overlap(self, patch):
        """Given None outright, windows abut."""
        out = patch.stft(time=8, samples=True, overlap=None)
        assert out.attrs["_tile_stride_time"] == 8

    def test_adaptive_spectral_filter_means_the_most_allowed(self, patch):
        """Windows of 16 overlap by 7, which is window // 2 - 1."""
        by_default = patch.adaptive_spectral_filter(time=16, distance=16, samples=True)
        by_hand = patch.adaptive_spectral_filter(
            time=16, distance=16, overlap=7, samples=True
        )
        assert np.allclose(by_default.data, by_hand.data)


class TestPercentRounding:
    """A percent of an odd window rounds half to even, as numpy does."""

    @pytest.mark.parametrize("window,hop", [(5, 3), (7, 3), (9, 5)])
    def test_stft_hop(self, patch, window, hop):
        """50% of 5 is 2, of 7 is 4, of 9 is 4; the hop is what is left."""
        out = patch.stft(time=window, samples=True, overlap=50 * percent)
        assert out.attrs["_tile_stride_time"] == hop

    def test_rolling_hop(self, patch):
        """Rolling rounds the same way: 5 samples at 50% steps by 3."""
        out = patch.rolling(time=5, samples=True, overlap=50 * percent).mean()
        assert out.shape[patch.get_axis("time")] == 667


class TestQuantitiesUnderSamples:
    """A quantity keeps its units even when the call said samples."""

    def test_rolling_overlap(self, patch):
        """8 ms is two samples whatever `samples` says."""
        by_quantity = patch.rolling(time=5, samples=True, overlap=0.008 * s).mean()
        by_samples = patch.rolling(time=5, samples=True, overlap=2).mean()
        assert by_quantity.shape == by_samples.shape
        assert np.allclose(by_quantity.data, by_samples.data, equal_nan=True)


class TestStepAndOverlap:
    """The two spellings of a hop, and what is refused."""

    def test_both_given_is_refused(self, patch):
        """One hop, one spelling."""
        with pytest.raises(ParameterError, match="mutually exclusive"):
            patch.rolling(time=5, samples=True, step=2, overlap=2)

    def test_complete_overlap_is_refused(self, patch):
        """A window which never advances is not a window."""
        with pytest.raises(ParameterError, match="greater than zero"):
            patch.rolling(time=5, samples=True, overlap=5)

    def test_negative_overlap_is_refused(self, patch):
        """Nor is one which retreats."""
        with pytest.raises(ParameterError, match="non-negative"):
            patch.rolling(time=5, samples=True, overlap=-1)

    def test_percent_out_of_range_is_refused(self, patch):
        """A percent is between 0 and 100."""
        with pytest.raises(ParameterError, match="between 0 and 100"):
            patch.rolling(time=5, samples=True, overlap=150 * percent)


class TestWarnings:
    """Hampel warns on the area of the window, the others say nothing."""

    def test_hampel_warns_on_a_large_area(self, patch):
        """11 by 11 is 121 samples, above the threshold of 100."""
        with pytest.warns(UserWarning, match="Large window"):
            patch.hampel_filter(time=11, distance=11, samples=True, approximate=False)

    def test_hampel_quiet_under_it(self, patch):
        """9 by 9 is 81."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            patch.hampel_filter(time=9, distance=9, samples=True, approximate=False)


class TestMinimumSamples:
    """Each function has its own floor."""

    def test_hampel_needs_three(self, patch):
        """A window of one sample has no neighbours to compare against."""
        with pytest.raises(ParameterError, match="at least 3 samples"):
            patch.hampel_filter(time=1, samples=True)

    def test_min_samples_message_names_the_step(self, patch):
        """Given in units, the message says what a sample is worth."""
        with pytest.raises(ParameterError, match="sampled every"):
            patch.hampel_filter(time=0.001)


class TestUnevenCoordinates:
    """A window in coordinate units needs an evenly sampled coordinate."""

    @pytest.mark.parametrize(
        "name", ["median_filter", "gaussian_filter", "hampel_filter", "wiener_filter"]
    )
    def test_refused_in_units(self, wacky, name):
        """Nothing to convert with."""
        with pytest.raises(CoordError):
            getattr(wacky, name)(time=3)


class TestUnevenCoordinatesInSamples:
    """A window in samples on an uneven coordinate: only AFK ever allowed it."""

    def test_adaptive_spectral_filter_reads_the_count(self, wacky):
        """It never consulted the coordinate for a sample count, and still does not."""
        out = wacky.adaptive_spectral_filter(time=16, samples=True, engine="scipy")
        assert out.shape == wacky.shape

    @pytest.mark.parametrize("name", ["median_filter", "hampel_filter"])
    def test_dense_filters_refuse(self, wacky, name):
        """They asked for an even coordinate whatever the units, and still do."""
        with pytest.raises(CoordError):
            getattr(wacky, name)(time=3, samples=True)

    def test_rolling_refuses(self, wacky):
        """As does rolling."""
        with pytest.raises(CoordError):
            wacky.rolling(time=3, samples=True)


class TestNoFloor:
    """The dense filters never had a minimum window."""

    def test_gaussian_zero_is_the_identity(self, patch):
        """A sigma of zero along an axis smooths nothing; scipy skips it."""
        out = patch.gaussian_filter(time=0, samples=True)
        assert np.allclose(out.data, patch.data)


class TestChangedOnPurpose:
    """
    What the resolver does differently, each a call which used to fail.

    Pinned so the difference is on record rather than discovered; each
    is named in the pull request which made it.
    """

    def test_hampel_takes_a_quantity_under_samples(self, patch):
        """Raised "must be integers" before; a quantity carries its units."""
        by_quantity = patch.hampel_filter(time=0.016 * s, samples=True)
        by_units = patch.hampel_filter(time=0.016)
        assert np.allclose(by_quantity.data, by_units.data)

    def test_rolling_takes_a_mapping(self, patch):
        """Raised a TypeError before; a mapping is one value per dimension."""
        mapped = patch.rolling(time=5, samples=True, overlap={"time": 2}).mean()
        plain = patch.rolling(time=5, samples=True, overlap=2).mean()
        assert np.allclose(mapped.data, plain.data, equal_nan=True)

    def test_adaptive_spectral_filter_reads_a_percent(self, patch):
        """Silently took int(25 * percent) == 0 before; 25% of 16 is 4."""
        by_percent = patch.adaptive_spectral_filter(
            time=16, distance=16, overlap=25 * percent, samples=True
        )
        by_count = patch.adaptive_spectral_filter(
            time=16, distance=16, overlap=4, samples=True
        )
        assert np.allclose(by_percent.data, by_count.data)

    def test_stft_refuses_a_zero_window(self, patch):
        """Reached scipy and failed on the window's shape before; now says why."""
        with pytest.raises(ParameterError, match="at least 1 samples"):
            patch.stft(time=0, samples=True, overlap=None)

    def test_rolling_zero_window_is_the_resolver_s_floor(self, patch):
        """Rolling had its own "can't be zero" check; the shared floor says it now."""
        with pytest.raises(ParameterError, match="at least 1 samples"):
            patch.rolling(time=0, samples=True)

    def test_hampel_with_no_window_refuses(self, patch):
        """Given no dimension, hampel used to return the data untouched."""
        with pytest.raises(ParameterError, match="at least one dimension"):
            patch.hampel_filter()

    def test_stft_array_window_of_the_wrong_length_refused(self, patch):
        """A 90-sample array for a 100-sample window used to be zero padded."""
        with pytest.raises(ParameterError, match="90 samples, not 100"):
            patch.stft(time=100, taper_window=np.ones(90), samples=True)

    def test_adaptive_spectral_filter_refuses_none_in_a_mapping(self, patch):
        """Raised a TypeError before; now says what to do instead."""
        with pytest.raises(ParameterError, match="leave it out"):
            patch.adaptive_spectral_filter(
                time=16, distance=16, overlap={"time": None}, samples=True
            )


class TestDimensionOrder:
    """Whether the order of the selected dimensions changes the answer."""

    def test_gaussian_does_not_care(self, patch):
        """Gaussian smoothing is separable, so order is nothing to it."""
        one = patch.gaussian_filter(time=2, distance=3, samples=True)
        other = patch.gaussian_filter(distance=3, time=2, samples=True)
        assert np.allclose(one.data, other.data)


class TestEmptySelection:
    """What a call with no dimension does."""

    def test_wiener_refuses(self, patch):
        """Wiener asks for a window."""
        with pytest.raises(ParameterError):
            patch.wiener_filter()

    def test_median_refuses_too(self, patch):
        """So does every function which reads its dimensions from kwargs."""
        with pytest.raises(ParameterError, match="at least one dimension"):
            patch.median_filter()


class TestTaperRamps:
    """
    The ramp each edge-tapering function builds, pinned by value.

    All take the first n samples of a window of 2n + 1, so a five-sample
    triangle is 1/6, 2/6, ... 5/6 and the plateau follows. `Patch.taper`
    used to take the first half of a window of 2n instead (1/10, 3/10,
    ...), and moved here on purpose: one ramp, everywhere.
    """

    @pytest.fixture(scope="class")
    def ones(self):
        """A patch of ones, so the taper is the output."""
        patch = dc.get_example_patch()
        return patch.new(data=np.ones_like(patch.data))

    def test_taper_uses_a_2n_plus_1_window(self, ones):
        """16 ms is five samples; the ramp is the first five of a triangle of 11."""
        out = ones.taper(time=(np.timedelta64(16, "ms"), None), window_type="triang")
        np.testing.assert_allclose(out.data[0, :6], np.arange(1, 7) / 6)

    def test_taper_hann_uses_a_2n_plus_1_window(self, ones):
        """The same construction for hann: the first five of a hann of 11."""
        out = ones.taper(time=(np.timedelta64(16, "ms"), None), window_type="hann")
        np.testing.assert_allclose(out.data[0, :5], hann(11)[:5], rtol=1e-6)
        assert out.data[0, 5] == 1.0

    def test_taper_end_is_the_start_reversed(self, ones):
        """The end ramp is the mirror of the start ramp."""
        out = ones.taper(time=np.timedelta64(16, "ms"), window_type="triang")
        np.testing.assert_allclose(out.data[0, -5:], (np.arange(1, 6) / 6)[::-1])

    def test_taper_range_uses_a_2n_plus_1_window(self, ones):
        """Four samples of triangle from a window of nine: 0.2, 0.4, 0.6, 0.8."""
        out = ones.taper_range(
            time=(0, 4, 100, 104), samples=True, window_type="triang"
        )
        np.testing.assert_allclose(out.data[0, :5], [0.2, 0.4, 0.6, 0.8, 1.0])

    def test_taper_range_hann_uses_a_2n_plus_1_window(self, ones):
        """And for hann."""
        out = ones.taper_range(time=(0, 4, 100, 104), samples=True, window_type="hann")
        expected = [0.0, 0.14644661, 0.5, 0.85355339, 1.0]
        np.testing.assert_allclose(out.data[0, :5], expected, rtol=1e-6)

    def test_stft_window_is_scipy_symmetric(self, patch):
        """Stft's whole-tile window is scipy's symmetric one."""
        out = patch.stft(time=8, samples=True, overlap=None)
        # The window travels as a coordinate for istft; it is the symmetric hann.
        np.testing.assert_allclose(
            out.get_coord("_tile_analysis_time").values, hann(8, sym=True), rtol=1e-6
        )

    def test_adaptive_spectral_filter_ramp(self, ones):
        """The tile taper's ramp is the 2n + 1 triangle, complementary by nature."""
        taper = get_taper("triang", (8,), (3,))
        np.testing.assert_allclose(taper, [0.25, 0.5, 0.75, 1, 1, 0.75, 0.5, 0.25])
