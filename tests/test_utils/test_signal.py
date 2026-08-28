"""Tests for windows, ramps, and tile tapers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal.windows import hann, triang

from dascore.exceptions import ParameterError
from dascore.utils.signal import WINDOW_FUNCTIONS, get_ramp, get_taper, get_window

# The shapes a blend might be asked for, from flat to bell.
SHAPES = ["boxcar", "triang", "hann", "hamming", "blackman", ("tukey", 0.5)]


class TestGetWindow:
    """One name, one window."""

    def test_dascore_name(self):
        """A registered name gives scipy's symmetric window."""
        np.testing.assert_allclose(get_window("hann", 9), hann(9, sym=True))

    def test_alias(self):
        """DASCore's own spellings are honoured."""
        np.testing.assert_allclose(get_window("cos", 9), hann(9))
        np.testing.assert_allclose(get_window("ramp", 9), triang(9))

    def test_scipy_name_and_tuple(self):
        """What scipy knows, this knows."""
        assert get_window(("tukey", 0.25), 16).shape == (16,)
        assert get_window("flattop", 16).shape == (16,)

    def test_periodic(self):
        """A periodic window, for a spectral estimate."""
        np.testing.assert_allclose(
            get_window("hann", 8, fftbins=True), hann(8, sym=False)
        )

    def test_array_passes_through(self):
        """An array of the right length is the window."""
        array = np.linspace(0, 1, 7)
        assert get_window(array, 7) is array

    def test_array_of_the_wrong_length_refused(self):
        """Of the wrong length, it is refused by count."""
        with pytest.raises(ParameterError, match="7 samples, not 8"):
            get_window(np.ones(7), 8)

    def test_scipy_parameter_errors_are_scipy_errors(self):
        """A window scipy knows but cannot build with those parameters says why."""
        with pytest.raises(ValueError, match="NW"):
            get_window(("dpss", 2.5), 3)

    def test_unknown_name_refused(self):
        """A name nobody knows says what the options are."""
        with pytest.raises(ParameterError, match="not a known window"):
            get_window("windowsXP", 8)

    def test_every_registered_name_builds(self):
        """The registry is not decorative."""
        for name in WINDOW_FUNCTIONS:
            assert get_window(name, 8).shape == (8,)


class TestGetRamp:
    """The rising edge of a window."""

    def test_triangle_values(self):
        """Four samples of triangle from a window of nine: 0.2, 0.4, 0.6, 0.8."""
        np.testing.assert_allclose(get_ramp("triang", 4), [0.2, 0.4, 0.6, 0.8])

    def test_zero_length(self):
        """No ramp is no samples."""
        assert get_ramp("hann", 0).shape == (0,)
        assert get_ramp("hann", 0, complementary=True).shape == (0,)

    def test_triangle_is_already_complementary(self):
        """A triangle's ramp and its reverse sum to one as built."""
        ramp = get_ramp("triang", 5)
        np.testing.assert_allclose(ramp + ramp[::-1], 1)
        np.testing.assert_allclose(get_ramp("triang", 5, complementary=True), ramp)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("length", [1, 2, 3, 7, 16])
    def test_complementary(self, shape, length):
        """Asked for, any shape's ramp and its reverse sum to one."""
        ramp = get_ramp(shape, length, complementary=True)
        np.testing.assert_allclose(ramp + ramp[::-1], 1, atol=1e-12)
        assert np.all(ramp >= 0) and np.all(ramp <= 1)

    def test_integer_array_ramp(self):
        """An integer array is weights too, once complementary."""
        ramp = get_ramp(np.arange(7), 3, complementary=True)
        np.testing.assert_allclose(ramp + ramp[::-1], 1)

    def test_complementary_keeps_the_shape_monotone(self):
        """Scaling does not turn a rise into a wobble."""
        ramp = get_ramp("hann", 16, complementary=True)
        assert np.all(np.diff(ramp) >= 0)


class TestGetTaper:
    """A tile of ones with ramps down every edge."""

    def test_values(self):
        """The one-dimensional taper is the ramp, ones, and the ramp reversed."""
        taper = get_taper("triang", (8,), (3,))
        expected = [0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25]
        np.testing.assert_allclose(taper, expected)
        assert taper.dtype == np.float32

    def test_two_dimensional_is_separable(self):
        """The 2-D taper is the outer product of its edges."""
        taper = get_taper("triang", (8, 8), (3, 3))
        edge = np.array([0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25])
        np.testing.assert_allclose(taper, edge[:, None] * edge[None, :])

    def test_three_dimensional(self):
        """And so for any number of axes."""
        taper = get_taper("hann", (6, 4, 8), (2, 1, 3))
        assert taper.shape == (6, 4, 8)
        assert taper[3, 2, 4] == 1.0

    def test_zero_overlap_is_all_ones(self):
        """No ramp along an axis is a boxcar along it."""
        np.testing.assert_array_equal(
            get_taper("hann", (8, 8), (0, 0)), np.ones((8, 8))
        )

    def test_odd_sizes_are_fine(self):
        """A tile need not be even."""
        assert get_taper("triang", (7, 9), (3, 4)).shape == (7, 9)

    def test_array_window(self):
        """An array window builds a taper, uncached."""
        taper = get_taper(np.hanning(7), (8,), (3,))
        assert taper.shape == (8,)
        assert taper[4] == 1.0

    def test_tuple_with_a_list_parameter(self):
        """A scipy tuple carrying a list is not hashable, and still builds."""
        taper = get_taper(("general_cosine", [0.5, 0.5]), (8,), (3,))
        assert taper.shape == (8,)

    def test_unknown_window_refused_even_at_zero_overlap(self):
        """No ramp is needed, and the name is still checked."""
        with pytest.raises(ParameterError, match="not a known window"):
            get_taper("windowsXP", (8,), (0,))

    def test_two_dimensional_is_float32_throughout(self):
        """Edges multiply in float32, so the corner is the float32 product."""
        taper = get_taper("triang", (8, 8), (2, 2))
        corner = np.float32(1 / 3) * np.float32(1 / 3)
        assert taper[0, 0] == corner

    def test_copies_are_handed_out(self):
        """Writing to one taper does not change the next."""
        taper = get_taper("triang", (16, 16), (2, 2))
        expected = taper.copy()
        taper[...] = -1.0
        np.testing.assert_array_equal(get_taper("triang", (16, 16), (2, 2)), expected)

    @pytest.mark.parametrize(
        "size,overlap,match",
        [
            ((16, 16), (2,), "same length"),
            ((16, 16), (-1, 2), "non-negative"),
            ((16, 16), (9, 2), "ramps would cross"),
        ],
    )
    def test_invalid_geometry_refused(self, size, overlap, match):
        """A taper which cannot be built says why."""
        with pytest.raises(ParameterError, match=match):
            get_taper("triang", size, overlap)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize(
        "size,overlap", [(16, 7), (16, 4), (8, 1), (9, 4), (32, 15)]
    )
    def test_tiles_sum_to_one(self, shape, size, overlap):
        """
        The invariant: tiles at a stride of size - overlap blend to exactly one.

        This is what lets an overlap-add of tapered tiles return the signal
        they were cut from, and it holds for every shape, not only the
        triangle, because the ramps are complementary.
        """
        taper = get_taper(shape, (size,), (overlap,))
        stride = size - overlap
        n_tiles = 6
        total = np.zeros(stride * (n_tiles - 1) + size)
        for tile in range(n_tiles):
            total[tile * stride : tile * stride + size] += taper
        # Away from the two ends, where no neighbour reaches.
        np.testing.assert_allclose(total[overlap:-overlap], 1, atol=1e-6)

    def test_two_dimensional_tiles_sum_to_one(self):
        """The same in two dimensions, where the corners see four tiles."""
        size, overlap = (8, 16), (3, 7)
        taper = get_taper("hann", size, overlap)
        stride = tuple(s - o for s, o in zip(size, overlap))
        n = 4
        total = np.zeros(tuple(st * (n - 1) + s for st, s in zip(stride, size)))
        for i in range(n):
            for j in range(n):
                total[
                    i * stride[0] : i * stride[0] + size[0],
                    j * stride[1] : j * stride[1] + size[1],
                ] += taper
        inner = total[overlap[0] : -overlap[0], overlap[1] : -overlap[1]]
        np.testing.assert_allclose(inner, 1, atol=1e-6)
