"""Tests for cutting an array into tiles and blending them back."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import ParameterError
from dascore.utils.signal import get_taper
from dascore.utils.tiles import TilePlan, get_tile_plan

# Shapes which are not multiples of anything, so the edge tiles are ragged.
CASES = [
    ((33,), (8,), (3,)),
    ((64, 64), (16, 16), (7, 7)),
    ((100, 257), (16, 16), (7, 7)),
    ((100, 257), (8, 32), (0, 15)),
    ((20, 30, 40), (8, 8, 16), (3, 3, 7)),
    ((5,), (16,), (7,)),
]


def _stride(size, overlap):
    return tuple(z - o for z, o in zip(size, overlap))


class TestGeometry:
    """What the plan says about where tiles sit."""

    def test_grid_and_extension(self):
        """Two strides of padding, tiles every stride, room for the last one."""
        plan = get_tile_plan((100,), (16,), (9,))
        assert plan.margin == (9,)
        assert plan.grid == (118 // 9,)
        assert plan.extended[0] >= (plan.grid[0] - 1) * 9 + 16
        assert plan.n_tiles == plan.grid[0]

    def test_colours(self):
        """Tiles two strides apart cannot overlap when the stride is over half."""
        assert get_tile_plan((64, 64), (16, 16), (9, 9)).colours == (2, 2)
        assert get_tile_plan((64,), (16,), (16,)).colours == (1,)
        assert get_tile_plan((64,), (16,), (5,)).colours == (4,)

    def test_cached(self):
        """One plan per geometry."""
        assert get_tile_plan((64,), (16,), (9,)) is get_tile_plan((64,), (16,), (9,))

    @pytest.mark.parametrize(
        "shape,size,stride,match",
        [
            ((64, 64), (16,), (9, 9), "same length"),
            ((64,), (16,), (0,), "positive"),
            ((64,), (0,), (1,), "positive"),
            ((64,), (16,), (17,), "leaves gaps"),
        ],
    )
    def test_refused(self, shape, size, stride, match):
        """A geometry which cannot tile says why."""
        with pytest.raises(ParameterError, match=match):
            TilePlan(shape, size, stride)


class TestExtract:
    """Cutting the stack."""

    def test_stack_shape(self):
        """One tile per grid cell, each the tile's size, in the array's dtype."""
        plan = get_tile_plan((100, 257), (16, 16), (9, 9))
        tiles = plan.extract(np.ones((100, 257), dtype=np.float32))
        assert tiles.shape == (plan.n_tiles, 16, 16)
        assert tiles.dtype == np.float32

    def test_dtype_is_kept_or_given(self):
        """Complex tiles for a complex array; a dtype asked for is used."""
        plan = get_tile_plan((40,), (8,), (5,))
        assert plan.extract(np.ones(40, dtype=np.complex64)).dtype == np.complex64
        assert (
            plan.extract(np.ones(40, dtype=np.int16), dtype=np.float32).dtype
            == np.float32
        )

    @pytest.mark.parametrize(
        "shape,size,stride", [((40,), (8,), (5,)), ((40, 30), (8, 8), (5, 5))]
    )
    def test_stack_is_writeable(self, shape, size, stride):
        """A stack is for transforming, in place if the caller likes."""
        plan = get_tile_plan(shape, size, stride)
        tiles = plan.extract(np.ones(shape, dtype=np.float32))
        tiles /= 2
        assert tiles.flags.writeable

    def test_pad_and_crop_round_trip(self):
        """`crop` undoes `pad`."""
        plan = get_tile_plan((40, 30), (8, 8), (5, 5))
        array = np.arange(1200, dtype=np.float64).reshape(40, 30)
        buffer = plan.pad(array)
        assert buffer.shape == plan.extended and buffer.dtype == np.float64
        np.testing.assert_array_equal(plan.crop(buffer), array)

    def test_edges_are_zero_padded(self):
        """The first tile starts one stride before the data, in zeros."""
        plan = get_tile_plan((100,), (16,), (9,))
        tiles = plan.extract(np.ones(100))
        assert np.all(tiles[0, :9] == 0) and np.all(tiles[0, 9:] == 1)

    def test_complex_round_trip(self):
        """A complex stack blends back to the complex array."""
        rng = np.random.default_rng(4)
        array = (rng.normal(size=(30, 40)) + 1j * rng.normal(size=(30, 40))).astype(
            np.complex64
        )
        plan = get_tile_plan(array.shape, (8, 8), (5, 5))
        taper = get_taper("hann", (8, 8), (3, 3))
        out = plan.overlap_add(plan.extract(array), taper)
        assert out.dtype == np.complex64
        np.testing.assert_allclose(out, array, atol=1e-5)

    def test_first_tile_is_the_padded_start(self):
        """A tile is a plain slice of the padded buffer."""
        rng = np.random.default_rng(0)
        array = rng.normal(size=(40, 50)).astype(np.float32)
        plan = get_tile_plan(array.shape, (8, 8), (5, 5))
        tiles = plan.extract(array).reshape((*plan.grid, 8, 8))
        # Tile (1, 1) starts one stride in on each axis, which is where the
        # data starts: its top-left corner is the array's top-left corner.
        np.testing.assert_array_equal(tiles[1, 1], array[:8, :8])


class TestOverlapAdd:
    """Blending the stack back."""

    @pytest.mark.parametrize("shape,size,overlap", CASES)
    def test_identity(self, shape, size, overlap):
        """An untouched stack under a complementary taper is the array."""
        rng = np.random.default_rng(1)
        array = rng.normal(size=shape).astype(np.float32)
        plan = get_tile_plan(shape, size, _stride(size, overlap))
        taper = get_taper("hann", size, overlap)
        out = plan.overlap_add(plan.extract(array), taper)
        np.testing.assert_allclose(out, array, atol=1e-5)
        assert out.shape == array.shape

    def test_matches_a_loop(self):
        """The colour-class adds equal adding every tile one by one."""
        rng = np.random.default_rng(2)
        array = rng.normal(size=(50, 70)).astype(np.float32)
        size, overlap = (16, 8), (7, 3)
        plan = get_tile_plan(array.shape, size, _stride(size, overlap))
        taper = get_taper("triang", size, overlap)
        tiles = plan.extract(array) * rng.normal(size=(plan.n_tiles, 1, 1))
        expected = np.zeros(plan.extended, dtype=np.float32)
        grid = tiles.reshape((*plan.grid, *size))
        for i in range(plan.grid[0]):
            for j in range(plan.grid[1]):
                b0, b1 = i * plan.stride[0], j * plan.stride[1]
                expected[b0 : b0 + 16, b1 : b1 + 8] += grid[i, j] * taper
        expected = plan.crop(expected)
        # float32, summed in a different order.
        np.testing.assert_allclose(plan.overlap_add(tiles, taper), expected, atol=1e-6)

    def test_apply_is_extract_func_blend(self):
        """`apply` is the three steps in one."""
        rng = np.random.default_rng(3)
        array = rng.normal(size=(64, 64)).astype(np.float32)
        plan = get_tile_plan(array.shape, (16, 16), (9, 9))
        taper = get_taper("triang", (16, 16), (7, 7))
        doubled = plan.apply(array, lambda tiles: tiles * 2, taper)
        np.testing.assert_allclose(doubled, 2 * array, atol=1e-5)
