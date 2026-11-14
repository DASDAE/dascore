"""Tests for block aggregation of a patch."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import get_coord
from dascore.exceptions import ParameterError


@pytest.fixture(scope="module")
def ramp():
    """Five samples counting up along distance, labelled by their position."""
    data, coords = np.arange(5.0), {"distance": np.arange(5)}
    return dc.Patch(data=data, coords=coords, dims=("distance",))


@pytest.fixture(scope="module")
def patch():
    """The example patch with random values."""
    example = dc.get_example_patch()
    data = np.random.default_rng(0).standard_normal(example.shape)
    return example.new(data=data)


class TestCoarsen:
    """Tests for Patch.coarsen."""

    def test_pairs_labelled_between(self):
        """Pairs average to their mean, labelled between the two channels."""
        patch = dc.get_example_patch().select(distance=(0, 6), samples=True)
        patch = patch.new(data=np.arange(6.0)[:, None] * np.ones(patch.shape))
        out = patch.coarsen(distance=2, samples=True).mean()
        np.testing.assert_array_equal(out.data[:, 0], [0.5, 2.5, 4.5])
        coord = out.get_coord("distance")
        np.testing.assert_array_equal(coord.values, [0.5, 2.5, 4.5])
        assert coord.step == 2
        assert out.attrs.history[-1].startswith("coarsen(")

    def test_units_match_samples(self, patch):
        """A block in the dimension's units is the same block in samples."""
        size = 25
        in_units = patch.coarsen(time=patch.get_coord("time").step * size).sum()
        in_samples = patch.coarsen(time=size, samples=True).sum()
        assert in_units.equals(in_samples)

    def test_trim(self, ramp):
        """A partial last block is dropped by default."""
        out = ramp.coarsen(distance=2, samples=True).max()
        np.testing.assert_array_equal(out.data, [1, 3])
        np.testing.assert_array_equal(out.get_coord("distance").values, [0.5, 2.5])

    def test_pad(self, ramp):
        """A partial last block is reduced over its samples, on the grid."""
        out = ramp.coarsen(distance=2, samples=True, boundary="pad").sum()
        np.testing.assert_array_equal(out.data, [1, 5, 4])
        coord = out.get_coord("distance")
        np.testing.assert_array_equal(coord.values, [0.5, 2.5, 4.5])
        assert coord.step == 2

    def test_time(self, patch):
        """Datetime labels move to each block's centre, on a coarser step."""
        before = patch.get_coord("time")
        out = patch.coarsen(time=4, samples=True).median()
        coord = out.get_coord("time")
        assert coord.step == 4 * before.step
        assert coord.min() == before.min() + 1.5 * before.step
        assert len(coord) == len(before) // 4
        blocks = patch.data[:, : len(coord) * 4].reshape(-1, len(coord), 4)
        np.testing.assert_allclose(out.data, np.median(blocks, axis=-1))

    def test_function(self, ramp):
        """Any function reducing the last axis is given each block."""
        out = ramp.coarsen(distance=2, samples=True).reduce(np.percentile, 100)
        np.testing.assert_array_equal(out.data, [1, 3])

    @pytest.mark.filterwarnings("ignore:No id could be derived")
    def test_unhashable_function(self, ramp):
        """A callable which cannot be hashed still reduces each block."""

        class Largest:
            __hash__ = None

            def __call__(self, data, axis):
                return data.max(axis=axis)

        out = ramp.coarsen(distance=2, samples=True).reduce(Largest())
        np.testing.assert_array_equal(out.data, [1, 3])

    def test_one_sample(self, ramp):
        """A block of one sample changes neither data nor labels."""
        out = ramp.coarsen(distance=1, samples=True).mean()
        np.testing.assert_array_equal(out.data, ramp.data)
        assert out.get_coord("distance") == ramp.get_coord("distance")

    def test_block_too_long(self, ramp):
        """A block longer than the dimension is refused."""
        with pytest.raises(ParameterError):
            ramp.coarsen(distance=6, samples=True).mean()

    def test_labels(self):
        """A dimension with no step is labelled by each block's mean label."""
        labels = np.array([0.0, 1.0, 3.0, 7.0, 8.0])
        patch = dc.Patch(data=np.ones(5), coords={"x": labels}, dims=("x",))
        out = patch.coarsen(x=2, samples=True, boundary="pad").mean()
        np.testing.assert_array_equal(out.get_coord("x").values, [0.5, 5.0, 8.0])

    @pytest.mark.parametrize("labels", [np.arange(5, -1, -1), np.arange(5.0, -1, -1)])
    def test_descending(self, labels):
        """A descending dimension is labelled between each pair, in order."""
        patch = dc.Patch(data=np.ones(6), coords={"x": labels}, dims=("x",))
        out = patch.coarsen(x=2, samples=True).mean()
        np.testing.assert_array_equal(out.get_coord("x").values, [4.5, 2.5, 0.5])

    def test_fractional_step(self):
        """An integer grid on a fractional step keeps it, times the block."""
        coord = get_coord(start=0, step=(304, 3), shape=(60,))
        patch = dc.Patch(data=np.ones(60), coords={"x": coord}, dims=("x",))
        out = patch.coarsen(x=2, samples=True).mean().get_coord("x")
        assert out.step == pytest.approx(608 / 3)
        assert out.min() == pytest.approx(304 / 6)

    def test_fractional_time_step(self):
        """A 1024 Hz grid in blocks of a second is labelled at exact centres."""
        start = np.datetime64("2020-01-01", "ns")
        coord = get_coord(start=start, step=(1, 1024), shape=(4096,))
        patch = dc.Patch(data=np.ones(4096), coords={"time": coord}, dims=("time",))
        out = patch.coarsen(time=1024, samples=True).mean().get_coord("time")
        # The centre is 1023/2048 s in, rounded down to a whole nanosecond.
        assert out.min() - start == np.timedelta64(499511718, "ns")
        assert out.step == np.timedelta64(1, "s")

    def test_unsigned_labels(self):
        """Descending unsigned labels with no step do not wrap."""
        labels = np.array([10, 9, 7, 3], dtype=np.uint32)
        patch = dc.Patch(data=np.ones(4), coords={"x": labels}, dims=("x",))
        out = patch.coarsen(x=2, samples=True).mean()
        np.testing.assert_array_equal(out.get_coord("x").values, [9.5, 5.0])

    def test_associated(self, patch):
        """A coordinate along the dimension keeps each block's first value."""
        patch = patch.update_coords(gauge=("distance", np.arange(patch.shape[0]) * 10))
        out = patch.coarsen(distance=3, samples=True).mean()
        np.testing.assert_array_equal(out.coords.get_array("gauge")[:2], [0, 30])

    @pytest.mark.parametrize(
        ("boundary", "function"), [("exact", "mean"), ("trim", "mode")]
    )
    def test_bad_option(self, ramp, boundary, function):
        """An unknown boundary or reduction name is refused."""
        blocks = ramp.coarsen(distance=2, samples=True, boundary=boundary)
        with pytest.raises(ParameterError):
            blocks.reduce(function)
