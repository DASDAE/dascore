"""Float slices retain the arithmetic which produced their parent labels."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import Grid, concat_coords, get_coord


@pytest.fixture(
    params=[
        (0.1, 0.1),
        (-3.3, 1.0209),
        (1e9, 0.03),
        (7.9, -0.1),
        (np.float32(0.1), 0.1),
    ]
)
def float_coord(request):
    """Grids with observable rounding, including descending and narrow floats."""
    start, step = request.param
    return get_coord(start=start, step=step, shape=(100,))


class TestFloatSlices:
    """Slicing never recomputes the selected labels on a different grid."""

    @pytest.mark.parametrize(
        "indexer",
        [
            slice(3, 73),
            slice(2, None, 3),
            slice(None, None, -1),
            slice(73, 3, -7),
            slice(-1, None),
            slice(0, 1),
        ],
    )
    def test_labels(self, float_coord, indexer):
        """Trimmed, strided, reversed, and singleton results match bit-for-bit."""
        expected = float_coord.values[indexer]
        result = float_coord[indexer]
        assert result.values.tobytes() == expected.tobytes()
        assert result.evenly_sampled
        assert not result.sources
        assert result[0] == expected[0]
        assert result[-1] == expected[-1]

    def test_nested_slices(self, float_coord):
        """Composed integer windows preserve the original expression."""
        expected = float_coord.values[3:91:2][::-1][2::3]
        result = float_coord[3:91:2][::-1][2::3]
        assert result.values.tobytes() == expected.tobytes()
        assert result.data_id == float_coord[85:2:-6].data_id

    def test_reverse_twice(self, float_coord):
        """Reversing twice restores both labels and grid identity."""
        result = float_coord[::-1][::-1]
        assert result.values.tobytes() == float_coord.values.tobytes()
        assert result.data_id == float_coord.data_id
        assert result == float_coord

    def test_dump(self, float_coord):
        """The original grid survives a dump independently of the parent object."""
        sliced = float_coord[93:2:-3]
        result = get_coord(**sliced.model_dump())
        assert result.values.tobytes() == sliced.values.tobytes()
        assert result.data_id == sliced.data_id

    def test_select(self, float_coord):
        """A value selection on a strided slice keeps its labels and positions."""
        sliced = float_coord[3:93:2]
        bounds = sorted(sliced.values[[3, 12]])
        result, indexer = sliced.select(bounds)
        expected = sliced.values[3:13]
        assert result.values.tobytes() == expected.tobytes()
        assert np.array_equal(sliced.values[indexer], expected)

    def test_join(self, float_coord):
        """Abutting windows fuse without changing the parent's rounding."""
        sliced = float_coord[3:93:2]
        result = concat_coords(sliced[:13], sliced[13:])
        assert result.runs_count == 1
        assert result.values.tobytes() == sliced.values.tobytes()
        assert result.data_id == sliced.data_id

    @pytest.mark.parametrize(
        "indexer", [slice(3, 65), slice(2, None, 3), slice(None, None, -1)]
    )
    def test_multiple_runs(self, float_coord, indexer):
        """Striding across a hole stays compact and keeps every selected label."""
        coord = concat_coords(float_coord[:30], float_coord[40:80])
        result = coord[indexer]
        assert result.values.tobytes() == coord.values[indexer].tobytes()
        assert all(isinstance(run, Grid) for run in result.runs)
        assert not result.sources

    def test_patch_indexing(self, float_coord):
        """Patch indexing keeps data paired with its exact coordinate labels."""
        patch = dc.Patch(
            data=np.arange(100), coords={"distance": float_coord}, dims=("distance",)
        )
        out = patch.isel(distance=slice(3, 73, 2))
        assert (
            out.get_coord("distance").values.tobytes()
            == float_coord.values[3:73:2].tobytes()
        )
        assert np.array_equal(out.data, patch.data[3:73:2])

    def test_large_slice_stays_compact(self, monkeypatch):
        """A billion-label axis can be sliced without evaluating its labels."""
        coord = get_coord(start=0.1, step=0.1, shape=(10**9,))
        original = Grid.labels

        def bounded_labels(self, indices, dtype):
            assert np.asarray(indices).size <= 4
            return original(self, indices, dtype)

        monkeypatch.setattr(Grid, "labels", bounded_labels)
        result = coord[3:-5:7][::-1]
        assert result.evenly_sampled and not result.sources
        assert len(result) == len(range(10**9)[3:-5:7])
        assert result[0] == coord[range(10**9)[3:-5:7][-1]]

    @pytest.mark.parametrize("field", ["parent_count", "stride"])
    def test_invalid_window(self, field):
        """A persisted window cannot use an empty parent or a zero stride."""
        coord = get_coord(start=0.1, step=0.1, shape=(100,))[::2]
        payload = coord.model_dump()
        payload["runs"][0][field] = 0
        with pytest.raises(ValueError, match="positive length and nonzero stride"):
            get_coord(**payload)
