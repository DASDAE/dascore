"""Tests for coordinate processing methods."""
from __future__ import annotations

import numpy as np


class TestSortCoords:
    """Test sorting patches' coordinates."""

    def test_forward_sort(self, wacky_dim_patch):
        """Test sort both dims forward."""
        # default sort should just sort both dims.
        out = wacky_dim_patch.sort_coords()
        for name, coord in out.coords.coord_map.items():
            if name not in out.dims:
                continue
            assert coord.sorted

    def test_reverse_sort(self, wacky_dim_patch):
        """Test sort both dims backward."""
        out = wacky_dim_patch.sort_coords(reverse=True)
        for name, coord in out.coords.coord_map.items():
            if name not in out.dims:
                continue
            assert coord.reverse_sorted

    def test_data_sorted_correctly(self, wacky_dim_patch):
        """Simple test to ensure data were sorted correctly."""
        patch = wacky_dim_patch
        shape = patch.shape
        dims = patch.dims
        # get array that counts cols and rows
        array_list = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
        )
        assert all([x.shape == patch.shape for x in array_list])
        # iterate each array and manually test sorting of data matches expected.
        for dim, array, ind in zip(dims, array_list, (1, 0)):
            coord = patch.coords.coord_map[dim]
            arg_sort = np.argsort(coord.values)
            new = patch.new(data=array).sort_coords(dim)
            data_along_slice = np.take(new.data, 0, ind)
            assert np.all(np.equal(arg_sort, data_along_slice))


class TestSnapDims:
    """Tests for snapping dimensions."""

    def test_snap_monotonic(self, wacky_dim_patch):
        """Ensure we can snap a single monotonic coordinate."""
        out = wacky_dim_patch.snap_coords("time")
        coord = out.coords.coord_map["time"]
        assert coord.sorted
        assert coord.evenly_sampled

    def test_snap_array(self, wacky_dim_patch):
        """Ensure we can snap a non monotonic coordinate."""
        out = wacky_dim_patch.snap_coords("distance")
        coord = out.coords.coord_map["distance"]
        assert coord.sorted
        assert coord.evenly_sampled

    def test_snap_dims(self, wacky_dim_patch):
        """Ensure we can snap a non monotonic coordinate."""
        out = wacky_dim_patch.snap_coords()
        for dim in out.dims:
            coord = out.coords.coord_map[dim]
            assert coord.sorted
            assert coord.evenly_sampled
