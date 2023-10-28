"""Tests for coordinate processing methods."""
from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import ParameterError


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


class TestDropCoords:
    """Tests for dropping coordinates."""

    def test_drop_non_dim(self, random_patch_with_lat_lon):
        """Ensure non_dim coords can be dropped."""
        out = random_patch_with_lat_lon.drop_coords("latitude")
        assert "latitude" not in out.coords.coord_map

    def test_drop_dim_raises(self, random_patch):
        """Ensure a dimensional coordinate can be dropped."""
        msg = "Cannot drop dimensional coordinates"
        with pytest.raises(ParameterError, match=msg):
            random_patch.drop_coords("time")


class TestCoordsFromDf:
    """Tests for attaching coordinate(s) to a dimension."""

    def test_no_extrapolte(self, random_patch, brady_hs_DAS_DTS_coords):
        """Ensure when out of range, we can use nans when extrapolate is False."""
        df_select = brady_hs_DAS_DTS_coords[
            brady_hs_DAS_DTS_coords["distance"] % 5 == 0
        ]
        brady_hs_DAS_DTS_coords = df_select[df_select["distance"] < 250]
        out = random_patch.coords_from_df(df_select)

        coords = out.coords
        X = coords.get_array("X")
        Y = coords.get_array("Y")
        Z = coords.get_array("Z")
        dist = coords.get_array("distance")

        assert (
            max((X[30:-50] - brady_hs_DAS_DTS_coords["X"][: len(dist) - 80]) / X[30])
            < 0.01
        )
        assert (
            max((Y[30:-50] - brady_hs_DAS_DTS_coords["Y"][: len(dist) - 80]) / Y[30])
            < 0.01
        )
        assert (
            max((Z[30:-50] - brady_hs_DAS_DTS_coords["Z"][: len(dist) - 80]) / Z[30])
            < 0.01
        )
        assert np.all(np.isnan(X[:30])) and np.all(np.isnan(X[-50:]))
        assert np.all(np.isnan(Y[:30])) and np.all(np.isnan(Y[-50:]))
        assert np.all(np.isnan(Z[:30])) and np.all(np.isnan(Z[-50:]))

    def test_extrapolate(self, random_patch, brady_hs_DAS_DTS_coords):
        """Ensure we can extrapolate outside coordinate range."""
        df_select = brady_hs_DAS_DTS_coords[
            brady_hs_DAS_DTS_coords["distance"] % 5 == 0
        ]
        brady_hs_DAS_DTS_coords = df_select[df_select["distance"] < 250]
        out = random_patch.coords_from_df(df_select, extrapolate=True)

        coords = out.coords
        X = coords.get_array("X")
        Y = coords.get_array("Y")
        Z = coords.get_array("Z")
        dist = coords.get_array("distance")

        assert (
            max((X[30:] - brady_hs_DAS_DTS_coords["X"][: len(dist) - 30]) / X[30])
            < 0.01
        )
        assert (
            max((Y[30:] - brady_hs_DAS_DTS_coords["Y"][: len(dist) - 30]) / Y[30])
            < 0.01
        )
        assert (
            max((Z[30:] - brady_hs_DAS_DTS_coords["Z"][: len(dist) - 30]) / Z[30])
            < 0.01
        )

    def test_units(self, random_patch, brady_hs_DAS_DTS_coords):
        """Ensure we can extrapolate outside coordinate range."""
        units = {}
        for a in brady_hs_DAS_DTS_coords.columns[1:]:
            units[a] = "m"

        df_select = brady_hs_DAS_DTS_coords[
            brady_hs_DAS_DTS_coords["distance"] % 5 == 0
        ]
        brady_hs_DAS_DTS_coords = df_select[df_select["distance"] < 250]
        out = random_patch.coords_from_df(df_select, units=units, extrapolate=True)

        coords = out.coords
        coords.get_array("X")
        coords.get_array("Y")
        coords.get_array("Z")
        coords.get_array("distance")
