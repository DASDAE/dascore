"""Tests for coordinate processing methods."""
from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import ParameterError
from dascore.units import get_quantity


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
            new = patch.update(data=array).sort_coords(dim)
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
    """Tests for attaching coordinate(s) to a patch."""

    def get_line_func(self, df, x_col, y_col):
        """Get a function which predicts value of y based on x."""
        assert len(df) == 2
        x1, x2 = df[x_col].min(), df[x_col].max()
        y1, y2 = df[y_col].min(), df[y_col].max()
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - x1 * slope

        def _func(x):
            return x * slope + intercept

        return _func

    @pytest.fixture()
    def coord_df(self, brady_hs_das_dts_coords, random_patch):
        """Get a coordinate dataframe that is compatible with random_patch."""
        dist_max = random_patch.coords.max("distance")
        df = (
            brady_hs_das_dts_coords.rename(columns={"Channel": "distance"})
            .sample(frac=1 / 5)
            .loc[lambda x: x["distance"] <= dist_max]
            .sort_values("distance")
            .reset_index(drop=True)
        )
        return df

    def test_interpolation_no_extrapolate(self, random_patch, coord_df):
        """Ensure interpolated values follow expected line without extrapolation."""
        # get a dataframe with only two points and interpolate.
        sub_df = coord_df.iloc[[0, -1]]
        out = random_patch.coords_from_df(sub_df, extrapolate=False)
        dist = out.coords.get_array("distance")
        for col in set(sub_df.columns[1:]) - set(out.dims):
            vals = out.coords.get_array(col)
            expected = self.get_line_func(sub_df, "distance", col)(dist)
            close = np.isclose(vals, expected)
            nan = np.isnan(vals)
            assert np.all(close | nan)

    def test_interpolation_with_extrapolate(self, random_patch, coord_df):
        """Ensure interpolated values follow expected line with extrapolation."""
        # get a dataframe with only two points and interpolate.
        sub_df = coord_df.iloc[[0, -1]]
        out = random_patch.coords_from_df(sub_df, extrapolate=True)
        dist = out.coords.get_array("distance")
        for col in set(sub_df.columns[1:]) - set(out.dims):
            vals = out.coords.get_array(col)
            expected = self.get_line_func(sub_df, "distance", col)(dist)
            assert np.allclose(vals, expected)

    def test_no_extrapolate(self, random_patch, coord_df):
        """Ensure when out of range, we use nans when extrapolate is False."""
        out = random_patch.coords_from_df(coord_df, extrapolate=False)
        coords = out.coords
        dist = coords.get_array("distance")
        df_dist = coord_df["distance"]
        # iterate each new coord and ensure out of bound values are NaN
        # but in bound are not.
        in_dist = (dist <= df_dist.max()) & (dist >= df_dist.min())
        for name in set(coord_df.columns) - set(random_patch.dims):
            vals = out.coords.get_array(name)
            assert np.all(np.isnan(vals[~in_dist]))
            assert np.all(~np.isnan(vals[in_dist]))

    def test_extrapolate(self, random_patch, coord_df):
        """Ensure we can extrapolate outside coordinate range."""
        out = random_patch.coords_from_df(coord_df, extrapolate=True)
        # All values should be filled in when extrapolating.
        for name in set(coord_df.columns) - set(random_patch.dims):
            vals = out.coords.get_array(name)
            assert np.all(~np.isnan(vals))

    def test_units(self, random_patch, coord_df):
        """Test passing in unit dictionary."""
        units = {x: "m" for x in coord_df.columns[1:]}
        out = random_patch.coords_from_df(coord_df, units=units, extrapolate=True)
        for char in "XYZ":
            coord = out.get_coord(char)
            assert coord.units == get_quantity("m")

    def test_no_dim_column_raises(self, random_patch, coord_df):
        """Ensure when no columns overlap with coords an error is raised."""
        bad_df = coord_df.drop(columns="distance")
        with pytest.raises(ParameterError, match="Exactly one column"):
            random_patch.coords_from_df(bad_df)
