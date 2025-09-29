"""Tests for coordinate processing methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.proc.coords
from dascore.compat import is_array
from dascore.core.coords import BaseCoord
from dascore.exceptions import (
    CoordError,
    ParameterError,
    PatchBroadcastError,
    PatchCoordinateError,
    PatchError,
)
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


class TestSelect:
    """Tests for selecting data from patch."""

    def _add_non_dim_coords(self, patch, coord_dict):
        """Helper to add non-dimensional coordinates to a patch."""
        new_coords = patch.coords.update(**coord_dict)
        return patch.new(coords=new_coords)

    def _assert_coord_unchanged(self, original_patch, selected_patch, coord_name):
        """Helper to assert a coordinate remains unchanged."""
        assert np.array_equal(
            original_patch.coords.get_array(coord_name),
            selected_patch.coords.get_array(coord_name),
        )

    def _assert_data_shape_unchanged(self, original_patch, selected_patch):
        """Helper to assert data shape remains unchanged."""
        assert selected_patch.data.shape == original_patch.data.shape
        assert np.array_equal(selected_patch.data, original_patch.data)

    def test_select_by_distance(self, random_patch):
        """Ensure distance can be used to filter patch."""
        dmin, dmax = 100, 200
        pa = random_patch.select(distance=(dmin, dmax))
        assert pa.data.shape < random_patch.data.shape
        # the attrs should have updated as well
        assert pa.attrs["distance_min"] >= 100
        assert pa.attrs["distance_max"] <= 200

    def test_select_by_absolute_time(self, random_patch):
        """Ensure the data can be sub-selected using absolute time."""
        shape = random_patch.data.shape
        t1 = random_patch.attrs["time_min"] + np.timedelta64(1, "s")
        t2 = t1 + np.timedelta64(3, "s")

        pa1 = random_patch.select(time=(None, t1))
        assert pa1.attrs["time_max"] <= t1
        assert pa1.data.shape < shape

        pa2 = random_patch.select(time=(t1, None))
        assert pa2.attrs["time_min"] >= t1
        assert pa2.data.shape < shape

        tr3 = random_patch.select(time=(t1, t2))
        assert tr3.attrs["time_min"] >= t1
        assert tr3.attrs["time_max"] <= t2
        assert tr3.data.shape < shape

    def test_select_out_of_bounds_time(self, random_patch):
        """Selecting out of coordinate range should leave patch unchanged."""
        # this equates to a timestamp of 1 (eg 1 sec after 1970)
        pa1 = random_patch.select(time=(1, None))
        assert pa1 == random_patch
        # it should also work with proper datetimes.
        t1 = random_patch.attrs["time_min"] - dc.to_timedelta64(1)
        pa2 = random_patch.select(time=(t1, None))
        assert pa2 == random_patch

    def test_select_distance_leaves_time_attr_unchanged(self, random_patch):
        """Ensure selecting on distance doesn't change time."""
        dist = random_patch.coords.get_array("distance")
        dist_max, dist_mean = np.max(dist), np.mean(dist)
        out = random_patch.select(distance=(dist_mean, dist_max - 1))
        assert out.attrs["time_max"] == out.coords.max("time")

    def test_select_emptify_array(self, random_patch):
        """If select range excludes data range patch should be emptied."""
        out = random_patch.select(distance=(-100, -10))
        assert len(out.shape) == len(random_patch.shape)
        deleted_axis = random_patch.get_axis("distance")
        assert out.shape[deleted_axis] == 0
        assert np.size(out.data) == 0

    def test_select_relative_start_end(self, random_patch):
        """Ensure relative select works on start to end."""
        patch1 = random_patch.select(time=(1, -1), relative=True)
        t1 = random_patch.attrs.time_min + dc.to_timedelta64(1)
        t2 = random_patch.attrs.time_max - dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_end_end(self, random_patch):
        """Ensure relative works for end to end."""
        patch1 = random_patch.select(time=(-3, -1), relative=True)
        t1 = random_patch.attrs.time_max - dc.to_timedelta64(1)
        t2 = random_patch.attrs.time_max - dc.to_timedelta64(3)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_start_start(self, random_patch):
        """Ensure relative start ot start."""
        patch1 = random_patch.select(time=(1, 3), relative=True)
        t1 = random_patch.attrs.time_min + dc.to_timedelta64(1)
        t2 = random_patch.attrs.time_min + dc.to_timedelta64(3)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_start_open(self, random_patch):
        """Ensure relative start to open end."""
        patch1 = random_patch.select(time=(1, None), relative=True)
        t1 = random_patch.attrs.time_min + dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, None))
        assert patch1 == patch2

    def test_select_relative_end_open(self, random_patch):
        """Ensure relative start to open end."""
        patch1 = random_patch.select(time=(-1, None), relative=True)
        t1 = random_patch.attrs.time_max - dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, None))
        assert patch1 == patch2

    def test_time_slice_samples(self, random_patch):
        """Ensure a simple time slice works."""
        pa1 = random_patch.select(time=(1, 5), samples=True)
        pa2 = random_patch.select(time=slice(1, 5), samples=True)
        assert pa1 == pa2

    def test_non_slice_samples(self, random_patch):
        """Ensure a non-slice doesnt change patch."""
        pa1 = random_patch.select(distance=(..., ...), samples=True)
        pa2 = random_patch.select(distance=(None, ...), samples=True)
        pa3 = random_patch.select(distance=slice(None, None), samples=True)
        pa4 = random_patch.select(distance=...)
        assert pa1 == pa2 == pa3 == pa4

    def test_iselect_deprecated(self, random_patch):
        """Ensure Patch.iselect raises deprecation error."""
        msg = "iselect is deprecated"
        with pytest.warns(DeprecationWarning, match=msg):
            _ = random_patch.iselect(time=(10, -10))

    def test_select_history_outside_bounds(self, random_patch):
        """Selecting outside the bounds should do nothing to history."""
        attrs = random_patch.attrs
        dt = dc.to_timedelta64(1)
        time = (attrs["time_min"] - dt, attrs["time_max"] + dt)
        dist = (attrs["distance_min"] - 1, attrs["distance_max"] + 1)
        new = random_patch.select(time=time, distance=dist)
        # if no select performed everything should be identical.
        assert new.equals(random_patch, only_required_attrs=False)

    def test_patch_non_coord(self, random_patch):
        """Test select for a patch with a non coord."""
        new_shape = tuple([*random_patch.shape, 10])
        patch = random_patch.append_dims("face_angle").make_broadcastable_to(new_shape)
        face_angle = patch.get_coord("face_angle")
        new = patch.select(face_angle=(face_angle.min(), face_angle.max()))
        assert new == patch

    def test_select_nonexistent_coordinate_raises_error(self, random_patch):
        """
        Test that selecting on a non-existing coordinate
        raises PatchCoordinateError.
        """
        # Try to select on a coordinate that doesn't exist
        with pytest.raises(PatchCoordinateError, match="nonexistent_coord"):
            random_patch.select(nonexistent_coord=(0, 10))

    def test_select_multiple_nonexistent_coordinates_raises_error(self, random_patch):
        """
        Test that selecting on multiple non-existing coordinates raises
        PatchCoordinateError.
        """
        # Try to select on multiple coordinates that don't exist
        with pytest.raises(PatchCoordinateError, match=r"coord1.*coord2"):
            random_patch.select(bad_coord1=(0, 10), bad_coord2=(5, 15))

    def test_select_mix_valid_invalid_coordinates_raises_error(self, random_patch):
        """
        Test that mixing valid and invalid coordinates raises
        PatchCoordinateError.
        """
        # Try to select on a mix of valid and invalid coordinates
        with pytest.raises(PatchCoordinateError, match="invalid_coord"):
            random_patch.select(time=(0, 1), invalid_coord=(0, 10))

    def test_select_non_dim_coord_with_boolean_mask(self, random_patch):
        """Test selecting non-dimensional coordinates using boolean masks on Patch."""
        # Add a non-dimensional coordinate to the patch
        quality_values = np.random.RandomState(42).rand(10)
        patch_with_coord = self._add_non_dim_coords(
            random_patch, {"quality": (None, quality_values)}
        )

        # Create boolean mask and select
        mask = quality_values > 0.5
        selected_patch = patch_with_coord.select(quality=mask)

        # Only the non-dimensional coordinate should be affected
        expected_quality = quality_values[mask]
        assert np.array_equal(
            selected_patch.coords.get_array("quality"), expected_quality
        )

        # Patch data and dimensional coordinates should remain unchanged
        self._assert_data_shape_unchanged(patch_with_coord, selected_patch)
        self._assert_coord_unchanged(patch_with_coord, selected_patch, "time")
        self._assert_coord_unchanged(patch_with_coord, selected_patch, "distance")

    def test_select_non_dim_coord_with_array_indices(self, random_patch):
        """Test selecting non-dimensional coordinates using array indices on Patch."""
        # Add non-dimensional coordinates to the patch
        sensor_ids = np.arange(100, 115)  # 15 values
        temperature_data = np.random.RandomState(42).rand(15) * 100
        patch_with_coords = self._add_non_dim_coords(
            random_patch,
            {"sensor_ids": (None, sensor_ids), "temperature": (None, temperature_data)},
        )

        # Select subset using array indices
        selected_indices = np.array([2, 5, 8, 12])
        selected_patch = patch_with_coords.select(
            sensor_ids=selected_indices, samples=True
        )

        # Check that only the selected coordinate was affected
        expected_ids = sensor_ids[selected_indices]
        assert np.array_equal(
            selected_patch.coords.get_array("sensor_ids"), expected_ids
        )
        # Temperature should remain unchanged since we only selected sensor_ids
        self._assert_coord_unchanged(patch_with_coords, selected_patch, "temperature")

        # Patch data and dimensional coordinates should be unchanged
        self._assert_data_shape_unchanged(patch_with_coords, selected_patch)
        for dim in patch_with_coords.dims:
            self._assert_coord_unchanged(patch_with_coords, selected_patch, dim)

    def test_select_mixed_dim_and_non_dim_coords(self, random_patch):
        """
        Test selecting both dimensional and non-dimensional coordinates
        simultaneously.
        """
        # Add non-dimensional coordinate with numeric values
        station_ids = np.array([101, 102, 103, 104, 105])
        patch_with_coord = self._add_non_dim_coords(
            random_patch, {"station_ids": (None, station_ids)}
        )

        # Select on both dimensional and non-dimensional coordinates
        time_coord = patch_with_coord.coords.get_array("time")
        time_subset = time_coord[: len(time_coord) // 2]  # First half of time
        station_mask = np.array(
            [True, False, True, False, True]
        )  # Select 101, 103, 105

        selected_patch = patch_with_coord.select(
            time=time_subset, station_ids=station_mask
        )

        # Check both selections worked
        expected_stations = station_ids[station_mask]  # [101, 103, 105]
        assert np.array_equal(
            selected_patch.coords.get_array("station_ids"), expected_stations
        )
        assert np.array_equal(selected_patch.coords.get_array("time"), time_subset)

        # Check dimensional selection affected patch shape
        assert selected_patch.data.shape != patch_with_coord.data.shape
        assert len(selected_patch.coords.get_array("time")) == len(time_subset)

    def test_select_non_dim_coord_associated_with_dimension(self, random_patch):
        """
        Test selecting non-dimensional coordinates that are associated with a
        dimension.
        """
        # Add a non-dimensional coordinate associated with distance dimension
        distance_coord = random_patch.coords.get_array("distance")
        elevation_values = np.random.RandomState(42).rand(len(distance_coord)) * 1000
        patch_with_coord = self._add_non_dim_coords(
            random_patch, {"elevation": ("distance", elevation_values)}
        )

        # Select using boolean mask on the elevation coordinate
        elevation_subset = elevation_values > 500  # Select high elevations
        selected_patch = patch_with_coord.select(elevation=elevation_subset)

        # Both elevation and distance coordinates should be affected
        expected_elevation = elevation_values[elevation_subset]
        expected_distance = distance_coord[elevation_subset]
        assert np.array_equal(
            selected_patch.coords.get_array("elevation"), expected_elevation
        )
        assert np.array_equal(
            selected_patch.coords.get_array("distance"), expected_distance
        )

        # Patch data shape should change because distance dimension changed
        assert selected_patch.data.shape != patch_with_coord.data.shape
        distance_axis = patch_with_coord.get_axis("distance")
        assert selected_patch.data.shape[distance_axis] == np.sum(elevation_subset)

        # Time coordinate should remain unchanged
        self._assert_coord_unchanged(patch_with_coord, selected_patch, "time")

    def test_select_non_dim_coord_with_array_values(self, random_patch):
        """Test selecting non-dimensional coordinates using specific array values."""
        # Add coordinates with same length as distance dimension
        distance_coord = random_patch.coords.get_array("distance")
        sensor_ids = np.arange(1000, 1000 + len(distance_coord))
        fiber_quality = np.random.RandomState(42).rand(len(distance_coord))
        patch_with_coord = self._add_non_dim_coords(
            random_patch,
            {
                "sensor_ids": ("distance", sensor_ids),
                "fiber_quality": ("distance", fiber_quality),
            },
        )

        # Select specific sensor IDs by array values (select a subset)
        selected_ids = sensor_ids[10:15]  # Select 5 sensors
        selected_patch = patch_with_coord.select(sensor_ids=selected_ids)

        # All coordinates tied to distance should be affected
        expected_quality = fiber_quality[10:15]
        expected_distance = distance_coord[10:15]
        assert np.array_equal(
            selected_patch.coords.get_array("sensor_ids"), selected_ids
        )
        assert np.array_equal(
            selected_patch.coords.get_array("fiber_quality"), expected_quality
        )
        assert np.array_equal(
            selected_patch.coords.get_array("distance"), expected_distance
        )

        # Data shape should change along distance axis
        distance_axis = patch_with_coord.get_axis("distance")
        assert selected_patch.data.shape[distance_axis] == 5

    def test_select_coord_tied_to_time_dimension(self, random_patch):
        """Test selecting coordinates associated with time dimension."""
        # Add a coordinate tied to time (e.g., measurement quality over time)
        time_coord = random_patch.coords.get_array("time")
        quality_over_time = np.random.RandomState(42).rand(len(time_coord))
        patch_with_coord = self._add_non_dim_coords(
            random_patch, {"measurement_quality": ("time", quality_over_time)}
        )

        # Select time periods with high quality measurements
        high_quality_mask = quality_over_time > 0.7
        selected_patch = patch_with_coord.select(measurement_quality=high_quality_mask)

        # Both quality and time coordinates should be affected
        expected_quality = quality_over_time[high_quality_mask]
        expected_time = time_coord[high_quality_mask]
        assert np.array_equal(
            selected_patch.coords.get_array("measurement_quality"), expected_quality
        )
        assert np.array_equal(selected_patch.coords.get_array("time"), expected_time)

        # Data shape should change along time axis
        time_axis = patch_with_coord.get_axis("time")
        assert selected_patch.data.shape[time_axis] == np.sum(high_quality_mask)

        # Distance coordinate should remain unchanged
        self._assert_coord_unchanged(patch_with_coord, selected_patch, "distance")


class TestOrder:
    """Tests for ordering Patches."""

    def test_simple_ordering(self, random_patch):
        """Ensure order changes to specify on patch."""
        dist = random_patch.get_array("distance")
        new_dist = dist[1:5][::-1]
        new = random_patch.order(distance=new_dist)
        assert np.all(new.get_array("distance") == new_dist)

    def test_duplicate_data(self, random_patch):
        """Duplicate the data along time dimension."""
        out = random_patch.order(time=[0, 0, 0], samples=True)
        assert isinstance(out, dc.Patch)

    def test_copy(self, random_patch):
        """Ensure copy creates a copy of the data array."""
        out = random_patch.order(time=[1, 2, 3], samples=True, copy=True)
        assert isinstance(out.data, np.ndarray)


class TestAppendDims:
    """Tests for appending dummy dimensions to data array."""

    def test_no_dims_unchanged_patch(self, random_patch):
        """Ensure no kwargs yields equal patches."""
        out = random_patch.append_dims()
        assert out == random_patch

    def test_flat_dimension(self, random_patch):
        """Ensure a flat dimension only expands dimensionality."""
        out = random_patch.append_dims(new=[1])
        assert len(out.shape) == (len(random_patch.shape) + 1)
        # New dim should show up at the end.
        assert out.dims[-1] == "new"
        coord = out.coords.get_array("new")
        assert np.all(coord == np.array([1]))
        # The flatten data should remain the same.
        assert np.allclose(out.data.flatten(), random_patch.data.flatten())

    def test_non_coordinate_dim(self, random_patch):
        """Ensure we can add non dimensional coordinates."""
        out = random_patch.append_dims(new=2)
        assert "new" in out.dims
        assert out.size == random_patch.size * 2
        assert out.shape[-1] == 2

    def test_expand_dims(self, random_patch):
        """Ensure dimensions can be expanded."""
        out = random_patch.append_dims(new=[1, 2])
        assert len(out.shape) == (len(random_patch.shape) + 1)
        # New dim should show up at the end.
        assert out.dims[-1] == "new"
        coord = out.coords.get_array("new")
        assert np.all(coord == np.array([1, 2]))

    def test_expand_multiple_dims(self, random_patch):
        """Ensure several dimensions can be expanded."""
        small_patch = random_patch.select(
            time=(1, 4),
            distance=(1, 6),
            samples=True,
        )
        out = small_patch.append_dims(new=[1, 2], old=[1, 2])
        assert len(out.shape) == (len(random_patch.shape) + 2)
        # New dim should show up at the end, in order.
        assert out.dims[-2:] == ("new", "old")

    def test_append_with_args(self, random_patch):
        """Ensure we can append with just the name of the dim."""
        out = random_patch.append_dims("new", "dim")
        assert list(out.dims) == [*list(random_patch.dims), "new", "dim"]

    def test_append_with_args_and_kwargs(self, random_patch):
        """Ensure we can use both kwargs and args."""
        out = random_patch.append_dims("new", new2=2)
        assert list(out.dims) == [*list(random_patch.dims), "new", "new2"]


class TestSqueeze:
    """Tests for squeeze."""

    @pytest.fixture(scope="class")
    def flat_patch(self):
        """Create a patch with a degenerate dimension."""
        data = np.atleast_2d(np.arange(10))
        coords = {"time": np.arange(10), "distance": np.array([1])}
        dims = ("distance", "time")
        out = dc.Patch(data=data, dims=dims, coords=coords)
        assert 1 in out.shape
        return out

    def test_remove_dimension(self, flat_patch):
        """Tests for removing degenerate dimensions."""
        out = flat_patch.squeeze("distance")
        assert "distance" not in out.dims
        assert len(out.data.shape) == 1, "data should be 1d"

    def test_tutorial_example(self, random_patch):
        """Ensure the tutorial snippet works."""
        patch = random_patch.select(distance=0, samples=True)
        squeezed = patch.squeeze()
        assert len(squeezed.dims) < len(patch.dims)

    def test_non_zero_length_raises(self, flat_patch):
        """Ensure squeezing a non-flat dim raises helpful error."""
        msg = "because it has non-zero length"
        with pytest.raises(CoordError, match=msg):
            flat_patch.squeeze(dim="time")

    def test_coord_summary(self, flat_patch):
        """Ensure the coordinate summary doesn't contain squeezed dim."""
        patch = flat_patch.squeeze()
        attrs = patch.attrs
        coords = attrs.get("coords", None)
        if coords:
            assert set(coords) == set(patch.coords.coord_map)


class TestGetCoord:
    """Tests for the get_coord convenience function."""

    def test_returns_coord(self, random_patch):
        """Simply ensure a coordinate is returned."""
        for dim in random_patch.dims:
            coord = random_patch.get_coord(dim)
            assert isinstance(coord, BaseCoord)

    def test_require_sorted(self, wacky_dim_patch):
        """Test required sorted raises if coord isn't sorted."""
        msg = "is not sorted"
        with pytest.raises(CoordError, match=msg):
            wacky_dim_patch.get_coord("distance", require_sorted=True)
        # but this should work
        coord = wacky_dim_patch.get_coord("time", require_sorted=True)
        assert isinstance(coord, BaseCoord)

    def test_non_existent_coord_raises(self, random_patch):
        """Ensure requesting non-existent coordinates raises CoordError."""
        msg = "not found in Patch"
        with pytest.raises(CoordError, match=msg):
            random_patch.get_coord("fire_house")

    def test_require_evenly_sampled(self, wacky_dim_patch):
        """Test required evenly sampled raises if coord isn't."""
        msg = "is not evenly sampled"
        with pytest.raises(CoordError, match=msg):
            wacky_dim_patch.get_coord("distance", require_evenly_sampled=True)
        with pytest.raises(CoordError, match=msg):
            wacky_dim_patch.get_coord("time", require_evenly_sampled=True)


class TestMakeBroadcastable:
    """Tests for making patches broadcastable to differnt shapes."""

    def test_broadcast_non_coords(self, random_patch):
        """Ensure non-coords of length 1 can broadcast."""
        collapsed_patch = random_patch.sum()
        shape = (2, 2)
        patch = collapsed_patch.make_broadcastable_to(shape)
        assert patch.shape == shape

    def test_raises_real_coord(self, random_patch):
        """If the dimension has values, it shouldn't be broadcastable."""
        patch = random_patch.select(time=1, distance=2, samples=True)
        # The shape is broadcastable, but the coords exist so it cant
        # broadcast.
        shape = (1, 2)
        msg = "Cannot broadcast non-empty coord"
        with pytest.raises(PatchBroadcastError, match=msg):
            patch.make_broadcastable_to(shape)

    def test_incompatible_shapes(self, random_patch):
        """Incompatible shapes should raise."""
        patch = random_patch.select(time=1, samples=True)
        shape = (12, 12)
        msg = "objects cannot be broadcast to a single shape"
        with pytest.raises(ValueError, match=msg):
            patch.make_broadcastable_to(shape)

    def test_broadcastable_to_current_shape(self, random_patch):
        """Making broadcastable to current shape should do nothing."""
        patch = random_patch
        out = patch.make_broadcastable_to(patch.shape)
        assert out == patch


class TestGetArray:
    """Tests for getting data/coordinate array."""

    def test_patch_data(self, random_patch):
        """Ensure no arguments returns patch data."""
        out = random_patch.get_array()
        assert out is random_patch.data

    def test_patch_coord_array(self, random_patch):
        """Ensure we can also get arrays from coordinates."""
        for dim in random_patch.dims:
            array = random_patch.get_array(dim)
            assert is_array(array)


class TestAddDistanceTo:
    """Tests for adding distance from a point to coords."""

    @pytest.fixture(scope="class")
    def shot_series(self):
        """Get the shot series."""
        shot = pd.Series({"x": 1000, "y": 42, "z": 15})
        return shot

    def test_coord(self, random_patch_with_xyz, shot_series):
        """Ensure coords have been added."""
        out = random_patch_with_xyz.add_distance_to(shot_series)
        assert "origin_distance" in out.coords.coord_map
        for name in shot_series.index:
            assert f"origin_{name}" in out.coords.coord_map

    def test_bad_name_raises(self, random_patch_with_xyz):
        """Ensure a PatchError is raised when index are not coords."""
        ser = pd.Series({"x": 1000, "y": 42, "q": 15})
        msg = "not patch coordinates"
        with pytest.raises(PatchError, match=msg):
            random_patch_with_xyz.add_distance_to(ser)

    def test_bad_association_raises(self, random_patch_with_xyz):
        """Ensure a PatchError is raised if coords dont share a dimension."""
        ser = pd.Series({"x": 1000, "y": 42, "time": 15})
        msg = "must be associated with the same dimension"
        with pytest.raises(PatchError, match=msg):
            random_patch_with_xyz.add_distance_to(ser)

    def test_sorting(self, random_patch_with_xyz, shot_series):
        """Ensure sorting can be done on the new patch."""
        out = random_patch_with_xyz.add_distance_to(shot_series)
        sorted_patch = out.sort_coords("origin_distance")
        coord = sorted_patch.get_array("origin_distance")
        assert np.all(np.sort(coord) == coord)


class TestGetAxis:
    """Tests for helper function to get get axis of dimension."""

    def test_index_comparable(self, random_patch):
        """Ensure get_axis is the same as patch.dims.index."""
        for dim in random_patch.dims:
            axis = random_patch.get_axis(dim)
            assert axis == random_patch.dims.index(dim)

    def test_raises(self, random_patch):
        """
        Ensure a nice error message is raised when asking for non-existent dim.
        """
        match = "has no dimension"
        with pytest.raises(CoordError, match=match):
            random_patch.get_axis(dim="money")
