"""Tests for coordinate processing methods."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import dascore as dc
import dascore.proc.coords
from dascore.compat import is_array
from dascore.core.coords import (
    BaseCoord,
    CoordMonotonicArray,
    CoordRange,
    _fill_layout,
    concat_coords,
    get_coord,
)
from dascore.exceptions import (
    CoordError,
    ParameterError,
    PatchBroadcastError,
    PatchCoordinateError,
    PatchError,
)
from dascore.units import get_quantity
from dascore.utils.gaps import GapTolerance
from dascore.warnings import DASCoreWarning


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

    def test_noop_sort_returns_self(self, random_patch):
        """Sorting already-sorted coords should return the same objects."""
        coords = random_patch.coords
        new_coords, array = coords.sort(array=random_patch.data)
        assert new_coords is coords
        assert array is random_patch.data
        assert random_patch.sort_coords() is random_patch


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

    @pytest.fixture(scope="class")
    def even_time_uneven_distance_patch(self):
        """A patch with an even (CoordRange) time and monotonic-uneven distance."""
        time = dc.to_datetime64(np.arange(20))
        distance = np.cumsum(np.arange(1, 11) ** 1.5)
        data = np.arange(len(time) * len(distance)).reshape(len(time), len(distance))
        patch = dc.Patch(
            data=data.astype(np.float64),
            coords={"time": time, "distance": distance},
            dims=("time", "distance"),
        )
        # sanity: time is even, distance is monotonic but not evenly sampled
        assert patch.coords.coord_map["time"].evenly_sampled
        assert not patch.coords.coord_map["distance"].evenly_sampled
        return patch

    def test_snap_already_even_returns_self(self, random_patch):
        """Snapping an already-even, sorted patch returns the same objects."""
        coords = random_patch.coords
        new_coords, array = coords.snap(array=random_patch.data)
        assert new_coords is coords
        assert array is random_patch.data
        assert random_patch.snap_coords() is random_patch
        assert random_patch.snap_coords("time", "distance") is random_patch

    def test_snap_changes_only_uneven_coord(self, even_time_uneven_distance_patch):
        """Only the coordinate that must change is replaced; others are reused."""
        patch = even_time_uneven_distance_patch
        out = patch.snap_coords()
        # distance was uneven, so it should now be evenly sampled
        assert out.coords.coord_map["distance"].evenly_sampled
        # time was already even; its coordinate object should be reused as-is
        assert out.coords.coord_map["time"] is patch.coords.coord_map["time"]
        # data is unchanged because nothing needed reordering
        assert np.array_equal(out.data, patch.data)

    def test_snap_reverse_sorted(self, even_time_uneven_distance_patch):
        """Reverse snapping sorts descending and reorders data accordingly."""
        patch = even_time_uneven_distance_patch
        out = patch.snap_coords("distance", reverse=True)
        coord = out.coords.coord_map["distance"]
        assert coord.reverse_sorted
        assert coord.evenly_sampled
        # data columns should be reversed relative to the ascending snap
        ascending = patch.snap_coords("distance")
        assert np.array_equal(out.data, ascending.data[:, ::-1])


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

    @pytest.mark.parametrize(
        "form", [["latitude"], ("latitude",), {"latitude"}, iter(["latitude"])]
    )
    def test_drop_sequence(self, random_patch_with_lat_lon, form):
        """A sequence of names should behave exactly like the bare name."""
        patch = random_patch_with_lat_lon
        out = patch.drop_coords(form)
        expected = patch.drop_coords("latitude")
        assert "latitude" not in out.coords.coord_map
        # Compared to the bare-name call so that dropping too much fails too.
        assert set(out.coords.coord_map) == set(expected.coords.coord_map)

    def test_drop_mixed_args(self, random_patch_with_lat_lon):
        """Names and sequences of names should be usable together."""
        patch = random_patch_with_lat_lon
        out = patch.drop_coords("latitude", ["longitude"])
        dropped = {"latitude", "longitude"}
        assert not dropped & set(out.coords.coord_map)
        # Everything else has to survive.
        assert set(out.coords.coord_map) == set(patch.coords.coord_map) - dropped

    def test_drop_dim_in_sequence_raises(self, random_patch):
        """A dimension inside a sequence should raise like a bare one."""
        msg = "Cannot drop dimensional coordinates"
        with pytest.raises(ParameterError, match=msg):
            random_patch.drop_coords(["time"])


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

    def test_select_infinite_bound(self, random_patch):
        """An infinite bound is an open one, like ... or None."""
        coord = random_patch.get_coord("distance")
        middle = coord.values[len(coord) // 2]
        expected = random_patch.select(distance=(middle, ...))
        assert random_patch.select(distance=(middle, np.inf)).equals(expected)
        assert random_patch.select(distance=(-np.inf, ...)).equals(random_patch)
        # The infinite side must not swallow the finite one.
        assert expected.shape != random_patch.shape

    def test_select_by_distance(self, random_patch):
        """Ensure distance can be used to filter patch."""
        dmin, dmax = 100, 200
        pa = random_patch.select(distance=(dmin, dmax))
        assert pa.data.shape < random_patch.data.shape
        # the attrs should have updated as well
        assert pa.get_coord("distance").min() >= 100
        assert pa.get_coord("distance").max() <= 200

    def test_select_by_absolute_time(self, random_patch):
        """Ensure the data can be sub-selected using absolute time."""
        shape = random_patch.data.shape
        t1 = random_patch.get_coord("time").min() + np.timedelta64(1, "s")
        t2 = t1 + np.timedelta64(3, "s")

        pa1 = random_patch.select(time=(None, t1))
        assert pa1.get_coord("time").max() <= t1
        assert pa1.data.shape < shape

        pa2 = random_patch.select(time=(t1, None))
        assert pa2.get_coord("time").min() >= t1
        assert pa2.data.shape < shape

        tr3 = random_patch.select(time=(t1, t2))
        assert tr3.get_coord("time").min() >= t1
        assert tr3.get_coord("time").max() <= t2
        assert tr3.data.shape < shape

    def test_select_out_of_bounds_time(self, random_patch):
        """Selecting out of coordinate range should leave patch unchanged."""
        # this equates to a timestamp of 1 (eg 1 sec after 1970)
        pa1 = random_patch.select(time=(1, None))
        assert pa1 == random_patch
        # it should also work with proper datetimes.
        t1 = random_patch.get_coord("time").min() - dc.to_timedelta64(1)
        pa2 = random_patch.select(time=(t1, None))
        assert pa2 == random_patch

    def test_select_distance_leaves_time_attr_unchanged(self, random_patch):
        """Ensure selecting on distance doesn't change time."""
        dist = random_patch.coords.get_array("distance")
        dist_max, dist_mean = np.max(dist), np.mean(dist)
        out = random_patch.select(distance=(dist_mean, dist_max - 1))
        assert out.get_coord("time").max() == out.coords.max("time")

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
        t1 = random_patch.get_coord("time").min() + dc.to_timedelta64(1)
        t2 = random_patch.get_coord("time").max() - dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_end_end(self, random_patch):
        """Ensure relative works for end to end."""
        patch1 = random_patch.select(time=(-3, -1), relative=True)
        t1 = random_patch.get_coord("time").max() - dc.to_timedelta64(1)
        t2 = random_patch.get_coord("time").max() - dc.to_timedelta64(3)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_start_start(self, random_patch):
        """Ensure relative start ot start."""
        patch1 = random_patch.select(time=(1, 3), relative=True)
        t1 = random_patch.get_coord("time").min() + dc.to_timedelta64(1)
        t2 = random_patch.get_coord("time").min() + dc.to_timedelta64(3)
        patch2 = random_patch.select(time=(t1, t2))
        assert patch1 == patch2

    def test_select_relative_start_open(self, random_patch):
        """Ensure relative start to open end."""
        patch1 = random_patch.select(time=(1, None), relative=True)
        t1 = random_patch.get_coord("time").min() + dc.to_timedelta64(1)
        patch2 = random_patch.select(time=(t1, None))
        assert patch1 == patch2

    def test_select_relative_end_open(self, random_patch):
        """Ensure relative start to open end."""
        patch1 = random_patch.select(time=(-1, None), relative=True)
        t1 = random_patch.get_coord("time").max() - dc.to_timedelta64(1)
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

    def test_select_history_outside_bounds(self, random_patch):
        """Selecting outside the bounds should do nothing to history."""
        patch = random_patch
        dt = dc.to_timedelta64(1)
        time_coord = patch.get_coord("time")
        distance_coord = patch.get_coord("distance")
        time = (time_coord.min() - dt, time_coord.max() + dt)
        dist = (
            distance_coord.min() - 1,
            distance_coord.max() + 1,
        )
        new = patch.select(time=time, distance=dist)
        # if no select performed everything should be identical.
        assert new.equals(patch, only_required_attrs=False)

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

    def test_select_single_numpy_int(self, random_patch):
        """Ensure a single numpy int behaves the same as a python int."""
        sub1 = random_patch.select(time=10, samples=True)
        sub2 = random_patch.select(time=np.int64(10), samples=True)
        assert sub1 == sub2

    def test_single_sample_retains_step(self, random_patch):
        """Single-sample select of an evenly sampled coord keeps step. See #567."""
        orig = random_patch.get_coord("time")
        sub_patch = random_patch.select(time=1, samples=1)
        time = sub_patch.get_coord("time")
        assert time.step is not None
        assert time.step == orig.step


class TestUnselect:
    """Keeping what a selection would have removed."""

    def test_complements_select(self, random_patch):
        """Together the two account for every sample, and share none."""
        selector = (50, 200)
        kept = random_patch.select(distance=selector).get_array("distance")
        dropped = random_patch.unselect(distance=selector).get_array("distance")
        whole = random_patch.get_array("distance")
        assert not set(kept) & set(dropped)
        assert sorted([*kept, *dropped]) == sorted(whole)

    def test_interior_range_leaves_a_hole(self, random_patch):
        """
        The property which makes a range complement wrong for a spool.

        A patch can have samples removed from its middle; a spool would
        have to cut every patch into the pieces on either side.
        """
        out = random_patch.unselect(distance=(50, 200))
        coord = out.get_coord("distance")
        assert coord.step is None
        values = out.get_array("distance")
        assert not ((values >= 50) & (values <= 200)).any()
        assert values.min() < 50 < 200 < values.max()

    def test_data_follows_the_coordinate(self, random_patch):
        """The rows removed are the rows the selection would have kept."""
        axis = random_patch.dims.index("distance")
        values = random_patch.get_array("distance")
        out = random_patch.unselect(distance=(50, 200))
        wanted = random_patch.data.take(
            np.flatnonzero(~((values >= 50) & (values <= 200))), axis=axis
        )
        assert np.array_equal(out.data, wanted)

    def test_samples(self, random_patch):
        """A sample range is complemented in samples too."""
        out = random_patch.unselect(distance=(..., 10), samples=True)
        whole = random_patch.get_array("distance")
        assert np.array_equal(out.get_array("distance"), whole[10:])

    def test_relative(self, random_patch):
        """Relative selectors mean what they mean in select."""
        kept = random_patch.select(time=(1, None), relative=True)
        dropped = random_patch.unselect(time=(1, None), relative=True)
        assert len(kept.get_array("time")) + len(dropped.get_array("time")) == len(
            random_patch.get_array("time")
        )

    def test_unselecting_everything_empties_the_dimension(self, random_patch):
        """Removing the whole span is legal, and says so with a shape."""
        coord = random_patch.get_coord("distance")
        out = random_patch.unselect(distance=(coord.min(), coord.max()))
        assert out.shape[random_patch.dims.index("distance")] == 0

    def test_each_named_coordinate_is_complemented(self, random_patch):
        """
        Two names remove two ranges rather than the one intersection.

        The complement of a block is a frame around it, which no array
        can hold, so unselect removes the part which is expressible.
        """
        time = random_patch.get_array("time")
        window = (time[0], time[4])
        out = random_patch.unselect(distance=(50, 60), time=window)
        assert len(out.get_array("distance")) == len(
            random_patch.get_array("distance")
        ) - len(random_patch.select(distance=(50, 60)).get_array("distance"))
        assert len(out.get_array("time")) == len(time) - 5

    def test_unknown_coordinate_raises(self, random_patch):
        """A misspelled name is the error it is in select."""
        with pytest.raises(PatchCoordinateError, match="not found in patch"):
            random_patch.unselect(not_a_coord=(1, 2))

    def test_a_multidimensional_coordinate_raises(self, random_patch):
        """
        A range of one names no samples of a single dimension to drop.

        Its complement is a shape spanning both, which is the same reason
        two coordinates cannot be complemented jointly.
        """
        size = random_patch.shape
        grid = np.arange(size[0] * size[1]).reshape(size)
        patch = random_patch.update_coords(quality=(("distance", "time"), grid))
        with pytest.raises(PatchCoordinateError, match="spans"):
            patch.unselect(quality=(0, 10))

    def test_non_dimensional_coordinate(self, random_patch):
        """A coordinate along a dimension trims that dimension."""
        size = random_patch.coord_shapes["distance"][0]
        patch = random_patch.update_coords(quality=("distance", np.arange(size)))
        out = patch.unselect(quality=(0, 9))
        assert len(out.get_array("quality")) == size - 10


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

    def test_noop_squeeze_returns_self(self, random_patch):
        """Squeeze on a patch with no length-1 dims returns the same patch."""
        assert 1 not in random_patch.shape
        assert random_patch.squeeze() is random_patch

    def test_noop_coord_squeeze_returns_self(self, random_patch):
        """CoordManager squeeze with no length-1 dims returns self."""
        coords = random_patch.coords
        assert coords.squeeze() is coords

    @pytest.mark.parametrize("dim", [None, ("distance", "time")])
    def test_squeeze_all_dimensions_raises(self, random_patch, dim):
        """Squeeze should not create an unsupported scalar patch."""
        patch = random_patch.select(distance=0, time=0, samples=True)
        msg = "at least one dimension"
        with pytest.raises(ParameterError, match=msg):
            patch.squeeze(dim)


class TestGetCoord:
    """Tests for the get_coord convenience function."""

    def test_returns_coord(self, random_patch):
        """Return a coordinate."""
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
    """Tests for making patches broadcastable to different shapes."""

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
        """Ensure a PatchError is raised if coords don't share a dimension."""
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


class TestTranspose:
    """Tests for transposing patches."""

    @pytest.fixture(scope="class")
    def patch_5d(self):
        """Create a 5D patch for testing transpose permutations."""
        shape = (3, 4, 5, 6, 7)
        dims = ("time", "distance", "x", "y", "z")
        data = np.arange(np.prod(shape)).reshape(shape)
        coords = {
            "time": np.arange(shape[0]),
            "distance": np.arange(shape[1]),
            "x": np.arange(shape[2]),
            "y": np.arange(shape[3]),
            "z": np.arange(shape[4]),
        }
        patch = dc.Patch(data=data, dims=dims, coords=coords)
        return patch

    def test_transpose_invalid_dimension_raises_parameter_error(self, random_patch):
        """Ensure transposing with invalid dimension raises clear ParameterError."""
        msg = "not_a_dim.*not found in Patch dimensions"
        with pytest.raises(ParameterError, match=msg):
            random_patch.transpose("time", "not_a_dim")

    def test_transpose_multiple_invalid_dimensions_raises(self, random_patch):
        """Ensure multiple invalid dimensions are reported."""
        msg = r"bad_dim1.*bad_dim2.*not found in Patch dimensions"
        with pytest.raises(ParameterError, match=msg):
            random_patch.transpose("bad_dim1", "bad_dim2")

    def test_transpose_valid_dimensions_works(self, random_patch):
        """Ensure transpose works with valid dimensions."""
        dims = random_patch.dims[::-1]
        # Transpose to reverse dimension order
        out = random_patch.transpose(*dims)
        # Should reverse the dimensions and shape
        assert out.dims == dims
        assert out.shape == (random_patch.shape[1], random_patch.shape[0])

    def test_transpose_no_args(self, random_patch):
        """Ensure transposing rotates dimensions."""
        pa = random_patch.transpose()
        assert pa.dims != random_patch.dims
        assert pa.dims == random_patch.dims[::-1]

    def test_transpose_5d_complete_reversal(self, patch_5d):
        """Test complete reversal of all dimensions."""
        out = patch_5d.transpose("z", "y", "x", "distance", "time")
        assert out.dims == ("z", "y", "x", "distance", "time")
        assert out.shape == (7, 6, 5, 4, 3)

    def test_transpose_5d_move_first_to_last(self, patch_5d):
        """Test moving first dimension to last."""
        out = patch_5d.transpose("distance", "x", "y", "z", "time")
        assert out.dims == ("distance", "x", "y", "z", "time")
        assert out.shape == (4, 5, 6, 7, 3)

    def test_transpose_5d_swap_adjacent(self, patch_5d):
        """Test swapping adjacent dimensions."""
        out = patch_5d.transpose("distance", "time", "x", "y", "z")
        assert out.dims == ("distance", "time", "x", "y", "z")
        assert out.shape == (4, 3, 5, 6, 7)

    def test_transpose_5d_arbitrary_permutation(self, patch_5d):
        """Test arbitrary permutation of dimensions."""
        out = patch_5d.transpose("y", "time", "z", "x", "distance")
        assert out.dims == ("y", "time", "z", "x", "distance")
        assert out.shape == (6, 3, 7, 5, 4)

    def test_transpose_5d_with_ellipsis(self, patch_5d):
        """Test using ellipsis to move last dimension to first."""
        out = patch_5d.transpose("z", ...)
        assert out.dims[0] == "z"
        assert out.shape[0] == 7
        # Remaining dims should be in original order
        assert out.dims == ("z", "time", "distance", "x", "y")

    def test_transpose_5d_data_integrity(self, patch_5d):
        """Test that transpose preserves data values."""
        out = patch_5d.transpose("z", "y", "x", "distance", "time")
        # Total size should be unchanged
        assert patch_5d.data.size == out.data.size
        # Data values should be the same, just rearranged
        assert np.array_equal(np.sort(patch_5d.data.flat), np.sort(out.data.flat))

    def test_noop_transpose_returns_self(self, random_patch):
        """Transposing to the current dim order returns the same patch."""
        assert random_patch.transpose(*random_patch.dims) is random_patch

    def test_noop_transpose_ellipsis_returns_self(self, patch_5d):
        """A trailing ellipsis that resolves to the same order returns self."""
        assert patch_5d.transpose(*patch_5d.dims[:-1], ...) is patch_5d

    def test_noop_coord_transpose_returns_self(self, random_patch):
        """CoordManager transpose to the current order returns self."""
        coords = random_patch.coords
        assert coords.transpose(*coords.dims) is coords


class TestCellBoundsOperations:
    """Bounds follow retained rows and are discarded for new grids."""

    @pytest.fixture
    def bounded_patch(self, random_patch):
        """A patch with non-centred cells on both dimensions."""
        updates = {}
        for dim in random_patch.dims:
            coord = random_patch.get_coord(dim)
            updates[f"{dim}_start"] = (dim, coord)
            updates[f"{dim}_stop"] = (dim, coord.update(min=coord.min() + coord.step))
        return random_patch.update_coords(**updates)

    @pytest.mark.parametrize("operation", ["select", "isel", "sel", "decimate"])
    def test_retained_rows(self, bounded_patch, operation):
        """Every selection path selects the same bounds as its labels."""
        patch = bounded_patch
        if operation == "select":
            out = patch.select(distance=(2, 8), samples=True)
        elif operation == "isel":
            out = patch.isel(distance=slice(2, 8))
        elif operation == "sel":
            values = patch.get_array("distance")
            out = patch.sel(distance=slice(values[2], values[7]))
        else:
            out = patch.decimate(distance=2, filter_type=None)
        np.testing.assert_array_equal(
            out.get_array("distance_start"), out.get_array("distance")
        )
        assert out.get_coord("time_start") == patch.get_coord("time_start")

    @pytest.mark.parametrize(
        "operation",
        ["resample", "decimate", "interpolate", "pad", "rolling", "rolling_stride"],
    )
    def test_new_grid(self, bounded_patch, operation):
        """Changing a grid drops only that dimension's pair."""
        patch = bounded_patch
        if operation == "resample":
            out = patch.resample(distance=2)
        elif operation == "decimate":
            out = patch.decimate(distance=2)
        elif operation == "interpolate":
            out = patch.interpolate(distance=patch.get_array("distance") + 0.25)
        elif operation == "pad":
            out = patch.pad(distance=2, samples=True)
        else:
            step = 2 if operation.endswith("stride") else 1
            out = patch.rolling(distance=4, step=step, samples=True).mean()
        assert "distance_start" not in out.coords
        assert "distance_stop" not in out.coords
        assert out.get_coord("time_start") == patch.get_coord("time_start")

    def test_convert_units(self, bounded_patch):
        """Converting a dimension transforms both bounds in the same call."""
        out = bounded_patch.convert_units(distance="km")
        for name in ("distance", "distance_start", "distance_stop"):
            np.testing.assert_allclose(
                out.get_array(name), bounded_patch.get_array(name) / 1000
            )
            assert out.get_coord(name).units == out.get_coord("distance").units

    @pytest.mark.parametrize("suffix", ["min", "max"])
    def test_translate(self, bounded_patch, suffix):
        """Time envelope updates shift bounds by the label translation."""
        coord = bounded_patch.get_coord("time")
        delta = np.timedelta64(3, "s")
        target = getattr(coord, suffix)() + delta
        out = bounded_patch.update_coords(**{f"time_{suffix}": target})
        for name in ("time", "time_start", "time_stop"):
            np.testing.assert_array_equal(
                out.get_array(name), bounded_patch.get_array(name) + delta
            )

    def test_step_update(self, bounded_patch):
        """Changing the declared step discards the old cells."""
        out = bounded_patch.update_coords(distance_step=2)
        assert "distance_start" not in out.coords
        assert "distance_stop" not in out.coords

    def test_rename(self, bounded_patch):
        """Renaming a dimension retains the reserved pair's meaning."""
        out = bounded_patch.rename_coords(distance="channel")
        assert "channel_start" in out.coords
        assert "distance_start" not in out.coords
        assert out.coords.dim_map["channel_start"] == ("channel",)

    def test_translate_values(self, bounded_patch):
        """Replacing labels by a pure translation also translates the cells."""
        out = bounded_patch.update_coords(
            distance=bounded_patch.get_array("distance") + 10
        )
        for name in ("distance", "distance_start", "distance_stop"):
            np.testing.assert_array_equal(
                out.get_array(name), bounded_patch.get_array(name) + 10
            )

    def test_replace_grid(self, bounded_patch):
        """A same-length replacement grid must not retain stale bounds."""
        out = bounded_patch.update_coords(
            distance=bounded_patch.get_array("distance") * 2
        )
        assert "distance_start" not in out.coords
        assert "distance_stop" not in out.coords

    def test_set_units(self, bounded_patch):
        """Unit assignment changes the pair's units with the dimension."""
        out = bounded_patch.set_units(distance="km")
        for name in ("distance_start", "distance_stop"):
            assert out.get_coord(name).units == out.get_coord("distance").units
            np.testing.assert_array_equal(
                out.get_array(name), bounded_patch.get_array(name)
            )

    def test_zero_pad(self, bounded_patch):
        """An unchanged grid keeps its cell bounds."""
        out = bounded_patch.pad(distance=0, samples=True)
        assert out.coords == bounded_patch.coords

    def test_coordinate_slices(self, bounded_patch):
        """Coordinate bracket slicing reproduces the retained selection rows."""
        out = bounded_patch.isel(distance=slice(1, 20, 3))
        for name in ("distance", "distance_start", "distance_stop"):
            np.testing.assert_array_equal(
                out.get_array(name), bounded_patch.get_coord(name)[1:20:3].values
            )

    def test_tile_round_trip(self, bounded_patch):
        """Windowing replaces input cells and reassembly restores their bounds."""
        out = bounded_patch.tile_apply(lambda x: x, mode="stack", time=16, samples=True)
        assert out.get_coord("time_start") != bounded_patch.get_coord("time_start")
        assert out.reassemble().equals(bounded_patch, close=True)


class TestBoundedGridReplacement:
    """Translation detection distinguishes roundoff from new coordinate grids."""

    @pytest.fixture
    def patch(self):
        """A simple bounded numeric grid."""
        return dc.Patch(
            data=np.zeros(3),
            dims=("x",),
            coords={
                "x": [0.0, 1.0, 2.0],
                "x_start": ("x", [-0.5, 0.5, 1.5]),
                "x_stop": ("x", [0.5, 1.5, 2.5]),
            },
        )

    @pytest.mark.parametrize("shift", [0.2, -0.2, 1e-12])
    def test_decimal_translation(self, patch, shift):
        """Decimal shifts retain physical edges despite canonicalization roundoff."""
        out = patch.update_coords(x=patch.get_array("x") + shift)
        for name in ("x_start", "x_stop"):
            np.testing.assert_allclose(
                out.get_array(name), patch.get_array(name) + shift, rtol=0, atol=1e-15
            )

    def test_small_stretch(self, patch):
        """A small, real cadence change is not classified as roundoff."""
        out = patch.update_coords(x=patch.get_array("x") * (1 + 1e-9) + 0.2)
        assert "x_start" not in out.coords

    @pytest.mark.parametrize("values", [["a", "b", "c"], dc.to_datetime64([0, 1, 2])])
    def test_incompatible_replacement(self, patch, values):
        """Categorical and temporal replacements discard numeric bounds."""
        out = patch.update_coords(x=values)
        assert "x_start" not in out.coords
        assert "x_stop" not in out.coords

    def test_time_to_numeric(self, patch):
        """Changing time labels to numeric labels is a valid grid replacement."""
        updates = {
            name: (
                "x",
                dc.get_coord(data=dc.to_datetime64(patch.get_array(name)), units="s"),
            )
            for name in ("x", "x_start", "x_stop")
        }
        timed = patch.update_coords(**updates)
        out = timed.update_coords(x=np.arange(3.0))
        assert "x_start" not in out.coords

    @pytest.mark.parametrize(
        "replacement", [[0.0, 1.0], dc.get_coord(data=[0.0, 1.0, 2.0], units="m")]
    )
    def test_other_grid_replacement(self, patch, replacement):
        """Different lengths and units also invalidate the old bounds."""
        cm = patch.coords.update(x=replacement)
        assert "x_start" not in cm
        assert "x_stop" not in cm

    def test_empty_replacement(self, patch):
        """Replacing an empty grid has no translation to infer."""
        empty = patch.isel(x=slice(0, 0))
        out = empty.update_coords(x=np.array([]))
        assert out.shape == (0,)
        assert "x_start" not in out.coords

    def test_integer_to_float_translation(self, patch):
        """A fractional offset can promote integer labels without losing bounds."""
        integer = patch.update_coords(x=np.arange(3))
        out = integer.update_coords(x=integer.get_array("x") + 0.2)
        np.testing.assert_allclose(
            out.get_array("x_start"),
            patch.get_array("x_start") + 0.2,
            rtol=0,
            atol=1e-15,
        )

    def test_time_translation(self, patch):
        """A direct time-label shift translates the physical time bounds."""
        updates = {
            name: (
                "x",
                dc.get_coord(data=dc.to_datetime64(patch.get_array(name)), units="s"),
            )
            for name in ("x", "x_start", "x_stop")
        }
        timed = patch.update_coords(**updates)
        delta = np.timedelta64(2, "s")
        out = timed.update_coords(x=timed.get_array("x") + delta)
        np.testing.assert_array_equal(
            out.get_array("x_start"), timed.get_array("x_start") + delta
        )

    @pytest.mark.parametrize(
        "edge_kind", ["centred", "unsigned_range", "unsigned_irregular"]
    )
    def test_unsigned_translation(self, edge_kind):
        """A backward shift never wraps unsigned labels or cell edges."""
        values = np.array(
            [2, 4, 5] if edge_kind.endswith("irregular") else [2, 3, 4], dtype=np.uint64
        )
        width = 0.5 if edge_kind == "centred" else np.uint64(1)
        original = dc.Patch(
            data=np.zeros(3),
            dims=("x",),
            coords={
                "x": values,
                "x_start": ("x", values - width),
                "x_stop": ("x", values + width),
            },
        )
        out = original.update_coords(x=values - np.uint64(1))
        np.testing.assert_array_equal(
            out.get_array("x_start"), original.get_array("x_start").astype(float) - 1
        )
        np.testing.assert_array_equal(
            out.get_array("x_stop"), original.get_array("x_stop").astype(float) - 1
        )

    def test_unsigned_edges_cross_zero(self):
        """Physical edges can promote to signed values when a shift crosses zero."""
        values = np.arange(1, 4, dtype=np.uint64)
        original = dc.Patch(
            data=np.zeros(3),
            dims=("x",),
            coords={
                "x": values,
                "x_start": ("x", values - np.uint64(1)),
                "x_stop": ("x", values + np.uint64(1)),
            },
        )
        out = original.update_coords(x=values - np.uint64(1))
        np.testing.assert_array_equal(out.get_array("x_start"), [-1, 0, 1])


class TestIntegerCellTranslation:
    """Integer physical bounds promote when translated outside their dtype."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64])
    @pytest.mark.parametrize("shift", [0.5, 200])
    def test_promoted_bounds(self, dtype, shift):
        """Fractional and out-of-range shifts retain the actual physical edges."""
        values = np.arange(2, 5, dtype=dtype)
        patch = dc.Patch(
            data=np.ones(3),
            dims=("x",),
            coords={
                "x": values,
                "x_start": ("x", values - 1),
                "x_stop": ("x", values + 1),
            },
        )
        out = patch.update_coords(x=values.astype(float) + shift)
        for name in ("x_start", "x_stop"):
            np.testing.assert_array_equal(
                out.get_array(name), patch.get_array(name).astype(float) + shift
            )


def _gapped_patch(coord, dim="time", dtype=np.float64):
    """A 3 x len(coord) patch whose data counts up along the coordinate."""
    size = len(coord)
    data = np.arange(3 * size).reshape(3, size).astype(dtype)
    coords = {"distance": np.arange(3), dim: coord}
    return dc.Patch(data=data, coords=coords, dims=("distance", dim))


class TestFillGaps:
    """Tests for filling holes along a dimension."""

    t0 = np.datetime64("2020-01-01", "ns")
    ms = np.timedelta64(1_000_000, "ns")

    @pytest.fixture()
    def gapped(self):
        """Five samples, three missing, then four more, at 1 ms."""
        first = get_coord(start=self.t0, step=self.ms, shape=(5,))
        second = get_coord(start=self.t0 + 8 * self.ms, step=self.ms, shape=(4,))
        return _gapped_patch(concat_coords(first, second))

    @pytest.fixture()
    def three_runs(self, gapped):
        """The gapped patch plus two samples after an 18-sample hole."""
        coord = gapped.get_coord("time")
        last = get_coord(start=self.t0 + 30 * self.ms, step=self.ms, shape=(2,))
        return _gapped_patch(concat_coords(coord, last))

    def test_fills_hole(self, gapped):
        """The hole becomes NaN and each run lands at its grid position."""
        out = gapped.fill_gaps("time")
        expected = get_coord(start=self.t0, step=self.ms, shape=(12,))
        assert out.get_coord("time") == expected
        assert np.isnan(out.data[:, 5:8]).all()
        assert np.array_equal(out.data[:, :5], gapped.data[:, :5])
        assert np.array_equal(out.data[:, 8:], gapped.data[:, 5:])

    @pytest.mark.parametrize(
        ("limit", "samples", "filled"),
        [
            (0.003, False, True),
            (0.0029, False, False),
            (np.timedelta64(3, "ms"), False, True),
            (3 * get_quantity("ms"), False, True),
            (3, True, True),
            (2, True, False),
            (0, True, False),
        ],
    )
    def test_limit(self, gapped, limit, samples, filled):
        """A hole is filled only when it is no wider than the limit."""
        out = gapped.fill_gaps(time=limit, samples=samples)
        assert out.shape[1] == (12 if filled else 9)

    def test_limit_leaves_wide_holes(self, three_runs):
        """Only the narrow hole is filled; the wide one stays a seam."""
        out = three_runs.fill_gaps(time=0.005)
        coord = out.get_coord("time")
        assert coord.segment_count == 2 and out.shape == (3, 14)
        assert coord.segments[-1] == three_runs.get_coord("time").segments[-1]
        assert np.array_equal(out.data[:, -2:], three_runs.data[:, -2:])

    @pytest.mark.parametrize(
        ("start", "step", "limit", "filled"),
        [
            (0.0, 0.5, 1.5, True),
            (0.0, 0.5, 1.4, False),
            (0, 2, 6, True),
            (0, 2, 5, False),
        ],
    )
    def test_limit_numeric_coord(self, start, step, limit, filled):
        """Float and integer coordinates take a limit in their own units."""
        first = get_coord(start=start, step=step, shape=(4,))
        second = get_coord(start=start + 7 * step, step=step, shape=(3,))
        patch = _gapped_patch(concat_coords(first, second), dim="x")
        assert patch.fill_gaps(x=limit).shape[1] == (10 if filled else 7)

    def test_value_not_castable(self, gapped):
        """A value which is not a number cannot fill numeric data."""
        with pytest.raises(ParameterError, match="Cannot fill"):
            gapped.fill_gaps("time", value="bob")

    @pytest.mark.parametrize("limit", [-1, 1.5])
    def test_bad_sample_limit(self, gapped, limit):
        """A sample limit must be a non-negative integer."""
        with pytest.raises(ParameterError, match="non-negative integer"):
            gapped.fill_gaps(time=limit, samples=True)

    def test_range_unchanged(self, random_patch):
        """A patch with nothing to fill comes back as it is."""
        assert random_patch.fill_gaps("time") is random_patch

    @pytest.mark.parametrize("shift_us", [300, 490, -300, -600])
    def test_off_grid_seam(self, shift_us):
        """A run off the grid moves to the nearest position, by at most half a step."""
        first = get_coord(start=self.t0, step=self.ms, shape=(5,))
        start = self.t0 + 8 * self.ms + np.timedelta64(shift_us, "us")
        second = get_coord(start=start, step=self.ms, shape=(4,))
        patch = _gapped_patch(concat_coords(first, second))
        out = patch.fill_gaps("time")
        coord = out.get_coord("time")
        assert isinstance(coord, CoordRange) and coord.step == self.ms
        # every sample keeps its value, at a label at most half a step away
        kept = ~np.isnan(out.data[0])
        assert np.array_equal(out.data[:, kept], patch.data)
        moved = np.abs(coord.values[kept] - patch.get_coord("time").values)
        assert moved.max() <= self.ms // 2

    def test_colliding_runs_raise(self):
        """A run within half a step of the previous last sample cannot be placed."""
        first = get_coord(start=self.t0, step=self.ms, shape=(5,))
        start = self.t0 + 4 * self.ms + np.timedelta64(400, "us")
        second = get_coord(start=start, step=self.ms, shape=(4,))
        patch = _gapped_patch(concat_coords(first, second))
        with pytest.raises(CoordError, match="same grid position"):
            patch.fill_gaps("time")

    def test_descending(self):
        """A descending coordinate fills in its own direction."""
        first = get_coord(start=10.0, step=-1.0, shape=(3,))
        second = get_coord(start=5.0, step=-1.0, shape=(3,))
        out = _gapped_patch(concat_coords(first, second), dim="depth")
        out = out.fill_gaps("depth")
        assert np.array_equal(out.get_coord("depth").values, np.arange(10.0, 2.0, -1))
        assert np.isnan(out.data[0, 3:5]).all()

    def test_float_steps_nearly_equal(self):
        """Float runs whose steps differ in the last bits share one grid."""
        first = get_coord(start=0.0, step=0.1, shape=(10,))
        second = get_coord(start=1.5, step=0.1 * (1 + 1e-12), shape=(5,))
        out = _gapped_patch(concat_coords(first, second), dim="x").fill_gaps("x")
        assert np.allclose(out.get_coord("x").values, np.arange(20) * 0.1)

    def test_different_steps_raise(self):
        """Runs sampled at different steps cannot share a grid."""
        first = get_coord(start=self.t0, step=self.ms, shape=(5,))
        second = get_coord(start=self.t0 + 8 * self.ms, step=2 * self.ms, shape=(4,))
        patch = _gapped_patch(concat_coords(first, second))
        with pytest.raises(CoordError, match="different steps"):
            patch.fill_gaps("time")

    def test_no_step_raises(self):
        """An array coordinate without a declared step has no grid to fill."""
        patch = _gapped_patch(get_coord(data=np.array([0.0, 1.0, 3.5])), dim="x")
        with pytest.raises(CoordError, match="declared step"):
            patch.fill_gaps("x")

    def test_dense_array_with_step(self):
        """A dense array kept whole with its declared step fills too."""
        values = np.delete(np.arange(3000), np.arange(5, 3000, 7))
        coord = get_coord(data=values, step=1)
        assert type(coord).__name__ == "CoordMonotonicArray"
        out = _gapped_patch(coord, dim="channel").fill_gaps("channel")
        assert out.get_coord("channel") == get_coord(start=0, stop=3000, step=1)
        assert np.isnan(out.data[0]).sum() == 3000 - len(values)
        assert np.array_equal(out.data[:, values], _gapped_patch(coord, "c").data)

    def test_integer_data(self, gapped):
        """NaN cannot fill integers; an integer value can."""
        patch = gapped.new(data=gapped.data.astype(np.int32))
        with pytest.raises(ParameterError, match="int32"):
            patch.fill_gaps("time")
        with pytest.raises(ParameterError, match="int32"):
            patch.fill_gaps("time", value=1.5)
        out = patch.fill_gaps("time", value=0)
        assert out.data.dtype == np.int32 and (out.data[:, 5:8] == 0).all()

    def test_drops_associated_coords(self, gapped):
        """A coordinate along the dimension is dropped, with a warning."""
        size = gapped.shape[1]
        patch = gapped.update_coords(quality=("time", np.arange(size)))
        with pytest.warns(DASCoreWarning, match="Filling gaps.*quality"):
            out = patch.fill_gaps("time")
        assert "quality" not in out.coords.coord_map

    def test_middle_axis(self, gapped):
        """The dimension may sit on any axis."""
        coord = gapped.get_coord("time")
        data = np.ones((2, len(coord), 4))
        coords = {"a": np.arange(2), "time": coord, "b": np.arange(4)}
        patch = dc.Patch(data=data, coords=coords, dims=("a", "time", "b"))
        out = patch.fill_gaps("time")
        assert out.shape == (2, 12, 4)
        assert np.isnan(out.data[:, 5:8]).all() and not np.isnan(out.data[:, 8:]).any()

    @pytest.mark.parametrize(
        "full",
        [
            get_coord(start=np.datetime64("2020-01-01"), step=(1, 1024), shape=(500,)),
            get_coord(start=0, step=3, shape=(500,)),
            get_coord(start=0.0, step=0.25, shape=(500,)),
            get_coord(start=2000, step=-3, shape=(500,)),
        ],
        ids=["1024Hz", "int", "float", "descending"],
    )
    def test_random_holes(self, full):
        """Random runs of a grid fill back to the grid with data in place."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            edges = np.sort(rng.choice(np.arange(1, 499), size=8, replace=False))
            keep = [(0, edges[0]), *zip(edges[1::2], edges[2::2]), (edges[-1], 500)]
            keep = [(a, b) for a, b in keep if b > a]
            coord = concat_coords(*(full[a:b] for a, b in keep))
            patch = _gapped_patch(coord, dim="x")
            out = patch.fill_gaps("x")
            span = full[keep[0][0] : keep[-1][1]]
            got = out.get_coord("x")
            if full._exact:
                assert got == span
            else:
                assert np.allclose(got.values, span.values)
            index = np.concatenate([np.arange(a, b) for a, b in keep])
            assert np.array_equal(out.data[:, index - keep[0][0]], patch.data)

    def test_jitter_seam_fuses(self):
        """A run landing on the next position fuses without a hole."""
        first = get_coord(start=self.t0, step=self.ms, shape=(5,))
        start = self.t0 + 5 * self.ms + np.timedelta64(400_000, "ns")
        second = get_coord(start=start, step=self.ms, shape=(4,))
        out = _gapped_patch(concat_coords(first, second)).fill_gaps("time")
        assert isinstance(out.get_coord("time"), CoordRange) and out.shape == (3, 9)
        assert not np.isnan(out.data).any()

    def test_float_off_grid(self):
        """A float run 8.6 steps on lands at position 9, the nearest."""
        first = get_coord(start=0.0, step=0.1, shape=(5,))
        second = get_coord(start=0.86, step=0.1, shape=(3,))
        patch = _gapped_patch(concat_coords(first, second), dim="x")
        out = patch.fill_gaps("x")
        assert np.allclose(out.get_coord("x").values, np.arange(12) * 0.1)
        assert np.isnan(out.data[:, 5:9]).all()
        assert np.array_equal(out.data[:, 9:], patch.data[:, 5:])

    def test_array_segment_offset(self):
        """An array segment inside a segmented coordinate keeps its data."""
        array = CoordMonotonicArray(values=np.array([10, 11, 13]), step=1)
        coord = concat_coords(get_coord(start=0, step=1, shape=(4,)), array)
        patch = _gapped_patch(coord, dim="x")
        out = patch.fill_gaps("x")
        assert out.get_coord("x") == get_coord(start=0, stop=14, step=1)
        assert np.array_equal(out.data[:, [10, 11, 13]], patch.data[:, 4:])
        assert np.isnan(out.data[:, [4, 5, 6, 7, 8, 9, 12]]).all()

    def test_lone_sample_takes_step(self):
        """A single sample without a step joins its neighbours' grid."""
        first = get_coord(start=0.0, step=1.0, shape=(5,))
        last = get_coord(start=10.0, step=1.0, shape=(2,))
        coord = concat_coords(first, get_coord(data=np.array([8.0])), last)
        out = _gapped_patch(coord, dim="x").fill_gaps("x")
        assert out.get_coord("x") == get_coord(start=0.0, stop=12.0, step=1.0)

    def test_narrow_hole_after_wide(self):
        """A narrow hole after a wide one is measured from its own group."""
        runs = [(0, 5), (30, 3), (35, 2)]
        coord = concat_coords(
            *(
                get_coord(start=self.t0 + a * self.ms, step=self.ms, shape=(n,))
                for a, n in runs
            )
        )
        out = _gapped_patch(coord).fill_gaps(time=0.005)
        new = out.get_coord("time")
        assert new.segment_count == 2 and out.shape == (3, 12)
        assert new.segments[-1] == get_coord(
            start=self.t0 + 30 * self.ms, step=self.ms, shape=(7,)
        )

    def test_float_different_steps_raise(self):
        """Float runs at clearly different steps cannot share a grid."""
        first = get_coord(start=0.0, step=0.1, shape=(5,))
        second = get_coord(start=1.0, step=0.2, shape=(3,))
        patch = _gapped_patch(concat_coords(first, second), dim="x")
        with pytest.raises(CoordError, match="different steps"):
            patch.fill_gaps("x")

    def test_float_step_drift_raises(self):
        """Steps close in ratio but drifting over a long run do not share a grid."""
        first = get_coord(start=0.0, step=1.0, shape=(5,))
        second = get_coord(start=10.0, step=1.0000009, shape=(1000,))
        patch = _gapped_patch(concat_coords(first, second), dim="x")
        with pytest.raises(CoordError, match="different steps"):
            patch.fill_gaps("x")

    @pytest.mark.parametrize(
        ("first", "second", "limit", "filled"),
        [
            ((10.0, -1.0), (5.0, -1.0), 2.0, True),
            ((10.0, -1.0), (5.0, -1.0), 1.5, False),
            ((2000, -3), (1988, -3), 3, True),
            ((2000, -3), (1988, -3), 2, False),
        ],
    )
    def test_descending_limit(self, first, second, limit, filled):
        """A descending coordinate measures holes by the step's magnitude."""
        coord = concat_coords(
            get_coord(start=first[0], step=first[1], shape=(3,)),
            get_coord(start=second[0], step=second[1], shape=(3,)),
        )
        patch = _gapped_patch(coord, dim="x")
        missing = 2 if isinstance(first[0], float) else 1
        assert patch.fill_gaps(x=limit).shape[1] == 6 + (missing if filled else 0)

    def test_float_limit_equal_to_hole(self):
        """A limit equal to a float hole's width fills it despite rounding."""
        first = get_coord(start=0.0, step=0.1, shape=(5,))
        second = get_coord(start=0.8, step=0.1, shape=(3,))
        patch = _gapped_patch(concat_coords(first, second), dim="x")
        assert patch.fill_gaps(x=0.3).shape == (3, 11)

    def test_fractional_limit(self):
        """A 1024 Hz hole of one sample fills with a limit of one step."""
        full = get_coord(start=self.t0, step=(1, 1024), shape=(20,))
        patch = _gapped_patch(concat_coords(full[:7], full[8:]))
        out = patch.fill_gaps(time=1 / 1024)
        assert out.get_coord("time") == full

    def test_unsigned_coordinate(self):
        """An unsigned run off the grid moves to the nearest position."""
        first = get_coord(start=np.uint32(0), step=np.uint32(4), shape=(3,))
        second = get_coord(start=np.uint32(13), step=np.uint32(4), shape=(2,))
        out = _gapped_patch(concat_coords(first, second), dim="channel")
        out = out.fill_gaps("channel")
        assert np.array_equal(out.get_coord("channel").values, [0, 4, 8, 12, 16])
        assert not np.isnan(out.data).any()

    def test_float32_overflow(self, gapped):
        """A fill value past float32's range raises; a rounded one does not."""
        patch = gapped.new(data=gapped.data.astype(np.float32))
        with pytest.raises(ParameterError, match="float32"):
            patch.fill_gaps("time", value=1e40)
        assert patch.fill_gaps("time", value=0.1).data.dtype == np.float32

    def test_sample_tolerance_raises(self, gapped):
        """A sample-count tolerance points to samples=True."""
        with pytest.raises(ParameterError, match="samples=True"):
            gapped.fill_gaps(time=GapTolerance.samples(4))

    def test_foreign_backend(self, gapped):
        """Data from another array backend fills as numpy data."""
        xp = pytest.importorskip("array_api_strict")
        patch = gapped.new(data=xp.asarray(gapped.data))
        out = patch.fill_gaps("time")
        assert out.shape == (3, 12) and np.isnan(np.asarray(out.data)[:, 5:8]).all()

    def test_cell_edges_dropped_quietly(self, gapped):
        """Cell edges along the dimension are dropped without a warning."""
        labels = gapped.get_coord("time").values
        patch = gapped.update_coords(
            time_start=("time", labels), time_stop=("time", labels + self.ms)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = patch.fill_gaps("time")
        assert "time_start" not in out.coords.coord_map

    def test_integer_nothing_to_fill(self, random_patch):
        """An integer patch with nothing to fill ignores the default NaN."""
        patch = random_patch.new(data=random_patch.data.astype(np.int32))
        assert patch.fill_gaps("time") is patch

    def test_float_array_drift_raises(self):
        """A float array whose spacings drift off its declared step raises."""
        values = np.arange(2_000_010) * (1 + 4e-7)
        values = np.delete(values, [5])
        coord = CoordMonotonicArray(values=values, step=1.0)
        with pytest.raises(CoordError, match="drift"):
            _fill_layout(coord)
