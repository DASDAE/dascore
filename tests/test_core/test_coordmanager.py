"""Tests for coordinate manager."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from rich.text import Text

import dascore as dc
import dascore.proc.coords
from dascore import to_datetime64
from dascore.compat import random_state
from dascore.core.coordmanager import (
    CoordManager,
    get_coord_manager,
)
from dascore.core.coords import (
    BaseCoord,
    get_coord,
)
from dascore.exceptions import (
    CoordDataError,
    CoordError,
    CoordSortError,
    ParameterError,
)
from dascore.units import get_quantity
from dascore.utils.misc import (
    all_close,
    get_middle_value,
    suppress_warnings,
)

COORDS = {
    "time": dc.to_datetime64(np.arange(10, 100, 10)),
    "distance": dc.get_coord(data=np.arange(0, 1_000, 10)),
}
DIMS = ("time", "distance")


class TestGetCoordManager:
    """Test suite for `get_coord_manager` helper function."""

    def test_coords_and_attrs_raise(self, random_patch):
        """Ensure using coords and attrs raises."""
        msg = "Cannot use both attrs and coords"
        coords, attrs = random_patch.coords, random_patch.attrs
        with pytest.raises(ParameterError, match=msg):
            get_coord_manager(coords=coords, attrs=attrs)

    def test_coords_from_attrs(self, random_patch):
        """Ensure we can get coordinates from patch attrs."""
        attrs = random_patch.attrs
        cm = get_coord_manager(attrs=attrs)
        assert "time" in cm.coord_map

    def test_non_coord_dims(self):
        """Ensure non coordinate dimensions can be created using shape."""
        coords = {"time": np.arange(10)}
        dims = ("time", "money")
        out = get_coord_manager(coords, dims, shape=(10, 2))
        assert out.shape == (10, 2)
        assert out.dims == ("time", "money")

    def test_not_associated_coord_1(self):
        """Ensure a not associated coord works as only input."""
        coords = {"time": (None, np.arange(10))}
        assert isinstance(get_coord_manager(coords), CoordManager)

    def test_not_associated_coord_2(self):
        """Ensure an empty string not associated coord works as only input."""
        coords = {"time": ("", np.arange(10))}
        assert isinstance(get_coord_manager(coords), CoordManager)


class TestBasicCoordManager:
    """Ensure basic things work with coord managers."""

    def test_init(self, coord_manager):
        """Ensure values can be init'ed."""
        assert isinstance(coord_manager, CoordManager)

    def test_to_dict(self, coord_manager):
        """CoordManager should be convertible to dict."""
        with suppress_warnings():
            c_dict = dict(coord_manager)
        expected = {x: coord_manager.get_array(x) for x in coord_manager.coord_map}
        assert set(expected) == set(c_dict)
        for key in set(expected):
            assert np.all(expected[key] == np.array(c_dict[key]))

    def test_membership(self, coord_manager):
        """Coord membership should work for coord names."""
        coords = list(coord_manager.coord_map)
        for name in coords:
            assert name in coords

    def test_empty(self):
        """And empty coord manager should be possible."""
        coord = get_coord_manager()
        assert isinstance(coord, CoordManager)
        assert dict(coord) == {}
        # shape should be the same as an empty array.
        assert coord.shape == np.array([]).shape

    def test_str(self, coord_manager):
        """Tests the str output for coord manager."""
        out = str(coord_manager)
        assert isinstance(out, str)
        assert len(out)

    def test_rich(self, coord_manager):
        """Tests the str output for coord manager."""
        out = coord_manager.__rich__()
        assert isinstance(out, Text)
        assert len(out)

    def test_cant_assign_new_coord_inplace(self, cm_basic):
        """The mappings inside the coord manager should be immutable."""
        cm = cm_basic
        expected_str = "does not support item assignment"
        # cant add new coord
        with pytest.raises(TypeError, match=expected_str):
            cm["bob"] = 10
        # cant modify existing coord
        with pytest.raises(TypeError, match=expected_str):
            cm[cm.dims[0]] = cm.get_array(cm.dims[0])

    def test_cant_modify_dim_map(self, cm_basic):
        """Ensure the dim map is immutable."""
        expected_str = "does not support item assignment"
        dim_map = cm_basic.dim_map
        # test new key
        with pytest.raises(TypeError, match=expected_str):
            dim_map["bob"] = 10
        # test existing key
        with pytest.raises(TypeError, match=expected_str):
            dim_map[cm_basic.dims[0]] = 10

    def test_cant_modify_coord_map(self, cm_basic):
        """Ensure the dim map is immutable."""
        expected_str = "does not support item assignment"
        coord_map = cm_basic.coord_map
        # test new key
        with pytest.raises(TypeError, match=expected_str):
            coord_map["bob"] = 10
        # test existing key
        with pytest.raises(TypeError, match=expected_str):
            coord_map[cm_basic.dims[0]] = 10

    def test_init_with_coord_manager(self, cm_basic):
        """Ensure initing coord manager works with a single coord manager."""
        out = get_coord_manager(cm_basic)
        assert out == cm_basic

    def test_init_with_cm_and_dims(self, cm_basic):
        """Ensure cm can be init'ed with coord manager and dims."""
        out = get_coord_manager(cm_basic, dims=cm_basic.dims)
        assert out == cm_basic

    def test_init_list_range(self):
        """Ensure the coord manager can be init'ed with a list and no dims."""
        input_dict = {
            "time": [10, 11, 12],
            "distance": ("distance", range(100)),
            "space": range(100),
        }
        out = get_coord_manager(input_dict, dims=list(input_dict))
        assert set(out.dims) == set(input_dict)

    def test_bad_datashape_raises(self, cm_basic):
        """Ensure a bad datashape raises."""
        match = "match the coordinate manager shape"
        cm = cm_basic
        data = np.ones((cm.shape[0], cm.shape[1] - 1))
        with pytest.raises(CoordDataError, match=match):
            cm.validate_data(data)

    def test_to_summary_dict(self, coord_manager):
        """Ensure coord managers can be converted to summary dicts."""
        sum_dict = coord_manager.to_summary_dict()
        assert set(sum_dict) == set(coord_manager.coord_map)

    def test_size(self, coord_manager):
        """Ensure all coord managers have a size."""
        assert isinstance(coord_manager.size, int | np.int_)

    def test_min(self, cm_basic):
        """Ensure we can get min value."""
        expected = np.min(cm_basic.time.data).astype(np.int64)
        got = cm_basic.min("time").astype(np.int64)
        assert np.isclose(got, expected)

    def test_max(self, cm_basic):
        """Ensure we can get max value."""
        expected = np.max(cm_basic.time.data).astype(np.int64)
        got = cm_basic.max("time").astype(np.int64)
        assert np.isclose(got, expected)

    def test_step(self, cm_basic):
        """Ensure we can get min value."""
        expected = cm_basic.time.step
        assert cm_basic.step("time") == expected

    def test_getattr(self, cm_basic):
        """Ensure getattr returns coordinate."""
        for dim in cm_basic.dims:
            coord = getattr(cm_basic, dim)
            assert coord == cm_basic.coord_map[dim]

    def test_get_item_warning(self, cm_basic):
        """Ensure get item emits a warning."""
        msg = "returns a numpy array"
        with pytest.warns(UserWarning, match=msg):
            _ = cm_basic["time"]

    def test_has_attr(self, cm_basic):
        """Ensure hasattr returns correct result."""
        dims = cm_basic.dims
        for dim in dims:
            assert hasattr(cm_basic, dim)
        assert not hasattr(cm_basic, "_NOT_A_DIM")

    def test_iterate(self, cm_basic):
        """Ensure coordinates yield name an coordinate when iterated."""
        for dim, coord in iter(cm_basic):
            expected = cm_basic.get_coord(dim)
            assert all_close(coord, expected)

    def test_coord_size(self, random_patch):
        """Ensure we can get size of the coordinate."""
        expected = len(random_patch.get_coord("time"))
        assert random_patch.coords.coord_size("time") == expected

    def test_coord_range(self, random_patch):
        """Ensure we can get a scaler value for the coordinate."""
        coord_array = random_patch.get_coord("time").data
        expected = (
            np.max(coord_array) - np.min(coord_array) + random_patch.attrs["time_step"]
        )
        assert random_patch.coords.coord_range("time") == expected


class TestCoordManagerInputs:
    """Tests for coordinates management."""

    def test_simple_inputs(self):
        """Simplest input case."""
        out = get_coord_manager(COORDS, DIMS)
        assert isinstance(out, CoordManager)

    def test_additional_coords(self):
        """Ensure a additional (non-dimensional) coords work."""
        coords = dict(COORDS)
        lats = random_state.rand(len(COORDS["distance"]))
        coords["latitude"] = ("distance", lats)
        out = get_coord_manager(coords, DIMS)
        assert isinstance(out.coord_map["latitude"], BaseCoord)

    def test_str(self, cm_basic):
        """Ensure a custom (readable) str is returned."""
        coord_str = str(cm_basic)
        assert isinstance(coord_str, str)

    def test_bad_coords(self):
        """Ensure specifying a bad coordinate raises."""
        coords = dict(COORDS)
        coords["bill"] = np.arange(10, 100, 10)
        with pytest.raises(CoordError, match="not named the same as dimension"):
            get_coord_manager(coords, DIMS)

    def test_nested_coord_too_long(self):
        """Nested coordinates that are gt 2 should fail."""
        coords = dict(COORDS)
        coords["time"] = ("time", to_datetime64(np.arange(10, 100, 10)), "space")
        with pytest.raises(CoordError, match="must be length two"):
            get_coord_manager(coords, DIMS)

    def test_invalid_dimensions(self):
        """Nested coordinates must specify valid dimensions."""
        coords = dict(COORDS)
        coords["time"] = ("bob", to_datetime64(np.arange(10, 100, 10)))
        with pytest.raises(CoordError, match="invalid dimension"):
            get_coord_manager(coords, DIMS)

    def test_missing_coordinates(self):
        """If all dims don't have coords an error should be raised."""
        coords = dict(COORDS)
        coords.pop("distance")
        msg = "All dimensions must have coordinates"
        with pytest.raises(ValidationError, match=msg):
            get_coord_manager(coords, DIMS)

    def test_secondary_coord_bad_lengths(self):
        """Ensure when coordinates don't line up an error is raised."""
        coords = dict(COORDS)
        coords["bad"] = ("time", np.ones(len(coords["time"]) - 1))
        with pytest.raises(ValidationError, match="does not match the dimension"):
            get_coord_manager(coords, DIMS)

    def test_dim_coord_wrong_dim(self):
        """It shouldn't be possible to init a dimension coord with the wrong dim."""
        coords = {
            "time": ((), np.array([1, 2, 3])),
            "distance": ("distance", np.array([1, 2, 3])),
        }
        dims = "time", "distance"
        with pytest.raises(ValidationError):
            get_coord_manager(coords, dims=dims)

    def test_mappings_immutable(self, coord_manager):
        """Ensure the mappings are immutable."""
        with pytest.raises(Exception):
            coord_manager.coord_map["bob"] = 2

    def test_init_coord_manager_with_non_coord_dim(self, cm_non_coord_dim):
        """A non-coordinate dimension should be allowed."""
        cm = cm_non_coord_dim
        assert isinstance(cm, CoordManager)
        assert cm.shape == (10, 5)


class TestCoordManagerWithAttrs:
    """Tests for initing coord managing with attribute dict."""

    def test_missing_dim(self):
        """Coord manager should be able to pull missing info from attributes."""
        attrs = dict(distance_min=1, distance_max=100, distance_step=10)
        new = get_coord_manager(None, ("distance",), attrs=attrs)
        assert "distance" in new.coord_map


class TestDrop:
    """Tests for dropping coords with coord manager."""

    def test_drop(self, cm_multidim):
        """Ensure coordinates can be dropped."""
        dim = "distance"
        coords, _ = cm_multidim.drop_coords(dim)
        assert dim not in coords.dims
        for _name, dims in coords.dim_map.items():
            assert dim not in dims

    def test_drop_doesnt_have_coord(self, cm_multidim):
        """Trying to drop a dim that doesnt exist should just return."""
        out, _ = cm_multidim.drop_coords("bob")
        assert out == cm_multidim

    def test_trims_array(self, cm_multidim):
        """Trying to drop a dim that doesnt exist should just return."""
        array = np.ones(cm_multidim.shape)
        axis = cm_multidim.get_axis("time")
        cm, new_array = cm_multidim.drop_coords("time", array=array)
        assert new_array.shape[axis] == 0

    def test_drop_non_dim_coord(self, cm_multidim):
        """Dropping a non-dim coord should not affect shape/dimensions."""
        cm = cm_multidim
        array = np.ones(cm.shape)
        coords_to_drop = set(cm.coord_map) - set(cm.dims)
        for coord in coords_to_drop:
            cm_new, array_new = cm.drop_coords(coord, array=array)
            # array should not have changed.
            assert array.shape == array_new.shape
            assert np.all(np.equal(array, array_new))
            assert coord not in set(cm_new.coord_map)


class TestDropPrivateCoords:
    """Tests for dropping private coordinates."""

    def test_drop_private(self, cm_basic):
        """Ensure private coords are removed."""
        cm = cm_basic.update_coords(_bad=(None, np.array([1, 2, 3])))
        out = cm.drop_private_coords()[0]
        assert "_bad" not in out.coord_map

    def test_no_private_coords(self, cm_basic):
        """Ensure this does nothing when no private attrs are found."""
        out = cm_basic.drop_private_coords()[0]
        assert out is cm_basic


class TestSelect:
    """Tests for filtering coordinates."""

    # index ranges to compare with normal select.
    slice_inds = (
        (None, 5),
        (..., 5),
        (1, 5),
        (1, None),
        (1, ...),
        (..., ...),
        (None, None),
        (3, 4),
        (-4, -1),
        (1, -1),
    )
    # single index values to test
    inds = (
        0,
        1,
        -1,
        -2,
    )

    def test_2d_coord_raises(self, cm_multidim):
        """Select shouldn't work on 2D coordinates."""
        with pytest.raises(CoordError, match="Only 1 dimensional"):
            cm_multidim.select(quality=(1, 2))

    def test_select_coord_dim(self, cm_basic):
        """Simple test for filtering dimension coord."""
        new, _ = cm_basic.select(distance=(100, 400))
        dist_ind = cm_basic.get_axis("distance")
        assert new.shape[dist_ind] < cm_basic.shape[dist_ind]

    def test_filter_array(self, cm_basic):
        """Ensure an array can be filtered."""
        data = np.ones(cm_basic.shape)
        new, trim = cm_basic.select(distance=(100, 400), array=data)
        assert trim.shape == trim.shape

    def test_select_emptying_dim(self, cm_basic):
        """Selecting a range outside of dim should empty the manager."""
        data = np.ones(cm_basic.shape)
        cm, trim = cm_basic.select(distance=(-100, -10), array=data)
        assert trim.shape[cm_basic.get_axis("distance")] == 0
        assert "distance" in cm.dims
        assert len(cm.get_array("distance")) == 0
        assert len(cm.coord_map["distance"]) == 0

    def test_select_trims_associated_coord_1(self, cm_multidim):
        """Ensure trimming a dimension also trims associated coordinate."""
        cm = cm_multidim
        coord_to_trim = "distance"
        distance = cm.get_array(coord_to_trim)
        out, _ = cm.select(distance=(distance[1], distance[-2]))
        # ensure all attrs with "distance" have been trimmed.
        expected_len = len(out.coord_map[coord_to_trim])
        for name in cm.dim_to_coord_map[coord_to_trim]:
            coord = out.coord_map[name]
            axis = cm.dim_map[name].index(coord_to_trim)
            assert coord.shape[axis] == expected_len

    def test_select_trims_associated_coords_2(self, cm_multidim):
        """Same as test #1, but now we check for trimming non-dimension coord."""
        cm = cm_multidim
        coord_to_trim = "latitude"
        array = cm.get_array(coord_to_trim)
        out, _ = cm.select(**{coord_to_trim: (array[1], array[-2])})
        dim = cm.dim_map[coord_to_trim][0]
        # ensure all attrs with shared dim have been trimmed.
        expected_len = len(out.coord_map[coord_to_trim])
        for name in cm.dim_to_coord_map[dim]:
            coord = out.coord_map[name]
            axis = cm.dim_map[name].index(dim)
            assert coord.shape[axis] == expected_len

    def test_select_handles_non_dim_kwargs(self, cm_basic):
        """The coord manager should handle (supress) non dim keyword args."""
        ar = np.ones(cm_basic.shape)
        out, new = cm_basic.select(bob=(10, 20), array=ar)
        assert new.shape == ar.shape
        assert out == cm_basic

    @pytest.mark.parametrize("slice_range", slice_inds)
    def test_compare_to_select(self, cm_basic, slice_range):
        """Ensure select with and without samples behaves the same with equiv. data."""
        cm = cm_basic
        for name, coord in cm.coord_map.items():
            ind_tuple = slice_range
            ind_1, ind_2 = slice_range
            val1 = coord[ind_1] if isinstance(ind_1, int) else ind_1
            val2 = coord[ind_2 - 1] if isinstance(ind_2, int) else ind_2
            value_tuple = (val1, val2)
            # first, just check coords are equal
            new_cm, _ = cm_basic.select(**{name: value_tuple})
            out1 = new_cm.get_coord(name)
            out2 = coord[slice(*ind_tuple)]
            if not out1 == out2:
                assert all_close(out1.values, out2.values)
            # then check that the whole coord_manager are equal
            cm1 = cm.select(**{name: value_tuple})
            cm2 = cm.select(**{name: ind_tuple}, samples=True)
            assert cm1 == cm2

    @pytest.mark.parametrize("index", inds)
    def test_single_values(self, cm_basic, index):
        """
        Single values should be treated like slice(val, val+1)
        as not to collapse the dimensions when samples=True.
        """
        cm = cm_basic
        data = np.empty(cm.shape)
        for dim in cm_basic.dims:
            kwargs = {dim: index}
            out1, new_data = cm.select(array=data, samples=True, **kwargs)
            dim_ind = cm.get_axis(dim)
            # now the array should have a len(1) in the selected dimension.
            assert out1.shape[dim_ind] == new_data.shape[dim_ind] == 1
            new_value = out1.coord_map[dim].values[0]
            expected_value = cm.get_array(dim)[index]
            all_close(new_value, expected_value)

    def test_trim_related_coords(self, cm_multidim):
        """Ensure trim also trims related dimensions."""
        cm = cm_multidim
        data = np.empty(cm.shape)
        out, new_data = cm.select(array=data, time=slice(2, 4), samples=True)
        for name, coord in out.coord_map.items():
            dims = cm.dim_map[name]
            if "time" not in dims:
                continue
            time_id = dims.index("time")
            assert coord.shape[time_id] == 2

    def test_samples_slice(self, coord_manager):
        """Ensure we can select when samples=True using ... or None."""
        new, _ = coord_manager.select(time=..., samples=True)
        assert new == coord_manager

    def test_select_shared_dims(self, coord_manager):
        """Ensure selections work when queries share a dimension."""
        dist = coord_manager.get_coord("distance")
        new_coord = np.arange(len(dist))
        cm = coord_manager.update_coords(
            d1=("distance", new_coord),
            d2=("distance", new_coord),
        )
        # Relative values should raise when the same dim is targeted by
        # multiple coords.
        with pytest.raises(CoordError):
            cm.select(d1=(3, None), d2=(None, 6), relative=True)
        # Same for samples.
        with pytest.raises(CoordError):
            cm.select(d1=(3, None), d2=(None, 6), samples=True)
        # But normal values should work and produce a shape of 4 for this case.
        out, _ = cm.select(d1=(3, None), d2=(None, 6))
        distance_dim = out.get_axis("distance")
        assert out.shape[distance_dim] == 4

    def test_array_by_values(self, coord_manager):
        """Ensure using an input array works for select."""
        name = coord_manager.dims[0]
        coord = coord_manager.get_coord(coord_manager.dims[0])
        values = coord.values[: int(np.ceil(len(coord) / 2))]
        out, _ = coord_manager.select(**{name: values})
        assert set(out.get_array(name)) == set(values)

    def test_array_by_ints(self, coord_manager):
        """Ensure int array also works."""
        name = coord_manager.dims[0]
        coord = coord_manager.get_coord(coord_manager.dims[0])
        if len(coord) <= 1:
            pytest.skip("Need non-one length coordinate")
        values = np.arange(len(coord))[:: len(coord) // 2]
        out, _ = coord_manager.select(**{name: values}, samples=True)
        assert len(out.get_coord(name)) == len(values)

    def test_array_subset(self, coord_manager):
        """Ensure a subset of values works with select."""
        if "time" not in coord_manager.dims:
            return
        dim_name = coord_manager.dims[0]
        coord = coord_manager.get_coord(dim_name)
        sub = coord[1:-1].values
        out, _ = coord_manager.select(**{dim_name: sub})
        assert out.shape[out.get_axis(dim_name)] == len(sub)

    def test_array_indices(self, coord_manager):
        """Ensure array indices work in select."""
        dims = coord_manager.dims
        dim_name = dims[1] if coord_manager.ndim > 1 else dims[0]
        coord = coord_manager.get_coord(dim_name)
        if len(coord) < 5:
            return
        inds = np.array([1, 2, 3])
        out, _ = coord_manager.select(**{dim_name: inds}, samples=True)
        coord_out = out.get_coord(dim_name)
        assert len(inds) == len(coord_out)

    def test_multiple_dims_array_select(self, cm_basic):
        """Ensure multiple arrays can be used for selection."""
        time = cm_basic.get_array("time")
        dist = cm_basic.get_array("distance")
        time_inds = np.array([1, 4, 0])
        dist_inds = np.array([10, 1, 2])
        new_time, new_dist = time[time_inds], dist[dist_inds]

        data = np.arange(cm_basic.size).reshape(cm_basic.shape)

        # select doesn't re-order so we need to sort inds
        ti, di = np.sort(time_inds), np.sort(dist_inds)
        expected_data = np.stack([data[x, di] for x in ti])
        cm1, new_data1 = cm_basic.select(
            time=new_time,
            distance=new_dist,
            array=data,
        )
        cm2, new_data2 = cm_basic.select(
            time=time_inds, distance=dist_inds, array=data, samples=True
        )
        assert cm1 == cm2
        assert np.all(new_data1 == new_data2)

        for cm, new_data in zip([cm1, cm2], [new_data1, new_data2]):
            assert cm.shape == (len(time_inds), len(dist_inds))
            # Ensure all the coordinates are correct.
            assert np.all(np.isin(cm.get_array("time"), new_time))
            assert np.all(np.isin(cm.get_array("distance"), new_dist))
            # Check that the data were re-arranged correctly
            assert np.all(expected_data == new_data)

    def test_select_non_dim_coord(self, cm_basic):
        """Ensure selecting on a non-dimension coord does nothing."""
        # TODO should this raise in the future?
        new_cm = cm_basic.update(new_dim=(None, [1, 2, 3]))
        out, _ = new_cm.select(new_dim=(1, 20))
        assert new_cm == out

    def test_select_non_dim_coord_shortens_coordinate(self, cm_basic):
        """Test that selecting non-dimensional coords shortens only that coordinate."""
        # Add a non-dimensional coordinate with numeric values
        quality_scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7])
        new_cm = cm_basic.update(quality=(None, quality_scores))
        # Select subset of quality scores using array indexing
        selected_indices = np.array([1, 3, 5])  # Select indices 1, 3, 5
        out, _ = new_cm.select(quality=selected_indices, samples=True)
        # Quality coordinate should be shortened
        expected_quality = quality_scores[selected_indices]
        assert np.array_equal(out.get_array("quality"), expected_quality)
        # Dimensional coordinates should be unchanged
        assert cm_basic.shape == out.shape
        assert np.array_equal(out.get_array("time"), new_cm.get_array("time"))
        assert np.array_equal(out.get_array("distance"), new_cm.get_array("distance"))

    def test_select_non_dim_coord_with_boolean_mask(self, cm_basic):
        """Test selecting non-dimensional coordinates using boolean arrays."""
        # Add a non-dimensional coordinate
        values = np.array([10, 20, 30, 40, 50, 60, 70])
        new_cm = cm_basic.update(sensor_values=(None, values))
        # Create boolean mask
        mask = values > 35  # Should select [40, 50, 60, 70]
        out, _ = new_cm.select(sensor_values=mask)
        # Only the non-dimensional coordinate should be affected
        expected_values = values[mask]
        assert np.array_equal(out.get_array("sensor_values"), expected_values)
        # Dimensional coordinates should remain unchanged
        for coord in set(cm_basic.coord_map) - {"sensor_values"}:
            assert cm_basic.get_coord(coord) == new_cm.get_coord(coord)

    def test_select_multi_dim_coord_raises(self, cm_multidim):
        """
        Coords that are associated with more than one dim cannot be selected
        because it could ruin the squareness of the patch.
        """
        # Non-dim coord associated with one dimension should work.
        lat = cm_multidim.get_array("latitude")
        lat_mean = np.mean(lat)
        out, _ = cm_multidim.select(latitude=(..., lat_mean))
        assert isinstance(out, dc.CoordManager)
        # Multi-dim coord should raise CoordError
        msg = "Only 1 dimensional coordinates"
        with pytest.raises(CoordError, match=msg):
            cm_multidim.select(quality=(1, 20))

    def test_select_coord_tied_to_dimension_affects_others(self, cm_multidim):
        """
        Test that selecting a coord tied to a dimension affects other coords
        on that dim.
        """
        # cm_multidim should have coordinates that share dimensions
        # Get a coordinate that's tied to a dimension and has other coords sharing
        # that dim
        lat = cm_multidim.get_array("latitude")
        lat_mean = np.mean(lat)
        out, _ = cm_multidim.select(latitude=(..., lat_mean))
        # Check that the new lat is what we expect.
        new_lat = out.get_array("latitude")
        expected = lat[lat <= lat_mean]
        assert np.array_equal(new_lat, expected)
        # And the other coord associated with that dimension have the same len.
        for name, coord in out.coord_map.items():
            coord_dims = out.dim_map[name]
            # Skip coords not tied to distance dimension
            if "distance" not in coord_dims:
                continue
            axis = coord_dims.index("distance")
            assert coord.shape[axis] == len(new_lat)

    def test_select_nonexistent_coordinate_ignores_gracefully(self, cm_basic):
        """Test that selecting on a non-existent coordinate is ignored gracefully."""
        # This tests line 122 in _get_indexers_and_new_coords_dict
        original_shape = cm_basic.shape
        out, _ = cm_basic.select(nonexistent_coord=(1, 10))

        # Should return unchanged coordinate manager
        assert out == cm_basic
        assert out.shape == original_shape

        # Should work with multiple nonexistent coordinates
        out2, _ = cm_basic.select(
            fake_coord1=(1, 2), fake_coord2=slice(0, 5), another_fake=(10, 20)
        )
        assert out2 == cm_basic
        assert out2.shape == original_shape

    def test_select_mix_valid_invalid_coordinates(self, cm_basic):
        """Test selecting with mix of valid and invalid coordinate names."""
        # This also exercises line 122 but with mixed scenarios
        time_vals = cm_basic.get_array("time")
        subset_time = (time_vals[1], time_vals[-2])

        out, _ = cm_basic.select(
            time=subset_time,  # valid coordinate
            nonexistent=(1, 10),  # invalid coordinate - should be ignored
            fake_dim=slice(0, 5),  # another invalid coordinate
        )

        # Only the valid coordinate selection should have been applied
        assert (
            out.shape[cm_basic.get_axis("time")]
            < cm_basic.shape[cm_basic.get_axis("time")]
        )
        # Distance should be unchanged since it wasn't selected
        assert (
            out.shape[cm_basic.get_axis("distance")]
            == cm_basic.shape[cm_basic.get_axis("distance")]
        )


class TestOrder:
    """Tests for ordering coordinate managers."""

    def test_order_by_values(self, cm_basic):
        """Ensure we can re-order the coord manager basic on coord values."""
        time = cm_basic.get_array("time")
        new_inds = [2, 1, 0, 3, 6]
        new_times = time[new_inds]
        data = np.arange(cm_basic.size).reshape(cm_basic.shape)
        cm, new_data = cm_basic.order(time=new_times, array=data)
        assert cm.shape == new_data.shape
        assert np.all(cm.get_array("time") == new_times)
        # Ensure the data were also arranged correctly.
        assert np.all(data[new_inds] == new_data)

    def test_order_by_samples(self, cm_basic):
        """Ensure we can re-order the coord manager based on coord samples."""
        time = cm_basic.get_array("time")
        new_inds = [2, 1, 0, 3, 6]
        new_times = time[new_inds]
        data = np.arange(cm_basic.size).reshape(cm_basic.shape)
        cm, new_data = cm_basic.order(time=new_inds, array=data, samples=True)
        assert cm.shape == new_data.shape
        assert np.all(cm.get_array("time") == new_times)
        # Ensure the data were also arranged correctly.
        assert np.all(data[new_inds] == new_data)

    def test_multiple_dim_order(self, cm_basic):
        """Ensure multiple dims can be used for re-ordering."""
        time = cm_basic.get_array("time")
        dist = cm_basic.get_array("distance")
        time_inds = [1, 4, 0]
        dist_inds = [10, 1, 2]
        new_time, new_dist = time[time_inds], dist[dist_inds]

        data = np.arange(cm_basic.size).reshape(cm_basic.shape)
        expected_data = np.stack([data[x, dist_inds] for x in time_inds])

        cm1, new_data1 = cm_basic.order(
            time=new_time,
            distance=new_dist,
            array=data,
        )
        cm2, new_data2 = cm_basic.order(
            time=time_inds, distance=dist_inds, array=data, samples=True
        )
        assert cm1 == cm2
        assert np.all(new_data1 == new_data2)

        for cm, new_data in zip([cm1, cm2], [new_data1, new_data2]):
            assert cm.shape == (len(time_inds), len(dist_inds))
            # Ensure all the coordinates are correct.
            assert np.all(cm.get_array("time") == new_time)
            assert np.all(cm.get_array("distance") == new_dist)
            # Check that the data were re-arranged correctly
            assert np.all(expected_data == new_data)


class TestEquals:
    """Tests for coord manager equality."""

    def test_basic_equals(self, coord_manager):
        """All coord managers should equal themselves."""
        assert coord_manager == coord_manager

    def test_unequal_float_coords(self, cm_multidim):
        """Ensure if coordinates are not equal false is returned."""
        coord = cm_multidim.coord_map["latitude"]
        new = get_coord(data=coord.values + 10)
        args = dict(latitude=("distance", new))
        new_coord = cm_multidim.update(**args)
        assert new_coord != cm_multidim

    def test_unequal_wrong_type(self, cm_basic):
        """Non-coord managers should not be considered equal."""
        cm = cm_basic
        assert cm != 10
        assert cm != "bob"
        assert cm != {1: 2, 2: 2}
        assert cm != cm.coord_map["distance"]

    def test_cm_with_non_coord(self, cm_basic):
        """Non coords should be not equal to other coords."""
        non_1 = cm_basic.update(time=1)
        assert non_1 != cm_basic
        assert cm_basic != non_1
        non_2 = cm_basic.update(time=len(cm_basic.get_coord("time")))
        assert non_2 != cm_basic
        assert cm_basic != non_2


class TestTranspose:
    """Test suite for transposing dimensions."""

    @pytest.fixture(scope="class")
    def many_dims_cm(self):
        """Create a coordmanager with many dimensions."""
        dims = ("one", "two", "three", "four")
        coords = {x: np.arange(10) for x in dims}
        many_dims = get_coord_manager(coords, dims=dims)
        return many_dims

    def test_simple_transpose(self, cm_basic):
        """Ensure the coord manager can be transposed."""
        dims = cm_basic.dims
        new_dims = dims[::-1]
        tran = cm_basic.transpose(*new_dims)
        assert tran.dims == new_dims
        assert tran.shape != cm_basic.shape
        assert tran.shape == cm_basic.shape[::-1]

    def test_empty_ellipses(self, many_dims_cm):
        """Empty transpose should just reverse order."""
        out = many_dims_cm.transpose()
        assert out.dims == many_dims_cm.dims[::-1]

    def test_ellipses_at_start(self, many_dims_cm):
        """Ensure ellipses works to just stick things at start/end."""
        new = many_dims_cm.transpose(..., "one", "two")
        assert new.dims == ("three", "four", "one", "two")

    def test_ellipses_at_end(self, many_dims_cm):
        """Ensure ellipses works to just stick things at start/end."""
        new = many_dims_cm.transpose("four", "two", ...)
        assert new.dims == ("four", "two", "one", "three")

    def test_ellipses_in_middle(self, many_dims_cm):
        """Ensure ellipses works to just stick things at start/end."""
        new = many_dims_cm.transpose("four", ..., "one")
        assert new.dims == ("four", "two", "three", "one")

    def test_duplicate_raises(self, many_dims_cm):
        """... can only be used once, and repeat dimensions must raise."""
        with pytest.raises(ParameterError, match="duplicate dimensions"):
            many_dims_cm.transpose(..., "two", ...)
        with pytest.raises(ParameterError, match="duplicate dimensions"):
            many_dims_cm.transpose("one", "one", "two", "three", "four")

    def test_raises_no_ellipses_missing_dim(self, many_dims_cm):
        """When ... is not used all dims must appear in args."""
        with pytest.raises(ParameterError, match="specify all dimensions"):
            many_dims_cm.transpose("one", "three", "four")
        with pytest.raises(ParameterError, match="specify all dimensions"):
            many_dims_cm.transpose("one")


class TestRenameDims:
    """Test case for renaming dimensions."""

    def test_rename_dims(self, cm_basic):
        """Ensure dimensions can be renamed."""
        rename_map = {x: x[:2] for x in cm_basic.dims}
        out = cm_basic.rename_coord(**rename_map)
        assert set(out.dims) == set(rename_map.values())

    def test_rename_extra_coords_kept(self, cm_multidim):
        """Ensure the extra dims are not dropped when a coord is renamed."""
        cm = cm_multidim
        # we should be able to rename distance to dist and
        # distance just gets swapped for dist
        out = cm.rename_coord(distance="dist")
        assert "dist" in out.dim_map
        assert "dist" in out.coord_map
        assert "dist" in out.dim_map["latitude"]

    def test_rename_same_dims_extra_coords(self, cm_multidim):
        """Ensure renaming dims the same doesn't mess with multidims."""
        cm = cm_multidim
        dim_map = {x: x for x in cm.dims}
        out = cm.rename_coord(**dim_map)
        assert set(out.dim_map) == set(cm.dim_map)


class TestUpdateFromAttrs:
    """Tests to ensure updating attrs can update coordinates."""

    def test_update_min(self, cm_basic):
        """Ensure min time in attrs updates appropriate coord."""
        for dim in cm_basic.dims:
            coord = cm_basic.coord_map[dim]
            attrs = {f"{dim}_max": coord.min()}
            new, _ = cm_basic.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            assert len(new_coord) == len(coord)
            assert new_coord.max() == coord.min()

    def test_update_max(self, cm_basic):
        """Ensure max time in attrs updates appropriate coord."""
        for dim in cm_basic.dims:
            coord = cm_basic.coord_map[dim]
            attrs = {f"{dim}_min": coord.max()}
            dist = coord.max() - coord.min()
            new, _ = cm_basic.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max() - new_coord.min()
            assert dist == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min() == coord.max()

    def test_update_step(self, cm_basic):
        """Ensure the step can be updated which changes endtime."""
        for dim in cm_basic.dims:
            coord = cm_basic.coord_map[dim]
            attrs = {f"{dim}_step": coord.step * 10}
            dist = coord.max() - coord.min()
            new, _ = cm_basic.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max() - new_coord.min()
            assert (dist * 10) == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min() == coord.min()

    def test_attrs_as_dict(self, cm_basic):
        """Ensure the attrs returned has coords attached."""
        coord = cm_basic.coord_map["time"]
        attrs = {"time_max": coord.min()}
        cm, attrs = cm_basic.update_from_attrs(attrs)
        assert attrs.coords == cm.to_summary_dict()
        assert attrs.dim_tuple == cm.dims

    def test_attrs_as_patch_attr(self, cm_basic):
        """Ensure this also works when attrs is a patch attr."""
        attrs = dc.PatchAttrs(time_min=to_datetime64("2022-01-01"))
        cm, new_attrs = cm_basic.update_from_attrs(attrs)
        assert new_attrs.coords == cm.to_summary_dict()
        assert new_attrs.dim_tuple == cm.dims

    def test_consistent_attrs_leaves_coords_unchanged(self, random_patch):
        """Attrs which are already consistent should leave coord unchanged."""
        attrs, coords = random_patch.attrs, random_patch.coords
        new_coords, new_attrs = coords.update_from_attrs(attrs)
        assert new_coords == coords


class TestUpdate:
    """Tests for updating coordinates."""

    def test_simple(self, cm_basic):
        """Ensure coordinates can be updated (replaced)."""
        new_time = cm_basic.get_array("time") + np.timedelta64(1, "s")
        out = cm_basic.update(time=new_time)
        assert np.all(np.equal(out.get_array("time"), new_time))

    def test_extra_coords_kept(self, cm_multidim):
        """Ensure extra coordinates are kept."""
        cm = cm_multidim
        new_time = cm.get_array("time") + np.timedelta64(1, "s")
        out = cm.update(time=new_time)
        assert set(out.coord_map) == set(cm_multidim.coord_map)
        assert set(out.coord_map) == set(cm_multidim.coord_map)

    def test_size_change(self, cm_basic):
        """Ensure sizes of dimensions can be changed."""
        new_time = cm_basic.get_array("time")[:10]
        out = cm_basic.update(time=new_time)
        assert np.all(np.equal(out.get_array("time"), new_time))

    def test_size_change_drops_old_coords(self, cm_multidim):
        """When the size changes on multidim, dims should be dropped."""
        cm = cm_multidim
        new_dist = cm.get_array("distance")[:10]
        dropped_coords = set(cm.coord_map) - set(cm.dims)
        out = cm.update(distance=new_dist)
        assert dropped_coords.isdisjoint(set(out.coord_map))

    def test_update_degenerate(self, coord_manager):
        """Tests for updating coord with degenerate coordinates."""
        cm = coord_manager
        out = cm.update(time=cm.coord_map["time"].empty())
        assert out.shape[out.get_axis("time")] == 0
        assert len(out.coord_map["time"]) == 0

    def test_update_degenerate_dim_multicoord(self, cm_multidim):
        """
        If one dim is degenerate all associated coords should be
        dropped.
        """
        cm = cm_multidim
        degen_time = cm.coord_map["time"].empty()
        new = cm.update(time=degen_time)
        assert new != cm
        # any coords with time as a dim (but not time itself) should be gone.
        has_time = [i for i, v in cm.dim_map.items() if ("time" in v and i != "time")]
        assert set(has_time).isdisjoint(set(new.coord_map))

    def test_update_none_drops(self, cm_basic):
        """Ensure when passing coord=None the coord is dropped."""
        cm1 = cm_basic.update(time=None)
        cm2, _ = cm_basic.drop_coords("time")
        assert cm1 == cm2

    def test_update_only_start(self, cm_basic):
        """Ensure start_coord can be used to update."""
        time1 = cm_basic.coord_map["time"]
        new_start = time1.max()
        cm = cm_basic.update(time_min=new_start)
        time2 = cm.coord_map["time"]
        assert time2.min() == new_start

    def test_dissociate(self, cm_basic):
        """Ensure coordinates can be dissociated with update_coords."""
        time = cm_basic.coord_map["time"]
        new_time = time.values + dc.to_timedelta64(1)
        new = cm_basic.update(new_time=("time", new_time))
        assert "new_time" in new.dim_map and "new_time" in new.coord_map
        dissociated = new.update(new_time=(None, new_time))
        assert dissociated.dim_map["new_time"] == ()

    def test_unchanged_coords(self, cm_with_units):
        """Ensure coordinates not updated are left unchanged."""
        cm = cm_with_units
        new_time = cm.coord_map["time"].update(min=0)
        new = cm.update(time=new_time)
        assert new.coord_map["distance"] == cm.coord_map["distance"]

    def test_coord_with_new_dim(self, coord_manager):
        """Ensure a new dimension can be added."""
        # should work for single name input as well as tuple.
        cm1 = coord_manager.update(bob=("bob", np.atleast_1d(1)))
        cm2 = coord_manager.update(bob=(("bob",), np.atleast_1d(1)))
        assert cm1 == cm2
        for cm in [cm1, cm2]:
            assert cm.ndim == (coord_manager.ndim + 1)
            assert cm.dims[-1] == "bob"

    def test_update_with_units(self, coord_manager):
        """Ensure units stick around when updated."""
        ft = dc.get_quantity("ft")
        dist = coord_manager.get_array("distance")
        new = coord_manager.update(distance=(dist * 2) * ft)
        new_coord = new.coord_map["distance"]
        assert dc.get_quantity(new_coord.units) == ft


class TestSqueeze:
    """Tests for squeezing degenerate dimensions."""

    def test_bad_dim_raises(self, cm_degenerate_time):
        """Ensure a bad dimension will raise CoordError."""
        with pytest.raises(CoordError, match="they don't exist"):
            cm_degenerate_time.squeeze("is_money")

    def test_non_zero_length_dim_raises(self, cm_degenerate_time):
        """Ensure a dim with length > 0 can't be squeezed."""
        with pytest.raises(CoordError, match="non-zero length"):
            cm_degenerate_time.squeeze("distance")

    def test_squeeze_single_degenerate(self, cm_degenerate_time):
        """Ensure a single degenerate dimension can be squeezed out."""
        cm = cm_degenerate_time
        out = cm.squeeze("time")
        assert "time" not in out.dims

    def test_squeeze_no_dim(self, cm_degenerate_time):
        """Ensure all degenerate dims are squeezed when no dim specified."""
        cm = cm_degenerate_time
        out = cm.squeeze()
        assert "time" not in out.dims


class TestNonDimCoords:
    """Tests for adding non-dimensional coordinates."""

    def test_update_with_1d_coordinate(self, cm_basic):
        """Ensure we can add coordinates."""
        lat = np.ones_like(cm_basic.get_array("distance"))
        out = cm_basic.update(latitude=("distance", lat))
        assert out is not cm_basic
        assert out.dims == cm_basic.dims, "dims shouldn't change"
        assert np.all(out.get_array("latitude") == lat)
        assert out.dim_map["latitude"] == out.dim_map["distance"]
        # the dim map shouldn't shrink; only things get added.
        assert set(out.dim_map).issuperset(set(cm_basic.dim_map))

    def test_init_with_1d_coordinate(self, cm_basic):
        """Ensure initing with 1D non-dim coords works."""
        with suppress_warnings():
            coords = dict(cm_basic)
        lat = np.ones_like(cm_basic.get_array("distance"))
        coords["latitude"] = ("distance", lat)
        out = get_coord_manager(coords, dims=cm_basic.dims)
        assert out is not cm_basic
        assert out.dims == cm_basic.dims, "dims shouldn't change"
        assert np.all(out.get_array("latitude") == lat)
        assert out.dim_map["latitude"] == out.dim_map["distance"]
        # the dim map shouldn't shrink; only things get added.
        assert set(out.dim_map).issuperset(set(cm_basic.dim_map))

    def test_update_2d_coord(self, cm_basic):
        """Ensure updating can be done with 2D coordinate."""
        dist = cm_basic.get_array("distance")
        time = cm_basic.get_array("time")
        quality = np.ones((len(dist), len(time)))
        dims = ("distance", "time")
        new = cm_basic.update(qual=(dims, quality))
        assert new.dims == cm_basic.dims
        assert new.dim_map["qual"] == dims
        assert "qual" in new


class TestDecimate:
    """Tests for evenly subsampling along a dimension."""

    def test_decimate_along_time(self, cm_basic):
        """Simply ensure decimation reduces shape."""
        cm = cm_basic
        ind = cm_basic.get_axis("distance")
        out, _ = cm.decimate(distance=2)
        assert len(out.coord_map["distance"]) < len(cm.coord_map["distance"])
        assert (out.shape[ind] * 2) == cm.shape[ind]


class TestSort:
    """Tests for sorting coord managers."""

    def test_sort_unsorted_dim_coord(self, cm_wacky_dims):
        """Simple test for single dim sort."""
        cm = cm_wacky_dims
        sorted_cm, _ = cm.sort("distance")
        assert sorted_cm.coord_map["distance"].sorted

    def test_sort_sorted_dim_coord(self, cm_wacky_dims):
        """Ensure sorting a sorted dim coord does nothing."""
        cm = cm_wacky_dims
        sorted_cm, _ = cm.sort("time")
        assert sorted_cm.coord_map["time"].sorted

    def test_r_sort_unsorted_dim_coord(self, cm_wacky_dims):
        """Simple test for single dim sort."""
        cm = cm_wacky_dims
        sorted_cm, _ = cm.sort("distance", reverse=True)
        assert sorted_cm.coord_map["distance"].reverse_sorted

    def test_r_sort_sorted_dim_coord(self, cm_wacky_dims):
        """Ensure sorting a sorted dim coord does nothing."""
        cm = cm_wacky_dims
        sorted_cm, _ = cm.sort("time", reverse=True)
        assert sorted_cm.coord_map["time"].reverse_sorted

    def test_simultaneous_sort(self, cm_wacky_dims):
        """Ensure all dimensions can be sorted at once."""
        sorted_cm, _ = cm_wacky_dims.sort()
        for _, coord in sorted_cm.coord_map.items():
            assert coord.sorted

    def test_simultaneous_r_sort(self, cm_wacky_dims):
        """Ensure all dimensions can be reverse sorted at once."""
        sorted_cm, _ = cm_wacky_dims.sort(reverse=True)
        for _, coord in sorted_cm.coord_map.items():
            assert coord.reverse_sorted

    def test_sort_sorted_cm(self, cm_basic):
        """Sorting a coord manager that is already sorted should do nothing."""
        cm = cm_basic
        cm_sorted, _ = cm.sort()
        assert cm == cm_sorted

    def test_sort_2d_coord_raises(self, cm_multidim):
        """Sorting on 2D coord is ambiguous, it should raise."""
        cm = cm_multidim
        with pytest.raises(CoordSortError, match="more than one dimension"):
            cm.sort("quality")

    def test_sort_two_coords_same_dim_raises(self, cm_multidim):
        """Trying to sort on two coords which share a dim should raise."""
        # need to make distance un sorted.
        cm, _ = cm_multidim.sort("distance", reverse=True)
        with pytest.raises(CoordSortError, match="they share a dimension"):
            cm.sort("latitude", "distance")
        # this should also raise since sorting latitude could unsort distance
        with pytest.raises(CoordSortError, match="they share a dimension"):
            cm_multidim.sort("distance", "latitude")

    def test_related_coords(self, cm_multidim):
        """Sorting on one coordinate should also sort others that share a dim."""
        cm = cm_multidim
        cm_sorted, _ = cm.sort("latitude")
        assert cm_sorted.coord_map["latitude"].sorted
        # distance should now be unsorted.
        assert not cm_sorted.coord_map["distance"].sorted


class TestSnap:
    """Tests for snapping coordinates."""

    def test_generic_snap(self, coord_manager):
        """All coord managers should support snap."""
        cm, _ = coord_manager.snap()
        for dim in cm.dims:
            coord = cm.coord_map[dim]
            if coord.degenerate or len(coord) < 2:
                continue
            assert coord.sorted
            assert not coord.reverse_sorted
            assert coord.evenly_sampled

    def test_generic_snap_reverse(self, coord_manager):
        """All coord managers should support snap in reverse direction."""
        cm, _ = coord_manager.snap(reverse=True)
        for dim in cm.dims:
            coord = cm.coord_map[dim]
            if coord.degenerate or len(coord) < 2:
                continue
            assert not coord.sorted
            assert coord.reverse_sorted
            assert coord.evenly_sampled

    def test_snap_dims(self, cm_wacky_dims):
        """Happy path for snapping dimensions."""
        cm = cm_wacky_dims
        data = np.ones(cm.shape)
        out, out_data = cm.snap(array=data)
        for coord_name in out.coord_map:
            coord1 = cm.coord_map[coord_name]
            coord2 = out.coord_map[coord_name]
            assert coord1.dtype == coord2.dtype
            assert len(coord1) == len(coord2)
        assert out_data.shape == data.shape

    def test_snap_expected_dt(
        self, cm_dt_small_diff, memory_spool_small_dt_differences
    ):
        """Ensure snapping creates expected dt from merged df."""
        spool = memory_spool_small_dt_differences
        expected_dt = get_middle_value([x.attrs.time_step for x in spool])
        snapped = cm_dt_small_diff.snap()[0]
        assert snapped.coord_map["time"].step == expected_dt


class TestConvertUnits:
    """Tests for converting coordinate units."""

    def test_convert_changes_labels(self, cm_basic):
        """Basic tests for converting units."""
        cm = cm_basic.convert_units(time="s", distance="furlong")
        time, dist = cm.coord_map["time"], cm.coord_map["distance"]
        assert time.units == get_quantity("s")
        assert dist.units == get_quantity("furlong")

    def test_convert_changes_values(self, cm_with_units):
        """Ensure values are scaled accordingly."""
        conv = get_quantity("m").to("ft").magnitude
        cm = cm_with_units.convert_units(distance="ft")
        dist1 = cm_with_units.coord_map["distance"]
        dist2 = cm.coord_map["distance"]
        assert np.isclose(dist1.step, dist2.step / conv)
        assert np.isclose(dist1.start, dist2.start / conv)
        assert np.isclose(dist1.stop, dist2.stop / conv)

    def test_convert_time(self, cm_with_units):
        """When time is already set and a datetime, units should just change."""
        # TODO, reconsider this; i am not sure its right.
        new = cm_with_units.convert_units(time="ms")
        time1 = cm_with_units.coord_map["time"]
        time2 = new.coord_map["time"]
        assert time1.dtype == time2.dtype
        assert np.all(np.equal(time1.values, time2.values))


class TestDisassociate:
    """Ensure coordinates can be disassociated from coordinate manager."""

    def test_basic_disassociate(self, cm_multidim):
        """Ensure coordinates can be dissociated."""
        cm = cm_multidim.disassociate_coord("time")
        # now time should still exist but have no dim, quality should drop
        assert "time" in cm.coord_map
        assert "quality" not in cm.coord_map
        time_dim = cm.dim_map["time"]
        assert time_dim == ()
        assert "time" not in cm.dims
        # but if both time and quality are dissociated nothing should be dropped.
        cm = cm_multidim.disassociate_coord("time", "quality")
        assert {"time", "quality"}.issubset(set(cm.coord_map))
        assert "time" not in cm.dims

    def test_disassociate_non_dim_coord(self, cm_multidim):
        """Ensure non-dim coords can be dissociated with no change to dims."""
        cm = cm_multidim.disassociate_coord("quality")
        assert cm.dim_map["quality"] == ()
        assert cm.dims == cm_multidim.dims


class TestSetDims:
    """Tests for setting dims to non-dimensional coordinates."""

    def test_swap_coords(self, cm_multidim):
        """Simple coord swap."""
        cm = cm_multidim
        out1 = cm.set_dims(distance="latitude")
        assert "latitude" in out1.dims
        assert "latitude" not in cm.dims, "original dim should be unchanged"
        # distance should now be associated with latitude.
        assert out1.dim_map["distance"][0] == "latitude"

    def test_bad_dimension(self, cm_multidim):
        """Ensure if a bad dimension is provided it raises."""
        cm = cm_multidim
        match = "is not a dimension or"
        with pytest.raises(CoordError, match=match):
            cm.set_dims(bob="latitude")
        with pytest.raises(CoordError, match=match):
            cm.set_dims(distance="bob")

    def test_wrong_shape(self, cm_multidim):
        """Ensure if a coord with wrong shape is chosen an error is raised."""
        cm = cm_multidim
        match = "does not match the shape of"
        with pytest.raises(CoordError, match=match):
            cm.set_dims(distance="quality")


class TestFlip:
    """
    Tests for flipping a coord manager.

    The flip method takes positional arguments: flip(*dims) and correctly
    flips both dimensional coordinates and their associated coordinates.
    """

    def test_flip_single_dim_coord(self, cm_basic):
        """Ensure we can flip a single dimensional coordinate."""
        cm = cm_basic
        original_time = cm.get_array("time")
        flipped_cm = cm.flip("time")
        # The flipped coordinate should be reversed
        flipped_time = flipped_cm.get_array("time")
        assert np.array_equal(flipped_time, original_time[::-1])
        # Other coordinates should remain unchanged
        assert np.array_equal(
            flipped_cm.get_array("distance"), cm.get_array("distance")
        )
        # Shape and dimensions should remain the same
        assert flipped_cm.shape == cm.shape
        assert flipped_cm.dims == cm.dims
        # Should not mutate original object
        assert flipped_cm is not cm

    def test_flip_multiple_dim_coords(self, cm_basic):
        """Ensure we can flip multiple dimensional coordinates."""
        cm = cm_basic
        original_time = cm.get_array("time")
        original_distance = cm.get_array("distance")
        flipped_cm = cm.flip("time", "distance")
        # Both coordinates should be reversed
        flipped_time = flipped_cm.get_array("time")
        flipped_distance = flipped_cm.get_array("distance")
        assert np.array_equal(flipped_time, original_time[::-1])
        assert np.array_equal(flipped_distance, original_distance[::-1])
        # Shape and dimensions should remain the same
        assert flipped_cm.shape == cm.shape
        assert flipped_cm.dims == cm.dims

    def test_flip_with_associated_coords(self, cm_multidim):
        """Ensure flipping a dim coord also flips associated coordinates."""
        cm = cm_multidim
        original_distance = cm.get_array("distance")
        original_latitude = cm.get_array("latitude")
        # Flip the distance coordinate
        flipped_cm = cm.flip("distance")
        # Distance should be flipped
        flipped_distance = flipped_cm.get_array("distance")
        assert np.array_equal(flipped_distance, original_distance[::-1])
        # Associated coordinates (like latitude) should also be flipped
        flipped_latitude = flipped_cm.get_array("latitude")
        assert np.array_equal(flipped_latitude, original_latitude[::-1])
        # Time should remain unchanged
        assert np.array_equal(flipped_cm.get_array("time"), cm.get_array("time"))
        # 2D associated coordinate (quality) should be flipped along distance axis
        q_before = cm.coord_map["quality"].values
        q_after = flipped_cm.coord_map["quality"].values
        assert np.array_equal(q_after, q_before[:, ::-1])

    def test_flip_2d_coord_raises(self, cm_multidim):
        """Ensure flipping a 2D coordinate directly raises an error."""
        cm = cm_multidim
        msg = "CoordManager can only flip 1D coords directly"
        with pytest.raises(CoordError, match=msg):
            cm.flip("quality")

    def test_flip_nonexistent_coord(self, cm_basic):
        """Ensure flipping a nonexistent coordinate raises an appropriate error."""
        cm = cm_basic
        # This should raise when trying to get the coordinate
        with pytest.raises(CoordError):
            cm.flip("nonexistent_coord")

    def test_flip_preserves_other_properties(self, cm_basic):
        """Ensure flip preserves coordinate properties like units, etc."""
        cm = cm_basic
        original_time_coord = cm.coord_map["time"]
        flipped_cm = cm.flip("time")
        flipped_time_coord = flipped_cm.coord_map["time"]
        # Properties other than values should be preserved
        assert original_time_coord.units == flipped_time_coord.units
        assert original_time_coord.dtype == flipped_time_coord.dtype
        assert len(original_time_coord) == len(flipped_time_coord)

    def test_flip_empty_coord(self, cm_basic):
        """Ensure we can flip an empty coordinate."""
        cm = cm_basic
        # Create a coordinate manager with an empty time coordinate
        empty_time = cm.coord_map["time"].empty()
        empty_cm = cm.update(time=empty_time)
        # Flipping should work even with empty coordinates
        flipped_cm = empty_cm.flip("time")
        assert len(flipped_cm.get_array("time")) == 0
        assert flipped_cm.shape[empty_cm.get_axis("time")] == 0

    def test_flip_preserves_coord_manager_equality(self, cm_basic):
        """Ensure flipping twice returns to original state."""
        cm = cm_basic
        double_flipped = cm.flip("time").flip("time")
        # Double flip should return to original state
        assert cm == double_flipped
        assert np.array_equal(cm.get_array("time"), double_flipped.get_array("time"))
        assert np.array_equal(
            cm.get_array("distance"), double_flipped.get_array("distance")
        )

    def test_flip_all_dimensions(self, cm_basic):
        """Ensure we can flip all dimensions at once."""
        cm = cm_basic
        all_flipped = cm.flip(*cm.dims)
        # All coordinates should be flipped
        for dim in cm.dims:
            original = cm.get_array(dim)
            flipped = all_flipped.get_array(dim)
            assert np.array_equal(flipped, original[::-1])
        # Shape and dimension order should remain the same
        assert all_flipped.shape == cm.shape
        assert all_flipped.dims == cm.dims

    def test_flip_all_dimensions_propagates_to_associated(self, cm_multidim):
        """Test that flipping all dims propagates to associated coordinates."""
        cm = cm_multidim
        out = cm.flip(*cm.dims)
        # latitude shares 'distance' -> should reverse
        lat0 = cm.get_array("latitude")
        assert np.array_equal(out.get_array("latitude"), lat0[::-1])
        # quality shares both -> reverse on both axes
        q0 = cm.coord_map["quality"].values
        q1 = out.coord_map["quality"].values
        assert np.array_equal(q1, q0[::-1, ::-1])
