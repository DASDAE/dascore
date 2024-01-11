"""Tests for coordinate manager."""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from rich.text import Text

import dascore as dc
from dascore import to_datetime64
from dascore.core.coordmanager import (
    CoordManager,
    get_coord_manager,
    merge_coord_managers,
)
from dascore.core.coords import (
    BaseCoord,
    CoordArray,
    CoordMonotonicArray,
    CoordRange,
    get_coord,
)
from dascore.exceptions import (
    CoordDataError,
    CoordError,
    CoordMergeError,
    CoordSortError,
    ParameterError,
)
from dascore.units import get_quantity
from dascore.utils.misc import (
    all_close,
    get_middle_value,
    register_func,
    suppress_warnings,
)

COORD_MANAGERS = []

COORDS = {
    "time": to_datetime64(np.arange(10, 100, 10)),
    "distance": get_coord(data=np.arange(0, 1_000, 10)),
}
DIMS = ("time", "distance")


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def basic_coord_manager():
    """The simplest coord manager."""
    return get_coord_manager(COORDS, DIMS)


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_manager_with_units(basic_coord_manager):
    """The simplest coord manager."""
    return basic_coord_manager.set_units(time="s", distance="m")


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def basic_degenerate_coord_manager(basic_coord_manager):
    """A degenerate coord manager on time axis."""
    time_coord = basic_coord_manager.coord_map["time"]
    degenerate = time_coord.empty()
    return basic_coord_manager.update(time=degenerate)


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_manager_multidim() -> CoordManager:
    """The simplest coord manager with several coords added."""
    coords = {
        "time": to_datetime64(np.arange(10, 110, 10)),
        "distance": get_coord(data=np.arange(0, 1000, 10)),
        "quality": (("time", "distance"), np.ones((10, 100))),
        "latitude": ("distance", np.random.rand(100)),
    }
    dims = ("time", "distance")

    return get_coord_manager(coords, dims)


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_manager_degenerate_time(coord_manager_multidim) -> CoordManager:
    """A coordinate manager with degenerate (length 1) time array."""
    new_time = to_datetime64(["2017-09-18T01:00:01"])
    out = coord_manager_multidim.update(time=new_time)
    return out


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_manager_wacky_dims() -> CoordManager:
    """A coordinate manager with non evenly sampled dims."""
    patch = dc.get_example_patch("wacky_dim_coords_patch")
    return patch.coords


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_dt_small_diff(memory_spool_small_dt_differences):
    """A list of coordinate managers with differences in dt merged."""
    spool = memory_spool_small_dt_differences
    coords = [x.coords for x in spool]
    out = merge_coord_managers(coords, dim="time")
    return out


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_non_associated_coord(basic_coord_manager):
    """A cm with coordinates that are not associated with a dimension."""
    new = basic_coord_manager.update(
        bob=(None, np.arange(10)),
        bill=((), np.arange(100)),
    )
    return new


@pytest.fixture(scope="class", params=COORD_MANAGERS)
def coord_manager(request) -> CoordManager:
    """Meta fixture for aggregating coordinates."""
    return request.getfixturevalue(request.param)


class TestGetCoordManager:
    """Test suite for `get_coord_manger` helper function."""

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

    def test_cant_assign_new_coord_inplace(self, basic_coord_manager):
        """The mappings inside the coord manager should be immutable."""
        cm = basic_coord_manager
        expected_str = "does not support item assignment"
        # cant add new coord
        with pytest.raises(TypeError, match=expected_str):
            cm["bob"] = 10
        # cant modify existing coord
        with pytest.raises(TypeError, match=expected_str):
            cm[cm.dims[0]] = cm.get_array(cm.dims[0])

    def test_cant_modify_dim_map(self, basic_coord_manager):
        """Ensure the dim map is immutable."""
        expected_str = "does not support item assignment"
        dim_map = basic_coord_manager.dim_map
        # test new key
        with pytest.raises(TypeError, match=expected_str):
            dim_map["bob"] = 10
        # test existing key
        with pytest.raises(TypeError, match=expected_str):
            dim_map[basic_coord_manager.dims[0]] = 10

    def test_cant_modify_coord_map(self, basic_coord_manager):
        """Ensure the dim map is immutable."""
        expected_str = "does not support item assignment"
        coord_map = basic_coord_manager.coord_map
        # test new key
        with pytest.raises(TypeError, match=expected_str):
            coord_map["bob"] = 10
        # test existing key
        with pytest.raises(TypeError, match=expected_str):
            coord_map[basic_coord_manager.dims[0]] = 10

    def test_init_with_coord_manager(self, basic_coord_manager):
        """Ensure initing coord manager works with a single coord manager."""
        out = get_coord_manager(basic_coord_manager)
        assert out == basic_coord_manager

    def test_init_with_cm_and_dims(self, basic_coord_manager):
        """Ensure cm can be init'ed with coord manager and dims."""
        out = get_coord_manager(basic_coord_manager, dims=basic_coord_manager.dims)
        assert out == basic_coord_manager

    def test_init_list_range(self):
        """Ensure the coord manager can be init'ed with a list and no dims."""
        input_dict = {
            "time": [10, 11, 12],
            "distance": ("distance", range(100)),
            "space": range(100),
        }
        out = get_coord_manager(input_dict, dims=list(input_dict))
        assert set(out.dims) == set(input_dict)

    def test_bad_datashape_raises(self, basic_coord_manager):
        """Ensure a bad datashape raises."""
        match = "match the coordinate manager shape"
        cm = basic_coord_manager
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

    def test_min(self, basic_coord_manager):
        """Ensure we can git min value."""
        expected = np.min(basic_coord_manager.time.data).astype(np.int64)
        got = basic_coord_manager.min("time").astype(np.int64)
        assert np.isclose(got, expected)

    def test_max(self, basic_coord_manager):
        """Ensure we can git max value."""
        expected = np.max(basic_coord_manager.time.data).astype(np.int64)
        got = basic_coord_manager.max("time").astype(np.int64)
        assert np.isclose(got, expected)

    def test_step(self, basic_coord_manager):
        """Ensure we can git min value."""
        expected = basic_coord_manager.time.step
        assert basic_coord_manager.step("time") == expected

    def test_getattr(self, basic_coord_manager):
        """Ensure getattr returns coordinate."""
        for dim in basic_coord_manager.dims:
            coord = getattr(basic_coord_manager, dim)
            assert coord == basic_coord_manager.coord_map[dim]

    def test_get_item_warning(self, basic_coord_manager):
        """Ensure get item emits a warning."""
        msg = "returns a numpy array"
        with pytest.warns(UserWarning, match=msg):
            _ = basic_coord_manager["time"]

    def test_has_attr(self, basic_coord_manager):
        """Ensure hasattr returns correct result."""
        dims = basic_coord_manager.dims
        for dim in dims:
            assert hasattr(basic_coord_manager, dim)
        assert not hasattr(basic_coord_manager, "_NOT_A_DIM")

    def test_iterate(self, basic_coord_manager):
        """Ensure coordinates yield name an coordinate when iterated."""
        for dim, coord in iter(basic_coord_manager):
            expected = basic_coord_manager.get_coord(dim)
            assert all_close(coord, expected)


class TestCoordManagerInputs:
    """Tests for coordinates management."""

    def test_simple_inputs(self):
        """Simplest input case."""
        out = get_coord_manager(COORDS, DIMS)
        assert isinstance(out, CoordManager)

    def test_additional_coords(self):
        """Ensure a additional (non-dimensional) coords work."""
        coords = dict(COORDS)
        lats = np.random.rand(len(COORDS["distance"]))
        coords["latitude"] = ("distance", lats)
        out = get_coord_manager(coords, DIMS)
        assert isinstance(out.coord_map["latitude"], BaseCoord)

    def test_str(self, basic_coord_manager):
        """Ensure a custom (readable) str is returned."""
        coord_str = str(basic_coord_manager)
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


class TestCoordManagerWithAttrs:
    """Tests for initing coord managing with attribute dict."""

    def test_missing_dim(self):
        """Coord manager should be able to pull missing info from attributes."""
        attrs = dict(distance_min=1, distance_max=100, distance_step=10)
        new = get_coord_manager(None, ("distance",), attrs=attrs)
        assert "distance" in new.coord_map


class TestDrop:
    """Tests for dropping coords with coord manager."""

    def test_drop(self, coord_manager_multidim):
        """Ensure coordinates can be dropped."""
        dim = "distance"
        coords, _ = coord_manager_multidim.drop_coords(dim)
        assert dim not in coords.dims
        for _name, dims in coords.dim_map.items():
            assert dim not in dims

    def test_drop_doesnt_have_coord(self, coord_manager_multidim):
        """Trying to drop a dim that doesnt exist should just return."""
        out, _ = coord_manager_multidim.drop_coords("bob")
        assert out == coord_manager_multidim

    def test_trims_array(self, coord_manager_multidim):
        """Trying to drop a dim that doesnt exist should just return."""
        array = np.ones(coord_manager_multidim.shape)
        axis = coord_manager_multidim.dims.index("time")
        cm, new_array = coord_manager_multidim.drop_coords("time", array=array)
        assert new_array.shape[axis] == 0

    def test_drop_non_dim_coord(self, coord_manager_multidim):
        """Dropping a non-dim coord should not affect shape/dimensions."""
        cm = coord_manager_multidim
        array = np.ones(cm.shape)
        coords_to_drop = set(cm.coord_map) - set(cm.dims)
        for coord in coords_to_drop:
            cm_new, array_new = cm.drop_coords(coord, array=array)
            # array should not have changed.
            assert array.shape == array_new.shape
            assert np.all(np.equal(array, array_new))
            assert coord not in set(cm_new.coord_map)


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

    def test_2d_coord_raises(self, coord_manager_multidim):
        """Select shouldn't work on 2D coordinates."""
        with pytest.raises(CoordError, match="Only 1 dimensional"):
            coord_manager_multidim.select(quality=(1, 2))

    def test_select_coord_dim(self, basic_coord_manager):
        """Simple test for filtering dimension coord."""
        new, _ = basic_coord_manager.select(distance=(100, 400))
        dist_ind = basic_coord_manager.dims.index("distance")
        assert new.shape[dist_ind] < basic_coord_manager.shape[dist_ind]

    def test_filter_array(self, basic_coord_manager):
        """Ensure an array can be filtered."""
        data = np.ones(basic_coord_manager.shape)
        new, trim = basic_coord_manager.select(distance=(100, 400), array=data)
        assert trim.shape == trim.shape

    def test_select_emptying_dim(self, basic_coord_manager):
        """Selecting a range outside of dim should empty the manager."""
        data = np.ones(basic_coord_manager.shape)
        cm, trim = basic_coord_manager.select(distance=(-100, -10), array=data)
        assert trim.shape[cm.dims.index("distance")] == 0
        assert "distance" in cm.dims
        assert len(cm.get_array("distance")) == 0
        assert len(cm.coord_map["distance"]) == 0

    def test_select_trims_associated_coord_1(self, coord_manager_multidim):
        """Ensure trimming a dimension also trims associated coordinate."""
        cm = coord_manager_multidim
        coord_to_trim = "distance"
        distance = cm.get_array(coord_to_trim)
        out, _ = cm.select(distance=(distance[1], distance[-2]))
        # ensure all attrs with "distance" have been trimmed.
        expected_len = len(out.coord_map[coord_to_trim])
        for name in cm.dim_to_coord_map[coord_to_trim]:
            coord = out.coord_map[name]
            axis = cm.dim_map[name].index(coord_to_trim)
            assert coord.shape[axis] == expected_len

    def test_select_trims_associated_coords_2(self, coord_manager_multidim):
        """Same as test #1, but now we check for trimming non-dimension coord."""
        cm = coord_manager_multidim
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

    def test_select_handles_non_dim_kwargs(self, basic_coord_manager):
        """The coord manager should handle (supress) non dim keyword args."""
        ar = np.ones(basic_coord_manager.shape)
        out, new = basic_coord_manager.select(bob=(10, 20), array=ar)
        assert new.shape == ar.shape
        assert out == basic_coord_manager

    @pytest.mark.parametrize("slice_range", slice_inds)
    def test_compare_to_select(self, basic_coord_manager, slice_range):
        """Ensure select with and without samples behaves the same with equiv. data."""
        cm = basic_coord_manager
        for name, coord in cm.coord_map.items():
            ind_tuple = slice_range
            ind_1, ind_2 = slice_range
            val1 = coord[ind_1] if isinstance(ind_1, int) else ind_1
            val2 = coord[ind_2 - 1] if isinstance(ind_2, int) else ind_2
            value_tuple = (val1, val2)
            # first, just check coords are equal
            out1 = coord.select(value_tuple)[0]
            out2 = coord[slice(*ind_tuple)]
            if not out1 == out2:
                assert all_close(out1.values, out2.values)
            # then check that the whole coord_manager are equal
            cm1 = cm.select(**{name: value_tuple})
            cm2 = cm.select(**{name: ind_tuple}, samples=True)
            assert cm1 == cm2

    @pytest.mark.parametrize("index", inds)
    def test_single_values(self, basic_coord_manager, index):
        """
        Single values should be treated like slice(val, val+1)
        as not to collapse the dimensions when samples=True.
        """
        cm = basic_coord_manager
        data = np.empty(cm.shape)
        for dim in basic_coord_manager.dims:
            kwargs = {dim: index}
            out1, new_data = cm.select(array=data, samples=True, **kwargs)
            dim_ind = cm.dims.index(dim)
            # now the array should have a len(1) in the selected dimension.
            assert out1.shape[dim_ind] == new_data.shape[dim_ind] == 1
            new_value = out1.coord_map[dim].values[0]
            expected_value = cm.get_array(dim)[index]
            all_close(new_value, expected_value)

    def test_trim_related_coords(self, coord_manager_multidim):
        """Ensure trim also trims related dimensions."""
        cm = coord_manager_multidim
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


class TestEquals:
    """Tests for coord manager equality."""

    def test_basic_equals(self, coord_manager):
        """All coord managers should equal themselves."""
        assert coord_manager == coord_manager

    def test_unequal_float_coords(self, coord_manager_multidim):
        """Ensure if coordinates are not equal false is returned."""
        coord = coord_manager_multidim.coord_map["latitude"]
        new = get_coord(data=coord.values + 10)
        args = dict(latitude=("distance", new))
        new_coord = coord_manager_multidim.update(**args)
        assert new_coord != coord_manager_multidim

    def test_unequal_wrong_type(self, basic_coord_manager):
        """Non-coord managers should not be considered equal."""
        cm = basic_coord_manager
        assert cm != 10
        assert cm != "bob"
        assert cm != {1: 2, 2: 2}
        assert cm != cm.coord_map["distance"]


class TestTranspose:
    """Test suite for transposing dimensions."""

    @pytest.fixture(scope="class")
    def many_dims_cm(self):
        """Create a coordmanager with many dimensions."""
        dims = ("one", "two", "three", "four")
        coords = {x: np.arange(10) for x in dims}
        many_dims = get_coord_manager(coords, dims=dims)
        return many_dims

    def test_simple_transpose(self, basic_coord_manager):
        """Ensure the coord manager can be transposed."""
        dims = basic_coord_manager.dims
        new_dims = dims[::-1]
        tran = basic_coord_manager.transpose(*new_dims)
        assert tran.dims == new_dims
        assert tran.shape != basic_coord_manager.shape
        assert tran.shape == basic_coord_manager.shape[::-1]

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

    def test_rename_dims(self, basic_coord_manager):
        """Ensure dimensions can be renamed."""
        rename_map = {x: x[:2] for x in basic_coord_manager.dims}
        out = basic_coord_manager.rename_coord(**rename_map)
        assert set(out.dims) == set(rename_map.values())

    def test_rename_extra_coords_kept(self, coord_manager_multidim):
        """Ensure the extra dims are not dropped when a coord is renamed."""
        cm = coord_manager_multidim
        # we should be able to rename distance to dist and
        # distance just gets swapped for dist
        out = cm.rename_coord(distance="dist")
        assert "dist" in out.dim_map
        assert "dist" in out.coord_map
        assert "dist" in out.dim_map["latitude"]

    def test_rename_same_dims_extra_coords(self, coord_manager_multidim):
        """Ensure renaming dims the same doesn't mess with multidims."""
        cm = coord_manager_multidim
        dim_map = {x: x for x in cm.dims}
        out = cm.rename_coord(**dim_map)
        assert set(out.dim_map) == set(cm.dim_map)


class TestUpdateFromAttrs:
    """Tests to ensure updating attrs can update coordinates."""

    def test_update_min(self, basic_coord_manager):
        """Ensure min time in attrs updates appropriate coord."""
        for dim in basic_coord_manager.dims:
            coord = basic_coord_manager.coord_map[dim]
            attrs = {f"{dim}_max": coord.min()}
            new, _ = basic_coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            assert len(new_coord) == len(coord)
            assert new_coord.max() == coord.min()

    def test_update_max(self, basic_coord_manager):
        """Ensure max time in attrs updates appropriate coord."""
        for dim in basic_coord_manager.dims:
            coord = basic_coord_manager.coord_map[dim]
            attrs = {f"{dim}_min": coord.max()}
            dist = coord.max() - coord.min()
            new, _ = basic_coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max() - new_coord.min()
            assert dist == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min() == coord.max()

    def test_update_step(self, basic_coord_manager):
        """Ensure the step can be updated which changes endtime."""
        for dim in basic_coord_manager.dims:
            coord = basic_coord_manager.coord_map[dim]
            attrs = {f"{dim}_step": coord.step * 10}
            dist = coord.max() - coord.min()
            new, _ = basic_coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max() - new_coord.min()
            assert (dist * 10) == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min() == coord.min()

    def test_attrs_as_dict(self, basic_coord_manager):
        """Ensure the attrs returned has coords attached."""
        coord = basic_coord_manager.coord_map["time"]
        attrs = {"time_max": coord.min()}
        cm, attrs = basic_coord_manager.update_from_attrs(attrs)
        assert attrs.coords == cm.to_summary_dict()
        assert attrs.dim_tuple == cm.dims

    def test_attrs_as_patch_attr(self, basic_coord_manager):
        """Ensure this also works when attrs is a patch attr."""
        attrs = dc.PatchAttrs(time_min=to_datetime64("2022-01-01"))
        cm, new_attrs = basic_coord_manager.update_from_attrs(attrs)
        assert new_attrs.coords == cm.to_summary_dict()
        assert new_attrs.dim_tuple == cm.dims

    def test_consistent_attrs_leaves_coords_unchanged(self, random_patch):
        """Attrs which are already consistent should leave coord unchanged."""
        attrs, coords = random_patch.attrs, random_patch.coords
        new_coords, new_attrs = coords.update_from_attrs(attrs)
        assert new_coords == coords


class TestUpdate:
    """Tests for updating coordinates."""

    def test_simple(self, basic_coord_manager):
        """Ensure coordinates can be updated (replaced)."""
        new_time = basic_coord_manager.get_array("time") + np.timedelta64(1, "s")
        out = basic_coord_manager.update(time=new_time)
        assert np.all(np.equal(out.get_array("time"), new_time))

    def test_extra_coords_kept(self, coord_manager_multidim):
        """Ensure extra coordinates are kept."""
        cm = coord_manager_multidim
        new_time = cm.get_array("time") + np.timedelta64(1, "s")
        out = cm.update(time=new_time)
        assert set(out.coord_map) == set(coord_manager_multidim.coord_map)
        assert set(out.coord_map) == set(coord_manager_multidim.coord_map)

    def test_size_change(self, basic_coord_manager):
        """Ensure sizes of dimensions can be changed."""
        new_time = basic_coord_manager.get_array("time")[:10]
        out = basic_coord_manager.update(time=new_time)
        assert np.all(np.equal(out.get_array("time"), new_time))

    def test_size_change_drops_old_coords(self, coord_manager_multidim):
        """When the size changes on multidim, dims should be dropped."""
        cm = coord_manager_multidim
        new_dist = cm.get_array("distance")[:10]
        dropped_coords = set(cm.coord_map) - set(cm.dims)
        out = cm.update(distance=new_dist)
        assert dropped_coords.isdisjoint(set(out.coord_map))

    def test_update_degenerate(self, coord_manager):
        """Tests for updating coord with degenerate coordinates."""
        cm = coord_manager
        out = cm.update(time=cm.coord_map["time"].empty())
        assert out.shape[out.dims.index("time")] == 0
        assert len(out.coord_map["time"]) == 0

    def test_update_degenerate_dim_multicoord(self, coord_manager_multidim):
        """
        If one dim is degenerate all associated coords should be
        dropped.
        """
        cm = coord_manager_multidim
        degen_time = cm.coord_map["time"].empty()
        new = cm.update(time=degen_time)
        assert new != cm
        # any coords with time as a dim (but not time itself) should be gone.
        has_time = [i for i, v in cm.dim_map.items() if ("time" in v and i != "time")]
        assert set(has_time).isdisjoint(set(new.coord_map))

    def test_update_none_drops(self, basic_coord_manager):
        """Ensure when passing coord=None the coord is dropped."""
        cm1 = basic_coord_manager.update(time=None)
        cm2, _ = basic_coord_manager.drop_coords("time")
        assert cm1 == cm2

    def test_update_only_start(self, basic_coord_manager):
        """Ensure start_coord can be used to update."""
        time1 = basic_coord_manager.coord_map["time"]
        new_start = time1.max()
        cm = basic_coord_manager.update(time_min=new_start)
        time2 = cm.coord_map["time"]
        assert time2.min() == new_start

    def test_dissociate(self, basic_coord_manager):
        """Ensure coordinates can be dissociated with update_coords."""
        time = basic_coord_manager.coord_map["time"]
        new_time = time.values + dc.to_timedelta64(1)
        new = basic_coord_manager.update(new_time=("time", new_time))
        assert "new_time" in new.dim_map and "new_time" in new.coord_map
        dissociated = new.update(new_time=(None, new_time))
        assert dissociated.dim_map["new_time"] == ()

    def test_unchanged_coords(self, coord_manager_with_units):
        """Ensure coordinates not updated are left unchanged."""
        cm = coord_manager_with_units
        new_time = cm.coord_map["time"].update(min=0)
        new = cm.update(time=new_time)
        assert new.coord_map["distance"] == cm.coord_map["distance"]


class TestSqueeze:
    """Tests for squeezing degenerate dimensions."""

    def test_bad_dim_raises(self, coord_manager_degenerate_time):
        """Ensure a bad dimension will raise CoordError."""
        with pytest.raises(CoordError, match="they don't exist"):
            coord_manager_degenerate_time.squeeze("is_money")

    def test_non_zero_length_dim_raises(self, coord_manager_degenerate_time):
        """Ensure a dim with length > 0 can't be squeezed."""
        with pytest.raises(CoordError, match="non-zero length"):
            coord_manager_degenerate_time.squeeze("distance")

    def test_squeeze_single_degenerate(self, coord_manager_degenerate_time):
        """Ensure a single degenerate dimension can be squeezed out."""
        cm = coord_manager_degenerate_time
        out = cm.squeeze("time")
        assert "time" not in out.dims

    def test_squeeze_no_dim(self, coord_manager_degenerate_time):
        """Ensure all degenerate dims are squeezed when no dim specified."""
        cm = coord_manager_degenerate_time
        out = cm.squeeze()
        assert "time" not in out.dims


class TestNonDimCoords:
    """Tests for adding non-dimensional coordinates."""

    def test_update_with_1d_coordinate(self, basic_coord_manager):
        """Ensure we can add coordinates."""
        lat = np.ones_like(basic_coord_manager.get_array("distance"))
        out = basic_coord_manager.update(latitude=("distance", lat))
        assert out is not basic_coord_manager
        assert out.dims == basic_coord_manager.dims, "dims shouldn't change"
        assert np.all(out.get_array("latitude") == lat)
        assert out.dim_map["latitude"] == out.dim_map["distance"]
        # the dim map shouldn't shrink; only things get added.
        assert set(out.dim_map).issuperset(set(basic_coord_manager.dim_map))

    def test_init_with_1d_coordinate(self, basic_coord_manager):
        """Ensure initing with 1D non-dim coords works."""
        with suppress_warnings():
            coords = dict(basic_coord_manager)
        lat = np.ones_like(basic_coord_manager.get_array("distance"))
        coords["latitude"] = ("distance", lat)
        out = get_coord_manager(coords, dims=basic_coord_manager.dims)
        assert out is not basic_coord_manager
        assert out.dims == basic_coord_manager.dims, "dims shouldn't change"
        assert np.all(out.get_array("latitude") == lat)
        assert out.dim_map["latitude"] == out.dim_map["distance"]
        # the dim map shouldn't shrink; only things get added.
        assert set(out.dim_map).issuperset(set(basic_coord_manager.dim_map))

    def test_update_2d_coord(self, basic_coord_manager):
        """Ensure updating can be done with 2D coordinate."""
        dist = basic_coord_manager.get_array("distance")
        time = basic_coord_manager.get_array("time")
        quality = np.ones((len(dist), len(time)))
        dims = ("distance", "time")
        new = basic_coord_manager.update(qual=(dims, quality))
        assert new.dims == basic_coord_manager.dims
        assert new.dim_map["qual"] == dims
        assert "qual" in new


class TestDecimate:
    """Tests for evenly subsampling along a dimension."""

    def test_decimate_along_time(self, basic_coord_manager):
        """Simply ensure decimation reduces shape."""
        cm = basic_coord_manager
        ind = cm.dims.index("distance")
        out, _ = cm.decimate(distance=2)
        assert len(out.coord_map["distance"]) < len(cm.coord_map["distance"])
        assert (out.shape[ind] * 2) == cm.shape[ind]


class TestMergeCoordManagers:
    """Tests for merging coord managers together."""

    def _get_offset_coord_manager(self, cm, from_max=True, **kwargs):
        """Get a new coord manager offset by some amount along a dim."""
        name, value = next(iter(kwargs.items()))
        coord = cm.coord_map[name]
        start = coord.max() if from_max else coord.min()
        attr_name = f"{name}_min"
        new, _ = cm.update_from_attrs({attr_name: start + value})
        return new

    def test_merge_simple(self, basic_coord_manager):
        """Ensure we can merge simple, contiguous, coordinates together."""
        cm1 = basic_coord_manager
        time = cm1.coord_map["time"]
        cm2 = self._get_offset_coord_manager(cm1, time=time.step)
        out = merge_coord_managers([cm1, cm2], dim="time")
        new_time = out.coord_map["time"]
        assert isinstance(new_time, CoordRange)
        assert new_time.min() == time.min()
        assert new_time.max() == cm2.coord_map["time"].max()

    def test_merge_offset_close_no_snap(self, basic_coord_manager):
        """When the coordinate don't line up, it should produce monotonic Coord."""
        cm1 = basic_coord_manager
        dt = cm1.coord_map["time"].step
        # try a little more than dt
        cm2 = self._get_offset_coord_manager(cm1, time=dt * 1.1)
        out = merge_coord_managers([cm1, cm2], dim="time")
        assert isinstance(out.coord_map["time"], CoordMonotonicArray)
        # try a little less
        cm2 = self._get_offset_coord_manager(cm1, time=dt * 0.9)
        out = merge_coord_managers([cm1, cm2], dim="time")
        assert isinstance(out.coord_map["time"], CoordMonotonicArray)

    def test_merge_offset_overlap(self, basic_coord_manager):
        """Ensure coordinates that have overlap produce Coord Array."""
        cm1 = basic_coord_manager
        dt = cm1.coord_map["time"].step
        cm2 = self._get_offset_coord_manager(cm1, time=-dt * 1.1)
        out = merge_coord_managers([cm1, cm2], dim="time")
        assert isinstance(out.coord_map["time"], CoordArray)

    def test_merge_snap_but_not_needed(self, basic_coord_manager):
        """Specifying a snap tolerance even if coords line up should work."""
        cm1 = basic_coord_manager
        time = cm1.coord_map["time"]
        cm2 = self._get_offset_coord_manager(cm1, time=time.step)
        out = merge_coord_managers([cm1, cm2], dim="time", snap_tolerance=1.3)
        new_time = out.coord_map["time"]
        assert isinstance(new_time, CoordRange)
        assert new_time.min() == time.min()
        assert new_time.max() == cm2.coord_map["time"].max()

    @pytest.mark.parametrize("factor", (1.1, 0.9, 1.3, 1.01, 0))
    def test_merge_snap_when_needed(self, basic_coord_manager, factor):
        """Snap should be applied because when other cm is close expected."""
        cm1 = basic_coord_manager
        time = cm1.coord_map["time"]
        nt = time.step * factor
        cm2 = self._get_offset_coord_manager(cm1, time=nt)
        out = merge_coord_managers([cm1, cm2], dim="time", snap_tolerance=1.3)
        new_time = out.coord_map["time"]
        assert isinstance(new_time, CoordRange)
        assert new_time.min() == time.min()
        new_dim_len = out.shape[cm1.dims.index("time")]
        expected_end = time.min() + (new_dim_len - 1) * time.step
        assert new_time.max() == expected_end

    @pytest.mark.parametrize("factor", (10, -10, 6, -6))
    def test_merge_raise_snap_too_big(self, basic_coord_manager, factor):
        """When snap is too big, an error should be raised."""
        cm1 = basic_coord_manager
        time = cm1.coord_map["time"]
        nt = time.step * factor
        cm2 = self._get_offset_coord_manager(cm1, time=nt)
        with pytest.raises(CoordMergeError, match="Snap tolerance"):
            merge_coord_managers([cm1, cm2], dim="time", snap_tolerance=1.3)

    def test_different_dims_raises(self, basic_coord_manager):
        """When dimensions differ merge should raise."""
        cm1 = basic_coord_manager
        cm2 = cm1.rename_coord(distance="dist")
        with pytest.raises(CoordMergeError, match="same dimensions"):
            merge_coord_managers([cm1, cm2], "time")
        with pytest.raises(CoordMergeError, match="same dimensions"):
            merge_coord_managers([cm1, cm2], "distance")

    def test_different_units_raises(self, basic_coord_manager):
        """When dimensions differ merge should raise."""
        cm1 = basic_coord_manager
        cm2 = cm1.set_units(distance="furlong")
        with pytest.raises(CoordMergeError, match="share the same units"):
            merge_coord_managers([cm1, cm2], "distance")

    def test_unequal_non_merge_coords(self, basic_coord_manager):
        """When coords that won't be merged arent equal merge should fail."""
        cm1 = basic_coord_manager
        dist = cm1.coord_map["distance"]
        new_dist = dist.update_limits(min=dist.max())
        cm2 = cm1.update(distance=new_dist)
        with pytest.raises(CoordMergeError, match="Non merging coordinates"):
            merge_coord_managers([cm1, cm2], "time")

    def test_unshared_coord_dropped(self, basic_coord_manager):
        """
        When one coord manager has coords and the other doesn't they should
        be dropped.
        """
        cm1 = basic_coord_manager
        cm2 = cm1.update(time2=("time", cm1.get_array("time")))
        out_no_range = merge_coord_managers([cm1, cm2], "time")
        assert "time2" not in out_no_range.coord_map
        out_with_range = merge_coord_managers([cm1, cm2], "time")
        assert "time2" not in out_with_range.coord_map

    def test_slightly_different_dt(self, coord_dt_small_diff):
        """
        Ensure coord managers with slightly different dt can still merge
        but produce uneven sampled dimension.
        """
        cm = coord_dt_small_diff
        coord = cm.coord_map["time"]
        assert coord.sorted


class TestSort:
    """Tests for sorting coord managers."""

    def test_sort_unsorted_dim_coord(self, coord_manager_wacky_dims):
        """Simple test for single dim sort."""
        cm = coord_manager_wacky_dims
        sorted_cm, _ = cm.sort("distance")
        assert sorted_cm.coord_map["distance"].sorted

    def test_sort_sorted_dim_coord(self, coord_manager_wacky_dims):
        """Ensure sorting a sorted dim coord does nothing."""
        cm = coord_manager_wacky_dims
        sorted_cm, _ = cm.sort("time")
        assert sorted_cm.coord_map["time"].sorted

    def test_r_sort_unsorted_dim_coord(self, coord_manager_wacky_dims):
        """Simple test for single dim sort."""
        cm = coord_manager_wacky_dims
        sorted_cm, _ = cm.sort("distance", reverse=True)
        assert sorted_cm.coord_map["distance"].reverse_sorted

    def test_r_sort_sorted_dim_coord(self, coord_manager_wacky_dims):
        """Ensure sorting a sorted dim coord does nothing."""
        cm = coord_manager_wacky_dims
        sorted_cm, _ = cm.sort("time", reverse=True)
        assert sorted_cm.coord_map["time"].reverse_sorted

    def test_simultaneous_sort(self, coord_manager_wacky_dims):
        """Ensure all dimensions can be sorted at once."""
        sorted_cm, _ = coord_manager_wacky_dims.sort()
        for _, coord in sorted_cm.coord_map.items():
            assert coord.sorted

    def test_simultaneous_r_sort(self, coord_manager_wacky_dims):
        """Ensure all dimensions can be reverse sorted at once."""
        sorted_cm, _ = coord_manager_wacky_dims.sort(reverse=True)
        for _, coord in sorted_cm.coord_map.items():
            assert coord.reverse_sorted

    def test_sort_sorted_cm(self, basic_coord_manager):
        """Sorting a coord manager that is already sorted should do nothing."""
        cm = basic_coord_manager
        cm_sorted, _ = cm.sort()
        assert cm == cm_sorted

    def test_sort_2d_coord_raises(self, coord_manager_multidim):
        """Sorting on 2D coord is ambiguous, it should raise."""
        cm = coord_manager_multidim
        with pytest.raises(CoordSortError, match="more than one dimension"):
            cm.sort("quality")

    def test_sort_two_coords_same_dim_raises(self, coord_manager_multidim):
        """Trying to sort on two coords which share a dim should raise."""
        # need to make distance un sorted.
        cm, _ = coord_manager_multidim.sort("distance", reverse=True)
        with pytest.raises(CoordSortError, match="they share a dimension"):
            cm.sort("latitude", "distance")
        # this should also raise since sorting latitude could unsort distance
        with pytest.raises(CoordSortError, match="they share a dimension"):
            coord_manager_multidim.sort("distance", "latitude")

    def test_related_coords(self, coord_manager_multidim):
        """Sorting on one coordinate should also sort others that share a dim."""
        cm = coord_manager_multidim
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

    def test_snap_dims(self, coord_manager_wacky_dims):
        """Happy path for snapping dimensions."""
        cm = coord_manager_wacky_dims
        data = np.ones(cm.shape)
        out, out_data = cm.snap(array=data)
        for coord_name in out.coord_map:
            coord1 = cm.coord_map[coord_name]
            coord2 = out.coord_map[coord_name]
            assert coord1.dtype == coord2.dtype
            assert len(coord1) == len(coord2)
        assert out_data.shape == data.shape

    def test_snap_expected_dt(
        self, coord_dt_small_diff, memory_spool_small_dt_differences
    ):
        """Ensure snapping creates expected dt from merged df."""
        spool = memory_spool_small_dt_differences
        expected_dt = get_middle_value([x.attrs.time_step for x in spool])
        snapped = coord_dt_small_diff.snap()[0]
        assert snapped.coord_map["time"].step == expected_dt


class TestConvertUnits:
    """Tests for converting coordinate units."""

    def test_convert_changes_labels(self, basic_coord_manager):
        """Basic tests for converting units."""
        cm = basic_coord_manager.convert_units(time="s", distance="furlong")
        time, dist = cm.coord_map["time"], cm.coord_map["distance"]
        assert time.units == get_quantity("s")
        assert dist.units == get_quantity("furlong")

    def test_convert_changes_values(self, coord_manager_with_units):
        """Ensure values are scaled accordingly."""
        conv = get_quantity("m").to("ft").magnitude
        cm = coord_manager_with_units.convert_units(distance="ft")
        dist1 = coord_manager_with_units.coord_map["distance"]
        dist2 = cm.coord_map["distance"]
        assert np.isclose(dist1.step, dist2.step / conv)
        assert np.isclose(dist1.start, dist2.start / conv)
        assert np.isclose(dist1.stop, dist2.stop / conv)

    def test_convert_time(self, coord_manager_with_units):
        """When time is already set and a datetime, units should just change."""
        # TODO, reconsider this; i am not sure its right.
        new = coord_manager_with_units.convert_units(time="ms")
        time1 = coord_manager_with_units.coord_map["time"]
        time2 = new.coord_map["time"]
        assert time1.dtype == time2.dtype
        assert np.all(np.equal(time1.values, time2.values))


class TestDisassociate:
    """Ensure coordinates can be disassociated from coordinate manager."""

    def test_basic_disassociate(self, coord_manager_multidim):
        """Ensure coordinates can be dissociated."""
        cm = coord_manager_multidim.disassociate_coord("time")
        # now time should still exist but have no dim, quality should drop
        assert "time" in cm.coord_map
        assert "quality" not in cm.coord_map
        time_dim = cm.dim_map["time"]
        assert time_dim == ()
        assert "time" not in cm.dims
        # but if both time and quality are dissociated nothing should be dropped.
        cm = coord_manager_multidim.disassociate_coord("time", "quality")
        assert {"time", "quality"}.issubset(set(cm.coord_map))
        assert "time" not in cm.dims

    def test_disassociate_non_dim_coord(self, coord_manager_multidim):
        """Ensure non-dim coords can be dissociated with no change to dims."""
        cm = coord_manager_multidim.disassociate_coord("quality")
        assert cm.dim_map["quality"] == ()
        assert cm.dims == coord_manager_multidim.dims


class TestSetDims:
    """Tests for setting dims to non-dimensional coordinates."""

    def test_swap_coords(self, coord_manager_multidim):
        """Simple coord swap."""
        cm = coord_manager_multidim
        out1 = cm.set_dims(distance="latitude")
        assert "latitude" in out1.dims
        assert "latitude" not in cm.dims, "original dim should be unchanged"
        # distance should now be associated with latitude.
        assert out1.dim_map["distance"][0] == "latitude"

    def test_bad_dimension(self, coord_manager_multidim):
        """Ensure if a bad dimension is provided it raises."""
        cm = coord_manager_multidim
        match = "is not a dimension or"
        with pytest.raises(CoordError, match=match):
            cm.set_dims(bob="latitude")
        with pytest.raises(CoordError, match=match):
            cm.set_dims(distance="bob")

    def test_wrong_shape(self, coord_manager_multidim):
        """Ensure if a coord with wrong shape is chosen an error is raised."""
        cm = coord_manager_multidim
        match = "does not match the shape of"
        with pytest.raises(CoordError, match=match):
            cm.set_dims(distance="quality")
