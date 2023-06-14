"""
Tests for coordinate managerment.
"""
from typing import Sequence

import numpy as np
import pytest
from pydantic import ValidationError
from rich.text import Text

from dascore import to_datetime64
from dascore.core.schema import PatchAttrs
from dascore.exceptions import CoordError
from dascore.utils.coordmanager import CoordManager, get_coord_manager
from dascore.utils.coords import BaseCoord, get_coord, get_coord_from_attrs
from dascore.utils.misc import register_func

COORD_MANAGERS = []

COORDS = {
    "time": to_datetime64(np.arange(10, 100, 10)),
    "distance": get_coord(values=np.arange(0, 1_000, 10)),
}
DIMS = ("time", "distance")


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def basic_coord_manager():
    """The simplest coord manager"""
    return get_coord_manager(COORDS, DIMS)


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_manager_multidim() -> CoordManager:
    """The simplest coord manager"""
    COORDS = {
        "time": to_datetime64(np.arange(10, 110, 10)),
        "distance": get_coord(values=np.arange(0, 1000, 10)),
        "quality": (("time", "distance"), np.ones((10, 100))),
        "latitude": ("distance", np.random.rand(100)),
    }
    DIMS = ("time", "distance")

    return get_coord_manager(COORDS, DIMS)


@pytest.fixture(scope="class", params=COORD_MANAGERS)
def coord_manager(request) -> CoordManager:
    """Meta fixture for aggregating coordinates."""
    return request.getfixturevalue(request.param)


class TestBasicCoordManager:
    """Ensure basic things work with coord managers."""

    def test_init(self, coord_manager):
        """Ensure values can be init'ed."""
        assert isinstance(coord_manager, CoordManager)

    def test_to_dict(self, coord_manager):
        """CoordManager should be convertible to dict."""
        c_dict = dict(coord_manager)
        assert c_dict == {x: coord_manager[x] for x in coord_manager.coord_map}

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
            cm[cm.dims[0]] = cm[cm.dims[0]]

    def test_cant_modify_dim_map(self, basic_coord_manager):
        """Ensure the dim map is immutable."""
        expected_str = "does not support item assignment"
        dim_map = basic_coord_manager.dim_map
        # test new key
        with pytest.raises(TypeError, match=expected_str):
            dim_map["bob"] = 10  # NOQA
        # test existing key
        with pytest.raises(TypeError, match=expected_str):
            dim_map[basic_coord_manager.dims[0]] = 10  # NOQA

    def test_cant_modify_coord_map(self, basic_coord_manager):
        """Ensure the dim map is immutable."""
        expected_str = "does not support item assignment"
        coord_map = basic_coord_manager.coord_map
        # test new key
        with pytest.raises(TypeError, match=expected_str):
            coord_map["bob"] = 10  # NOQA
        # test existing key
        with pytest.raises(TypeError, match=expected_str):
            coord_map[basic_coord_manager.dims[0]] = 10  # NOQA


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
        """Ensure specifying a bad coordinate raises"""
        coords = dict(COORDS)
        coords["bill"] = np.arange(10, 100, 10)
        with pytest.raises(CoordError, match="not named the same as dimension"):
            get_coord_manager(coords, DIMS)

    def test_nested_coord_too_long(self):
        """Nested coordinates that are gt 2 should fail"""
        coords = dict(COORDS)
        coords["time"] = ("time", to_datetime64(np.arange(10, 100, 10)), "space")
        with pytest.raises(CoordError, match="must be length two"):
            get_coord_manager(coords, DIMS)

    def test_invalid_dimensions(self):
        """Nested coordinates must specify valid dimensions"""
        coords = dict(COORDS)
        coords["time"] = ("bob", to_datetime64(np.arange(10, 100, 10)))
        with pytest.raises(CoordError, match="invalid dimension"):
            get_coord_manager(coords, DIMS)

    def test_missing_coordinates(self):
        """If all dims don't have coords an error should be raised."""
        coords = dict(COORDS)
        coords.pop("distance")
        with pytest.raises(CoordError, match="All dimensions must have coordinates"):
            get_coord_manager(coords, DIMS)

    def test_secondary_coord_bad_lengths(self):
        """Ensure when coordinates don't line up an error is raised."""
        coords = dict(COORDS)
        coords["bad"] = ("time", np.ones(len(coords["time"]) - 1))
        with pytest.raises(ValidationError, match="does not match the dimension"):
            get_coord_manager(coords, DIMS)

    def test_mappings_immutable(self, coord_manager):
        """Ensure the mappings are immutable."""
        with pytest.raises(Exception):
            coord_manager.coord_map["bob"]


class TestCoordManagerWithAttrs:
    """Tests for initing coord managing with attribute dict."""

    def test_missing_dim(self):
        """Coord manager should be able to pull missing info from attributes."""
        attrs = dict(distance_min=1, distance_max=100, d_distance=10)
        coord = {"time": COORDS["time"]}
        new = get_coord_manager(coord, DIMS, attrs=attrs)
        assert "distance" in new.coord_map


class TestDrop:
    """Tests for dropping coords with coord manager."""

    def test_drop(self, coord_manager_multidim):
        """Ensure coordinates can be dropped."""
        dim = "distance"
        coords, index = coord_manager_multidim.drop_dim(dim)
        # ensure the index corresponding to distance is 0
        ind = coord_manager_multidim.dims.index(dim)
        assert index[ind] == 0
        assert dim not in coords.dims
        for name, dims in coords.dim_map.items():
            assert dim not in dims


class TestSelect:
    """Tests for filtering coordinates."""

    def test_2d_coord_raises(self, coord_manager_multidim):
        """Select shouldn't work on 2D coordinates."""
        with pytest.raises(CoordError, match="Only 1 dimensional"):
            coord_manager_multidim.select(quality=(1, 2))

    def test_select_coord_dim(self, basic_coord_manager):
        """Simple test for filtering dimension coord."""
        new, inds = basic_coord_manager.select(distance=(100, 400))
        dist_ind = basic_coord_manager.dims.index("distance")
        assert new.shape[dist_ind] < basic_coord_manager.shape[dist_ind]
        assert len(inds) == len(basic_coord_manager.dims)
        assert inds[dist_ind] != slice(None, None)


class TestTranspose:
    """Test suite for transposing dimensions."""

    def test_missing_dim_raises(self, basic_coord_manager):
        """All dimensions must be specified, else raise."""
        with pytest.raises(CoordError, match="specify all dimensions"):
            basic_coord_manager.transpose(["time"])

    def test_simple_transpose(self, basic_coord_manager):
        """Ensure the coord manager can be transposed"""
        dims = basic_coord_manager.dims
        new_dims = dims[::-1]
        tran = basic_coord_manager.transpose(new_dims)
        assert tran.dims == new_dims
        assert tran.shape != basic_coord_manager.shape
        assert tran.shape == basic_coord_manager.shape[::-1]


class TestRenameDims:
    """Test case for renaming dimensions."""

    def test_rename_dims(self, basic_coord_manager):
        """Ensure dimensions can be renamed."""
        rename_map = {x: x[:2] for x in basic_coord_manager.dims}
        out = basic_coord_manager.rename_dims(**rename_map)
        assert set(out.dims) == set(rename_map.values())


class TestUpdateFromAttrs:
    """Tests to ensure updating attrs can update coordinates."""

    def test_update_min(self, basic_coord_manager):
        """Ensure min time in attrs updates appropriate coord."""
        for dim in basic_coord_manager.dims:
            coord = basic_coord_manager.coord_map[dim]
            attrs = {f"{dim}_max": coord.min}
            new = basic_coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            assert len(new_coord) == len(coord)
            assert new_coord.max == coord.min

    def test_update_max(self, basic_coord_manager):
        """Ensure max time in attrs updates appropriate coord."""
        for dim in basic_coord_manager.dims:
            coord = basic_coord_manager.coord_map[dim]
            attrs = {f"{dim}_min": coord.max}
            dist = coord.max - coord.min
            new = basic_coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max - new_coord.min
            assert dist == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min == coord.max

    def test_update_step(self, basic_coord_manager):
        """Ensure the step can be updated which changes endtime."""
        for dim in basic_coord_manager.dims:
            coord = basic_coord_manager.coord_map[dim]
            attrs = {f"d_{dim}": coord.step * 10}
            dist = coord.max - coord.min
            new = basic_coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max - new_coord.min
            assert (dist * 10) == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min == coord.min


class TestUpdateToAttrs:
    """Tests to ensure coordinate manager can update attributes."""

    def assert_dim_coords_consistent(self, coord_manager, attrs):
        """Ensure attrs are consistent with coords."""
        for dim in coord_manager.dims:
            coord = coord_manager.coord_map[dim]
            start = getattr(attrs, f"{dim}_min")
            stop = getattr(attrs, f"{dim}_max")
            step = getattr(attrs, f"d_{dim}")
            assert start == coord.min
            assert stop == coord.max
            assert step == coord.step
            vals_from_coord = coord.values
            vals_from_attrs = get_coord_from_attrs(attrs, dim).values
            eqs = np.all(np.equal(vals_from_coord, vals_from_attrs))
            assert eqs or np.allclose(vals_from_attrs, vals_from_coord)

    def test_empty(self, basic_coord_manager):
        """Ensure attributes can be generated coords."""
        attrs = basic_coord_manager.update_to_attrs()
        assert isinstance(attrs, PatchAttrs)
        self.assert_dim_coords_consistent(basic_coord_manager, attrs)

    def test_unrelated(self, basic_coord_manager, random_patch):
        """Passing an unrelated attrs should wipe relevant fields."""
        old_attrs = random_patch.attrs
        attrs = basic_coord_manager.update_to_attrs(old_attrs)
        assert isinstance(attrs, PatchAttrs)
        self.assert_dim_coords_consistent(basic_coord_manager, attrs)
        non_dim_keys = set(
            x
            for x in dict(attrs)
            if not (x.endswith("_max") or x.endswith("_min") or x.startswith("d_"))
        )
        for key in non_dim_keys:
            v1, v2 = getattr(old_attrs, key), getattr(attrs, key)
            if isinstance(v1, Sequence):
                assert set(v1) == set(v2)
            else:
                assert v1 == v2


class TestNonDimCoords:
    """Tests for adding non-dimensional coordinates."""

    def test_update_with_1d_coordinate(self, basic_coord_manager):
        """Ensure we can add coordinates."""
        lat = np.ones_like(basic_coord_manager["distance"])
        out = basic_coord_manager.update_coords(latitude=("distance", lat))
        assert out is not basic_coord_manager
        assert out.dims == basic_coord_manager.dims, "dims shouldn't change"
        assert np.all(out["latitude"] == lat)
        assert out.dim_map["latitude"] == out.dim_map["distance"]

    def test_init_with_1d_coordinate(self, basic_coord_manager):
        """Ensure initing with 1D non-dim coords works."""
        coords = dict(basic_coord_manager)
        lat = np.ones_like(basic_coord_manager["distance"])
        coords["latitude"] = ("distance", lat)
        out = get_coord_manager(coords, dims=basic_coord_manager.dims)
        assert out is not basic_coord_manager
        assert out.dims == basic_coord_manager.dims, "dims shouldn't change"
        assert np.all(out["latitude"] == lat)
        assert out.dim_map["latitude"] == out.dim_map["distance"]

    def test_update_2d_coord(self, basic_coord_manager):
        """Ensure updating can be done with 2D coordinate."""
        dist, time = basic_coord_manager["distance"], basic_coord_manager["time"]
        quality = np.ones((len(dist), len(time)))
        dims = ("distance", "time")
        new = basic_coord_manager.update_coords(qual=(dims, quality))
        assert new.dims == basic_coord_manager.dims
        assert new.dim_map["qual"] == dims
        assert "qual" in new
