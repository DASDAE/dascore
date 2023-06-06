"""
Tests for coordinate managerment.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from dascore import to_datetime64
from dascore.exceptions import CoordError
from dascore.utils.coordmanager import CoordManager, get_coord_manager
from dascore.utils.coords import BaseCoord, get_coord
from dascore.utils.misc import register_func

COORD_MANAGERS = []

COORDS = {
    "time": to_datetime64(np.arange(10, 100, 10)),
    "distance": get_coord(values=np.arange(0, 1_000, 10)),
}
DIMS = ("time", "distance")


@pytest.fixture(scope="class")
@register_func(COORD_MANAGERS)
def coord_manager():
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
def coord(request):
    """Meta fixture for aggregating coordinates."""
    return request.getfixturevalue(request.param)


class TestBasicCoordManager:
    """Ensure basic things work with coord managers."""

    def test_init(self, coord):
        """Ensure values can be init'ed."""
        assert isinstance(coord, CoordManager)


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
        assert isinstance(out["latitude"], BaseCoord)

    def test_str(self, coord_manager):
        """Ensure a custom (readable) str is returned."""
        coord_str = str(coord_manager)
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

    def test_mappings_immutable(self, coord):
        """Ensure the mappings are immutable."""
        with pytest.raises(Exception):
            coord.coord_map["bob"]


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

    def test_select_coord_dim(self, coord_manager):
        """Simple test for filtering dimension coord."""
        new, inds = coord_manager.select(distance=(100, 400))
        dist_ind = coord_manager.dims.index("distance")
        assert new.shape[dist_ind] < coord_manager.shape[dist_ind]
        assert len(inds) == len(coord_manager.dims)
        assert inds[dist_ind] != slice(None, None)


class TestTranspose:
    """Test suite for transposing dimensions."""

    def test_missing_dim_raises(self, coord_manager):
        """All dimensions must be specified, else raise."""
        with pytest.raises(CoordError, match="specify all dimensions"):
            coord_manager.transpose(["time"])

    def test_simple_transpose(self, coord_manager):
        """Ensure the coord manager can be transposed"""
        dims = coord_manager.dims
        new_dims = dims[::-1]
        tran = coord_manager.transpose(new_dims)
        assert tran.dims == new_dims
        assert tran.shape != coord_manager.shape
        assert tran.shape == coord_manager.shape[::-1]


class TestRenameDims:
    """Test case for renaming dimensions."""

    def test_rename_dims(self, coord_manager):
        """Ensure dimensions can be renamed."""
        rename_map = {x: x[:2] for x in coord_manager.dims}
        out = coord_manager.rename_dims(**rename_map)
        assert set(out.dims) == set(rename_map.values())


class TestUpdateFromAttrs:
    """Tests to ensure updating attrs can update coordinates."""

    def test_update_min(self, coord_manager):
        """Ensure min time in attrs updates appropriate coord."""
        for dim in coord_manager.dims:
            coord = coord_manager.coord_map[dim]
            attrs = {f"{dim}_max": coord.min}
            new = coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            assert len(new_coord) == len(coord)
            assert new_coord.max == coord.min

    def test_update_max(self, coord_manager):
        """Ensure max time in attrs updates appropriate coord."""
        for dim in coord_manager.dims:
            coord = coord_manager.coord_map[dim]
            attrs = {f"{dim}_min": coord.max}
            dist = coord.max - coord.min
            new = coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max - new_coord.min
            assert dist == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min == coord.max

    def test_update_step(self, coord_manager):
        """Ensure the step can be updated which changes endtime."""
        for dim in coord_manager.dims:
            coord = coord_manager.coord_map[dim]
            attrs = {f"d_{dim}": coord.step * 10}
            dist = coord.max - coord.min
            new = coord_manager.update_from_attrs(attrs)
            new_coord = new.coord_map[dim]
            new_dist = new_coord.max - new_coord.min
            assert (dist * 10) == new_dist
            assert len(new_coord) == len(coord)
            assert new_coord.min == coord.min
