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


class TestUpdateFromAttrs:
    """Tests to ensure updating attrs can update coordinates."""

    def test_update_min(self, coord_manager):
        """Ensure min time in attrs updates appropriate coord."""
        for dim in coord_manager.dims:
            coord = coord_manager.coord_map[dim]
            attrs = {f"{dim}_max": coord.min}
            new = coord_manager.update_from_attrs(attrs)
            assert new.coord_map[dim].max == coord.min


#
#
# class TestAttrsCoordsMixer:
#     """Tests for handling complex interaction between attrs and coords."""
#
#     @pytest.fixture()
#     def attrs(self, random_patch):
#         """return an attrs from a patch"""
#         return random_patch.attrs
#
#     @pytest.fixture()
#     def coords(self, random_patch):
#         """return an attrs from a patch"""
#         return random_patch.coords
#
#     @pytest.fixture()
#     def mixer(self, attrs, coords, random_patch):
#         """Return a mixer instance."""
#         return _AttrsCoordsMixer(attrs, coords, random_patch.dims)
#
#     def test_original_attrs_unchanged(self, mixer, attrs):
#         """ensure the original attrs don't change."""
#         t1 = attrs["time_min"]
#         td = np.timedelta64(1, "s")
#         mixer.update_attrs(time_min=t1 + td)
#         new_attr, _ = mixer()
#         assert new_attr is not attrs
#         assert attrs["time_min"] + td == new_attr["time_min"]
#
#     def test_original_coords_unchanged(self, mixer, coords, attrs):
#         """ensure the original coords don't change."""
#         t1 = attrs["time_min"]
#         td = np.timedelta64(10, "s")
#         mixer.update_attrs(time_min=t1 + td)
#         _, new_coords = mixer()
#         assert new_coords is not coords
#         np.all(np.equal(coords["time"] + td, new_coords["time"]))
#
#     def test_coords_unchanged(self, coords, attrs, random_patch):
#         """ensure the original coords don't change."""
#         # this was added to track down some mutation issues
#         assert coords["time"].min() == attrs["time_min"]
#
#     def test_starttime_updates_endtime(self, mixer, attrs, coords):
#         """Ensure the end time gets updated when setting time_min"""
#         t1 = attrs["time_min"]
#         t_new = t1 + np.timedelta64(10_000_000, "s")
#         mixer.update_attrs(time_min=t_new)
#         attr, coords = mixer()
#         assert attr["time_min"] == t_new
#         # make sure time min was updated in coords
#         time = coords["time"]
#         assert np.min(time) == t_new
#
#     def test_endtime_updates_starttime(self, mixer, attrs):
#         """Ensure the start time gets updated when setting time_max."""
#         tdist1 = attrs["time_max"] - attrs["time_min"]
#         t2 = attrs["time_max"]
#         t_new = t2 - np.timedelta64(10_000_000, "s")
#         mixer.update_attrs(time_max=t_new)
#         attr, coords = mixer()
#         assert attr["time_max"] == t_new
#         # make sure time min was updated in coords
#         time = coords["time"]
#         assert np.max(time) == t_new
#         # ensure distance between start/end is the same
#         tdist2 = attr["time_max"] - attr["time_min"]
#         assert tdist2 == tdist1
#
#     def test_coords_updates_times_and_distance(self, mixer, coords, attrs):
#         """
#         Ensure updating coords also updates attributes in attrs.
#         """
#         td = np.timedelta64(10, "s")
#         dx = 10
#         new_coords_kwarg = {
#             "time": coords["time"] + td,
#             "distance": coords["distance"] + dx,
#         }
#
#         mixer.update_coords(**new_coords_kwarg)
#         new_attrs, new_coords = mixer()
#         # first ensure coords actually updated
#         assert np.all(new_coords["time"] == new_coords_kwarg["time"])
#         assert np.all(new_coords["distance"] == new_coords_kwarg["distance"])
#         # check attrs time are updated
#         assert attrs["time_min"] + td == new_attrs["time_min"]
#         assert attrs["time_max"] + td == new_attrs["time_max"]
#         # check distance
#         assert attrs["distance_min"] + dx == new_attrs["distance_min"]
#         assert attrs["distance_max"] + dx == new_attrs["distance_max"]
#
#     def test_relative_time_coord_no_absolute_time(self):
#         """Ensure a time axis is still obtained with relative time and no start"""
#         coords = dict(
#             time=np.arange(10) * 0.1,
#             distance=np.arange(10) * 10,
#         )
#         attrs = {}
#         mixer = _AttrsCoordsMixer(attrs, coords, ("time", "distance"))
#         new_attrs, new_coords = mixer()
#         assert not np.any(pd.isnull(new_coords["time"]))
#
#     def test_update_startttime_string(self, mixer, coords, attrs):
#         """check start/end time relationship when starttime is a string."""
#         new_start = np.datetime64("2000-01-01")
#         duration = attrs["time_max"] - attrs["time_min"]
#         mixer.update_attrs(time_min=str(new_start))
#         new_attrs, _ = mixer()
#         assert new_attrs["time_min"] == new_start
#         assert new_attrs["time_max"] == new_start + duration
#
#     def test_update_time_delta(self, mixer, coords, attrs):
#         """Updating the time delta should also update endtimes."""
#         one_sec = np.timedelta64(1, "s")
#         td_old = attrs["d_time"]
#         td_new = td_old * 2
#         mixer.update_attrs(d_time=td_new)
#         new_attrs, new_coords = mixer()
#         assert new_attrs["d_time"] == td_new
#         # first ensure new time coords are approximately new time delta
#         new_time = new_coords["time"]
#         tdiff = (new_time[1:] - new_time[:-1]) / one_sec
#         assert np.allclose(tdiff, td_new / one_sec)
#         # also ensure the end time has increased proportionately.
#         old_duration = attrs["time_max"] - attrs["time_min"]
#         new_duration = new_attrs["time_max"] - new_attrs["time_min"]
#         assert np.isclose(old_duration / new_duration, td_old / td_new)
#
