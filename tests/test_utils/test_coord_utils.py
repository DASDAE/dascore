"""
Tests for coordinate utilities.
"""
import numpy as np
import pytest

from dascore.utils.coords import Coords
from dascore.utils.misc import register_func


class TestCoords:
    """Tests for DASCore's custom coord class."""

    coord_fixtures = []
    time = [1, 2, 3]
    distance = [1, 2, 3, 4, 5]
    shape = (len(time), len(distance))
    dims = ("time", "distance")

    @pytest.fixture(scope="class")
    @register_func(coord_fixtures)
    def simple_dict_coords(self):
        """Create a simple version of coords from dict."""
        cdict = {"time": self.time, "distance": self.distance}
        return Coords(cdict, self.dims)

    @pytest.fixture(scope="class")
    @register_func(coord_fixtures)
    def simple_dict_coords_dim_name_tuple(self):
        """Create a simple version of coords from dict."""
        input_dict = {
            "time": ("time", self.time),
            "distance": ("distance", self.distance),
        }
        return Coords(input_dict, self.dims)

    @pytest.fixture(scope="class")
    @register_func(coord_fixtures)
    def simple_dict_coords_other_coords(self):
        """Create a simple version of coords from dict."""
        input_dict = {
            "time": ("time", self.time),
            "distance": self.distance,
            "latitude": ("distance", np.ones_like(self.distance)),
        }
        return Coords(input_dict, self.dims)

    @pytest.fixture(scope="class")
    @register_func(coord_fixtures)
    def coord_from_xarray(self, random_patch):
        """Return a Coord instance from xarray coords."""
        dar = random_patch.to_xarray().transpose("time", "distance")
        return Coords(dar.coords)

    @pytest.fixture(scope="class", params=coord_fixtures)
    def coord(self, request):
        """meta-fixture to aggregate all coords."""
        return request.getfixturevalue(request.param)

    def test_dims_inferred(self, coord):
        """Ensure dimensions can be inferred."""
        assert coord.dims == ("time", "distance")

    def test_update(self, coord_from_xarray):
        """Ensure update returns a new coord with updated values."""
        time = coord_from_xarray["time"]
        new_time = time + np.timedelta64(1, "s")
        out = coord_from_xarray.update(time=new_time)
        assert np.all(new_time == out["time"])
        assert isinstance(out, Coords)
        assert out is not coord_from_xarray
