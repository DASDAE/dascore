"""
Tests for coordinate object.
"""
import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import BaseCoord, get_coord
from dascore.exceptions import CoordError
from dascore.utils.misc import register_func
from dascore.utils.time import to_datetime64

COORDS = []


@pytest.fixture(scope="class")
@register_func(COORDS)
def evenly_sampled_coord():
    """Create coordinates which are evenly sampled."""
    ar = np.arange(1, 100, 1)
    return get_coord(data=ar)


@pytest.fixture(scope="class")
@register_func(COORDS)
def evenly_sampled_date_coord():
    """Create coordinates which are evenly sampled."""
    ar = to_datetime64(np.arange(1, 100_000, 1_000))
    return get_coord(data=ar)


@pytest.fixture(scope="class", params=COORDS)
def coord(request):
    """Meta-fixture for returning all coords"""
    return request.getfixturevalue(request.param)


class TestBasics:
    """A suite of basic tests for coordinates."""

    def test_coord_init(self, coord):
        """simply run to insure all coords initialize."""
        assert isinstance(coord, BaseCoord)

    def test_bad_init(self):
        """Ensure no parameters raises error"""
        with pytest.raises(CoordError):
            get_coord()


class TestCoordRange:
    """Tests for coords from array."""

    def test_init_array(self, evenly_sampled_coord):
        """Ensure the array can be initialized."""
        assert evenly_sampled_coord.step == 1.0
        assert evenly_sampled_coord.units is None

    def test_values(self, evenly_sampled_coord):
        """Ensure we can get an array with values attribute."""
        vals = evenly_sampled_coord.values
        assert len(vals) == len(evenly_sampled_coord)

    def test_set_units(self, evenly_sampled_coord):
        """Ensure units can be set."""
        out = evenly_sampled_coord.set_units("m")
        assert out.units == dc.Unit("m")
        assert out is not evenly_sampled_coord

    def test_convert_units(self, evenly_sampled_coord):
        """Ensure units can be converted/set."""
        # if units are not set convert_units should set them.
        out_m = evenly_sampled_coord.convert_units("m")
        assert out_m.units == dc.Unit("m")
        out_mm = out_m.convert_units("mm")
        assert out_mm.units == dc.Unit("mm")
        assert len(out_mm) == len(out_m)
        assert np.allclose(out_mm.values / 1000, out_m.values)

    def test_dtype(self, evenly_sampled_coord, evenly_sampled_date_coord):
        """Ensure datatypes are sensible."""
        dtype1 = evenly_sampled_coord.dtype
        assert np.issubdtype(dtype1, np.int_)
        dtype2 = evenly_sampled_date_coord.dtype
        assert np.issubdtype(dtype2, np.datetime64)

    def test_select_tuple_ints(self, evenly_sampled_coord):
        """Ensure a tuple works as a limit."""
        assert evenly_sampled_coord.select((50, None)) == slice(50, None)
        assert evenly_sampled_coord.select((0, None)) == slice(0, None)
        assert evenly_sampled_coord.select((-10, None)) == slice(None, None)
        assert evenly_sampled_coord.select((None, None)) == slice(None, None)
        assert evenly_sampled_coord.select((None, 1_000_000)) == slice(None, None)

    def test_select_date_string(self):
        """Ensure string selection works with datetime objects."""
