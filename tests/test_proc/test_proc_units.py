"""Tests for unit dealings on patches."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnitError
from dascore.units import get_quantity
from dascore.utils.time import dtype_time_like


class TestSetUnits:
    """Tests for setting units without conversions."""

    def test_data_units(self, random_patch_with_lat_lon):
        """Simply test setting units for data."""
        patch = random_patch_with_lat_lon
        unit_str = "km/µs"
        out = patch.set_units(unit_str)
        assert get_quantity(out.attrs.data_units) == get_quantity(unit_str)

    def test_convert_dim_units(self, random_patch_with_lat_lon):
        """Ensure we can convert dimensional coordinate units."""
        patch = random_patch_with_lat_lon
        unit_str = "furlong"  # A silly unit that wont be used on patch
        for dim in patch.dims:
            expected_quant = get_quantity(unit_str)
            coord = patch.get_coord(dim)
            new = patch.set_units(**{dim: unit_str})
            attr = new.attrs
            # first check attributes
            attr_name = f"{dim}_units"
            assert hasattr(attr, attr_name)
            value = getattr(new.attrs, attr_name)
            # time shouldn't ever be a anything other than s
            if dtype_time_like(coord.dtype):
                expected_quant = get_quantity("s")
            assert value == expected_quant
            # then check coord
            coord = new.coords.coord_map[dim]
            assert coord.units == expected_quant

    def test_update_non_dim_coord_units(self, random_patch_with_lat_lon):
        """Ensure non-dim coords can also have their units updated."""
        patch = random_patch_with_lat_lon
        unit_str = "10 furlongs"
        cmap = patch.coords.coord_map
        non_dims = set(cmap) - set(patch.dims)
        for coord_name in non_dims:
            new = patch.set_units(**{coord_name: unit_str})
            # coord string should show up in the attrs
            attr_name = f"{coord_name}_units"
            assert getattr(new.attrs, attr_name) == get_quantity(unit_str)
            # and the coord should be set
            coord = new.get_coord(coord_name)
            assert coord.units == get_quantity(unit_str)

    def test_set_data_and_coord_units(self, random_patch):
        """Ensure we can set data and coordinate units in 1 go."""
        out = random_patch.set_units("m/s", distance="ft")
        assert isinstance(random_patch, dc.Patch)
        assert get_quantity(out.attrs.data_units) == get_quantity("m/s")
        assert get_quantity(out.attrs.distance_units) == get_quantity("ft")

    def test_remove_units(self, random_patch):
        """Ensure set coords can remove units."""
        patch = random_patch.set_units("m", distance="m")
        # ensure data units can be removed
        new_none = patch.set_units(None)
        assert new_none.attrs.data_units is None
        new_empty_str = patch.set_units("")
        assert new_empty_str.attrs.data_units is None
        # ensure coord units can be removed
        new_dist_none = random_patch.set_units(distance=None)
        assert new_dist_none.get_coord("distance").units is None
        new_dist_empty = random_patch.set_units(distance="")
        assert new_dist_empty.get_coord("distance").units is None


class TestConvertUnits:
    """Tests for converting from one unit to another."""

    @pytest.fixture()
    def unit_patch(self, random_patch_with_lat_lon):
        """Get a patch with units indicated."""
        patch = random_patch_with_lat_lon
        unit_patch = patch.set_units("m/s", distance="m", time="s")
        return unit_patch

    def test_convert_data_units(self, unit_patch):
        """Ensure simple conversions work."""
        out_km = unit_patch.convert_units("km/s")
        assert np.allclose(out_km.data * 1_000, unit_patch.data)
        assert out_km.attrs.data_units == get_quantity("km/s")
        out_mm = unit_patch.convert_units("mm/s")
        assert np.allclose(out_mm.data / 1_000, unit_patch.data)
        assert out_mm.attrs.data_units == get_quantity("mm/s")

    def test_convert_data_quantity(self, unit_patch):
        """Ensure quantities can be used for units."""
        unit_str = "10 m/s"
        out = unit_patch.convert_units(unit_str)
        assert out.attrs.data_units == get_quantity(unit_str)
        assert np.allclose(out.data * 10, unit_patch.data)

    def test_attrs_preserved(self, unit_patch):
        """The attributes should not change."""
        patch = unit_patch.update_attrs(tag="bob")
        out = patch.convert_units("km/s")
        assert patch.attrs.tag == out.attrs.tag

    def test_bad_data_conversion_raises(self, unit_patch):
        """Ensure asking for incompatible units raises."""
        with pytest.raises(UnitError):
            unit_patch.convert_units("degC")
        with pytest.raises(UnitError):
            unit_patch.convert_units("M")
        with pytest.raises(UnitError):
            unit_patch.convert_units("s")

    def test_update_data_and_coord_units(self, random_patch):
        """Ensure we can update data and coordinate units in 1 go."""
        patch = random_patch.set_units("m/s", distance="ft")
        out = patch.convert_units("ft/s", distance="m")
        assert isinstance(random_patch, dc.Patch)
        assert get_quantity(out.attrs.data_units) == get_quantity("ft/s")
        assert get_quantity(out.attrs.distance_units) == get_quantity("m")


class TestSimplifyUnits:
    """Ensure units can be simplified."""

    @pytest.fixture(scope="class")
    def patch_complicated_units(self, random_patch):
        """Get a patch with very complicated units for simplifying."""
        out = random_patch.set_units(
            "km/us * (rad/degrees)",
            time="s * (kfurlongs)/(furlongs)",
            distance="miles",
        )
        return out

    def test_simplify_everything(self, patch_complicated_units):
        """Test to simplify all parts of the patch."""
        out = patch_complicated_units.simplify_units()
        attrs = out.attrs
        assert isinstance(out, dc.Patch)
        # test attrs
        assert get_quantity(attrs.data_units) == get_quantity("m/s")
        assert get_quantity(attrs.time_units) == get_quantity("s")
        assert get_quantity(attrs.distance_units) == get_quantity("m")
