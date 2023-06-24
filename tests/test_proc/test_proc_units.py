"""
Tests for unit dealings on patches.
"""
import numpy as np
import pytest
from pint import DimensionalityError


class TestSetUnits:
    """Tests for setting units without conversions."""

    def test_data_units(self, random_patch_with_lat_lon):
        """Simply test setting units for data."""
        patch = random_patch_with_lat_lon
        unit_str = "km/µs"
        out = patch.set_units(unit_str)
        assert str(out.attrs.data_units) == unit_str

    def test_convert_dim_units(self, random_patch_with_lat_lon):
        """Ensure we can convert dimensional coordinate units."""
        patch = random_patch_with_lat_lon
        unit_str = "furlong"  # A silly unit that wont be used on patch
        for dim in patch.dims:
            new = patch.set_units(**{dim: unit_str})
            attr = new.attrs
            # first check attributes
            attr_name = f"{dim}_units"
            assert hasattr(attr, attr_name)
            assert getattr(new.attrs, attr_name) == unit_str
            # then check coord
            coord = new.coords.coord_map[dim]
            assert coord.units == unit_str

    def test_update_non_dim_coord_units(self, random_patch_with_lat_lon):
        """Ensure non-dim coords can also have their units updated."""
        patch = random_patch_with_lat_lon
        unit_str = "10 furlongs"
        cmap = patch.coords.coord_map
        non_dims = set(cmap) - set(patch.dims)
        for coord_name in non_dims:
            new = patch.set_units(**{coord_name: unit_str})
            # coord string should not show up in the attrs
            attr_name = f"{coord_name}_units"
            assert not hasattr(new.attrs, attr_name)
            # the but coord should be set
            coord = new.coords.coord_map[coord_name]
            assert coord.units == unit_str


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
        assert out_km.attrs.data_units == "km/s"
        out_mm = unit_patch.convert_units("mm/s")
        assert np.allclose(out_mm.data / 1_000, unit_patch.data)
        assert out_mm.attrs.data_units == "mm/s"

    def test_convert_data_quantity(self, unit_patch):
        """Ensure quantities can be used for units."""
        unit_str = "10 m/s"
        out = unit_patch.convert_units(unit_str)
        assert out.attrs.data_units == unit_str
        assert np.allclose(out.data * 10, unit_patch.data)

    def test_bad_data_conversion_raises(self, unit_patch):
        """Ensure asking for incompatible units raises."""
        with pytest.raises(DimensionalityError):
            unit_patch.convert_units("degC")
        with pytest.raises(DimensionalityError):
            unit_patch.convert_units("M")
        with pytest.raises(DimensionalityError):
            unit_patch.convert_units("s")
