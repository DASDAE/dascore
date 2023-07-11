"""
Tests for transformatter.
"""
import pytest

from dascore.units import get_quantity
from dascore.utils.transformatter import FourierTransformatter


@pytest.fixture()
def ft_reformatter():
    """Simple fourier transform formatter."""
    return FourierTransformatter()


class TestFTDimensionRename:
    """Tests for renaming dimensions using FT transformer"""

    dims = ("distance", "time")

    def test_forward_rename_one_index(self, ft_reformatter):
        """Ensure the name can be reassigned."""
        out = ft_reformatter.rename_dims(self.dims, 1)
        assert out == ("distance", "ft_time")

    def test_forward_rename_all_index(self, ft_reformatter):
        """Ensure all indices can be renamed."""
        out = ft_reformatter.rename_dims(self.dims)
        assert out == tuple(f"{ft_reformatter.forward_prefix}{x}" for x in self.dims)

    def test_forward_undo_inverse(self, ft_reformatter):
        """Ensure the inverse is correctly undone."""
        dims = tuple([f"{ft_reformatter.inverse_prefix}{x}" for x in self.dims])
        out = ft_reformatter.rename_dims(dims)
        assert out == self.dims

    def test_inverse_rename_one_index(self, ft_reformatter):
        """Ensure the name can be reassigned."""
        out = ft_reformatter.rename_dims(self.dims, 1, forward=False)
        assert out == ("distance", "ift_time")

    def test_undo_forward_index(self, ft_reformatter):
        """Ensure forward prefex is undone by inverse."""
        dims = tuple(f"{ft_reformatter.forward_prefix}{x}" for x in self.dims)
        out = ft_reformatter.rename_dims(dims, forward=False)
        assert out == self.dims

    def test_double_forward(self, ft_reformatter):
        """Prefixes should stack."""
        pre = ft_reformatter.forward_prefix
        dims1 = ft_reformatter.rename_dims(self.dims)
        dims2 = ft_reformatter.rename_dims(dims1)
        assert dims2 == tuple(f"{pre}{pre}{x}" for x in self.dims)


class TestFTUnitRename:
    """Ensure units can be renamed."""

    dims = ("distance", "time")
    attrs = {
        "distance_units": str(get_quantity("1.0 m")),
        "time_units": str(get_quantity("1.0 s")),
        "data_units": str(get_quantity("1.0 V")),
    }

    def test_forward_unit_transform(self, ft_reformatter):
        """Simple forward unit transformation"""
        out = ft_reformatter.rename_attrs(self.dims, self.attrs, index=0)
        key = f"{self.dims[0]}_units"
        value = self.attrs[key]
        assert get_quantity(f"1/({value})") == get_quantity(out[key])
        out_2 = ft_reformatter.rename_attrs(self.dims, out, index=0)
        assert out_2 == self.attrs

    def test_backward_unit_transform(self, ft_reformatter):
        """Simple forward unit transformation"""
        out = ft_reformatter.rename_attrs(self.dims, self.attrs, index=0, forward=False)
        key = f"{self.dims[0]}_units"
        value = self.attrs[key]
        assert 1 / get_quantity(value) == get_quantity(out[key])
        out_2 = ft_reformatter.rename_attrs(self.dims, out, index=0, forward=False)
        assert out_2 == self.attrs

    def test_data_units_single_index(self, ft_reformatter):
        """Ensure data units are renamed when a single index is specified."""
        for index, dim in enumerate(self.dims):
            out = ft_reformatter.rename_attrs(
                self.dims, self.attrs, index=index, forward=False
            )
            new_units = get_quantity(out["data_units"])
            old_units = get_quantity(self.attrs["data_units"])
            old_coord_units = get_quantity(self.attrs[f"{dim}_units"])
            assert new_units == old_units * old_coord_units
            # now check round tripping
            rt = ft_reformatter.rename_attrs(
                self.dims,
                out,
                index=index,
                forward=True,
            )
            assert get_quantity(rt["data_units"]) == old_units

    def test_data_units_no_index(self, ft_reformatter):
        """Ensure data units are renamed when all units are specified."""
        out = ft_reformatter.rename_attrs(self.dims, self.attrs)
        dunit = get_quantity(self.attrs["data_units"])
        time_unit = get_quantity(self.attrs["time_units"])
        dist_units = get_quantity(self.attrs["distance_units"])
        assert get_quantity(out["data_units"]) == dunit * time_unit * dist_units
        reverse = ft_reformatter.rename_attrs(self.dims, out)
        assert get_quantity(reverse["data_units"]) == dunit
