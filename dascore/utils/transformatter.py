"""
Implements logic to apply formatting changes to patches from transformations.
"""
import abc
from abc import abstractmethod

from dascore.units import get_quantity


class BaseTransformatter(abc.ABC):
    """
    Base model for helping to apply transformation format changes.
    """

    forward_prefix: str = ""
    inverse_prefix: str = ""

    def _forward_rename(self, name):
        """Rename the dimension for forward transform."""
        if name.startswith(self.inverse_prefix):
            return name[len(self.inverse_prefix) :]
        return f"{self.forward_prefix}{name}"

    def _inverse_rename(self, name):
        """Rename the dimension for backward transform."""
        if name.startswith(self.forward_prefix):
            return name[len(self.forward_prefix) :]
        return f"{self.inverse_prefix}{name}"

    @abstractmethod
    def _forward_unit_label(self, unit_name):
        """Adjust the unit label for forward transform."""

    @abstractmethod
    def _inverse_unit_label(self, unit_name):
        """Adjust the init label for inverse transform."""

    def rename_dims(self, dims, index=None, forward=True):
        """Rename the dimensions."""
        func = self._forward_rename if forward else self._inverse_rename
        new = list(dims)
        index_list = [index] if index is not None else range(len(dims))
        for index in index_list:
            new[index] = func(new[index])
        return tuple(new)

    def rename_attrs(self, dims, attrs, index=None, forward=True):
        """Rename the unit attribute names."""
        func = self._forward_unit_label if forward else self._inverse_unit_label
        index_list = [index] if index is not None else range(len(dims))
        out = dict(attrs)  # this ensures we have a dict and a copy
        new_units = []
        for ind in index_list:
            dim_unit_attr_name = f"{dims[ind]}_units"
            if dim_unit_attr_name in out:
                quant = func(out[dim_unit_attr_name])
                out[dim_unit_attr_name] = quant
                new_units.append(get_quantity(quant))
        if "data_units" in out and (dunits := out["data_units"]) is not None:
            data_units = get_quantity(dunits)
            for new_unit in new_units:
                data_units /= new_unit
            out["data_units"] = str(data_units)
        return out

    def transform_dims_and_attrs(self, dims, attrs, index=None, forward=True):
        """Create new dims and attrs."""
        new_attrs = self.rename_attrs(dims, attrs, index=index, forward=forward)
        new_dims = self.rename_dims(dims, index=index, forward=forward)
        return new_dims, new_attrs


class FourierTransformatter(BaseTransformatter):
    """
    Formatters
    """

    forward_prefix: str = "ft_"
    inverse_prefix: str = "ift_"

    def _toggle_unit_str(self, unit_str):
        """Toggle the unit string."""
        quantity = get_quantity(unit_str)
        return str(1 / quantity)

    def _forward_unit_label(self, unit_name):
        """Adjust the unit label for forward transform."""
        return self._toggle_unit_str(unit_name)

    def _inverse_unit_label(self, unit_name):
        """Adjust the init label for inverse transform."""
        return self._toggle_unit_str(unit_name)
