"""
Implements logic to apply formatting changes to patches from transformations.
"""
import abc
import fnmatch
from abc import abstractmethod


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
        for ind in index_list:
            dim_unit_attr_name = f"{dims[ind]}_units"
            if dim_unit_attr_name in out:
                out[dim_unit_attr_name] = func(out[dim_unit_attr_name])
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
        if fnmatch.fnmatch(unit_str, "1/(*)"):
            out = unit_str[3:-1]
        else:
            out = f"1/({unit_str})"
        return out

    def _forward_unit_label(self, unit_name):
        """Adjust the unit label for forward transform."""
        return self._toggle_unit_str(unit_name)

    def _inverse_unit_label(self, unit_name):
        """Adjust the init label for inverse transform."""
        return self._toggle_unit_str(unit_name)
