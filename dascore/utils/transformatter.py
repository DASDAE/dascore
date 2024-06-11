"""Implements logic to apply formatting changes to patches from transformations."""

from __future__ import annotations

import abc

from dascore.utils.misc import iterate


class BaseTransformatter(abc.ABC):
    """Base model for helping to apply transformation format changes."""

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

    def rename_dims(self, dims, index=None, forward=True):
        """Rename the dimensions."""
        func = self._forward_rename if forward else self._inverse_rename
        new = list(iterate(dims))
        index_list = iterate(index) if index is not None else range(len(new))
        for index in index_list:
            new[index] = func(new[index])
        return tuple(new)


class FourierTransformatter(BaseTransformatter):
    """Formatters."""

    forward_prefix: str = "ft_"
    inverse_prefix: str = "ift_"
