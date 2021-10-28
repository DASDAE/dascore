"""
A 2D trace object.
"""
from typing import Union, Optional, Mapping

import numpy as np
from numpy.typing import ArrayLike

from dfs.utils.mapping import FrozenDict
from dfs.constants import DEFAULT_ATTRS


def _get_attrs(attr=None):
    """Get the attribute dict, add required keys if not yet defined."""
    out = {} if attr is None else dict(attr)
    # add default values
    for missing in set(DEFAULT_ATTRS) - set(attr):
        out[missing] = DEFAULT_ATTRS[missing]
    return FrozenDict(out)


class Trace2D:
    """
    A Class for storing and accessing 2D fiber data.
    """

    def __init__(
        self,
        data: Union[ArrayLike, "Trace2D"],
        time: Optional[np.ndarray] = None,
        distance: Optional[np.ndarray] = None,
        attrs: Optional[Mapping] = None,
    ):
        self._data = data
        self.time = time
        self.distance = distance
        self._attrs = _get_attrs(attrs)

    def __eq__(self, other):
        """
        Compare one Trace2D to another.

        Parameters
        ----------
        other

        Returns
        -------

        """

    def equals(self, other: "Trace2D", only_required_attrs=True) -> bool:
        """
        Determine if the current trace equals the other trace.

        Parameters
        ----------
        other
            A Trace2D object
        only_required_attrs
            If True, only compare required attributes.
        """
        return self.data.equals(other.data)

    def update_attrs(self, **kwargs) -> "Trace2D":
        """
        Update attrs and return a new trace2D.
        """
        attrs = dict(self._attrs)
        attrs.update(**kwargs)
        return self.__class__(
            self.data, time=self.time, distance=self.distance, attrs=attrs
        )

    @property
    def data(self):
        return self._data
