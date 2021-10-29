"""
A module for storing streams of fiber data.
"""
from collections import UserList
from typing import Sequence, Union
from fios.core import Trace2D


class Stream(UserList):
    """
    A stream of fiber data.
    """

    def __init__(self, data: Union[Trace2D, Sequence[Trace2D]]):
        if isinstance(data, Trace2D):
            data = [data]
        self._data: Sequence[Trace2D] = data
