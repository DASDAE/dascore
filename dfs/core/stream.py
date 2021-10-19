"""
A module for storing streams of fiber data.
"""
from collections import UserList
from typing import Sequence, Union
from dfs.core import DataArray


class Stream(UserList):
    """
    A stream of fiber data.
    """

    def __init__(self, data: Union[DataArray, Sequence[DataArray]]):
        if isinstance(data, DataArray):
            data = [data]
        self.data: Sequence[DataArray] = data
