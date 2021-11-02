"""
A module for storing streams of fiber data.
"""
from collections import UserList
from typing import Sequence, Union

import fios


class Stream(UserList):
    """
    A stream of fiber data.
    """

    def __init__(self, data: Union["fios.Patch", Sequence["fios.Patch"]]):
        if isinstance(data, fios.Patch):
            data = [data]
        self.data: Sequence[fios.Patch] = data
