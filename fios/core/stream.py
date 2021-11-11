"""
A module for storing streams of fiber data.
"""
from collections import UserList
from typing import Sequence, Union

import pandas as pd

import fios
from fios.io.base import scan_patches


class Stream:
    """
    A stream of fiber data.
    """

    def __init__(self, data: Union["fios.Patch", Sequence["fios.Patch"]]):
        if isinstance(data, fios.Patch):
            data = [data]
        df = pd.DataFrame(scan_patches(data))
        df["patch"] = data
        self._df = df

    def __getitem__(self, item):
        return self._df.iloc[0]["patch"]

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        patches = self._df["patch"].values
        return iter(patches)
