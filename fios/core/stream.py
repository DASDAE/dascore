"""
A module for storing streams of fiber data.
"""
from collections import UserList
from typing import Sequence, Union

import pandas as pd

import fios


class Stream(UserList):
    """
    A stream of fiber data.
    """

    def __init__(self, data: Union["fios.Patch", Sequence["fios.Patch"]]):
        # if isinstance(data, fios.Patch):
        #     data = [data]
        # df = pd.DataFrame(fios.scan(data))
        # df['patch'] = data
        # self._df = df
        self.data = data

    # def __getitem__(self, item):
    #     return self._df.iloc[0]['patch']
