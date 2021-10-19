"""
Core routines and functionality for processing distributed fiber data.
"""
from xarray import DataArray, Dataset

from .constructor import create_das_array
from .trim import trim_by_time, trim_by_distance
from .stream import Stream
