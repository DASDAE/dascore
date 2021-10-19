"""
Core routines and functionality for processing distributed fiber data.
"""
from xarray import DataArray, Dataset

from .constructor import create_das_array
from .reshape import trim_by_time
