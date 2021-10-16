"""
Generic Constructors for datasets.
"""
import xarray as xr
import numpy as np


def create_array(
    data: np.array, time: np.array, channel: np.array, attrs=None
) -> xr.DataArray:
    """
    Create a dataset from numpy arrays.
    """
    coords = {"time": time, "channel": channel}
    out = xr.DataArray(data=data, dims=list(coords), coords=coords, attrs=attrs)
    return out
