"""
Generic Constructors for datasets.
"""
import numpy as np
from dfs.core import Dataset, DataArray

def create_das_array(
    data: np.array,
    time: np.array,
    channel: np.array,
    sample_time: np.timedelta64,
    sample_lenth: float,
    attrs=None,
) -> DataArray:
    """
    Create a DAS DataArray from numpy arrays and metadata.
    """
    coords = {
        "time": time,
        "channel": channel,
        "sample_time": sample_time,
        "sample_lenth": sample_lenth,
        "data_type": 'DAS',
    }
    out = DataArray(data=data, dims=list(coords)[:2], coords=coords, attrs=attrs)
    return out
