"""
Generic Constructors for datasets.
"""
import numpy as np
from dfs.core import Dataset, DataArray


def create_das_array(
    data: np.array,
    time: np.array,
    distance: np.array,
    sample_time: np.timedelta64,
    sample_lenth: float,
    attrs=None,
    datatype="",
) -> DataArray:
    """
    Create a DAS DataArray from numpy arrays and metadata.
    """
    coords = {
        "time": time,
        "distance": distance,
        "sample_time": sample_time,
        "sample_length": sample_lenth,
        "category": "DAS",
        "datatype": datatype,
    }
    out = DataArray(data=data, dims=list(coords)[:2], coords=coords, attrs=attrs)
    return out
