"""
Read functionality for fiber files.
"""
from typing import Union, Optional
from pathlib import Path

import numpy as np
import xarray as xr


def read(
    path: Union[str, Path],
    format: Optional[str] = None,
    start_time: Optional[np.datetime64] = None,
    end_time: Optional[np.datetime64] = None,
    start_distance: Optional[float] = None,
    end_distance: Optional[float] = None,
) -> xr.DataArray:
    """
    Read a fiber file.
    """
