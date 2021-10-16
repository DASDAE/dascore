"""
IO functionality for DAS (distributed acoustic sensing)
"""
from typing import Union, Optional
from pathlib import Path

import numpy as np
import xarray as xr


def read_das(
    path: Union[str, Path],
    format: Optional[str] = None,
    starttime: Optional[np.datetime64] = None,
    endtime: Optional[np.datatime64] = None,
) -> xr.DataArray:
    """Read a das file."""
