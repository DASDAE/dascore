"""NetCDF IO support for DASCore using CF (Climate and Forecast) conventions."""

from __future__ import annotations

from dascore.io.netcdf.core import NetCDFCFV18
from dascore.io.netcdf.utils import (
    cf_time_to_datetime64,
    datetime64_to_cf_time,
    get_cf_data_attrs,
    get_cf_global_attrs,
)
