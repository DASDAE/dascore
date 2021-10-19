"""
Just some scratch space for playing around with xarray dataset.
"""
import numpy as np
import xarray as xr

# create first example data array

data = np.random.rand(1000, 100)
t1 = np.datetime64("2015-01-01")
td = np.timedelta64(1_000_00, "ns")
time_1 = np.arange(data.shape[0]) * td + t1
distance = np.arange(data.shape[1]) * 1

coords = {
    "time": time_1,
    "distance": distance,
}
ar_1 = xr.DataArray(data, dims=list(coords), coords=coords)

# create second example data array

t1 = np.datetime64("2019-01-01")
td = np.timedelta64(1_000_000, "ns")
time_2 = np.arange(data.shape[0]) * td + t1

coords = {
    "time": time_2,
    "distance": distance,
}
ar_2 = xr.DataArray(data, dims=list(coords), coords=coords)

# create dataset

ds = xr.Dataset({1: ar_1, 2: ar_2})

print(ds[1].shape)  # (2000, 100)
print(ds[1].isnull().sum)  # 100_000
breakpoint()
