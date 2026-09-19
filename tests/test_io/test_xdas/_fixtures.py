"""Small synthetic XDAS files, independent of any supplied recording."""

from __future__ import annotations

import numpy as np

from dascore.utils.misc import optional_import


def xdas_dataset(*, legacy=False, name="strain rate", shape=(31, 9)):
    """Create named data with compact coordinates and an absolute time origin."""
    xr = optional_import("xarray")
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    start = np.datetime64("2025-02-03T01:02:03.123456789", "ns")
    times = start + np.arange(shape[0]) * np.timedelta64(4, "ms")
    distances = -4.0 + np.arange(shape[1]) * 2.0
    ds = xr.Dataset(
        {name: (("time", "distance"), data)},
        attrs={"Conventions": "CF-1.9" if legacy else "CF-1.13"},
    )
    mapping = []
    for dim, values in (("time", times), ("distance", distances)):
        ds[f"{dim}_indices"] = (f"{dim}_points", [0, len(values) - 1])
        ds[f"{dim}_values"] = (f"{dim}_points", values[[0, -1]])
        ds[f"{dim}_interpolation"] = xr.DataArray(
            np.nan,
            attrs={
                "interpolation_name": "linear",
                "tie_points_mapping"
                if legacy
                else "tie_point_mapping": f"{dim}_points: {dim}_indices {dim}_values"
                if legacy
                else f"{dim}: {dim}_indices {dim}_points",
            },
        )
        mapping.append(
            f"{dim}: {dim}_indices {dim}_values"
            if legacy
            else f"{dim}_values: {dim}_interpolation"
        )
    ds[name].attrs = {
        "coordinate_interpolation": " ".join(mapping),
        "tag": "synthetic",
        "data_units": "1/s",
    }
    ds.distance_values.attrs["units"] = "m"
    return ds, data, times, distances


def write_xdas(path, **kwargs):
    """Write one small materialized file and return its independently known labels."""
    dataset, data, times, distances = xdas_dataset(**kwargs)
    dataset.to_netcdf(path, engine="h5netcdf")
    return data, times, distances
