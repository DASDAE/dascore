"""Utilities for reading Febus T1 data"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore import get_coord_manager
from dascore.constants import timeable_types
from dascore.io.core import _make_scan_payload
from dascore.utils.hdf5 import H5Reader

_DATA = "Data"


def _get_h5_attr(fi: H5Reader, key: str) -> np.ndarray:
    return fi[f"{_DATA}/{key}"][()]


def _is_t1_file(fi: H5Reader) -> bool:
    """Minimal fingerprint check — T1 files always have these datasets."""
    required = {"Temperature", "Distance", "DistanceSignal", "Time"}
    present = set(fi.get(_DATA, {}).keys())
    return required.issubset(present)


def _get_distance_coord(fi):
    """Get the distances from the T1 file"""
    dist = fi["Data/Distance"][()]
    return dc.get_coord(values=dist, units="m")


def _get_time_coord(fi):
    """Get the times from the T1 file"""
    ts = fi["Data/Time"][()].squeeze()
    times = (ts * 1e9).astype("datetime64[ns]")
    return dc.get_coord(values=times, units="s")


def _get_coords(fi) -> dc.CoordManager:
    time_coord = _get_time_coord(fi)
    distance_coord = _get_distance_coord(fi)
    dims = ("time", "distance")
    return get_coord_manager(
        {"time": time_coord, "distance": distance_coord},
        dims=dims,
    )


def _get_attrs(path, format, version):
    """Build PatchAttrs from an already-constructed CoordManager."""
    return dict(
        path=path,
        file_format=format,
        file_version=str(version),
        data_type="temperature",
        data_units="°C",
        data_category="DTS",
        instrument_make="FEBUS",
        instrument_model="T1",
    )


def _scan_t1(fi: H5Reader, format, version):
    """Get the coordinates and attributes for a T1 data patch"""
    coords = _get_coords(fi)
    attrs = _get_attrs(path="", format="", version="")
    for name in ("path", "file_format", "file_version"):
        attrs.pop(name)
    return _make_scan_payload(
        attrs=attrs,
        coords=coords,
        dims=coords.dims,
        shape=coords.shape,
        dtype=str(_get_h5_attr(fi, "Temperature").dtype),
    )


def _get_t1_patch(
    fi: H5Reader,
    format: str,
    version: str,
    time: tuple[timeable_types, timeable_types] | None = None,
    distance: tuple[float, float] | None = None,
) -> dc.Patch:
    """Core builder shared by read() and scan()."""
    coords = _get_coords(fi)
    # Slice the coordinates
    time_coord, time_slice = coords.get_coord("time").select(time)
    distance_coord, distance_slice = coords.get_coord("distance").select(distance)
    coords = coords.new(coord_map={"time": time_coord, "distance": distance_coord})
    # Get the temperature data
    temp = _get_h5_attr(fi, "Temperature")[
        time_slice, distance_slice
    ]  # (n_time, n_dist)
    # Construct the patch
    attrs = _get_attrs(path=fi.filename, format=format, version=version)
    attrs = dc.PatchAttrs(**attrs)
    return dc.Patch(data=temp, coords=coords, dims=coords.dims, attrs=attrs)
