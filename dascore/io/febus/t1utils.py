"""Utilities for reading Febus T1 data"""
import numpy as np

import dascore as dc
from dascore.utils.hdf5 import H5Reader


_DATA = "Data"


def _get_h5_attr(fi: H5Reader, key: str) -> np.ndarray:
    return fi[f"{_DATA}/{key}"][()]


def _timestamps(fi: H5Reader) -> np.ndarray:
    """Return 1-D datetime64[ns] array from the (n_time, 1) Unix-sec dataset."""
    ts = _get_h5_attr(fi, "Time").squeeze()          # (n_time,)
    return (ts * 1e9).astype("datetime64[ns]")


def _is_t1_file(fi: H5Reader) -> bool:
    """Minimal fingerprint check — T1 files always have these datasets."""
    required = {"Temperature", "Distance", "DistanceSignal", "Time"}
    present  = set(fi.get(_DATA, {}).keys())
    return required.issubset(present)


def _get_coords(fi: H5Reader):
    """Get coordinates from T1 file datasets."""

    def _get_distance_coord(fi):
        dist = fi["Data/Distance"][()]
        start, stop, step = dist[0], dist[-1], np.diff(dist).mean()
        return dc.get_coord(start=start, stop=stop+step, step=step, units="m")

    def _get_time_coord(fi):
        ts = fi["Data/Time"][()].squeeze()
        times = (ts * 1e9).astype("datetime64[ns]")
        if times.ndim == 0 or len(times) == 1:
            # the file just has a single reading
            t = times.flat[0]
            return dc.get_coord(start=t, stop=t, step=np.timedelta64(0, "ns"), units="s")
        start, stop, step = times[0], times[-1], np.diff(times).mean()
        return dc.get_coord(start=start, stop=stop+step, step=step, units="s")

    coords = dict(
        time=_get_time_coord(fi),
        distance=_get_distance_coord(fi),
    )
    return dc.get_coord_manager(coords, dims=tuple(coords))


def _get_attrs(cm, path, format, version):
    """Build PatchAttrs from an already-constructed CoordManager."""
    time_coord = cm.coord_map["time"]
    dist_coord = cm.coord_map["distance"]
    return dc.PatchAttrs(
        path=path,
        file_format=format,
        file_version=str(version),
        data_type="temperature",
        data_units="°C",
        data_category="DTS",
        instrument_make="FEBUS",
        instrument_model="T1",
        time_min=time_coord.min(),
        time_max=time_coord.max(),
        d_time=time_coord.step,
        distance_min=dist_coord.min(),
        distance_max=dist_coord.max(),
        d_distance=dist_coord.step,
        distance_units="m",
    )


def _get_t1_coords_and_attrs(fi: H5Reader, format, version) -> tuple[dc.CoordManager, dc.PatchAttrs]:
    """ Get the coordinates and attributes for a T1 data patch"""
    coords = _get_coords(fi)
    attrs = _get_attrs(coords, path=fi.filename, format=format, version=version)
    return coords, attrs


def _get_t1_patch(
    fi: H5Reader,
    format: str,
    version: str,
) -> dc.Patch:
    """Core builder shared by read() and scan()."""
    coords, attrs = _get_t1_coords_and_attrs(fi, format, version)
    temp  = _get_h5_attr(fi, "Temperature")  # (n_time, n_dist)

    return dc.Patch(data=temp, coords=coords, dims=("time", "distance"), attrs=attrs)