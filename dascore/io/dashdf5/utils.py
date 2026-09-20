"""Utilities for DASHDF5."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.core import get_coord
from dascore.io.utils import get_exact_coord, should_snap
from dascore.utils.misc import maybe_get_items

# --- Getting format/version

_REQUIRED_GROUPS = frozenset({"channel", "trace", "das", "t", "x", "y", "z"})
_COORD_GROUPS = ("channel", "trace", "t", "x", "y", "z")


# maps a stored attribute's name onto the patch attr it becomes.
_ROOT_ATTR_MAPPING = {"project": "project"}
_DAS_ATTR_MAPPING = {"long_name": "data_type"}
_CRS_MAPPING = {"epsg_code": "epsg_code"}


def _get_cf_version_str(hdf_fi) -> str | Literal[False]:
    """Return the version string for dashdf5 files."""
    conventions = hdf_fi.attrs.get("Conventions", [])
    cf_str = [x for x in conventions if x.startswith("CF-")]
    das_hdf_str = [x for x in conventions if x.startswith("DAS-HDF5")]
    has_req_groups = _REQUIRED_GROUPS.issubset(set(hdf_fi))
    # if CF convention not found or not all
    if len(cf_str) == 0 or len(das_hdf_str) == 0 or not has_req_groups:
        return False
    return das_hdf_str[0].replace("DAS-HDF5-", "")


_CF_DIMS = ("channel", "time")


def _get_cf_coords(hdf_fi, minimal=False, snap=True) -> dc.core.CoordManager:
    """
    Get a coordinate manager of full file range.

    Parameters
    ----------
    minimal
        If True, only return queryable parameters.

    """

    def _coord(values, name, units=None):
        """Return a tolerant or exact coordinate from stored values."""
        values = np.asarray(values)
        if should_snap(snap, name):
            return get_coord(data=values, units=units)
        return get_exact_coord(values, units=units)

    def _get_spatialcoord(hdf_fi, code):
        """Get spatial coord."""
        return _coord(hdf_fi[code], code, units=hdf_fi[code].attrs["units"])

    coords_map = {
        "channel": _coord(hdf_fi["channel"][:], "channel"),
        "trace": _coord(hdf_fi["trace"][:], "trace"),
        "time": _coord(dc.to_datetime64(hdf_fi["t"][:]), "time"),
        "x": _get_spatialcoord(hdf_fi, "x"),
        "y": _get_spatialcoord(hdf_fi, "y"),
        "z": _get_spatialcoord(hdf_fi, "z"),
    }
    dim_map = {
        "time": ("time",),
        "trace": ("time",),
        "channel": ("channel",),
        "x": ("channel",),
        "y": ("channel",),
        "z": ("channel",),
    }
    cm = dc.core.CoordManager(
        coord_map=coords_map,
        dim_map=dim_map,
        dims=_CF_DIMS,
    )
    if cm.dims != _get_cf_dims(hdf_fi):
        cm = cm.transpose()
    return cm


def _get_cf_dims(hdf_fi) -> tuple[str, str]:
    """
    The stored dimension order of the ``das`` dataset.

    The format does not state it, so it is read off the dataset's shape:
    the channel-major order unless only the other one fits.
    """
    shape = (len(hdf_fi["channel"]), len(hdf_fi["t"]))
    return _CF_DIMS if hdf_fi["das"].shape == shape else _CF_DIMS[::-1]


def _get_cf_attrs(hdf_fi, coords=None, extras=None):
    """Get attributes for CF file."""
    out = {}
    out.update(extras or {})
    # These files spell a scalar attribute as a length-1 array, so
    # `project` is a name and `epsg_code` a number, not arrays of one.
    out.update(maybe_get_items(hdf_fi.attrs, _ROOT_ATTR_MAPPING))
    das_attrs = getattr(hdf_fi.get("das", {}), "attrs", {})
    out.update(maybe_get_items(das_attrs, _DAS_ATTR_MAPPING))
    crs_attrs = getattr(hdf_fi.get("crs", {}), "attrs", {})
    out.update(maybe_get_items(crs_attrs, _CRS_MAPPING))
    return dc.PatchAttrs.from_dict(out)
