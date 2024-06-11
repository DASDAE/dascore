"""Utilities for terra15."""

from __future__ import annotations

import dascore as dc
from dascore.core import get_coord

# --- Getting format/version

_REQUIRED_GROUPS = frozenset({"channel", "trace", "das", "t", "x", "y", "z"})
_COORD_GROUPS = ("channel", "trace", "t", "x", "y", "z")


# maps attributes on DAS group to attrs stored in patch.
_ROOT_ATTR_MAPPING = {"project": "project"}
_DAS_ATTR_MAPPING = {"long_name": "data_type"}
_CRS_MAPPING = {"epsg_code": "epsg_code"}


def _get_cf_version_str(hdf_fi) -> str | bool:
    """Return the version string for dashdf5 files."""
    conventions = hdf_fi.attrs.get("Conventions", [])
    cf_str = [x for x in conventions if x.startswith("CF-")]
    das_hdf_str = [x for x in conventions if x.startswith("DAS-HDF5")]
    has_req_groups = _REQUIRED_GROUPS.issubset(set(hdf_fi))
    # if CF convention not found or not all
    if len(cf_str) == 0 or len(das_hdf_str) == 0 or not has_req_groups:
        return False
    return das_hdf_str[0].replace("DAS-HDF5-", "")


def _get_cf_coords(hdf_fi, minimal=False) -> dc.core.CoordManager:
    """
    Get a coordinate manager of full file range.

    Parameters
    ----------
    minimal
        If True, only return queryable parameters.

    """

    def _get_spatialcoord(hdf_fi, code):
        """Get spatial coord."""
        return get_coord(
            data=hdf_fi[code],
            units=hdf_fi[code].attrs["units"],
        )

    coords_map = {
        "channel": get_coord(data=hdf_fi["channel"][:]),
        "trace": get_coord(data=hdf_fi["trace"][:]),
        "time": get_coord(data=dc.to_datetime64(hdf_fi["t"][:])),
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
    dims = ("channel", "time")
    cm = dc.core.CoordManager(
        coord_map=coords_map,
        dim_map=dim_map,
        dims=dims,
    )
    # a bit of a hack to make sure data and coords align.
    if cm.shape != hdf_fi["das"].shape:
        cm = cm.transpose()
    return cm


def _get_cf_attrs(hdf_fi, coords=None, extras=None):
    """Get attributes for CF file."""
    out = {"coords": coords or _get_cf_coords(hdf_fi)}
    out.update(extras or {})
    for n1, n2 in _ROOT_ATTR_MAPPING.items():
        out[n1] = hdf_fi.attrs.get(n2)
    for n1, n2 in _DAS_ATTR_MAPPING.items():
        out[n1] = getattr(hdf_fi.get("das", {}), "attrs", {}).get(n2)
    for n1, n2 in _CRS_MAPPING.items():
        out[n1] = getattr(hdf_fi.get("crs", {}), "attrs", {}).get(n2)
    return dc.PatchAttrs(**out)
