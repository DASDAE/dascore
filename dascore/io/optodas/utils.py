"""Utilities for OptoDAS."""

from __future__ import annotations

import dascore as dc
import dascore.core
from dascore.core.coords import get_coord
from dascore.io.utils import build_patches, get_exact_coord
from dascore.utils.misc import _maybe_unpack, unbyte

# --- Getting format/version


def _get_opto_das_version_str(hdf_fi) -> str:
    """Return the version string for OptoDAS file."""
    # define a few root attrs that act as a "fingerprint"
    expected_attrs = (
        "acqSpec",
        "header",
        "cableSpec",
        "data",
        "fileVersion",
    )
    if not all([x in hdf_fi for x in expected_attrs]):
        return ""
    version_str = str(unbyte(hdf_fi["fileVersion"][()]))
    return version_str


def _get_coord_manager(fi, snap=True):
    """Get the distance ranges and spacing."""
    header = fi["header"]
    dims = tuple(unbyte(x) for x in header["dimensionNames"])
    units = tuple(unbyte(x) for x in header["dimensionUnits"])
    coords = {}
    for index, (dim, unit) in enumerate(zip(dims, units)):
        crange = header["dimensionRanges"][f"dimension{index}"]
        step = _maybe_unpack(crange["unitScale"])

        # Special case for time.
        if dim == "time":
            step = dc.to_timedelta64(step)
            t1 = dc.to_datetime64(_maybe_unpack(header["time"]))
            start = t1 + _maybe_unpack(crange["min"]) * step
            stop = t1 + (_maybe_unpack(crange["max"]) + 1) * step
            coord = get_coord(min=start, max=stop, step=step, units=unit)
        else:  # and distance
            # The channels are ints so we multiply by step to get distance.
            distance = fi["/header/channels"][:] * step
            if snap:
                coord = get_coord(data=distance, units=unit)
            else:
                coord = get_exact_coord(distance, units=unit)
        coords[dim] = coord
    out = dascore.core.get_coord_manager(coords=coords, dims=dims)
    return out


def _get_attr_dict(header):
    """Map header info to DAS attrs."""
    attr_map = {
        "gaugeLength": "gauge_length",
        "unit": "data_units",
        "instrument": "interrogator.name",
        "experiment": "experiment",
    }
    out = {"data_category": "DAS"}
    for head_name, attr_name in attr_map.items():
        value = header[head_name]
        if hasattr(value, "shape"):
            value = _maybe_unpack(value)
        out[attr_name] = unbyte(value)
    return out


def _get_opto_das_attrs(fi, snap=True) -> tuple[dict, dascore.core.CoordManager]:
    """Scan a OptoDAS file, return metadata and coordinates."""
    cm = _get_coord_manager(fi, snap=snap)
    attrs = _get_attr_dict(fi["header"])
    return attrs, cm


def _read_opto_das(fi, distance=None, time=None, attr_cls=dc.PatchAttrs):
    """Read the OptoDAS values into a patch."""
    attrs, coords = _get_opto_das_attrs(fi)
    return build_patches(
        coords,
        fi["data"],
        attrs,
        attr_cls=attr_cls,
        selection={"time": time, "distance": distance},
    )
