"""Utilities for OptoDAS."""

from __future__ import annotations

import dascore as dc
import dascore.core
from dascore.core.coords import get_coord
from dascore.utils.hdf5 import unpack_scalar_h5_dataset
from dascore.utils.misc import get_path, unbyte

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


def _get_coord_manager(fi):
    """Get the distance ranges and spacing."""
    header = fi["header"]
    dims = tuple(unbyte(x) for x in header["dimensionNames"])
    units = tuple(unbyte(x) for x in header["dimensionUnits"])
    coords = {}
    for index, (dim, unit) in enumerate(zip(dims, units)):
        crange = header["dimensionRanges"][f"dimension{index}"]
        step = unpack_scalar_h5_dataset(crange["unitScale"])

        # Special case for time.
        if dim == "time":
            step = dc.to_timedelta64(step)
            t1 = dc.to_datetime64(unpack_scalar_h5_dataset(header["time"]))
            start = t1 + unpack_scalar_h5_dataset(crange["min"]) * step
            stop = t1 + (unpack_scalar_h5_dataset(crange["max"]) + 1) * step
            coord = get_coord(min=start, max=stop, step=step, units=unit)
        else:  # and distance
            # The channels are ints so we multiply by step to get distance.
            distance = fi["/header/channels"][:] * step
            coord = get_coord(values=distance)
        coords[dim] = coord
    out = dascore.core.get_coord_manager(coords=coords, dims=dims)
    return out


def _get_attr_dict(header, path, format_name, format_version):
    """Map header info to DAS attrs."""
    attr_map = {
        "gaugeLength": "gauge_length",
        "unit": "data_units",
        "instrument": "instrument_id",
        "experiment": "acquisition_id",
    }
    out = {
        "data_category": "DAS",
        "path": path,
        "format_name": format_name,
        "format_version": format_version,
    }
    for head_name, attr_name in attr_map.items():
        value = header[head_name]
        if hasattr(value, "shape"):
            value = unpack_scalar_h5_dataset(value)
        out[attr_name] = unbyte(value)
    return out


def _get_opto_das_coords_attrs(fi, format_name) -> tuple[dc.CoordManager, dict]:
    """Scan a OptoDAS file, return metadata."""
    cm = _get_coord_manager(fi)
    path = get_path(fi)
    version = _get_opto_das_version_str(fi)
    attrs = _get_attr_dict(fi["header"], path, format_name, version)
    return cm, attrs


def _read_opto_das(
    fi, distance=None, time=None, attr_cls=dc.PatchAttrs, format_name=""
):
    """Read the OptoDAS values into a patch."""
    coords, attrs = _get_opto_das_coords_attrs(fi, format_name)
    data_node = fi["data"]
    cm, data = coords.select(array=data_node, distance=distance, time=time)
    return [dc.Patch(data=data, coords=cm, attrs=attr_cls(**attrs))]
