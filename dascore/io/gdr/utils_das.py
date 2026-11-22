"""
Utilities functions for GDR DAS format.

See: https://gdr.openei.org/das_data_standard for more info.
"""

from __future__ import annotations

import warnings
from contextlib import suppress

import numpy as np

import dascore as dc
from dascore.core import get_coord
from dascore.exceptions import UnitError
from dascore.io.utils import convert_attr_units, parse_units, stored_coord
from dascore.units import convert_units
from dascore.utils.hdf5 import extract_h5_attrs, h5_matches_structure
from dascore.utils.misc import _maybe_unpack, unbyte

# This defines the metadata/rawdata version combinations that define GDR versions.
_COMPOSITE_VERSIONS = {
    ("DAS-RCN v1.10", "PRODML v2.2"): "1",
}

_BASE_STRUCTURE = (
    "DasMetadata/Interrogator/Acquisition",
    "DasRawData/DasTimeArray",
    "DasRawData/RawData",
    "DasMetadata.MetadataStandard",
    "DasMetadata.RawDataStandard",
)

# Attribute map for version 1. {current_name: new_name}
ACQ = "DasMetadata/Interrogator/Acquisition"
_V1_ATTR_MAP = {
    f"{ACQ}.GaugeLength": "gauge_length",
    f"{ACQ}.GaugeLengthUnit": "gauge_length_units",
    f"{ACQ}.UnitOfMeasure": "data_units",
    "DasMetadata/Interrogator.SerialNumber": "interrogator.serial_number",
}


def _get_version(h5fi):
    """Get the version code of the GDR file."""
    if not h5_matches_structure(h5fi, _BASE_STRUCTURE):
        return False
    meta = h5fi["DasMetadata"].attrs
    data_fmt = meta["RawDataStandard"]
    meta_fmt = meta["MetadataStandard"]
    return "GDR_DAS", _COMPOSITE_VERSIONS[(meta_fmt, data_fmt)]


def _get_dims(dataset) -> tuple[str, ...]:
    """
    The dataset's dimension names, in DASCore's spelling.

    The file states them in ``DasDimensions``, where the distance axis is
    called ``locus``.
    """
    das_dims = dataset.attrs["DasDimensions"]
    out = [""] * 2
    for num, dim in enumerate(das_dims):
        dim = unbyte(dim)
        if dim.startswith("time"):
            out[num] = "time"
        elif dim == "locus":
            out[num] = "distance"
    assert all(out), f"unexpected DasDimensions {das_dims}"
    return tuple(out)


def _get_attrs_coords_and_data(resource, snap):
    """
    Get attributes, coordinates, and data from the file.
    """
    fill = {"NaN": "", "nan": ""}
    attrs = extract_h5_attrs(resource, _V1_ATTR_MAP, fill_values=fill)
    convert_attr_units(attrs, "gauge_length", "m")
    coords = _get_coord_manager(resource, snap)
    data = resource["DasRawData/RawData"]
    return attrs, coords, data


def _get_distance_start(resource, units) -> float:
    """
    Return where the channels start along the fiber, in the spacing's units.

    GDR files state it, if at all, as the fiber's FiberStartLocation
    (PRODML's StartLocus); a NaN or missing one is 0, and a unit left
    unstated is the spacing's.
    """
    attrs = getattr(resource.get("DasMetadata/Cable/Fiber"), "attrs", {})
    stated = unbyte(_maybe_unpack(attrs.get("FiberStartLocation", "NaN")))
    try:
        start = float(stated)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(start):
        return 0.0
    from_units = unbyte(_maybe_unpack(attrs.get("FiberStartLocationUnit", "")))
    from_units = from_units or units
    if parse_units(from_units) is not None:
        with suppress(UnitError):
            return float(convert_units(start, units, from_units))
    msg = (
        f"FiberStartLocation {stated!r} is in units {from_units!r}, which do "
        f"not convert to {units!r}, so distance starts at 0."
    )
    warnings.warn(msg, UserWarning)
    return 0.0


def _nanoseconds(values) -> np.ndarray:
    """Decode stored times, integer nanoseconds since the epoch."""
    return np.asarray(values).astype(np.int64).astype("datetime64[ns]")


def _get_coord_manager(resource, snap=True):
    """Get a coordinate manager from the file."""

    def get_time_coord(resource, snap):
        """Get the time coordinate."""
        # TODO: I am not sure if time will always be in ns, check on it.
        time = resource["DasRawData/DasTimeArray"]
        return stored_coord(
            time, snap, name="time", where=resource, decode=_nanoseconds
        )

    def get_dist_coord(resource, length):
        """Get distance coordinates."""
        # Note: Channels are often not included, so the distance starts
        # where the fiber's metadata says, else at 0, since the location of
        # each channel must be attached later anyway. This at least includes
        # correct dx information.
        group = resource["DasMetadata/Interrogator/Acquisition"]
        dx = float(unbyte(group.attrs["SpatialSamplingInterval"]))
        units = unbyte(group.attrs["SpatialSamplingIntervalUnit"])
        start = _get_distance_start(resource, units)
        return get_coord(start=start, step=dx, shape=(length,), units=units)

    time_coord = get_time_coord(resource, snap)
    dataset = resource["DasRawData/RawData"]
    dims = _get_dims(dataset)
    # Get distance coord.
    dist_axis = dims.index("distance")
    dist_length = dataset.shape[dist_axis]
    dist_coord = get_dist_coord(resource, dist_length)

    coords = {
        "time": time_coord,
        "distance": dist_coord,
    }

    return dc.get_coord_manager(coords=coords, dims=dims)
