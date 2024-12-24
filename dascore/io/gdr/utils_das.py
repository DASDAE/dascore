"""
Utilities functions for GDR DAS format.

See: https://gdr.openei.org/das_data_standard for more info.
"""

import numpy as np

import dascore as dc
from dascore.core import get_coord
from dascore.utils.hdf5 import extract_h5_attrs, h5_matches_structure
from dascore.utils.misc import unbyte

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
    "DasMetadata/Interrogator.SerialNumber": "instrument_id",
}


def _get_version(h5fi):
    """Get the version code of the GDR file."""
    if not h5_matches_structure(h5fi, _BASE_STRUCTURE):
        return False
    meta = h5fi["DasMetadata"].attrs
    data_fmt = meta["RawDataStandard"]
    meta_fmt = meta["MetadataStandard"]
    return "GDR_DAS", _COMPOSITE_VERSIONS[(meta_fmt, data_fmt)]


def _get_attrs_coords_and_data(resource, snap):
    """
    Get attributes, coordinates, and data from the file.
    """
    fill = {"NaN": "", "nan": ""}
    attrs = extract_h5_attrs(resource, _V1_ATTR_MAP, fill_values=fill)
    coords = _get_coord_manager(resource, snap)
    data = resource["DasRawData/RawData"]
    return attrs, coords, data


def _get_coord_manager(resource, snap=True):
    """Get a coordinate manager from the file."""

    def get_time_coord(resource, snap):
        """Get the time coordinate."""
        # TODO: I am not sure if time will always be in ns, check on it.
        time = resource["DasRawData/DasTimeArray"]
        if not snap:
            return get_coord(data=np.array(time).astype("datetime64[ns]"))
        t1 = np.int64(time[0]).astype("datetime64[ns]")
        t2 = np.int64(time[-1]).astype("datetime64[ns]")
        step = (t2 - t1) / (len(time) - 1)
        return get_coord(start=t1, stop=t2, step=step).change_length(len(time))

    def get_dist_coord(resource, length):
        """Get distance coordinates."""
        # Note: There is not enough info to correctly infer the start of
        # distance coordinate since Channels are often not included. In this
        # case we just assume the distance starts at 0 since the location of
        # each channel must be attached alter anyway. This at least includes
        # correct dx information.
        group = resource["DasMetadata/Interrogator/Acquisition"]
        dx = float(unbyte(group.attrs["SpatialSamplingInterval"]))
        units = unbyte(group.attrs["SpatialSamplingIntervalUnit"])
        start = 0
        stop = length * dx
        coord = get_coord(start=start, stop=stop, step=dx, units=units)
        return coord.change_length(length)

    def get_dims(dataset):
        """Get the dimension names."""
        das_dims = dataset.attrs["DasDimensions"]
        out = [""] * 2
        for num, dim in enumerate(das_dims):
            if dim.startswith("time"):
                out[num] = "time"
            elif dim == "locus":
                out[num] = "distance"
        assert all(out)
        return tuple(out)

    time_coord = get_time_coord(resource, snap)
    dataset = resource["DasRawData/RawData"]
    dims = get_dims(dataset)
    # Get distance coord.
    dist_axis = dims.index("distance")
    dist_length = dataset.shape[dist_axis]
    dist_coord = get_dist_coord(resource, dist_length)

    coords = {
        "time": time_coord,
        "distance": dist_coord,
    }

    return dc.get_coord_manager(coords=coords, dims=dims)


def _maybe_trim_data(cm, data, time=None, distance=None, **kwargs):
    """Maybe trim the data."""
    if time is not None or distance is not None:
        cm, data = cm.select(time=time, distance=distance, array=data)
    return cm, data
