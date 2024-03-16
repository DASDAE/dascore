"""Utilities for Febus."""
from __future__ import annotations

import dascore as dc
import dascore.core
from dascore.core.coords import get_coord
from dascore.utils.hdf5 import unpack_scalar_h5_dataset
from dascore.utils.misc import unbyte

# --- Getting format/version


def _get_febus_version_str(hdf_fi) -> str:
    """Return the version string for febus file."""
    # define a few root attrs that act as a "fingerprint"
    # all Febus DAS files have folders that start with fa
    inst_keys = sorted(hdf_fi.keys())
    expected_source_attrs = {
        "AmpliPower",
        "Hostname",
        "WholeExtent",
        "SamplingRate",
    }
    # iterate instrument keys
    is_febus = all([x.startswith("fa") for x in inst_keys])
    # Version 1, or what I think is version one (eg Valencia PubDAS data)
    # did not include a Version attr in Source dataset, so we use that as
    # the default.
    version = "1"
    for inst_key in inst_keys:
        inst = hdf_fi[inst_key]
        source_keys = set(inst.keys())
        is_febus = is_febus and all(x.startswith("Source") for x in source_keys)
        for source_key in source_keys:
            source = inst[source_key]
            # If the version is set in a Source use that version.
            # Hopefully this is the file version...
            version = source.attrs.get("Version", version)
            is_febus = is_febus and expected_source_attrs.issubset(set(source.attrs))
    if is_febus:
        return version
    return ""


def _get_coord_manager(header):
    """Get the distance ranges and spacing."""
    dims = tuple(unbyte(x) for x in header["dimensionNames"])
    units = tuple(unbyte(x) for x in header["dimensionUnits"])

    coords = {}
    for index, (dim, unit) in enumerate(zip(dims, units)):
        crange = header["dimensionRanges"][f"dimension{index}"]
        step = unpack_scalar_h5_dataset(crange["unitScale"])

        # special case for time.
        if dim == "time":
            step = dc.to_timedelta64(step)
            t1 = dc.to_datetime64(unpack_scalar_h5_dataset(header["time"]))
            start = t1 + unpack_scalar_h5_dataset(crange["min"]) * step
            stop = t1 + (unpack_scalar_h5_dataset(crange["max"]) + 1) * step
        else:
            # The min/max values appear to be int ranges so we need to
            # multiply by step.
            start = unpack_scalar_h5_dataset(crange["min"]) * step
            stop = (unpack_scalar_h5_dataset(crange["max"]) + 1) * step

        coords[dim] = get_coord(min=start, max=stop, step=step, units=unit)
    return dascore.core.get_coord_manager(coords=coords, dims=dims)


def _get_attr_dict(header):
    """Map header info to DAS attrs."""
    attr_map = {
        "gaugeLength": "gauge_length",
        "unit": "data_units",
        "instrument": "intrument_id",
        "experiment": "acquisition_id",
    }
    out = {"data_category": "DAS"}
    for head_name, attr_name in attr_map.items():
        value = header[head_name]
        if hasattr(value, "shape"):
            value = unpack_scalar_h5_dataset(value)
        out[attr_name] = unbyte(value)
    return out


def _get_febus_attrs(fi) -> dict:
    """Scan a febus file, return metadata."""
    header = fi["header"]
    cm = _get_coord_manager(header)
    attrs = _get_attr_dict(header)
    attrs["coords"] = cm
    return attrs


def _read_febus(fi, distance=None, time=None, attr_cls=dc.PatchAttrs):
    """Read the febus values into a patch."""
    attrs = _get_febus_attrs(fi)
    data_node = fi["data"]
    coords = attrs.pop("coords")
    cm, data = coords.select(array=data_node, distance=distance, time=time)
    attrs["coords"] = cm.to_summary_dict()
    attrs["dims"] = cm.dims
    return [dc.Patch(data=data, coords=cm, attrs=attr_cls(**attrs))]
