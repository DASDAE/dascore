"""Utilities functions for Neubrex DAS IO support"""

import dascore as dc
from dascore.utils.misc import maybe_get_items, unbyte


def _is_neubrex(h5fi):
    """Determine if the file is of Neubrex origin."""
    expected_attrs = {
        "NEUBRESCOPE.DAS.Model",
        "NEUBRESCOPE.DAS.Model",
        "StartPosition",
        "StopPosition",
        "SpatialSamplingInterval",
        "GPSTimeStamp(UTC)",
        "CPUTimeStamp(UTC)",
    }
    acoustic = h5fi.get("Acoustic", None)
    attrs = set(getattr(acoustic, "attrs", set()))
    return expected_attrs.issubset(attrs)


def _get_coord_manager(acoustic):
    """Get a coordinate manager from the file."""

    def _get_time_coord(acoustic):
        """Get the time coordinate."""
        attrs = acoustic.attrs

        # We havent encountered a time decimated file yet; raise over guess
        assert attrs["TimeDecimationFilter"] in {0, 1}, "not implemented"

        gps = unbyte(attrs["GPSTimeStamp(UTC)"])
        cpu = unbyte(attrs["CPUTimeStamp(UTC)"])
        start = dc.to_datetime64(gps if gps else cpu)
        step = dc.to_timedelta64(attrs["TimeSamplingInterval(seconds)"])
        time_len = acoustic.shape[0]
        stop = start + step * time_len
        return dc.get_coord(start=start, step=step, stop=stop)

    def _get_dist_coord(acoustic):
        """Get the distance (depth) coordinate."""
        dist_len = acoustic.shape[1]
        attrs = acoustic.attrs

        # We havent encountered a distance decimated file yet; raise over guess
        assert attrs["DistanceDecimationFilter"] in {0, 1}, "not yet implemented"

        step = attrs["SpatialSamplingInterval"]
        units = dc.get_quantity(attrs["StartStopPositionUnit"])
        start = attrs["StartPosition"]
        stop = dist_len * step + start
        return dc.get_coord(start=start, step=step, units=units, stop=stop)

    coords = {
        "time": _get_time_coord(acoustic),
        "distance": _get_dist_coord(acoustic),
    }
    assert len(acoustic.shape) == 2, "Expecting 2D Neubrex array."
    return dc.get_coord_manager(coords=coords, dims=("time", "distance"))


def _get_data_units_and_type(attrs):
    """Get the units from contained string."""
    data_unit_str = unbyte(attrs["DataPhysicalUnit"]).replace("-", "")
    return dc.get_quantity(data_unit_str)


def _get_attr_dict(acoustic):
    """Get a dict of neubrex attributes."""
    mapping = {
        "GaugeLength": "gauge_length",
        "GaugeLengthUnits": "gauge_length_units",
        "IndexOfRefraction": "index_of_refraction",
        "PhaseToStrainConversion(MicroStrainPerRadian)": "phase_to_strain",
        "NEUBRESCOPE.DAS.SerialNum": "instrument_id",
        "NEUBRESCOPE.DAS.Model": "instrument_model",
        "DistanceDecimationFilter": "distance_decimation_filter",
        "TimeDecimationFilter": "time_decimation_filter",
    }
    attrs = dict(acoustic.attrs)
    out = maybe_get_items(attrs, mapping)
    out["data_units"] = _get_data_units_and_type(attrs)
    return out


def _maybe_trim_data(cm, data, time=None, distance=None, **kwargs):
    """Maybe trim the data."""
    if time is not None or distance is not None:
        cm, data = cm.select(time=time, distance=distance, array=data)
    return cm, data


def _get_attrs_coords_and_data(h5fi):
    """Return the attributes, coordinates, and data array."""
    acoustic = h5fi["Acoustic"]
    cm = _get_coord_manager(acoustic)
    attrs = _get_attr_dict(acoustic)
    return attrs, cm, acoustic
