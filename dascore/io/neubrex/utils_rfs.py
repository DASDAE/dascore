"""Utilities functions for Neubrex IO support"""

import dascore as dc
from dascore.io.utils import get_exact_coord, should_snap
from dascore.utils.misc import maybe_get_items


def _is_neubrex(h5fi):
    """Determine if the file is of Neubrex origin."""
    expected_keys = {"data", "depth", "stamps"}
    keys = set(h5fi.keys())
    if not expected_keys.issubset(keys):
        return False
    expected_attrs = {"DataUnitLabel", "StartDateTime", "EndDateTime"}
    data_attrs = set(h5fi["data"].attrs)
    if expected_attrs.issubset(data_attrs):
        return True


def _get_coord_manager(h5fi, snap=True):
    """Get a coordinate manager from the file."""

    def _get_time_coord(h5fi, snap):
        """Get the time coordinate."""
        # Unix stamps are in us for test files, not sure if always true.
        unix_stamps = dc.to_datetime64(h5fi["stamps_unix"][:] / 1_000_000)
        if should_snap(snap, "time"):
            time_coord = dc.get_coord(data=unix_stamps).snap()
        else:
            time_coord = get_exact_coord(unix_stamps)
        return time_coord

    def _get_dist_coord(h5fi):
        """Get the distance (depth) coordinate."""
        depth = h5fi["depth"][:]
        return (
            dc.get_coord(data=depth)
            if should_snap(snap, "distance")
            else get_exact_coord(depth)
        )

    coords = {
        "time": _get_time_coord(h5fi, snap=snap),
        "distance": _get_dist_coord(h5fi),
    }
    return dc.get_coord_manager(coords=coords, dims=("time", "distance"))


def _get_data_units_and_type(data_unit_label):
    """Get the units from contained string."""
    quantity = dc.get_quantity(data_unit_label.replace("-", ""))
    return quantity


def _get_attr_dict(h5fi):
    """Get a dict of neubrex attributes."""
    mapping = {
        "API": "api",
        # "DataUnitLabel": "data_unit_label",
        "FieldName": "field_name",
        "WellID": "well_id",
        "WellName": "well_name",
        "WellBoreID": "well_bore_id",
    }
    data_attrs = dict(h5fi["data"].attrs)
    out = maybe_get_items(data_attrs, mapping)
    out["data_units"] = _get_data_units_and_type(data_attrs["DataUnitLabel"])
    return out
