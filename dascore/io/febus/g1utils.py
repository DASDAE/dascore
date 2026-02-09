"""
Utilities for working with Febus' G1 DSTS system files.
"""

from pathlib import Path

import numpy as np

import dascore as dc

param_parse_dict = {
    "start time": lambda x: dc.to_datetime64(float(x[0])),
    "end time": lambda x: dc.to_datetime64(float(x[0])),
    "mode": lambda x: str(x[0]),
    "channel": lambda x: int(x[0]),
}

# Values in the attr dict to not
attr_exclude = frozenset({"start_time", "end_time", "fiberfrom", "fiberto"})


def _make_param_dict(resource):
    """Yield the parameters out from the header in the file."""
    resource.seek(0)  # reset resource position to iterate from start.
    out = {}
    for num, line in enumerate(resource):
        current = line.strip()
        split = current.split(";")
        # We have reached the end of the params.
        if not line.startswith("Param;") or len(split) < 3:
            out["_data_start_line"] = num
            break
        _, name, *vals = current.split(";")
        key_name = name.lower().replace(" ", "_")
        # Handle special parsing or just convert to float
        if name in param_parse_dict:
            out[key_name] = param_parse_dict[name](vals)
        else:
            out[key_name] = float(vals[0])
    suffix = Path(getattr(resource, "name", "")).suffix
    out["_spectra"] = True if suffix == ".mtx" else False
    out["data_units"] = "microstrain" if out["mode"] == "strain" else "celsius"
    return out


def _is_g1_file(resource) -> bool:
    """Get the format tuple for a potential febus G1 file or return False."""
    name = Path(getattr(resource, "name", "")).stem
    if len(split := name.split("_")) != 3:
        return False
    _, _, time_str = split
    try:
        # We only need to know that part of the date string is valid; later
        # we read the time stamps directory.
        _ = dc.to_datetime64(time_str.replace(".", ":")[:19])
    except (ValueError, IndexError):
        return False
    # Next read the first few lines, ensure they start with param;
    first_lines = []
    for _ in range(3):
        line = resource.readline()
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        first_lines.append(line)
    return all((x.startswith("Param;")) for x in first_lines)


def _get_coords(params):
    """Get coordinates from header info."""

    def _get_distance_coord(params):
        """The fiber from/to/sampling defines the length in m"""
        start = params["fiberfrom"]
        end = params["fiberto"]
        step = params["sampling_resolution"]
        return dc.get_coord(start=start, stop=end, step=step, units="m")

    def _get_time_coord(params):
        """The time coord is really just one value."""
        t1 = params["start_time"]
        t2 = params["end_time"]
        step = t2 - t1
        return dc.get_coord(start=t1, stop=t2, step=step)

    spectra = params["_spectra"]
    if spectra:
        msg = "DASCore cannot yet parse spectra Febus G1 files."
        raise NotImplementedError(msg)
    coords = dict(
        distance=_get_distance_coord(params),
        time=_get_time_coord(params),
    )
    return dc.get_coord_manager(coords, dims=tuple(coords))


def _get_attrs(params):
    """Get the attributes from header info."""
    return {i: v for (i, v) in params.items() if i not in attr_exclude}


def _get_g1_coords_and_attrs(resource):
    """Placeholder parser for g1 scan/read paths."""
    resource_name = Path(getattr(resource, "name", "")).stem
    name, channel, _ = resource_name.split("_")
    params = _make_param_dict(resource)
    params["instrument_id"] = name
    params["data_type"] = params.pop("mode", None)
    params["data_category"] = "DSS"

    coords = _get_coords(params)
    attrs = _get_attrs(params)
    return coords, attrs


def _get_g1_patch(resource, attr_cls):
    """Get a patch from the g1 file."""
    coords, attrs = _get_g1_coords_and_attrs(resource)
    data_start_line = int(attrs.pop("_data_start_line", 0))
    resource.seek(0)
    data = np.loadtxt(resource, skiprows=data_start_line)
    data = np.asarray(data).reshape(coords.shape)
    attrs = attr_cls(**attrs)
    return dc.Patch(data=data, coords=coords, attrs=attrs)
