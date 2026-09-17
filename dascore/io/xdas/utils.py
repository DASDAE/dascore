"""Decode XDAS's coordinate metadata without depending on XDAS itself."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import pairwise
from math import gcd
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

import dascore as dc
from dascore.core.coords import _validate_segment_chain
from dascore.exceptions import CoordError, MissingOptionalDependencyError
from dascore.io.utils import get_exact_coord
from dascore.utils.hdf5 import get_h5py_file
from dascore.utils.misc import optional_import, unbyte

_MAPPING_ATTRS = ("coordinate_interpolation", "coordinate_sampling")
_TIME_UNITS: dict[str, Literal["D", "h", "m", "s", "ms", "us", "ns"]] = {
    "days": "D",
    "hours": "h",
    "minutes": "m",
    "seconds": "s",
    "milliseconds": "ms",
    "microseconds": "us",
    "nanoseconds": "ns",
}


def iter_groups(group):
    """Walk stored groups, keeping collection paths as native identifiers."""
    yield group
    for node in group.values():
        if isinstance(node, h5py.Group):
            yield from iter_groups(node)


def is_xdas_file(resource) -> bool:
    """Recognize XDAS metadata without opening xarray or reading samples."""
    root_has_arrays = any(isinstance(node, h5py.Dataset) for node in resource.values())
    for group in iter_groups(resource):
        if "CF-" not in unbyte(group.attrs.get("Conventions", "")):
            continue
        # Collection leaves carry CF metadata even when the root does not.
        if group.name != "/" and not root_has_arrays:
            return True
        for name, node in group.items():
            if isinstance(node, h5py.Dataset) and (
                name == "__values__"
                or any(key in node.attrs for key in _MAPPING_ATTRS)
                or "__tiling__" in node.attrs
            ):
                return True
    return False


def _mapping_groups(mapping):
    """Parse labelled groups, retaining the actual referenced variable names."""
    groups = []
    for word in mapping.split():
        if word.endswith(":"):
            groups.append((word[:-1], []))
        elif groups:
            groups[-1][1].append(word)
        else:
            raise ValueError(f"Invalid XDAS coordinate mapping: {mapping!r}")
    if not groups or any(not name or not refs for name, refs in groups):
        raise ValueError(f"Invalid XDAS coordinate mapping: {mapping!r}")
    return groups


@dataclass(frozen=True)
class CoordSpec:
    """References describing one compressed coordinate of a signal."""

    name: str
    dim: str
    values: str
    indices: str
    descriptor: str | None
    sampled: bool = False
    legacy_sampled: bool = False


def _coordinate_specs(dataset, variable):
    """Resolve both XDAS's original grammar and the newer CF-shaped grammar."""
    specs = []
    for attr in _MAPPING_ATTRS:
        if attr not in variable.attrs:
            continue
        sampled = attr == "coordinate_sampling"
        for label, refs in _mapping_groups(variable.attrs[attr]):
            if len(refs) == 2 and not sampled:
                # Older XDAS: dimension: index_variable value_variable.
                indices, values = refs
                descriptor = f"{label}_interpolation"
                descriptor = descriptor if descriptor in dataset else None
                spec = CoordSpec(label, label, values, indices, descriptor)
            elif len(refs) == 1:
                descriptor = refs[0]
                meta = dataset[descriptor].attrs
                mapping = _mapping_groups(meta["tie_point_mapping"])
                if len(mapping) != 1 or len(mapping[0][1]) != 2:
                    raise ValueError(
                        "XDAS requires a one-dimensional coordinate mapping"
                    )
                dim, (indices, points) = mapping[0]
                legacy = sampled and "sampling_interval" not in meta
                name = (
                    label
                    if legacy
                    else descriptor.removesuffix(
                        "_sampling" if sampled else "_interpolation"
                    )
                )
                values, indices = (indices, points) if legacy else (label, indices)
                spec = CoordSpec(
                    name, dim, values, indices, descriptor, sampled, legacy
                )
            else:
                raise ValueError(
                    f"Unsupported XDAS coordinate mapping: {label}: {refs}"
                )
            for key in (spec.values, spec.indices):
                if key not in dataset:
                    raise ValueError(
                        f"XDAS coordinate {spec.name!r} references missing {key!r}"
                    )
            specs.append(spec)
    return specs


@contextmanager
def open_signals(resource):
    """Open leaf datasets over the managed handle and yield their signal map."""
    xr = optional_import("xarray")
    optional_import("h5netcdf")
    handle = get_h5py_file(resource)
    with ExitStack() as stack:
        signals = {}
        groups = iter_groups(handle) if is_xdas_file(handle) else (handle,)
        for group in groups:
            if "CF-" not in unbyte(group.attrs.get("Conventions", "")):
                continue
            dataset = stack.enter_context(
                xr.open_dataset(
                    handle,
                    engine="h5netcdf",
                    group=group.name,
                    decode_timedelta=False,
                    mask_and_scale=False,
                )
            )
            specs = {
                name: _coordinate_specs(dataset, var)
                for name, var in dataset.data_vars.items()
            }
            helpers = {
                ref
                for entries in specs.values()
                for spec in entries
                for ref in (spec.values, spec.indices, spec.descriptor)
                if ref is not None
            }
            for name, variable in dataset.data_vars.items():
                if name in helpers:
                    continue
                if "__tiling__" in variable.attrs:
                    raise NotImplementedError("XDAS tile manifests are not supported")
                stored_name = "__values__" if name is None else name
                key = f"{group.name.rstrip('/')}/{stored_name}"
                signals[key] = (dataset, variable, specs[name], group[stored_name])
        yield signals


def _ramp(start, numerator, denominator, length, dtype):
    """Interpolate with XDAS's nearest-even rounding for integer/time labels."""
    if np.issubdtype(dtype, np.floating):
        return np.asarray(
            float(start) + np.arange(length) * (float(numerator) / denominator),
            dtype=dtype,
        )
    temporal = np.issubdtype(dtype, np.datetime64)
    origin = (
        int(start.astype("datetime64[ns]").astype(np.int64)) if temporal else int(start)
    )
    # Reduce the ratio before multiplying. Normal grids use int64 temporaries;
    # an extreme span still uses Python integers to avoid intermediate overflow.
    common = gcd(int(numerator), int(denominator))
    numerator, denominator = int(numerator) // common, int(denominator) // common
    safe = abs(numerator) * max(0, length - 1) < np.iinfo(np.int64).max // 2
    products = np.arange(length, dtype=np.int64 if safe else object) * numerator
    quotient = products // int(denominator)
    remainder = products % int(denominator)
    increment = (2 * remainder > denominator) | (
        (2 * remainder == denominator) & (quotient % 2 != 0)
    )
    result = np.asarray(
        quotient + increment + origin, dtype=np.int64 if temporal else dtype
    )
    return result.astype("datetime64[ns]") if temporal else result


def _coord_segment(start, numerator, denominator, length, dtype, units, endpoint=None):
    """Keep integer/time grids compact when their step is an exact whole tick."""
    if (
        not np.issubdtype(dtype, np.floating)
        and numerator % denominator == 0
        and numerator != 0
    ):
        step = int(numerator) // int(denominator)
        if np.issubdtype(dtype, np.datetime64):
            step = np.timedelta64(step, "ns")
        return dc.get_coord(start=start, step=step, shape=length, units=units)
    values = _ramp(start, numerator, denominator, length, dtype)
    if endpoint is not None:
        values[-1] = endpoint
    return get_exact_coord(values, units=units)


def _join_segments(pieces, units):
    """Join compact runs without sorting the file's labels into a new order."""
    try:
        _validate_segment_chain(tuple(pieces))
        return dc.get_coord(segments=pieces)
    except CoordError:
        return get_exact_coord(
            np.concatenate([piece.values for piece in pieces]), units=units
        )


def interpolate_coord(
    values: np.ndarray, indices: np.ndarray, length: int, *, compact=False, units=None
):
    """Expand linear tie points, preserving endpoints and discontinuities."""
    if length == 0 and not len(values) and not len(indices):
        return get_exact_coord(values, units=units) if compact else values
    if (
        indices.ndim != 1
        or values.ndim != 1
        or len(indices) != len(values)
        or not np.issubdtype(indices.dtype, np.integer)
        or not len(indices)
        or indices[0] != 0
        or indices[-1] != length - 1
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            "Invalid XDAS tie points: indices must span the dimension "
            "in increasing order"
        )
    pieces = []
    temporal = np.issubdtype(values.dtype, np.datetime64)
    for index, (left, right) in enumerate(pairwise(indices)):
        span = int(right) - int(left)
        if temporal:
            delta = int(values[index + 1].astype(np.int64)) - int(
                values[index].astype(np.int64)
            )
        elif np.issubdtype(values.dtype, np.integer):
            delta = int(values[index + 1]) - int(values[index])
        else:
            delta = float(values[index + 1]) - float(values[index])
        count = span + (index == len(indices) - 2)
        pieces.append(
            _coord_segment(
                values[index],
                delta,
                span,
                count,
                values.dtype,
                units,
                endpoint=values[index + 1] if index == len(indices) - 2 else None,
            )
        )
    if not pieces:
        out = get_exact_coord(values, units=units)
    else:
        out = _join_segments(pieces, units)
    return out if compact else out.values


def _sampled_values(dataset, spec, values, lengths, length):
    """Expand constant-rate segments using their stored rational sampling rate."""
    coord_units = dataset[spec.values].attrs.get("units")
    if length == 0 and not len(values) and not len(lengths):
        return get_exact_coord(values, units=coord_units)
    meta = dataset[spec.descriptor].attrs
    if spec.legacy_sampled:
        numerator = dataset[spec.descriptor].values[()]
        units = meta.get("units")
        denominator = 1
    else:
        key = (
            "sampling_numerator"
            if "sampling_denominator" in meta
            else "sampling_interval"
        )
        numerator = meta[key]
        units = meta.get(f"{key}_units")
        denominator = meta.get("sampling_denominator", 1)
    if units:
        numerator = (
            np.timedelta64(int(numerator), _TIME_UNITS[units])
            .astype("timedelta64[ns]")
            .astype(np.int64)
        )
    if (
        lengths.ndim != 1
        or values.ndim != 1
        or len(lengths) != len(values)
        or not np.issubdtype(lengths.dtype, np.integer)
        or np.any(lengths <= 0)
        or sum(map(int, lengths)) != length
        or denominator <= 0
    ):
        raise ValueError(
            "Invalid XDAS sampled coordinate lengths or sampling denominator"
        )
    pieces = [
        _coord_segment(
            value, numerator, denominator, int(count), values.dtype, coord_units
        )
        for value, count in zip(values, lengths, strict=True)
    ]
    return _join_segments(pieces, coord_units)


def get_coords(dataset, variable, specs, snap=True):
    """Build coordinates identically for scans and reads, including auxiliaries."""
    coords = {}
    for name, coord in variable.coords.items():
        values = coord.values
        units = coord.attrs.get("units")
        # Explicit stored labels are authoritative, even when nearly regular.
        coords[name] = (coord.dims, get_exact_coord(values, units=units))
    for spec in specs:
        values = dataset[spec.values].values
        if np.issubdtype(values.dtype, np.datetime64):
            values = values.astype("datetime64[ns]")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"XDAS coordinate {spec.name!r} contains non-finite tie values"
            )
        indices = dataset[spec.indices].values
        length = variable.sizes[spec.dim]
        if spec.sampled:
            expanded = _sampled_values(dataset, spec, values, indices, length)
        else:
            if (
                spec.descriptor
                and dataset[spec.descriptor].attrs.get("interpolation_name", "linear")
                != "linear"
            ):
                raise NotImplementedError(
                    "Only linear XDAS coordinate interpolation is supported"
                )
            expanded = interpolate_coord(
                values,
                indices,
                length,
                compact=True,
                units=dataset[spec.values].attrs.get("units"),
            )
        # Tie points define the labels; snapping must not change their spacing.
        coords[spec.name] = (
            (spec.dim,),
            expanded,
        )
    for dim, length in variable.sizes.items():
        if dim not in coords:
            coords[dim] = dc.get_coord(start=0, step=1, shape=length)
    return dc.get_coord_manager(coords=coords, dims=variable.dims)


def get_attrs(variable, key):
    """Keep user attrs and the stable source key, consuming storage metadata."""
    attrs = {
        name: value
        for name, value in variable.attrs.items()
        if name not in _MAPPING_ATTRS
        and not (name.startswith("__") and name.endswith("__"))
    }
    return attrs | {"_source_patch_key": key}


def require_filters(node):
    """Register optional HDF5 codecs only when the signal needs them."""
    if node.is_virtual:
        # HDF5 silently substitutes fill values for missing virtual sources.
        # Check the sources and register their codecs before reading the VDS.
        for source in node.virtual_sources():
            filename = unbyte(source.file_name)
            if filename == ".":
                require_filters(node.file[source.dset_name])
            else:
                path = Path(filename)
                if not path.is_absolute():
                    path = Path(node.file.filename).parent / path
                with h5py.File(path, "r") as source_file:
                    require_filters(source_file[source.dset_name])
    properties = node.id.get_create_plist()
    missing = [
        properties.get_filter(i)[0]
        for i in range(properties.get_nfilters())
        if not h5py.h5z.filter_avail(properties.get_filter(i)[0])
    ]
    if missing:
        optional_import(
            "hdf5plugin",
            required_for="reading XDAS signals compressed with optional HDF5 filters",
        )
        if any(not h5py.h5z.filter_avail(code) for code in missing):
            raise MissingOptionalDependencyError(
                f"XDAS signal requires unavailable HDF5 filters {missing}"
            )
