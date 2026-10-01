"""Decode the signals and tie-point coordinates of XDAS NetCDF files."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

import dascore as dc
from dascore.core import get_coord
from dascore.core.source import ArraySource
from dascore.exceptions import InvalidFiberFileError
from dascore.io.netcdf.utils import XDAS_MAPPINGS, XDAS_TILING
from dascore.io.utils import cf_time_values, should_snap
from dascore.utils.misc import _maybe_unpack, unbyte

# The stand-in netCDF writes for a dimension with no coordinate variable.
_PLACEHOLDER = "This is a netCDF dimension but not a netCDF variable"


def _attr(attrs, name, default=""):
    """Return one attr as a plain scalar or string."""
    return unbyte(_maybe_unpack(attrs.get(name, default)))


def _groups(text) -> list[tuple[str, list[str]]]:
    """Split ``"label: ref ref label: ref"`` into (label, refs) pairs."""
    out = []
    for word in str(text).split():
        if word.endswith(":"):
            out.append((word[:-1], []))
        else:
            out[-1][1].append(word)
    return out


def _values(node) -> np.ndarray:
    """Return a stored coordinate's values, decoding CF time."""
    if " since " in str(_attr(node.attrs, "units")):
        return cf_time_values(node)
    return np.atleast_1d(node[()])


def _units(node):
    """Return a coordinate's units, None for a CF time reference."""
    units = _attr(node.attrs, "units") or None
    return None if units and " since " in units else units


def _step(delta, count, dtype):
    """Return `delta` spread over `count` samples as get_coord takes a step."""
    if dtype.kind == "M":  # delta in ns; an exact fraction of a second
        return Fraction(int(delta), int(count) * 10**9)
    if dtype.kind in "iu":
        return Fraction(int(delta), int(count))
    return float(delta) / float(count)


def _join(starts, counts, steps, units):
    """Join evenly sampled pieces; a lone sample continues its predecessor."""
    pieces, step = [], None
    for start, count, own in zip(starts, counts, steps, strict=True):
        step = own if count > 1 or step is None else step
        pieces.append(get_coord(start=start, step=step, shape=int(count)))
    return get_coord(segments=pieces).set_units(units)


def _tie_point_coord(group, values_name, indices_name, length, rate=None):
    """
    Decode one tie-point coordinate.

    Without a `rate`, the indices are tie points interpolated linearly;
    with one, they are the lengths of segments sampled at that rate.
    """
    values, node = _values(group[values_name]), group[values_name]
    indices = np.atleast_1d(group[indices_name][()])
    if values.dtype.kind == "f":  # float32 steps would drift across a piece
        values = values.astype(np.float64)
    if rate is not None:
        if indices.sum() != length:
            msg = f"Segment lengths of {values_name} do not sum to {length}."
            raise InvalidFiberFileError(msg)
        interval, denominator = rate
        step = _step(interval, denominator, values.dtype)
        return _join(values, indices, [step] * len(values), _units(node))
    if indices[0] != 0 or indices[-1] != length - 1 or np.any(np.diff(indices) < 1):
        msg = f"Tie points of {values_name} do not span its {length} samples."
        raise InvalidFiberFileError(msg)
    if length == 1:
        return get_coord(data=values, units=_units(node))
    deltas = np.diff(values)
    spans = np.diff(indices)
    steps = [_step(d, s, values.dtype) for d, s in zip(deltas, spans, strict=True)]
    # Each pair of tie points starts a piece; the last tie point ends the last.
    return _join(values, [*spans, 1], [*steps, None], _units(node))


def _rate(descriptor, legacy: bool) -> tuple[Any, Any]:
    """Return the (interval, denominator) a sampling descriptor states."""
    attrs = descriptor.attrs
    if legacy:  # the interval was the descriptor's own value
        value, units, denominator = descriptor[()], _attr(attrs, "units"), 1
    else:
        key = "sampling_interval"
        if "sampling_denominator" in attrs:
            key = "sampling_numerator"
        value, units = _attr(attrs, key), _attr(attrs, f"{key}_units")
        denominator = _attr(attrs, "sampling_denominator", 1)
    if units:
        value = pd.to_timedelta(value, unit=units).value
    return value, denominator


def _specs(group, node):
    """
    Yield (name, dim, values, indices, rate) for each tie-point coordinate.

    Both spellings are read: the original names the dimension and both
    tie-point variables, the CF-shaped one names the tie-point values and
    a descriptor whose ``tie_point_mapping`` names the rest.
    """
    for label, refs in _groups(_attr(node.attrs, "coordinate_interpolation")):
        if len(refs) == 2:
            yield label, label, refs[1], refs[0], None
            continue
        dim, (indices, _) = _groups(_attr(group[refs[0]].attrs, "tie_point_mapping"))[0]
        yield refs[0].removesuffix("_interpolation"), dim, label, indices, None
    for label, (ref,) in _groups(_attr(node.attrs, "coordinate_sampling")):
        descriptor = group[ref]
        dim, refs = _groups(_attr(descriptor.attrs, "tie_point_mapping"))[0]
        legacy = "sampling_interval" not in descriptor.attrs
        name, values, lengths = (label, *refs) if legacy else (ref, label, refs[0])
        rate = _rate(descriptor, legacy)
        yield name.removesuffix("_sampling"), dim, values, lengths, rate


def _dims(node) -> tuple[str, ...]:
    """Return the names of a dataset's netCDF dimensions."""
    return tuple(node.dims[i][0].name.rsplit("/", 1)[-1] for i in range(node.ndim))


def _signal_coords(group, node, aux, snap):
    """Return the coordinates of one signal: tie-point, stored, or positional."""
    dims = _dims(node)
    coords = {}
    for name, dim, values, indices, rate in _specs(group, node):
        length = node.shape[dims.index(dim)]
        coords[name] = (dim, _tie_point_coord(group, values, indices, length, rate))
    stored = [(x, _dims(x)) for x in aux if x.ndim == 1]
    for axis, dim in enumerate(dims):
        if dim in coords:
            continue
        scale = node.dims[axis][0]
        if str(_attr(scale.attrs, "NAME")).startswith(_PLACEHOLDER):
            coords[dim] = (dim, get_coord(start=0, stop=node.shape[axis], step=1))
        else:
            stored.append((scale, (dim,)))
    for other, other_dims in stored:
        if set(other_dims) <= set(dims):
            name = other.name.rsplit("/", 1)[-1]
            coord = get_coord(
                data=_values(other), units=_units(other), snap=should_snap(snap, name)
            )
            coords[name] = (other_dims, coord)
    return dc.get_coord_manager(coords=coords, dims=dims)


def _signal_attrs(node) -> dict:
    """Return a signal's own attrs, without netCDF or layout bookkeeping."""
    skip = {*XDAS_MAPPINGS, "DIMENSION_LIST", "coordinates"}
    return {
        name: _attr(node.attrs, name)
        for name in node.attrs
        if name not in skip and not name.startswith("_")
    }


def _iter_signals(group):
    """Yield (group, dataset, auxiliary coordinates) for every signal."""
    datasets = [x for x in group.values() if isinstance(x, h5py.Dataset)]
    helpers, aux = set(), set()
    for node in [group, *datasets]:
        aux |= set(_attr(node.attrs, "coordinates").split())
        if node is not group:
            for _, _, values, indices, _ in _specs(group, node):
                helpers |= {values, indices}
    for node in datasets:
        name = node.name.rsplit("/", 1)[-1]
        if XDAS_TILING in node.attrs:
            raise NotImplementedError("XDAS tile manifests are not supported.")
        is_scale = _attr(node.attrs, "CLASS") == "DIMENSION_SCALE"
        if node.ndim and not is_scale and name not in helpers | aux:
            yield group, node, {group[x] for x in aux if x in group}
    for child in group.values():
        if isinstance(child, h5py.Group):
            yield from _iter_signals(child)


def get_pieces(h5, snap=True) -> dict[str, tuple]:
    """
    Map each patch key to its (metadata, dataset, sample offsets).

    A signal whose tie points leave a hole, or change the sampling step,
    is one patch per evenly sampled stretch, keyed ``<path>#<number>``.
    """
    out = {}
    for group, node, aux in _iter_signals(h5):
        key = node.name.lstrip("/")
        cm = _signal_coords(group, node, aux, snap)
        attrs = _signal_attrs(node)
        bounds = []
        for dim in cm.dims:
            coord = cm.coord_map[dim]
            counts = [len(x) for x in getattr(coord, "segments", (coord,))]
            stops = np.cumsum(counts)
            bounds.append(list(zip(stops - counts, stops, strict=True)))
        windows = list(product(*bounds))
        for number, window in enumerate(windows):
            piece = dict(zip(cm.dims, (slice(*x) for x in window), strict=True))
            name = key if len(windows) == 1 else f"{key}#{number}"
            meta = dc.PatchMeta(
                attrs=attrs,
                coords=cm.isel(piece)[0],
                dims=cm.dims,
                dtype=str(node.dtype),
                source=ArraySource(key=name),
            )
            out[name] = (meta, node, tuple(x[0] for x in window))
    return out


def check_virtual_sources(node):
    """Refuse a virtual dataset whose source file is gone; HDF5 fills it."""
    if not node.is_virtual:
        return
    folder = Path(node.file.filename).parent
    for source in node.virtual_sources():
        name = unbyte(source.file_name)
        if not (folder / name).exists():
            msg = f"{node.name} reads from {name}, which does not exist."
            raise FileNotFoundError(msg)
