"""Decode the signals and tie-point coordinates of XDAS NetCDF files."""

from __future__ import annotations

import os
from fractions import Fraction
from itertools import pairwise, product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import dascore as dc
from dascore.core import get_coord
from dascore.core.source import ArraySource
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import cf_time_values, should_snap
from dascore.utils.misc import _maybe_unpack, unbyte

_MAPPINGS = ("coordinate_interpolation", "coordinate_sampling")
_TILING = "__tiling__"
_FILLS = ("_FillValue", "missing_value")
_PACKING = ("scale_factor", "add_offset")
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
        elif out:
            out[-1][1].append(word)
        else:
            raise InvalidFiberFileError(f"Cannot parse the mapping {text!r}.")
    return out


def _delta(attrs, key):
    """Return a spacing stated as ``key`` (and ``key_units``), in ns if timed."""
    value, units = _attr(attrs, key), _attr(attrs, f"{key}_units")
    return pd.to_timedelta(value, unit=units).value if units else value


def _specs(group, node):
    """
    Yield (name, dim, values, indices, sampled, exact step, tolerance).

    Reads the original ``dim: indices values`` spelling, the CF-shaped
    ``name_values: name_interpolation`` one, and sampled segments. Any
    other mapping, such as CF interpolation this layout does not write,
    is refused.
    """
    for label, refs in _groups(_attr(node.attrs, "coordinate_interpolation")):
        if len(refs) == 2:
            yield label, label, refs[1], refs[0], False, None, 0
            continue
        name = refs[0].removesuffix("_interpolation") if len(refs) == 1 else ""
        meta = group[refs[0]].attrs if name and refs[0] in group else {}
        mapping = _groups(_attr(meta, "tie_point_mapping"))
        linear = _attr(meta, "interpolation_name") == "linear"
        shape = [len(x[1]) for x in mapping]
        if label != f"{name}_values" or not linear or shape != [2]:
            msg = f"Only XDAS's linear tie points are supported, not {label!r}."
            raise NotImplementedError(msg)
        dim, (indices, _) = mapping[0]
        step = None
        if "sampling_denominator" in meta:
            step = Fraction(_delta(meta, "sampling_numerator"))
            step /= _attr(meta, "sampling_denominator")
        elif "sampling_interval" in meta:
            step = Fraction(_delta(meta, "sampling_interval"))
        tolerance = _delta(meta, "tolerance") if "tolerance" in meta else 0
        yield name, dim, label, indices, False, step, tolerance
    for label, (ref,) in _groups(_attr(node.attrs, "coordinate_sampling")):
        descriptor = group[ref]
        attrs = descriptor.attrs
        dim, refs = _groups(_attr(attrs, "tie_point_mapping"))[0]
        if "sampling_interval" not in attrs:  # the interval was its own value
            interval = descriptor[()]
            if units := _attr(attrs, "units"):
                interval = pd.to_timedelta(interval, unit=units).value
            yield label, dim, *refs, True, Fraction(interval), 0
            continue
        key = "sampling_interval"
        if "sampling_denominator" in attrs:
            key = "sampling_numerator"
        step = Fraction(_delta(attrs, key)) / _attr(attrs, "sampling_denominator", 1)
        yield ref.removesuffix("_sampling"), dim, label, refs[0], True, step, 0


def is_xdas_file(h5) -> bool:
    """Return True if a NetCDF-4 file holds a signal with an XDAS mapping."""
    return "_NCProperties" in h5.attrs and _has_signal(h5)


def _has_signal(group) -> bool:
    """Search a group's datasets or, if it holds only groups, those groups."""
    nodes = list(group.values())
    datasets = [x for x in nodes if isinstance(x, h5py.Dataset)]
    for node in datasets:
        if any(x in node.attrs for x in _MAPPINGS):
            try:
                list(_specs(group, node))
            except (InvalidFiberFileError, NotImplementedError, KeyError):
                return False
            return True
    return not datasets and any(_has_signal(x) for x in nodes)


def _values(node) -> np.ndarray:
    """Return a stored coordinate's values, decoding CF time."""
    if " since " in str(_attr(node.attrs, "units")):
        return cf_time_values(node)
    return np.atleast_1d(node[()])


def _units(node):
    """Return a coordinate's units, None for a CF time reference."""
    units = _attr(node.attrs, "units") or None
    return None if units and " since " in units else units


def _runs(segments, tolerance):
    """
    Merge (start, count, step) segments which lie on one line.

    A segment continues the run before it when its first value is where
    the run's line puts it and its own end is on the line joining them;
    anything else (a hole, an overlap, a jump back) starts a new run.
    """
    runs = []
    for start, count, step in segments:
        if runs:
            first, total, run_step = runs[-1]
            line = (start - first) / total
            ok = run_step is None or abs(first + total * run_step - start) <= tolerance
            end = first + (total + count - 1) * line
            if ok and (
                count == 1 or abs(start + (count - 1) * step - end) <= tolerance
            ):
                runs[-1] = (first, total + count, line)
                continue
        runs.append((start, count, step if count > 1 else None))
    return runs


def _run_coord(first, count, step, dtype, units):
    """Return one run as a coordinate in the stored dtype."""
    if step is None:
        start = np.datetime64(first, "ns") if dtype.kind == "M" else first
        return get_coord(data=np.array([start]), units=units)
    if dtype.kind == "M":  # ns ticks; the step is an exact fraction of a second
        start = np.datetime64(first, "ns")
        return get_coord(start=start, step=Fraction(step) / 10**9, shape=count)
    if dtype.kind in "iu" and Fraction(step).denominator != 1:
        # labels the format rounds to integers are stored, not gridded
        labels = np.array([round(first + i * step) for i in range(count)])
        return get_coord(data=labels.astype(dtype), units=units, snap=False)
    step = int(step) if dtype.kind in "iu" else float(step)
    return get_coord(start=first, step=step, shape=count, units=units)


def _tie_point_runs(group, spec, length):
    """Return [(first sample, coordinate)] for each run of a tie-point coord."""
    _, _, values_name, indices_name, sampled, exact, tolerance = spec
    node = group[values_name]
    values = _values(node)
    indices = np.atleast_1d(group[indices_name][()]).tolist()
    dtype = values.dtype
    # Python numbers: ns as integers, no unsigned wrap, float64 floats.
    ticks = values.astype(np.int64) if dtype.kind == "M" else values
    ticks = ticks.tolist()
    exact_kind = dtype.kind in "Miu"
    div = Fraction if exact_kind else lambda a, b: a / b
    if sampled:
        if sum(indices) != length:
            msg = f"Segment lengths of {values_name} do not sum to {length}."
            raise InvalidFiberFileError(msg)
        segments = [(v, n, exact) for v, n in zip(ticks, indices, strict=True)]
    else:
        steps = np.diff(indices)
        if indices[0] != 0 or indices[-1] != length - 1 or np.any(steps < 1):
            msg = f"Tie points of {values_name} do not span its {length} samples."
            raise InvalidFiberFileError(msg)
        pairs = zip(pairwise(ticks), pairwise(indices), strict=True)
        segments = [
            (v0, i1 - i0, div(v1 - v0, i1 - i0)) for (v0, v1), (i0, i1) in pairs
        ]
        segments.append((ticks[-1], 1, None))
    # stored labels are within a tick (half of one for rounded integers)
    tick = 1 if dtype.kind == "M" else 0.5 if exact_kind else 0
    tolerance += tick or 1e-9 * max(abs(x) for x in ticks)
    out, first = [], 0
    for start, count, step in _runs(segments, tolerance):
        if exact is not None and step is not None:
            step = exact if abs(step - exact) * count <= tolerance else step
        out.append((first, _run_coord(start, count, step, dtype, _units(node))))
        first += count
    return out


def _dims(node) -> tuple[str, ...]:
    """Return the names of a dataset's netCDF dimensions."""
    return tuple(node.dims[i][0].name.rsplit("/", 1)[-1] for i in range(node.ndim))


def _stored(node, snap):
    """Return a coordinate from stored labels."""
    name = node.name.rsplit("/", 1)[-1]
    snap = should_snap(snap, name)
    return get_coord(data=_values(node), units=_units(node), snap=snap)


def _signal_pieces(group, node, aux, snap):
    """Yield (coords, sample offsets) for each evenly sampled piece of a signal."""
    dims = _dims(node)
    runs, other = {}, {}
    for spec in _specs(group, node):
        name, dim = spec[:2]
        found = _tie_point_runs(group, spec, node.shape[dims.index(dim)])
        if name == dim:
            runs[dim] = found
        else:  # a coordinate along another's dimension
            labels = np.concatenate([x.values for _, x in found])
            other[name] = ((dim,), get_coord(data=labels, snap=False))
    for axis, dim in enumerate(dims):
        if dim not in runs:
            scale = node.dims[axis][0]
            if str(_attr(scale.attrs, "NAME")).startswith(_PLACEHOLDER):
                size = node.shape[axis]
                runs[dim] = [(0, get_coord(start=0, stop=size, step=1))]
            else:
                runs[dim] = [(0, _stored(scale, snap))]
    for name in aux:
        if name in group and group[name].ndim == 1:
            other_dims = _dims(group[name])
            if set(other_dims) <= set(dims):
                other[name] = (other_dims, _stored(group[name], snap))
    for combo in product(*(runs[x] for x in dims)):
        coords = {dim: coord for dim, (_, coord) in zip(dims, combo, strict=True)}
        firsts = dict(zip(dims, (first for first, _ in combo), strict=True))
        for name, (other_dims, coord) in other.items():
            dim = other_dims[0]
            start = firsts[dim]
            coords[name] = (other_dims, coord[start : start + len(coords[dim])])
        cm = dc.get_coord_manager(coords=coords, dims=dims)
        yield cm, tuple(firsts.values())


def _signal_attrs(node) -> dict:
    """Return a signal's own attrs, without netCDF or layout bookkeeping."""
    skip = {*_MAPPINGS, *_PACKING, *_FILLS, "DIMENSION_LIST", "coordinates"}
    return {
        name: _attr(node.attrs, name)
        for name in node.attrs
        if name not in skip and not name.startswith("_")
    }


def _decoded_dtype(node) -> np.dtype:
    """Return the dtype a signal has once CF packing and fill are applied."""
    if not any(x in node.attrs for x in (*_PACKING, *_FILLS)):
        return node.dtype
    packing = [np.asarray(node.attrs[x]).dtype for x in _PACKING if x in node.attrs]
    return np.result_type(node.dtype, np.float32, *packing)


def unpack(node, data: np.ndarray) -> np.ndarray:
    """Apply a signal's CF fill values, scale and offset to stored samples."""
    dtype = _decoded_dtype(node)
    if dtype == node.dtype and not any(x in node.attrs for x in _FILLS):
        return data
    out = data.astype(dtype)
    for name in _FILLS:
        if name in node.attrs:
            out[data == _attr(node.attrs, name)] = np.nan
    scale = dtype.type(_attr(node.attrs, "scale_factor", 1))
    return out * scale + dtype.type(_attr(node.attrs, "add_offset", 0))


def _iter_signals(group):
    """Yield (group, dataset, auxiliary coordinate names) for every signal."""
    datasets = [x for x in group.values() if isinstance(x, h5py.Dataset)]
    helpers, aux = set(), set(_attr(group.attrs, "coordinates").split())
    for node in datasets:
        aux |= set(_attr(node.attrs, "coordinates").split())
        for spec in _specs(group, node):
            helpers |= {spec[2], spec[3]}
    for node in datasets:
        name = node.name.rsplit("/", 1)[-1]
        if _TILING in node.attrs:
            raise NotImplementedError("XDAS tile manifests are not supported.")
        # a netCDF variable which is not a dimension's own coordinate
        if "DIMENSION_LIST" in node.attrs and name not in helpers | aux:
            own = set(_attr(node.attrs, "coordinates").split())
            yield group, node, own | set(_attr(group.attrs, "coordinates").split())
    for child in group.values():
        if isinstance(child, h5py.Group):
            yield from _iter_signals(child)


def get_pieces(h5, snap=True) -> dict[str, tuple]:
    """
    Map each patch key to its (metadata, dataset, sample offsets).

    A signal whose tie points leave a hole, overlap or change the step is
    one patch per evenly sampled stretch, keyed ``<path>#<number>``.
    """
    out = {}
    for group, node, aux in _iter_signals(h5):
        key = node.name.lstrip("/")
        attrs, dtype = _signal_attrs(node), str(_decoded_dtype(node))
        pieces = list(_signal_pieces(group, node, aux, snap))
        for number, (cm, offsets) in enumerate(pieces):
            name = key if len(pieces) == 1 else f"{key}#{number}"
            source = ArraySource(key=name)
            meta = dc.PatchMeta(attrs=attrs, coords=cm, dtype=dtype, source=source)
            out[name] = (meta, node, offsets)
    return out


def _holds(path: Path, name: str) -> bool:
    """Return True if an HDF5 file exists at path and holds a dataset name."""
    if not path.is_file():
        return False
    with h5py.File(path, "r") as handle:
        return name in handle


def check_virtual_sources(node):
    """Refuse a virtual dataset whose source HDF5 cannot find; it reads fill."""
    if not node.is_virtual:
        return
    here = Path(node.file.filename).parent
    prefix = os.environ.get("HDF5_VDS_PREFIX", "").replace("${ORIGIN}", str(here))
    for source in node.virtual_sources():
        name = unbyte(source.file_name)
        if name == ".":
            found = source.dset_name in node.file
        else:
            folders = [Path(x) for x in (prefix, here, Path.cwd()) if str(x)]
            found = any(_holds(x / name, source.dset_name) for x in folders)
        if not found:
            msg = f"{node.name} reads {source.dset_name} from {name}, not found."
            raise FileNotFoundError(msg)
