"""Translate xarray-style write encodings into h5py dataset options."""

from __future__ import annotations

import h5py
import numpy as np

import dascore as dc

# The h5netcdf encoding keys which map onto h5py's create_dataset.
_KEYS = (
    "chunksizes",
    "complevel",
    "compression",
    "compression_opts",
    "fletcher32",
    "shuffle",
    "zlib",
)


def _translate(name, encoding) -> dict:
    """Translate one variable's encoding as xarray's h5netcdf backend does."""
    if unknown := sorted(set(encoding) - set(_KEYS)):
        msg = (
            f"Unexpected encoding parameters for variable {name!r}: {unknown}. "
            f"Valid encodings are: {list(_KEYS)}."
        )
        raise ValueError(msg)
    out = dict(encoding)
    if out.pop("zlib", False):
        if out.get("compression") not in (None, "gzip"):
            raise ValueError("'zlib' and 'compression' encodings mismatch")
        out.setdefault("compression", "gzip")
    level, opts = out.get("complevel"), out.get("compression_opts")
    if "complevel" in out and "compression_opts" in out and level != opts:
        raise ValueError("'complevel' and 'compression_opts' encodings mismatch")
    if complevel := out.pop("complevel", 0):
        out.setdefault("compression_opts", complevel)
    # JSON and YAML give lists; h5py filters need tuples.
    if isinstance(out.get("compression_opts"), list):
        out["compression_opts"] = tuple(out["compression_opts"])
    if (chunks := out.pop("chunksizes", None)) is not None:
        out["chunks"] = tuple(chunks)
    return out


def _get_h5_options(encoding, spool) -> dict[str, dict]:
    """Validate an encoding for a spool; return h5py options per variable."""
    if not encoding:
        return {}
    # The contents frame has a "{name}_min" column for every coordinate.
    columns = dc.spool(spool).get_contents().columns
    names = {"data"} | {x.removesuffix("_min") for x in columns if x.endswith("_min")}
    if unknown := sorted(set(encoding) - names):
        msg = f"Unexpected encoding for variables not in the patches: {unknown}."
        raise ValueError(msg)
    options = {name: _translate(name, enc) for name, enc in encoding.items()}
    # Only compression settings can fail in h5py; try them before writing.
    filtered = [
        x
        for x in options.values()
        if x.get("compression") is not None or x.get("compression_opts") is not None
    ]
    if filtered:
        _trial(filtered)
    return options


def _trial(options_list):
    """Create a small in-memory dataset with each set of options."""
    with h5py.File("trial", "w", driver="core", backing_store=False) as h5:
        for num, opts in enumerate(options_list):
            h5.create_dataset(str(num), data=np.zeros(2), **{**opts, "chunks": None})


def _check_chunks(options, patch):
    """Refuse chunk sizes whose length differs from their array's dimensions."""
    shapes = {name: coord.shape for name, coord in patch.coords.coord_map.items()}
    shapes["data"] = patch.shape
    for name, opts in options.items():
        chunks, shape = opts.get("chunks"), shapes.get(name)
        if chunks is not None and shape is not None and len(chunks) != len(shape):
            msg = (
                f"chunksizes for {name!r} has {len(chunks)} values but the "
                f"array has {len(shape)} dimensions."
            )
            raise ValueError(msg)


def _dataset_kwargs(options, shape) -> dict:
    """Return create_dataset kwargs for an array, clamping chunks to its shape."""
    if not (options and shape and all(shape)):  # h5py refuses filters on these
        return {}
    if (chunks := options.get("chunks")) is None:
        return options
    return {
        **options,
        "chunks": tuple(min(c, n) for c, n in zip(chunks, shape, strict=True)),
    }
