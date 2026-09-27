"""Translate xarray-style write encodings into h5py dataset options."""

from __future__ import annotations

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
_DROPPED = ("source", "original_shape")


def _translate(name, encoding) -> dict:
    """Translate one variable's encoding as xarray's h5netcdf backend does."""
    # xarray silently drops these, which an opened dataset's encoding holds.
    out = {k: v for k, v in encoding.items() if k not in _DROPPED}
    if unknown := sorted(set(out) - set(_KEYS)):
        msg = (
            f"Unexpected encoding parameters for variable {name!r}: {unknown}. "
            f"Valid encodings are: {list(_KEYS)}."
        )
        raise ValueError(msg)
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


def _get_h5_options(encoding) -> dict[str, dict]:
    """Return h5py dataset options for each variable of an encoding."""
    return {name: _translate(name, enc) for name, enc in (encoding or {}).items()}


def _check_variables(options, patch):
    """Refuse an encoding naming a variable the patch lacks, as xarray does."""
    if unknown := sorted(set(options) - {"data", *patch.coords.coord_map}):
        raise KeyError(f"Encoding names variables not in the patch: {unknown}")


def _dataset_kwargs(options, shape) -> dict:
    """Return create_dataset kwargs for an array, clamping chunks to its shape."""
    if not (options and shape and all(shape)):  # h5py refuses filters on these
        return {}
    chunks = options.get("chunks")
    if chunks is None or len(chunks) != len(shape):  # h5py reports a bad rank
        return options
    return {
        **options,
        "chunks": tuple(min(c, n) for c, n in zip(chunks, shape, strict=True)),
    }
