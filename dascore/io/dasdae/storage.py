"""Storage options for DASDAE files."""

from __future__ import annotations

from typing import ClassVar

import tables
from pydantic import SerializeAsAny, field_validator, model_validator

from dascore.io.codec import get_codec
from dascore.io.core import BaseCodec, BaseStorage
from dascore.io.hdf5 import HDF5Codec


class DASDAEStorage(BaseStorage):
    """
    Storage options for DASDAE HDF5 files.

    Parameters
    ----------
    codec
        The compression codec. May be a codec instance, a codec name string
        (e.g. ``"gzip"``), or a dict such as ``{"name": "gzip", "level": 3}``.
        ``None`` (default) writes uncompressed arrays. Codec names are resolved
        against the codec registry, so codecs added by plugins are usable here
        as long as DASDAE can store them (see ``supported_codec_bases``).
    chunks
        A mapping of dimension name to chunk length used for the on-disk array
        layout, e.g. ``{"time": 2000, "distance": 64}``. Dimensions absent from
        the mapping use the full array length. Required by HDF5 for compression
        but also usable on its own for faster partial reads.
    """

    name: ClassVar[str] = "dasdae"
    presets: ClassVar[dict[str, dict]] = {
        "compressed": {"codec": {"name": "blosc:zstd", "level": 5}},
    }
    # DASDAE stores arrays via native PyTables/HDF5 filters, so it supports
    # HDF5-filter codecs. Byte-transform codecs would need a blob storage path.
    supported_codec_bases: ClassVar[tuple[type[BaseCodec], ...]] = (HDF5Codec,)

    codec: SerializeAsAny[BaseCodec] | None = None
    chunks: dict[str, int] | None = None

    @field_validator("codec", mode="before")
    @classmethod
    def _resolve_codec(cls, value):
        """Resolve a codec from an instance, a name string, or a dict."""
        if value is None:
            return None
        if isinstance(value, str):
            value = {"name": value}
        if isinstance(value, BaseCodec):
            codec = value
        elif isinstance(value, dict):
            if "name" not in value:
                msg = "codec dict must include a 'name' key."
                raise ValueError(msg)
            codec_cls = get_codec(value["name"])
            codec = codec_cls(**{k: v for k, v in value.items() if k != "name"})
        else:
            msg = f"Cannot interpret codec value {value!r}."
            raise ValueError(msg)
        if not isinstance(codec, cls.supported_codec_bases):
            allowed = ", ".join(c.__name__ for c in cls.supported_codec_bases)
            msg = (
                f"{cls.__name__} cannot store codec {codec.name!r}; it supports "
                f"codecs of type: {allowed}."
            )
            raise ValueError(msg)
        return codec

    @model_validator(mode="after")
    def _validate_chunks(self):
        """Ensure chunk sizes are positive."""
        for dim, size in (self.chunks or {}).items():
            if size <= 0:
                msg = f"chunk sizes must be positive, got {size} for {dim!r}."
                raise ValueError(msg)
        return self

    def _validate_chunk_dims(self, dims) -> None:
        """
        Ensure configured chunk dims correspond to real dimensions.

        Chunk dim names are not known to be valid at construction time (the
        patch is not available), so a typo (e.g. ``"tim"`` for ``"time"``)
        would otherwise be silently ignored, leaving the array un-chunked.
        """
        if not self.chunks:
            return
        unknown = set(self.chunks) - set(dims)
        if unknown:
            unknown_str = ", ".join(repr(x) for x in sorted(unknown))
            valid_str = ", ".join(repr(x) for x in dims) or "(none)"
            msg = (
                f"Unknown chunk dimension(s): {unknown_str}. "
                f"Valid dimensions: {valid_str}."
            )
            raise ValueError(msg)

    def _get_filters(self) -> tables.Filters | None:
        """Return PyTables filters for the configured codec (or None)."""
        if self.codec is None:
            return None
        return self.codec._to_pytables_filters()

    def _resolve_chunkshape(self, dims, shape) -> tuple[int, ...] | None:
        """
        Resolve the chunk shape for a single array given its dims and shape.

        Returns None (let PyTables decide, or stay contiguous) when no chunks
        are configured, the array is scalar, or dims don't match the array.
        """
        if self.chunks is None or not shape or len(dims) != len(shape):
            return None
        return tuple(
            min(self.chunks.get(dim, size), size) for dim, size in zip(dims, shape)
        )
