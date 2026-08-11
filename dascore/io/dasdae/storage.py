"""Storage options for DASDAE files."""

from __future__ import annotations

from typing import ClassVar

from pydantic import SerializeAsAny, field_validator, model_validator

from dascore.io.codec import _codec_name, get_codec
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

    Notes
    -----
    The codec and chunk layout apply to the patch data arrays and to the
    coordinate arrays (chunks match coordinate arrays by dimension name).
    """

    name: ClassVar[str] = "dasdae"
    presets: ClassVar[dict[str, dict]] = {
        "compressed": {"codec": {"name": "gzip", "level": 5}},
    }
    # DASDAE stores arrays via native HDF5 dataset filters, so it supports
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
            codec = codec_cls(**value)
        else:
            msg = f"Cannot interpret codec value {value!r}."
            raise ValueError(msg)
        if not isinstance(codec, cls.supported_codec_bases):
            allowed = ", ".join(c.__name__ for c in cls.supported_codec_bases)
            msg = (
                f"{cls.__name__} cannot store codec of type "
                f"{type(codec).__name__}; it supports codecs of type: {allowed}."
            )
            raise ValueError(msg)
        # Reject codec instances with no registered discriminator (e.g. a bare
        # HDF5Codec()) here rather than erroring opaquely mid-write, and mark
        # the discriminator as explicitly set so exclude_unset dumps (used by
        # DascoreBaseModel.new) still include it and can round-trip.
        name = getattr(codec, "name", None) or _codec_name(type(codec))
        if "name" not in codec.model_fields_set:
            codec = codec.model_copy(update={"name": name})
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

    def _resolve_chunkshape(self, dims, shape) -> tuple[int, ...] | None:
        """
        Resolve the chunk shape for a single array given its dims and shape.

        Returns None (stay contiguous, or let h5py decide when compressing)
        when no chunks are configured, the array is scalar, or dims don't
        match the array.
        """
        if self.chunks is None or not shape or len(dims) != len(shape):
            return None
        return tuple(
            min(self.chunks.get(dim, size), size)
            for dim, size in zip(dims, shape, strict=True)
        )

    def _dataset_options(self, dims, shape) -> dict:
        """
        Return h5py ``create_dataset`` options for one array.

        Combines the codec's compression kwargs with the chunk shape resolved
        for the array's dims. An empty dict means default contiguous storage.
        """
        codec = self.codec
        if codec is None:
            out = {}
        else:
            # _resolve_codec guarantees only HDF5-filter codecs get this far.
            assert isinstance(codec, HDF5Codec)
            out = codec._dataset_kwargs()
        chunkshape = self._resolve_chunkshape(dims, shape)
        if chunkshape is not None:
            out["chunks"] = chunkshape
        return out
