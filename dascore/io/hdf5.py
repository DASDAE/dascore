"""HDF5 IO storage codecs."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import model_validator

from dascore.io.core import BaseCodec


class HDF5Codec(BaseCodec):
    """
    Base class for codecs which map to native HDF5 dataset filters.

    Concrete subclasses set ``_compression`` (the h5py ``compression`` value)
    and declare a unique ``name`` Literal field.
    """

    level: int = 5
    shuffle: bool = True

    # The h5py compression name; set by concrete subclasses.
    _compression: ClassVar[str] = ""

    @model_validator(mode="after")
    def _validate_level(self):
        """Ensure compression level fits the HDF5 range."""
        if not 0 <= self.level <= 9:
            msg = "level must be between 0 and 9."
            raise ValueError(msg)
        return self

    def _dataset_kwargs(self) -> dict:
        """Return h5py dataset creation kwargs, empty when disabled."""
        if self.level == 0:
            return {}
        return {
            "compression": self._compression,
            "compression_opts": self.level,
            "shuffle": self.shuffle,
        }


class Gzip(HDF5Codec):
    """Portable gzip (zlib/deflate) HDF5 compression."""

    name: Literal["gzip"] = "gzip"
    _compression: ClassVar[str] = "gzip"
