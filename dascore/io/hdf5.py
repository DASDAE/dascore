"""HDF5 IO storage codecs."""

from __future__ import annotations

from typing import ClassVar, Literal

import tables
from pydantic import model_validator

from dascore.io.core import BaseCodec


class HDF5Codec(BaseCodec):
    """
    Base class for codecs which map to PyTables HDF5 filters.

    Concrete subclasses set ``_complib`` (the PyTables ``complib``) and declare
    a unique ``name`` Literal field.
    """

    level: int = 5
    shuffle: bool = True

    # The PyTables complib name; set by concrete subclasses.
    _complib: ClassVar[str] = ""

    @model_validator(mode="after")
    def _validate_level(self):
        """Ensure compression level fits the HDF5/PyTables range."""
        if not 0 <= self.level <= 9:
            msg = "level must be between 0 and 9."
            raise ValueError(msg)
        return self

    def _to_pytables_filters(self) -> tables.Filters | None:
        """Return PyTables filters, or None when compression is disabled."""
        if self.level == 0:
            return None
        return tables.Filters(
            complevel=self.level,
            complib=self._complib,
            shuffle=self.shuffle,
        )


class Gzip(HDF5Codec):
    """Portable gzip (zlib) HDF5 compression."""

    name: Literal["gzip"] = "gzip"
    _complib: ClassVar[str] = "zlib"


class BloscZstd(HDF5Codec):
    """Blosc Zstandard HDF5 compression."""

    name: Literal["blosc:zstd"] = "blosc:zstd"
    _complib: ClassVar[str] = "blosc:zstd"
