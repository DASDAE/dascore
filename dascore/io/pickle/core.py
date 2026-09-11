"""Core module for reading and writing pickle format."""

from __future__ import annotations

import pickle

import numpy as np

import dascore
import dascore as dc
from dascore.core.summary import normalize_source_patch_key
from dascore.io import BinaryReader, BinaryWriter, FiberIO, PatchSource
from dascore.io.utils import resolve_keyed_source, slice_dataset


def _read_patches(resource):
    """Load positional entries, retaining unambiguous keys from legacy pickles."""
    patches = list(dascore.spool(pickle.load(resource)))
    # PatchSource describes the previous file, not positions in this container.
    keys = [
        normalize_source_patch_key(p.attrs.get("_source_patch_key", ""))
        for p in patches
    ]
    resolved = [key or str(index) for index, key in enumerate(keys)]
    if len(set(resolved)) != len(patches):
        keys = [""] * len(patches)
    return list(zip(patches, keys, strict=True))


class PickleIO(FiberIO):
    """
    Provides IO support for the pickle format.

    Warning
    -------
    The pickle format is discouraged due to potential security and
    compatibility issues.
    """

    name = "PICKLE"
    preferred_extensions = ("pkl", "pickle")
    multi_patch_write = True

    def _header_is_dascore(self, byte_stream):
        """Return True if the first few bytes mention dascore classes."""
        has_dascore = b"dascore.core" in byte_stream
        spool_or_patch = b"Spool" in byte_stream or b"Patch" in byte_stream
        return has_dascore and spool_or_patch

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        try:
            start = resource.read(100)
            if self._header_is_dascore(start):
                getattr(resource, "seek", lambda x: None)(0)
                pickle.load(resource)
                return self.version
            else:
                return None
        except (pickle.UnpicklingError, FileNotFoundError, IndexError):
            return None

    def get_metadata(
        self, resource: BinaryReader, *, snap: bool = True
    ) -> list[dc.Patch]:
        """Decode a pickle to describe its patches without retaining their arrays."""
        return [
            dc.Patch(
                coords=p.coords,
                attrs=p.attrs.drop("_source_patch_key"),
                dtype=p.dtype,
                source=PatchSource(key=key),
            )
            for p, key in _read_patches(resource)
        ]

    def read_array(
        self, resource: BinaryReader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Decode one pickled logical patch and slice its array."""
        patch = resolve_keyed_source(
            [
                (native or str(i), p)
                for i, (p, native) in enumerate(_read_patches(resource))
            ],
            key,
        )
        return slice_dataset(patch.data, patch.dims, windows)

    def write(self, spool, resource: BinaryWriter, **kwargs):
        """Write a Patch/Spool to disk."""
        pickle.dump(spool, resource)
