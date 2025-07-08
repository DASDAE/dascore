"""Core module for reading and writing pickle format."""

from __future__ import annotations

import pickle

import dascore
from dascore.io import BinaryReader, BinaryWriter, FiberIO


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

    def _header_is_dascore(self, byte_stream):
        """Return True if the first few bytes mention dascore classes."""
        has_dascore = b"dascore.core" in byte_stream
        spool_or_patch = b"Spool" in byte_stream or b"Patch" in byte_stream
        return has_dascore and spool_or_patch

    def get_format(self, resource: BinaryReader, **kwargs) -> tuple[str, str] | bool:
        """
        Return True if file contains a pickled Patch or Spool.

        Parameters
        ----------
        resource
            A path to the file which may contain terra15 data.
        """
        try:
            start = resource.read(100)  # read first 100 bytes, look for class names
            if self._header_is_dascore(start):
                getattr(resource, "seek", lambda x: None)(0)
                pickle.load(resource)
                return ("PICKLE", self.version)  # TODO add pickle protocol
            else:
                return False
        except (pickle.UnpicklingError, FileNotFoundError, IndexError):
            return False

    def read(self, resource: BinaryReader, **kwargs):
        """Read a Patch/Spool from disk."""
        out = pickle.load(resource)
        return dascore.spool(out)

    def write(self, patch, resource: BinaryWriter, **kwargs):
        """Write a Patch/Spool to disk."""
        pickle.dump(patch, resource)
