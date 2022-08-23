"""
Core module for reading and writing pickle format.
"""
import pickle
from pathlib import Path
from typing import Union

import dascore
from dascore.io.core import FiberIO


class PickleIO(FiberIO):
    """
    Provides IO support for the pickle format.

    Warning
    -------
    The pickle format is discouraged due to potential security and
    compatibility issues.
    """

    name = "PICKLE"

    def _header_is_dascore(self, byte_stream):
        """Return True if the first few bytes mention dascore classes."""
        has_dascore = b"dascore.core" in byte_stream
        spool_or_stream = b"Spool" in byte_stream or b"Patch" in byte_stream
        return has_dascore and spool_or_stream

    def get_format(self, path: Union[str, Path]) -> Union[tuple[str, str], bool]:
        """
        Return True if file contains a pickled Patch or Spool.

        Parameters
        ----------
        path
            A path to the file which may contain terra15 data.
        """
        with open(path, "rb") as fp:
            try:
                start = fp.read(100)  # read first 100 bytes, look for class names
                if self._header_is_dascore(start):
                    fp.seek(0)
                    pickle.load(fp)
                    return ("PICKLE", self.version)  # TODO add pickle protocol
                else:
                    return False
            except (pickle.UnpicklingError, FileNotFoundError, IndexError):
                return False

    def read(self, path, **kwargs):
        """Read a Patch/Stream from disk."""
        with open(path, "rb") as fi:
            out = pickle.load(fi)
        return dascore.MemorySpool(out)

    def write(self, patch, path, **kwargs):
        """Read a Patch/Stream from disk."""
        with open(path, "wb") as fi:
            pickle.dump(patch, fi)
