"""
Core module for reading and writing pickle format.
"""
import pickle
from pathlib import Path
from typing import Union


def _is_pickle(path: Union[str, Path]) -> Union[tuple[str, str], bool]:
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
            if b"dascore.core.patch" in start or b"dascore.core.spool" in start:
                fp.seek(0)
                pickle.load(fp)
                return ("PICKLE", "")
            else:
                return False
        except (pickle.UnpicklingError, FileNotFoundError):
            return False
