"""IO module for reading simple h5 data."""
from __future__ import annotations

from dascore.io import FiberIO


class H5Simple(FiberIO):
    """Support for bare-bones h5 format."""

    name = "H5Simple"
    preferred_extensions = ("hdf5", "h5")
    version = "1.0"
