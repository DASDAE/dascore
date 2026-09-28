"""
Support for the DASDAE format.

Version 1 reads both legacy PyTables files and h5py files that mark their patch attributes and coordinate metadata as separate. Coordinate units, sampling intervals, and datetime flags are restored from the coordinate datasets in the latter layout.

Pass a file path when reading the separate-metadata layout. Open PyTables handles are supported only for legacy files; the reader rejects marked files through those handles to avoid losing metadata.

Note
----
This is an experimental format and is subject to change.
"""
from __future__ import annotations
from .core import DASDAEV1
