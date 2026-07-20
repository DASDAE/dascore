"""
Core routines and functionality for processing distributed fiber data.
"""
from __future__ import annotations
from .patch import Patch  # noqa
from .summary import PatchSummary  # noqa
from .coords import CoordSegmented, CoordSummary, concat_coords, get_coord
from .coordmanager import get_coord_manager, CoordManager
