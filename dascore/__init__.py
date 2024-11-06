"""DASCore - A library for fiber optic sensing."""
from __future__ import annotations
from rich import print  # noqa

from dascore.core.patch import Patch
from dascore.core.attrs import PatchAttrs
from dascore.core.spool import BaseSpool, spool
from dascore.core.coordmanager import get_coord_manager, CoordManager
from dascore.core.coords import get_coord
from dascore.examples import get_example_patch, get_example_spool
from dascore.io.core import get_format, read, scan, scan_to_df, write
from dascore.units import get_quantity, get_unit
from dascore.utils.patch import patch_function
from dascore.utils.time import to_datetime64, to_timedelta64, to_float
from dascore.version import __last_version__, __version__

# flag for disabling progress bar when debugging
_debug = False
