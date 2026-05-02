"""DASCore - A library for fiber optic sensing."""
from __future__ import annotations

import warnings

from rich import print  # noqa

from dascore.core.attrs import PatchAttrs as PatchAttrs
from dascore.core.coordmanager import CoordManager as CoordManager
from dascore.core.coordmanager import get_coord_manager as get_coord_manager
from dascore.core.coords import get_coord as get_coord
from dascore.core.inventory import Inventory as Inventory
from dascore.core.patch import Patch as Patch
from dascore.core.spool import BaseSpool as BaseSpool
from dascore.core.spool import spool as spool
from dascore.core.summary import PatchSummary as PatchSummary
from dascore.examples import get_example_inventory as get_example_inventory
from dascore.examples import get_example_patch as get_example_patch
from dascore.examples import get_example_spool as get_example_spool
from dascore.io.core import get_format as get_format
from dascore.io.core import read as read
from dascore.io.core import scan as scan
from dascore.io.core import scan_to_df as scan_to_df
from dascore.io.core import write as write
from dascore.units import get_quantity as get_quantity
from dascore.units import get_unit as get_unit
from dascore.utils.patch import patch_function as patch_function
from dascore.utils.time import to_datetime64 as to_datetime64
from dascore.utils.time import to_float as to_float
from dascore.utils.time import to_timedelta64 as to_timedelta64
from dascore.version import __last_version__ as __last_version__
from dascore.version import __version__ as __version__

# Ensure warnings are issued only once (per warning/line)
warnings.filterwarnings("once", category=UserWarning, module=r"dascore\..*")
