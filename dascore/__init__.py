"""DASCore - A library for fiber optic sensing."""
from __future__ import annotations

# Aliased because importing dascore.warnings binds the submodule to the
# `warnings` name in this namespace, which would shadow the standard library.
import warnings as std_warnings

from rich import print  # noqa

from dascore.core.patch import Patch
from dascore.core.attrs import PatchAttrs
from dascore.core.summary import PatchSummary
from dascore.core.spool import BaseSpool, Spool, spool
from dascore.core.annotation_loader import annotations
from dascore.core.annotations import AnnotationSet
from dascore.core.inventory import Inventory
from dascore.core.inventory_loader import inventory
from dascore.core.coordmanager import get_coord_manager, CoordManager
from dascore.core.coords import get_coord
from dascore.config import (
    DascoreConfig,
    config_context,
    get_config,
    reset_config,
    set_config,
)
from dascore.examples import (
    get_example_inventory,
    get_example_patch,
    get_example_spool,
)
from dascore.io.core import get_format, read, scan, scan_payloads, scan_to_df, write
from dascore.units import get_quantity, get_unit
from dascore.utils.patch import patch_function
from dascore.utils.time import to_datetime64, to_timedelta64, to_float
from dascore.version import __last_version__, __version__

# flag for disabling progress bar when debugging
_debug = False


def __getattr__(name: str):
    """
    Lazily import select submodules on attribute access (PEP 562).

    The viz module imports matplotlib, which is slow, so it is deferred
    until first use to keep `import dascore` fast.
    """
    if name == "viz":
        import dascore.viz

        return dascore.viz
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Ensure warnings are issued only once (per warning/line)
std_warnings.filterwarnings("once", category=UserWarning, module=r"dascore\..*")
