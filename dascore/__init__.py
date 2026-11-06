"""DASCore - A library for fiber optic sensing."""

# No `from __future__ import annotations` here: it would bind `annotations`
# in this namespace, leaving dc.annotations correct only while the loader
# import below happens to rebind it.

# Aliased because importing dascore.warnings binds the submodule to the
# `warnings` name in this namespace, which would shadow the standard library.
import warnings as _std_warnings

from dascore.core.patch import Patch
from dascore.core.patch_meta import PatchMeta
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
from dascore.io.core import (
    get_format,
    read,
    read_array,
    scan,
    scan_payloads,
    scan_to_df,
    write,
)
from dascore.units import get_quantity, get_unit
from dascore.utils.patch import patch_function
from dascore.core.processor import PatchProcessor
from dascore.utils.time import to_datetime64, to_timedelta64, to_float
from dascore.version import __last_version__, __version__

# The supported top-level API. `tests/test_public_api.py` pins it, together
# with the public modules and the Patch and Spool methods. The `io` and
# `warnings` subpackages stay attributes (dc.io, dc.warnings) but are not
# star-exported, since they would shadow the standard library modules.
__all__ = [
    "AnnotationSet",
    "BaseSpool",
    "CoordManager",
    "DascoreConfig",
    "Inventory",
    "Patch",
    "PatchAttrs",
    "PatchMeta",
    "PatchProcessor",
    "PatchSummary",
    "Spool",
    "__version__",
    "annotations",
    "config",
    "config_context",
    "constants",
    "core",
    "examples",
    "exceptions",
    "get_config",
    "get_coord",
    "get_coord_manager",
    "get_example_inventory",
    "get_example_patch",
    "get_example_spool",
    "get_format",
    "get_quantity",
    "get_unit",
    "inventory",
    "models",
    "patch_function",
    "proc",
    "read",
    "read_array",
    "reset_config",
    "scan",
    "scan_payloads",
    "scan_to_df",
    "set_config",
    "spool",
    "to_datetime64",
    "to_float",
    "to_timedelta64",
    "transform",
    "units",
    "utils",
    "viz",
    "write",
    "xarray",
]


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
_std_warnings.filterwarnings("once", category=UserWarning, module=r"dascore\..*")
