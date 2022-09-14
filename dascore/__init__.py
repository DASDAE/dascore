"""
DASCore - A library for fiber optic sensing.
"""
# set xarray settings
from xarray import set_options

from dascore.clients.dirspool import DirectorySpool
from dascore.core.patch import Patch
from dascore.core.spool import MemorySpool, spool
from dascore.examples import get_example_patch, get_example_spool
from dascore.io.core import get_format, read, scan, scan_to_df, write
from dascore.utils.patch import patch_function
from dascore.version import __last_version__, __version__

# keep attrs on xarray DataArray
set_options(keep_attrs=True)

# flag for disabling progress bar when debugging
_debug = False
