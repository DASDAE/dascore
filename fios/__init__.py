"""
fios - (Fi)ber (O)ptic (S)esning library.
"""
# set xarray settings
from xarray import set_options

from fios.utils.patch import patch_function
from fios.core.patch import Patch
from fios.core.stream import Stream
from fios.io.base import get_format, read, scan_file
from fios.version import __version__

# keep attrs on xarray DataArray
set_options(keep_attrs=True)
