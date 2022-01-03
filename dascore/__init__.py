"""
dascore - (Fi)ber (O)ptic (S)esning library.
"""
# set xarray settings
from xarray import set_options

from dascore.core.patch import Patch
from dascore.core.stream import Stream
from dascore.examples import get_example_patch, get_example_stream
from dascore.io.base import get_format, read, scan_file, write
from dascore.utils.patch import patch_function
from dascore.version import __version__

# keep attrs on xarray DataArray
set_options(keep_attrs=True)
