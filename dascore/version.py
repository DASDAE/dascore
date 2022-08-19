""" Version of dascore. """
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dascore")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

__last_version__ = __version__.split("+")[0].split("-")[0].replace("v", "")
