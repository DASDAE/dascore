"""
Simple script for downloading external files.
"""
from pathlib import Path

import pkg_resources
import pooch

from fios.constants import DATA_VERSION

# Create a pooch for fetching data files
fetcher = pooch.create(
    path=pooch.os_cache("fios"),
    base_url="https://github.com/d-chambers/fios",
    version=DATA_VERSION,
    version_dev="master",
    env="DFS_DATA_DIR",
)
fetcher.load_registry(pkg_resources.resource_stream("fios", "data_registry.txt"))


def fetch(name: str, **kwargs) -> Path:
    """
    Fetch a data file.

    Parameters
    ----------
    name
    kwargs

    Returns
    -------

    """
    return Path(fetcher.fetch(name, **kwargs))
