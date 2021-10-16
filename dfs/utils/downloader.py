"""
Simple script for downloading external files.
"""
from pathlib import Path

import pkg_resources
import pooch

from dfs.constants import DATA_VERSION

# Create a pooch for fetching data files
fetcher = pooch.create(
    path=pooch.os_cache("dfs"),
    base_url="https://github.com/d-chambers/dfs",
    version=DATA_VERSION,
    version_dev="master",
    env="DFS_DATA_DIR",
)
fetcher.load_registry(pkg_resources.resource_stream("dfs", "data_registry.txt"))


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
