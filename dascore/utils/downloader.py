"""
Simple script for downloading external files.
"""
from importlib.resources import files
from pathlib import Path

import pooch

from dascore.constants import DATA_VERSION

# Create a pooch for fetching data files
fetcher = pooch.create(
    path=pooch.os_cache("dascore"),
    base_url="https://github.com/d-chambers/dascore",
    version=DATA_VERSION,
    version_dev="master",
    env="DFS_DATA_DIR",
)
fetcher.load_registry(files("dascore").joinpath("data_registry.txt"))


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
