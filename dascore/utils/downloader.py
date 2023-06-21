"""
Simple script for downloading external files.
"""
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pooch

from dascore.constants import DATA_VERSION

REGISTRY_PATH = Path(files("dascore").joinpath("data_registry.txt"))

# Create a pooch for fetching data files
fetcher = pooch.create(
    path=pooch.os_cache("dascore"),
    base_url="https://github.com/d-chambers/dascore",
    version=DATA_VERSION,
    version_dev="master",
    env="DFS_DATA_DIR",
)
fetcher.load_registry(REGISTRY_PATH)


def get_registry_df() -> pd.DataFrame:
    """
    Returns a dataframe of all files in the data registry.
    """
    names = (
        "name",
        "hash",
        "url",
    )
    df = pd.read_csv(REGISTRY_PATH, sep=r"\s+", skiprows=1, names=names)
    return df


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
