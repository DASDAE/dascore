"""Simple script for downloading external files."""

from __future__ import annotations

from functools import cache
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


@cache
def get_registry_df() -> pd.DataFrame:
    """Returns a dataframe of all files in the data registry."""
    names = (
        "name",
        "hash",
        "url",
    )
    df = pd.read_csv(REGISTRY_PATH, sep=r"\s+", skiprows=1, names=names)
    return df


@cache
def fetch(name: Path | str, **kwargs) -> Path:
    """
    Fetch a data file from the registry.

    Parameters
    ----------
    name
        The name of the file to fetch. Must be in the data registry or a
        path which exists.
    kwargs
        Left for compatibility reasons.

    Returns
    -------
    A path to the downloaded file.
    """
    if (existing_path := Path(name)).exists():
        return existing_path
    return Path(fetcher.fetch(name, **kwargs))
