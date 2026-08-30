"""Simple script for downloading external files."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pooch

from dascore.config import get_config
from dascore.constants import DATA_VERSION
from dascore.exceptions import UnknownExampleError
from dascore.utils.paths import EXAMPLE_SCHEME, is_example_uri

REGISTRY_PATH = Path(str(files("dascore").joinpath("data_registry.txt")))

# An example uri becomes a Path; anything else is handed back as it came.
_T = TypeVar("_T")


@cache
def _get_fetcher(cache_dir: str) -> pooch.Pooch:
    """Create and cache one pooch fetcher for a specific cache directory."""
    fetcher = pooch.create(
        path=Path(cache_dir),
        base_url="https://github.com/d-chambers/dascore",
        version=DATA_VERSION,
        version_dev="master",
        env="DFS_DATA_DIR",
        # Retry transient network failures; without this a single read
        # timeout aborts a bulk fetch of the registry.
        retry_if_failed=3,
    )
    fetcher.load_registry(REGISTRY_PATH)
    return fetcher


def get_fetcher() -> pooch.Pooch:
    """Return the downloader fetcher for the active runtime configuration."""
    return _get_fetcher(str(get_config().downloader_cache_dir))


class _FetcherProxy:
    """Proxy ``fetcher`` access through the active runtime configuration."""

    def __getattr__(self, item):
        """Delegate attribute access to the active fetcher."""
        return getattr(get_fetcher(), item)


fetcher = _FetcherProxy()


@cache
def get_registry_df() -> pd.DataFrame:
    """Return a dataframe of files in the data registry."""
    names = (
        "name",
        "hash",
        "url",
    )
    df = pd.read_csv(REGISTRY_PATH, sep=r"\s+", skiprows=1, names=names)
    return df


@cache
def _fetch_cached(name: str, cache_dir: str) -> Path:
    """Fetch one named file for a specific downloader cache directory."""
    return Path(_get_fetcher(cache_dir).fetch(name))


def fetch(name: Path | str, **kwargs) -> Path:
    """
    Fetch a data file from the registry.

    Parameters
    ----------
    name
        The name of the file to fetch. Must be in the data registry or a
        path which exists.
    kwargs
        Ignored and kept only for compatibility with older call sites.

    Returns
    -------
    A path to the downloaded file.
    """
    if (existing_path := Path(name)).exists():
        return existing_path
    return _fetch_cached(
        name=str(name), cache_dir=str(get_config().downloader_cache_dir)
    )


def resolve_example_uri(resource: _T) -> _T | Path:
    """
    Resolve an ``examples://{name}`` URI to a local path.

    Parameters
    ----------
    resource
        Any resource. Anything which is not an example URI is returned
        unchanged so call sites need no branch of their own.

    Returns
    -------
    A path to the downloaded file, or the unmodified resource.

    Raises
    ------
    [`UnknownExampleError`](`dascore.exceptions.UnknownExampleError`) if the
    name is not a file in the data registry.

    Examples
    --------
    >>> from dascore.utils.downloader import resolve_example_uri
    >>> path = resolve_example_uri("examples://terra15_das_1_trimmed.hdf5")
    >>> path.exists()
    True
    """
    if not is_example_uri(resource):
        return resource
    name = str(resource)[len(EXAMPLE_SCHEME) :]
    if name in set(get_registry_df()["name"]):
        # Deliberately not fetch: it returns a same-named file in the working
        # directory in preference to the registry entry, and an examples://
        # name asked for the registry entry.
        return _fetch_cached(
            name=name, cache_dir=str(get_config().downloader_cache_dir)
        )
    # Imported here rather than at module top because dascore.examples
    # imports fetch from this module.
    from dascore.examples import EXAMPLE_PATCHES, EXAMPLE_SPOOLS  # noqa: PLC0415

    if name in EXAMPLE_PATCHES or name in EXAMPLE_SPOOLS:
        msg = (
            f"'{name}' is a generated example, not a file in the data "
            f"registry, so it has no path to open. Use "
            f"dc.get_example_patch('{name}') or "
            f"dc.get_example_spool('{name}') instead."
        )
    else:
        msg = (
            f"No example file named '{name}'. Names come from the data "
            f"registry; see dascore.utils.downloader.get_registry_df for "
            f"the full list."
        )
    raise UnknownExampleError(msg)
