"""Simple script for downloading external files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pooch

from dascore.config import get_config
from dascore.constants import DATA_VERSION

REGISTRY_PATH = Path(files("dascore").joinpath("data_registry.txt"))
LARGE_REGISTRY_FILES = frozenset({"whale_1.hdf5"})


@cache
def _get_fetcher(cache_dir: str) -> pooch.Pooch:
    """Create and cache one pooch fetcher for a specific cache directory."""
    fetcher = pooch.create(
        path=Path(cache_dir),
        base_url="https://github.com/d-chambers/dascore",
        version=DATA_VERSION,
        version_dev="master",
        env="DFS_DATA_DIR",
        # Priming the CI cache pulls every registry file in one pass, so a
        # single read timeout from the host used to fail the whole job.
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


@dataclass(frozen=True)
class TestDataCacheInfo:
    """Metadata needed to restore or prime the CI test-data cache."""

    registry_path: Path
    cache_path: Path
    data_version: str
    registry_hash: str

    def get_restore_prefix(self, runner_os: str, cache_number: str | int) -> str:
        """
        Return the cache-key prefix used to fall back to an older cache.

        A registry change makes the exact key miss. Restoring the previous
        cache under this prefix leaves pooch fetching only the added files
        rather than the whole registry again.

        ``cache_number`` is part of the prefix, not just the full key, so
        bumping it still resets the cache; a prefix which stopped at the
        data version would fall back onto the cache the bump meant to drop.
        """
        return f"data-{runner_os}-{self.data_version}-{cache_number}-"

    def get_key(self, runner_os: str, cache_number: str | int) -> str:
        """Return the GitHub Actions cache key for the given OS and cache number."""
        prefix = self.get_restore_prefix(runner_os, cache_number)
        return f"{prefix}{self.registry_hash}"


@cache
def _get_test_data_cache_info(cache_dir: str) -> TestDataCacheInfo:
    """Return the metadata needed to populate the CI test-data cache."""
    registry_path = Path(REGISTRY_PATH)
    fetcher = _get_fetcher(cache_dir)
    return TestDataCacheInfo(
        registry_path=registry_path,
        cache_path=Path(fetcher.path).parent,
        data_version=DATA_VERSION,
        registry_hash=sha256(registry_path.read_bytes()).hexdigest(),
    )


def get_test_data_cache_info() -> TestDataCacheInfo:
    """Return the metadata needed to populate the CI test-data cache."""
    return _get_test_data_cache_info(str(get_config().downloader_cache_dir))


@cache
def get_registry_df(*, exclude_large: bool = False) -> pd.DataFrame:
    """Return a dataframe of files in the data registry."""
    names = (
        "name",
        "hash",
        "url",
    )
    df = pd.read_csv(REGISTRY_PATH, sep=r"\s+", skiprows=1, names=names)
    if exclude_large:
        df = df.loc[~df["name"].isin(LARGE_REGISTRY_FILES)]
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
