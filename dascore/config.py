"""Runtime configuration for DASCore."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir
from typing import Literal

import numpy as np
import pooch
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _get_cache_root() -> Path:
    """Return the base cache directory for DASCore."""
    return Path(pooch.os_cache("dascore"))


def _get_remote_cache_root() -> Path:
    """Return the OS-appropriate temp directory used for remote-file caching."""
    return Path(gettempdir()) / "dascore" / "remote_cache"


class DascoreConfig(BaseModel):
    """Container for runtime configuration values."""

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # General behavior.
    debug: bool = Field(default=False, description="Enable DASCore debug behavior.")

    # Display and history rendering.
    display_float_precision: int = Field(
        default=3,
        description="Number of decimal places to show in numeric displays.",
    )
    display_array_threshold: int = Field(
        default=100,
        description="Maximum array size to display before summarizing values.",
    )
    display_patch_history_array_threshold: int = Field(
        default=10,
        description="Maximum history length to display before summarizing entries.",
    )
    patch_history: Literal["standard", "disabled"] = Field(
        default="standard",
        description="Controls whether DASCore appends processing history to patches.",
    )

    # Local cache and index locations.
    downloader_cache_dir: Path = Field(
        default_factory=lambda: _get_cache_root() / "data",
        description="Persistent directory used to cache downloaded example data.",
    )
    directory_index_map_path: Path = Field(
        default_factory=lambda: _get_cache_root() / "indexes" / "cache_paths.json",
        description="Path to the cache that records external index-file locations.",
    )
    index_query_buffer: np.timedelta64 = Field(
        default=np.timedelta64(1, "s"),
        description="Time buffer applied when querying cached directory indexes.",
    )

    # HDF index writing.
    hdf_index_complib: str = Field(
        default="blosc:lz4",
        description="Compression library used when writing DASCore HDF index files.",
    )
    hdf_index_complevel: int = Field(
        default=5,
        description="Compression level used when writing DASCore HDF index files.",
    )
    hdf_index_max_retries: int = Field(
        default=3,
        description="Maximum number of retries for concurrent HDF index access.",
    )

    # Progress display.
    progress_basic_refresh_per_second: float = Field(
        default=0.25,
        description="Refresh rate for basic progress display updates.",
    )

    # Remote IO and local materialization policy.
    remote_cache_dir: Path = Field(
        default_factory=_get_remote_cache_root,
        description="Temporary directory used to materialize remote files locally.",
    )
    allow_remote_cache: bool = Field(
        default=True,
        description="Allow DASCore to cache remote files to local temporary storage.",
    )
    allow_remote_cache_for_metadata: bool = Field(
        default=False,
        description="Allow local caching of remote files when only metadata is needed.",
    )
    warn_on_remote_cache: bool = Field(
        default=True,
        description="Warn when DASCore falls back to caching a remote file locally.",
    )
    allow_dasdae_format_unpickle: bool = Field(
        default=False,
        description=(
            "Allow legacy DASDAE files to unpickle embedded coord metadata for "
            "compatibility with trusted historical files."
        ),
    )
    remote_download_block_size: int = Field(
        default=1_048_576,
        description="Block size in bytes for general remote file downloads.",
    )
    remote_download_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for blocking remote file downloads.",
    )
    remote_hdf5_block_size: int = Field(
        default=5_242_880,
        description="Block size in bytes for remote HDF5 access on tuned protocols.",
    )

    @field_validator(
        "downloader_cache_dir",
        "directory_index_map_path",
        "remote_cache_dir",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, value):
        """Normalize configured path values."""
        return Path(value).expanduser()

    @field_validator("index_query_buffer", mode="before")
    @classmethod
    def _coerce_timedelta(cls, value):
        """Normalize timedelta-like config values."""
        return np.timedelta64(value)


_CONFIG = DascoreConfig()


class _ConfigDescriptor:
    """Descriptor for attributes that should always reflect runtime config."""

    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def __get__(self, _instance, _owner=None):
        """Return the current configured value for the target attribute."""
        return getattr(get_config(), self.attr_name)


def config_attr(attr_name: str):
    """Return a descriptor bound to one field on the active runtime config."""
    return _ConfigDescriptor(attr_name)


def get_config() -> DascoreConfig:
    """Return the active runtime configuration."""
    return _CONFIG


@contextmanager
def _restore_config(previous: DascoreConfig):
    """Restore the previous config when exiting a context manager."""
    global _CONFIG
    try:
        yield _CONFIG
    finally:
        _CONFIG = previous


def set_config(
    new_config: DascoreConfig | None = None, **kwargs
):  # pragma: no cover - exercised via context manager use
    """Set the active runtime config and return a restoring context manager."""
    global _CONFIG
    previous = _CONFIG
    if new_config is not None and kwargs:
        msg = "Cannot supply both new_config and keyword overrides."
        raise ValueError(msg)
    if new_config is None:
        payload = previous.model_dump()
        payload.update(kwargs)
        new_config = DascoreConfig(**payload)
    elif not isinstance(new_config, DascoreConfig):
        msg = "new_config must be an instance of DascoreConfig."
        raise TypeError(msg)
    _CONFIG = new_config
    return _restore_config(previous)


def reset_config() -> DascoreConfig:
    """Reset the active runtime config to defaults."""
    global _CONFIG
    _CONFIG = DascoreConfig()
    return _CONFIG
