"""Runtime configuration for DASCore."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from typing import Literal

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
        # Reject unknown fields so misspelled overrides raise rather than
        # silently doing nothing.
        extra="forbid",
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
    display_max_items: int = Field(
        default=10,
        ge=0,
        description="Maximum children a repr lists per level before eliding.",
    )
    display_max_patches: int = Field(
        default=1_000,
        ge=0,
        description=(
            "Patches a spool may hold before its repr stops summarizing "
            "what it covers and prints its count, its path, and the "
            "limit it exceeded instead. Summarizing realizes the whole "
            "index relation, which past this many rows is more work "
            "than a glance is worth; the limit holds whether or not "
            "that relation happens to be realized already."
        ),
    )
    display_html: bool = Field(
        default=True,
        description=(
            "Whether a display which renders HTML -- a notebook, the docs "
            "site -- is offered the collapsible repr. False sends every "
            "display back to the text repr, which says the same words."
        ),
    )
    display_html_open_lines: int = Field(
        default=12,
        ge=0,
        description=(
            "Body lines a section of an HTML repr may hold and still open "
            "on its own. A short section is worth a glance; a long one -- "
            "an array, a deep tree -- is folded until it is asked for. "
            "0 folds every section."
        ),
    )
    patch_history: Literal["standard", "disabled"] = Field(
        default="standard",
        description="Controls whether DASCore appends processing history to patches.",
    )
    patch_provenance: Literal["ids", "disabled"] = Field(
        default="ids",
        description=(
            "Controls whether DASCore maintains the patch_id and "
            "processing_id which say which data a patch is and what was "
            "done to it."
        ),
    )
    sampling_group_tolerance: float = Field(
        default=0.05,
        gt=0,
        description=(
            "Relative sampling-interval difference above which patches are "
            "never combined during chunk/merge operations. E.g. the default "
            "0.05 keeps patches whose steps differ by more than 5% in "
            "separate groups."
        ),
    )
    patch_kind_attrs: tuple[str, ...] = Field(
        default=(
            "acquisition_key",
            "data_category",
            "tag",
            # Legacy: these are not patch attrs any more, but archives
            # predating acquisition_key partition by them and a name
            # missing from a spool is ignored. Grouping too finely only
            # leaves patches unmerged; too coarsely merges patches which
            # describe different places.
            "network",
            "station",
        ),
        description=(
            "Attributes which decide whether patches are the same kind; they "
            "are categorical labels (strings), not quantities. Patches "
            "holding conflicting values for any of these are never "
            "combined: operators and ufuncs raise, concatenate and stack "
            "skip them, and chunk puts them in separate outputs (its "
            "per-call `group` argument overrides this default). Operators "
            "and ufuncs read a missing or empty value as a wildcard which "
            "matches anything; the operations which partition a collection "
            "read it as a value, equal to another missing value and nothing "
            "else. The legacy `network` and `station` can stay because a "
            "name no patch in a spool states is ignored, and an archive "
            "which predates `acquisition_key` states them throughout."
        ),
    )

    # Local cache and index locations.
    downloader_cache_dir: Path = Field(
        default_factory=lambda: _get_cache_root() / "data",
        description="Persistent directory used to cache downloaded example data.",
    )
    directory_index_map_dir: Path = Field(
        default_factory=lambda: _get_cache_root() / "indexes" / "path_map",
        description="Directory of per-data-directory external index-location entries.",
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
    remote_hdf5_block_size: int = Field(
        default=5_242_880,
        gt=0,
        description=(
            "Block size in bytes for remote HDF5 access on tuned protocols. "
            "Zero would make fsspec return a non-seekable streaming file and "
            "download the whole thing, so it is rejected."
        ),
    )
    xarray_block_size: int = Field(
        default=268_435_456,  # 256 MiB
        ge=0,
        description=(
            "Largest a single dask block may be, in bytes, when a spool "
            "converts to an xarray tree. A source patch bigger than this is "
            "read in several windows along the merged dimension rather than "
            "whole, so one block never has to hold a whole large file. Zero "
            "reads every source patch as one block, which makes any "
            "selection touching a patch read all of it."
        ),
    )
    remote_hdf5_max_blocks: int = Field(
        default=8,
        gt=0,
        description=(
            "Blocks each open HTTP HDF5 handle may keep cached. Retained memory "
            "is this times `remote_hdf5_block_size`."
        ),
    )
    warn_on_gc_pause: bool = Field(
        default=True,
        description=(
            "Warn the first time DASCore pauses automatic garbage collection "
            "for a remote HDF5 read."
        ),
    )

    @field_validator(
        "downloader_cache_dir",
        "directory_index_map_dir",
        "remote_cache_dir",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, value):
        """Normalize configured path values."""
        return Path(value).expanduser()


# Runtime configuration has two tiers. `_GLOBAL_CONFIG` is the process-wide
# base, visible from every thread and task; `set_config(...)` swaps it. Scoped
# overrides from `config_context(...)` live in a ContextVar so concurrent
# blocks stay isolated per thread/task and never clobber one another.
_GLOBAL_CONFIG: DascoreConfig = DascoreConfig()
_GLOBAL_CONFIG_LOCK = Lock()
_CONFIG_OVERRIDE: ContextVar[DascoreConfig | None] = ContextVar(
    "dascore_config_override", default=None
)


def _reinit_config_lock():
    """Install a fresh config lock, used after a fork.

    A fork can copy the lock while another thread holds it. That thread does
    not exist in the child, so the inherited copy would never be released and
    any config change in the child would hang.
    """
    global _GLOBAL_CONFIG_LOCK
    _GLOBAL_CONFIG_LOCK = Lock()


if hasattr(os, "register_at_fork"):  # not available on windows
    os.register_at_fork(after_in_child=_reinit_config_lock)


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
    """Return the active runtime configuration.

    A scoped override from [`config_context`](`dascore.config.config_context`)
    in the current thread/task takes precedence over the process-wide base set
    by [`set_config`](`dascore.config.set_config`).
    """
    override = _CONFIG_OVERRIDE.get()
    return override if override is not None else _GLOBAL_CONFIG


def _build_config(base: DascoreConfig, new_config, kwargs) -> DascoreConfig:
    """Validate and build a config from a full replacement or field overrides."""
    if new_config is not None and kwargs:
        msg = "Cannot supply both new_config and keyword overrides."
        raise ValueError(msg)
    if new_config is None:
        payload = base.model_dump()
        payload.update(kwargs)
        return DascoreConfig(**payload)
    if not isinstance(new_config, DascoreConfig):
        msg = "new_config must be an instance of DascoreConfig."
        raise TypeError(msg)
    return new_config


def set_config(new_config: DascoreConfig | None = None, **kwargs) -> DascoreConfig:
    """
    Set the process-wide runtime config, visible from every thread and task.

    Parameters
    ----------
    new_config
        A complete [`DascoreConfig`](`dascore.config.DascoreConfig`) to install.
        Mutually exclusive with keyword overrides.
    **kwargs
        Individual field overrides applied on top of the current base config.

    Notes
    -----
    This is a permanent change to the process-wide base (it is not restored
    automatically). For a temporary, thread/task-local override that restores
    on exit, use [`config_context`](`dascore.config.config_context`) instead.

    Examples
    --------
    >>> import dascore as dc
    >>> _ = dc.set_config(display_float_precision=5)
    >>> assert dc.get_config().display_float_precision == 5
    >>> _ = dc.reset_config()
    """
    global _GLOBAL_CONFIG
    # Serialize the read-modify-write so concurrent keyword updates cannot lose
    # each other's fields or return a config another thread just installed.
    with _GLOBAL_CONFIG_LOCK:
        _GLOBAL_CONFIG = _build_config(_GLOBAL_CONFIG, new_config, kwargs)
        return _GLOBAL_CONFIG


@contextmanager
def config_context(
    new_config: DascoreConfig | None = None, **kwargs
) -> Iterator[DascoreConfig]:
    """
    Temporarily override the runtime config for the current thread/task.

    Parameters
    ----------
    new_config
        A complete [`DascoreConfig`](`dascore.config.DascoreConfig`) to install.
        Mutually exclusive with keyword overrides.
    **kwargs
        Individual field overrides applied on top of the active config.

    Notes
    -----
    The override is stored in a ``ContextVar``, so it is isolated per thread and
    task and restored when the block exits. Whether a newly started OS thread
    inherits a copy of the override is runtime-dependent
    (``sys.flags.thread_inherit_context`` -- normally enabled on free-threaded
    builds and disabled otherwise); an inherited copy is not undone by this
    block's exit. For deterministic propagation, capture
    ``contextvars.copy_context()`` and run the worker with it, or rely on APIs
    that bind the config for you such as
    [`Spool.map`](`dascore.core.spool.Spool.map`).

    Examples
    --------
    >>> import dascore as dc
    >>> with dc.config_context(debug=True):
    ...     assert dc.get_config().debug
    >>> assert not dc.get_config().debug
    """
    config = _build_config(get_config(), new_config, kwargs)
    token = _CONFIG_OVERRIDE.set(config)
    try:
        yield config
    finally:
        _CONFIG_OVERRIDE.reset(token)


def reset_config() -> DascoreConfig:
    """Reset the process-wide runtime config base to defaults."""
    return set_config(DascoreConfig())
