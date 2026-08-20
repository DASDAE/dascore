"""Core module for reading and writing DASDAE format."""

from __future__ import annotations

import contextlib
from typing import Literal

import pandas as pd

import dascore as dc
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader, H5Writer
from dascore.utils.io import _normalize_source_patch_keys
from dascore.utils.misc import unbyte
from dascore.utils.patch import get_patch_names

from .utils import (
    _get_contents_from_patch_groups_generic,
    _get_patch_attrs,
    _is_legacy_file,
    _is_legacy_group,
    _kwargs_empty,
    _matches_attr_filters,
    _read_patch,
    _save_patch,
    _write_meta,
)


class DASDAEV1(FiberIO):
    """
    Provides IO support for the DASDAE format version 1.

    DASDAE format is loosely based on the Adaptable Seismic Data Format (ASDF)
    which uses hdf5. The hdf5 structure is the following:

    /root
    /root.attrs
        __format__ = "DASDAE"
        __DASDAE_version__ = '1'  # version str
    /root/waveforms/
        DAS__{net}__{sta}__{tag}__{start}__{end}
            data   # patch data array
            data.attrs
            _coords_{coord_name}  # each coordinate array is saved here
        DAS__{net}__{sta}__{tag}__{start}__{end}.attrs
            _attrs_{attr_nme}  # each patch attribute
            _dims  # a str of 'dim1, dim2, dim3'
    """

    name = "DASDAE"
    preferred_extensions = ("h5", "hdf5")
    version = "1"
    multi_patch_write = True

    def write(
        self,
        spool: dc.Patch | dc.Spool,
        resource: H5Writer,
        **kwargs,
    ):
        """
        Write a collection of patches to a DASDAE file.

        Parameters
        ----------
        spool:
            A collection of patches or a spool (same thing).
        resource
            The path to the file.
        """
        # write out patches
        _write_meta(resource, self.version)
        # get an iterable of patches and save them
        patches = [spool] if isinstance(spool, dc.Patch) else spool
        with contextlib.suppress(ValueError):
            resource.create_group("waveforms")
        waveforms = resource["waveforms"]
        # write new patches to file, ensuring unique group names within this
        # batch so same-named patches (e.g. gap-split siblings that differ
        # only along a non-named dimension) don't overwrite each other.
        # strict zip keeps streaming (no spool materialization) while failing
        # loudly if the name pass and patch pass ever disagree in length.
        patch_names = get_patch_names(patches).values
        counts: dict[str, int] = {}
        for patch, name in zip(patches, patch_names, strict=True):
            num = counts.get(name, 0)
            counts[name] = num + 1
            unique_name = name if num == 0 else f"{name}__{num}"
            _save_patch(patch, waveforms, unique_name)

    def _get_patch_summary(self, patches) -> pd.DataFrame:
        """Get a patch summary to put into index."""
        df = (
            dc.scan_to_df(patches)
            .assign(
                source_patch_key=lambda x: get_patch_names(x),
                source_format=self.name,
                source_version=self.version,
            )
            .dropna(subset=["time_min", "time_max", "distance_min", "distance_max"])
        )
        return df

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """Return the format from a dasdae file."""
        is_dasdae, version = False, ""  # NOQA
        attrs = resource.attrs
        file_format = unbyte(attrs.get("__format__", ""))
        if file_format != self.name:
            return False
        version = unbyte(attrs.get("__DASDAE_version__", ""))
        return file_format, version

    def read(self, resource: H5Reader, source_patch_key=(), **kwargs) -> dc.Spool:
        """Read a dascore file."""
        patches = []
        source_patch_keys = _normalize_source_patch_keys(source_patch_key)
        try:
            waveform_group = resource["waveforms"]
        except (KeyError, IndexError):
            return dc.spool([])
        file_legacy = _is_legacy_file(resource)
        for patch_group in waveform_group.values():
            patch_name = str(patch_group.name).rsplit("/", maxsplit=1)[-1]
            if source_patch_keys and patch_name not in source_patch_keys:
                continue
            legacy = _is_legacy_group(patch_group, file_legacy)
            attrs = _get_patch_attrs(patch_group, legacy)
            if not _matches_attr_filters(attrs, kwargs):
                continue
            patch = _read_patch(patch_group, legacy=legacy, **kwargs)
            if not patch.data.size and not _kwargs_empty(kwargs):
                continue
            patches.append(patch)
        return dc.spool(patches)

    def scan(self, resource: H5Reader, snap: bool = True, **kwargs):
        """
        Get patch info by iterating waveform groups in the file.

        Parameters
        ----------
        resource
            A path to the file.
        """
        return _get_contents_from_patch_groups_generic(resource, snap=snap)
