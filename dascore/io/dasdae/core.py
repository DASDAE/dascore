"""Core module for reading and writing DASDAE format."""

from __future__ import annotations

import contextlib

import pandas as pd

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.utils.hdf5 import H5Reader, H5Writer
from dascore.utils.io import _normalize_source_patch_ids
from dascore.utils.misc import unbyte
from dascore.utils.patch import get_patch_names

from .utils import (
    _get_contents_from_patch_groups_generic,
    _kwargs_empty,
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

    def write(self, spool: SpoolType, resource: H5Writer, **kwargs):
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
        # write new patches to file
        patch_names = get_patch_names(patches).values
        for patch, name in zip(patches, patch_names):
            _save_patch(patch, waveforms, name)

    def _get_patch_summary(self, patches) -> pd.DataFrame:
        """Get a patch summary to put into index."""
        df = (
            dc.scan_to_df(patches)
            .assign(
                source_patch_id=lambda x: get_patch_names(x),
                file_format=self.name,
                file_version=self.version,
            )
            .dropna(subset=["time_min", "time_max", "distance_min", "distance_max"])
        )
        return df

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Return the format from a dasdae file."""
        is_dasdae, version = False, ""  # NOQA
        attrs = resource.attrs
        file_format = unbyte(attrs.get("__format__", ""))
        if file_format != self.name:
            return False
        version = unbyte(attrs.get("__DASDAE_version__", ""))
        return file_format, version

    def read(self, resource: H5Reader, source_patch_id=(), **kwargs) -> SpoolType:
        """Read a dascore file."""
        patches = []
        source_patch_ids = _normalize_source_patch_ids(source_patch_id)
        try:
            waveform_group = resource["waveforms"]
        except (KeyError, IndexError):
            return dc.spool([])
        for patch_group in waveform_group.values():
            patch_name = str(patch_group.name).rsplit("/", maxsplit=1)[-1]
            if source_patch_ids and patch_name not in source_patch_ids:
                continue
            patch = _read_patch(patch_group, **kwargs)
            if not patch.data.size and not _kwargs_empty(kwargs):
                continue
            patches.append(patch)
        return dc.spool(patches)

    def scan(self, resource: H5Reader, **kwargs):
        """
        Get patch info by iterating waveform groups in the file.

        Parameters
        ----------
        resource
            A path to the file.
        """
        return _get_contents_from_patch_groups_generic(resource)
