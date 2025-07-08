"""Core module for reading and writing DASDAE format."""

from __future__ import annotations

import contextlib

import pandas as pd

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.utils.hdf5 import (
    H5Reader,
    HDFPatchIndexManager,
    NodeError,
    PyTablesReader,
    PyTablesWriter,
)
from dascore.utils.misc import unbyte
from dascore.utils.patch import get_patch_names

from .utils import (
    _get_contents_from_patch_groups,
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

    def write(self, spool: SpoolType, resource: PyTablesWriter, index=False, **kwargs):
        """
        Write a collection of patches to a DASDAE file.

        Parameters
        ----------
        spool:
            A collection of patches or a spool (same thing).
        resource
            The path to the file.
        index
            If True, create an index for the file. This slows down
            writing but will make reading/scanning the file more efficient.
            This is recommended for files with many patches and not recommended
            for files with few patches.
        """
        # write out patches
        _write_meta(resource, self.version)
        # get an iterable of patches and save them
        patches = [spool] if isinstance(spool, dc.Patch) else spool
        # create new node called waveforms, else suppress error if it
        # already exists.
        with contextlib.suppress(NodeError):
            resource.create_group(resource.root, "waveforms")
        waveforms = resource.get_node("/waveforms")
        # write new patches to file
        patch_names = get_patch_names(patches).values
        for patch, name in zip(patches, patch_names):
            _save_patch(patch, waveforms, resource, name)
        indexer = HDFPatchIndexManager(resource)
        if index or indexer.has_index:
            df = self._get_patch_summary(patches)
            indexer.write_update(df)

    def _get_patch_summary(self, patches) -> pd.DataFrame:
        """Get a patch summary to put into index."""
        df = (
            dc.scan_to_df(patches)
            .assign(
                path=lambda x: get_patch_names(x),
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

    def read(self, resource: PyTablesReader, **kwargs) -> SpoolType:
        """Read a dascore file."""
        patches = []
        try:
            waveform_group = resource.root["/waveforms"]
        except (KeyError, IndexError):
            return dc.spool([])
        for patch_group in waveform_group:
            patches.append(_read_patch(patch_group, **kwargs))
        return dc.spool(patches)

    def scan(self, resource: PyTablesReader, **kwargs):
        """
        Get the patch info from the file.

        First we check if the file is using an index. If so, the index is
        returned. Otherwise, iterate over each waveform group and assemble
        content information.

        Parameters
        ----------
        resource
            A path to the file.
        """
        indexer = HDFPatchIndexManager(resource.filename)
        if indexer.has_index:
            # We need to change the path back to the file rather than internal
            # HDF5 path so it works with FileSpool and such.
            records = indexer.get_index().assign(path=str(resource)).to_dict("records")
            return [dc.PatchAttrs(**x) for x in records]
        else:
            file_format = self.name
            version = resource.root._v_attrs.__DASDAE_version__
            return _get_contents_from_patch_groups(resource, version, file_format)

    def index(self, path):
        """Index the dasdae file."""
        indexer = HDFPatchIndexManager(path)
        if not indexer.has_index:
            df = dc.scan_to_df(path)
            indexer.write_update(df)
