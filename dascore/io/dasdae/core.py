"""
Core module for reading and writing pickle format.
"""
import contextlib
from typing import Union

import pandas as pd

import dascore as dc
from dascore.constants import SpoolType, path_types
from dascore.core.schema import PatchFileSummary
from dascore.io.core import FiberIO
from dascore.utils.hdf5 import HDFPatchIndexManager, NodeError, open_hdf5_file
from dascore.utils.patch import get_default_patch_name

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

    def write(self, spool: SpoolType, path: path_types, index=False, **kwargs):
        """
        Write a collection of patches to a DASDAE file.

        Parameters
        ----------
        spool:
            A collection of patches or a spool (same thing).
        path
            The path to the file.
        index
            If True, create an index for the file. This slows down
            writing but will make reading/scanning the file more efficient.
            This is recommended for files with many patches and not recommended
            for files with few patches.
        """
        # write out patches
        with open_hdf5_file(path, mode="a") as h5:
            _write_meta(h5, self.version)
            # get an iterable of patches and save them
            patches = [spool] if isinstance(spool, dc.Patch) else spool
            # create new node called waveforms, else suppress error if it
            # already exists.
            with contextlib.suppress(NodeError):
                h5.create_group(h5.root, "waveforms")
            waveforms = h5.get_node("/waveforms")
            # write new patches to file
            for patch in patches:
                _save_patch(patch, waveforms, h5)
        # if the index exists, or its creation requested, update it.
        indexer = HDFPatchIndexManager(path)
        if index or indexer.has_index:
            df = self._get_patch_summary(patches)
            indexer.write_update(df)

    def _get_patch_summary(self, patches) -> pd.DataFrame:
        """Get a patch summary to put into index."""
        df = (
            dc.scan_to_df(patches)
            .assign(
                path=[f"waveforms/{get_default_patch_name(x)}" for x in patches],
                file_format=self.name,
                file_version=self.version,
            )
            .dropna(subset=["time_min", "time_max", "distance_min", "distance_max"])
        )
        return df

    def get_format(self, path) -> Union[tuple[str, str], bool]:
        """Return the format from a dasdae file."""
        with open_hdf5_file(path, mode="r") as fi:
            is_dasdae, version = False, ""  # NOQA
            with contextlib.suppress(KeyError):
                is_dasdae = fi.root._v_attrs["__format__"] == "DASDAE"
                dasdae_file_version = fi.root._v_attrs["__DASDAE_version__"]
            if is_dasdae:
                return (self.name, dasdae_file_version)
            return False

    def read(self, path, **kwargs) -> SpoolType:
        """
        Read a dascore file.
        """
        patches = []
        with open_hdf5_file(path, mode="r") as fi:
            try:
                waveform_group = fi.root["/waveforms"]
            except KeyError:
                return dc.MemorySpool([])
            for patch_group in waveform_group:
                patches.append(_read_patch(patch_group, **kwargs))
        return dc.MemorySpool(patches)

    def scan(self, path):
        """
        Get the patch info from the file.

        First we check if the file is using an index. If so, the index is
        returned. Otherwise, iterate over each waveform group and assemble
        content information.

        Parameters
        ----------
        Path
            A path to the file.
        """
        indexer = HDFPatchIndexManager(path)
        if indexer.has_index:
            # We need to change the path back to the file rather than internal
            # HDF5 path so it works with FileSpool and such.
            records = indexer.get_index().assign(path=str(path)).to_dict("records")
            return [PatchFileSummary(**x) for x in records]
        else:
            with open_hdf5_file(path) as h5:
                file_format = self.name
                version = h5.root._v_attrs.__DASDAE_version__
                return _get_contents_from_patch_groups(path, version, file_format)

    def index(self, path):
        """Index the dasdae file."""
        indexer = HDFPatchIndexManager(path)
        if not indexer.has_index:
            df = dc.scan_to_df(path)
            indexer.write_update(df)
