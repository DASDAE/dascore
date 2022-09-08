"""
Core module for reading and writing pickle format.
"""
import contextlib
from typing import Union

import dascore as dc
from dascore.constants import SpoolType, path_types
from dascore.core.schema import PatchFileSummary
from dascore.io.core import FiberIO
from dascore.io.dasdae.utils import _read_patch, _save_patch, _write_meta
from dascore.utils.docs import compose_docstring
from dascore.utils.hdf5 import HDFPatchIndexManager, open_hdf5_file
from dascore.utils.patch import get_default_patch_name


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

    def write(self, spool: SpoolType, path: path_types, **kwargs):
        """Read a Patch/Spool from disk."""
        with open_hdf5_file(path, mode="a") as h5:
            _write_meta(h5, self.version)
            # get an iterable of patches and save them
            patches = [spool] if isinstance(spool, dc.Patch) else spool
            waveforms = h5.create_group(h5.root, "waveforms")
            for spool in patches:
                _save_patch(spool, waveforms, h5)

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


@compose_docstring(table_columns=list(PatchFileSummary.__fields__))
class DASDAEV2(DASDAEV1):
    """
    Provides IO support for the DASDAE format version 2.

    DASDAE V2 adds a query table located at /root/waveforms/.index which
    allows large files to be scanned quickly and read operations planed.

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
    /root/waveforms/.index - Index table of waveform contents. The columns
    in the table are:
    {table_columns}

    Notes
    -----
    Unlinke V1, empty data patches are not saved.
    """

    version = "2"

    def write(self, spool: SpoolType, path: path_types, **kwargs):
        """Read a Patch/Spool from disk."""
        # update index of table, get patch names
        df = (
            dc.scan_to_df(spool)
            .assign(
                path=[f"waveforms/{get_default_patch_name(x)}" for x in spool],
                file_format=self.name,
                file_version=self.version,
            )
            .dropna(subset=["time_min", "time_max", "distance_min", "distance_max"])
        )
        # write out patches
        with open_hdf5_file(path, mode="a") as h5:
            _write_meta(h5, self.version)
            # get an iterable of patches and save them
            patches = [spool] if isinstance(spool, dc.Patch) else spool
            waveforms = h5.create_group(h5.root, "waveforms")
            for ind in df.index:
                _save_patch(patches[ind], waveforms, h5)
        # write new index to file.
        HDFPatchIndexManager(path).write_update(df)

    def scan(self, path):
        """Return the info from the file."""
        indexer = HDFPatchIndexManager(path)
        # We need to change the path back to the file rather than internal
        # HDF5 path so it works with FileSpool and such.
        records = indexer.get_index().assign(path=str(path)).to_dict("records")
        return [PatchFileSummary(**x) for x in records]
