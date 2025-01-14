"""Core module for reading and writing DASDAE format."""

from __future__ import annotations

import dascore as dc
from dascore.constants import SpoolType
from dascore.io import FiberIO
from dascore.utils.fs import get_path
from dascore.utils.hdf5 import (
    H5Reader,
    H5Writer,
)
from dascore.utils.misc import unbyte
from dascore.utils.patch import get_patch_names

from .utils import (
    _get_summary_from_patch_groups,
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

    def write(self, spool: SpoolType, resource: H5Writer, index=False, **kwargs):
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
        if "waveforms" not in resource:
            resource.create_group(resource, "waveforms")
        waveforms = resource["/waveforms"]
        # write new patches to file
        patch_names = get_patch_names(patches).values
        for patch, name in zip(patches, patch_names):
            _save_patch(patch, waveforms, resource, name)

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """Return the format from a dasdae file."""
        is_dasdae, version = False, ""  # NOQA
        attrs = resource.attrs
        file_format = unbyte(attrs.get("__format__", ""))
        if file_format != self.name:
            return False
        version = unbyte(attrs.get("__DASDAE_version__", ""))
        return file_format, version

    def read(self, resource: H5Reader, **kwargs) -> SpoolType:
        """Read a DASDAE file."""
        patches = []
        path = get_path(resource)
        format_version = unbyte(resource.attrs["__DASDAE_version__"])
        format_name = self.name
        try:
            waveform_group = resource["/waveforms"]
        except (KeyError, IndexError):
            return dc.spool([])
        for patch_group in waveform_group:
            pa = _read_patch(
                patch_group,
                path=path,
                format_name=format_name,
                format_version=format_version,
                **kwargs,
            )
            patches.append(pa)
        return dc.spool(patches)

    def scan(self, resource: H5Reader, **kwargs) -> list[dc.PatchSummary]:
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
        file_format = self.name
        return _get_summary_from_patch_groups(resource, file_format)
