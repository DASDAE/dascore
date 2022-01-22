"""
Core module for reading and writing pickle format.
"""
import contextlib
from typing import Union

import tables

import dascore as dc
from dascore.constants import StreamType
from dascore.io.core import FiberIO
from dascore.io.dasdae.utils import _read_patch, _save_patch, _write_meta


class DASDAEIO(FiberIO):
    """
    Provides IO support for the DASDAE format.
    """

    name = "DASDAE"
    preferred_extensions = ("h5", "hdf5")

    def write(self, patch, path, **kwargs):
        """Read a Patch/Stream from disk."""
        with tables.open_file(path, mode="a") as h5:
            _write_meta(h5)
            # get an iterable of patches and save them
            patches = [patch] if isinstance(patch, dc.Patch) else patch
            waveforms = h5.create_group(h5.root, "waveforms")
            for patch in patches:
                _save_patch(patch, waveforms, h5)

    def get_format(self, path) -> Union[tuple[str, str], bool]:
        """Return the format from a dasdae file."""
        with tables.open_file(path, mode="r") as fi:
            is_dasdae, version = False, ""  # NOQA
            with contextlib.suppress(KeyError):
                is_dasdue = fi.root._v_attrs["__format__"] == "DASDAE"
                dasdae_version = fi.root._v_attrs["__DASDAE_version__"]
            if is_dasdue:
                return (self.name, dasdae_version)
            return False

    def read(self, path, **kwargs) -> StreamType:
        """
        Read a dascore file.
        """
        patches = []
        with tables.open_file(path, mode="r") as fi:
            try:
                waveform_group = fi.root["/waveforms"]
            except KeyError:
                return dc.Stream([])
            for patch_group in waveform_group:
                patches.append(_read_patch(patch_group, **kwargs))
        return dc.Stream(patches)
