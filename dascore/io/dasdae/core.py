"""Core module for reading and writing DASDAE format."""

from __future__ import annotations

import contextlib

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset
from dascore.utils.hdf5 import H5Reader, H5Writer
from dascore.utils.misc import unbyte
from dascore.utils.patch import get_patch_names

from .utils import (
    _get_contents_from_patch_groups_generic,
    _get_dims,
    _get_patch_group,
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
    # Version 2 writes ranges and segments as descriptions, not values.
    _compact_coords = False

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
            _save_patch(patch, waveforms, unique_name, compact=self._compact_coords)

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        attrs = resource.attrs
        file_format = unbyte(attrs.get("__format__", ""))
        if file_format != self.name:
            return None
        version = unbyte(attrs.get("__DASDAE_version__", ""))
        return version

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice one patch's data dataset directly.

        Only the group's ``_dims`` attribute and the requested hyperslab
        leave the file; patch attrs and coordinates are never parsed. See
        `FiberIO.read_array` for the window contract. ``source_patch_key``
        is the waveform group name `scan` reports; a positional index is
        not accepted, because DASDAE never synthesizes one.
        """
        group = _get_patch_group(resource, key)
        return slice_dataset(group["data"], _get_dims(group), windows)

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """
        Get patch info by iterating waveform groups in the file.

        Parameters
        ----------
        resource
            A path to the file.
        """
        return _get_contents_from_patch_groups_generic(resource, snap=snap)


class DASDAEV2(DASDAEV1):
    """
    DASDAE format version 2.

    Reads and writes as version 1, except that each coordinate node
    states its class (``object_type``) and describes itself: a range is stored as its
    start, extent, and step (the exact grid where the coordinate holds
    one, so a fractional sampling rate never drifts) at the cost of a few
    attributes however long it is; a segmented coordinate as a group of
    its segments; and only irregular coordinates as arrays of values.
    Gapped patches are therefore stored as they are rather than split.
    """

    version = "2"
    segmented_write = True
    _compact_coords = True
