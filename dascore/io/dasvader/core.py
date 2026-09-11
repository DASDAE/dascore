"""IO module for reading DASVader JLD2 data."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, slice_dataset
from dascore.utils.hdf5 import H5Reader

from .utils import (
    DATA_NAMES,
    _dereference,
    _get_attr_dict,
    _get_coord_manager,
    _get_data_and_dims,
    _get_reference_names,
    _is_dasvader_jld2,
)


class DASVaderV1(FiberIO):
    """
    Support for DASVader JLD2 files.

    Notes
    -----
    Legacy DASVader files may contain anonymous JLD2 object references. DASCore
    reads these references when supported by HDF5 and raises
    `DASVaderCompatibilityError` with compatibility instructions when
    dereferencing fails. A known working stack for
    such legacy files is `h5py<3.16` with `HDF5 1.14.x`.
    """

    name = "DASVader"
    preferred_extensions = ("jld2",)
    version = "1"

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        if _is_dasvader_jld2(resource):
            return self.version
        return None

    def get_metadata(self, resource: H5Reader, *, snap: bool = True) -> list[dc.Patch]:
        """Scan a DASVader file, return summary information about the file."""
        rec = resource["dDAS"][()]
        cm = _get_coord_manager(resource, rec)
        ref_names = set(_get_reference_names(resource))
        attrs = (
            _get_attr_dict(_dereference(resource, rec["atrib"], "atrib"))
            if "atrib" in ref_names
            else {}
        )
        data_ref = next(iter(DATA_NAMES & ref_names), None)
        dtype = (
            str(_dereference(resource, rec[data_ref], data_ref).dtype)
            if data_ref
            else ""
        )
        attrs = dc.PatchAttrs.from_dict(attrs)
        return [dc.Patch(attrs=attrs, coords=cm, dtype=dtype)]

    def read_array(
        self, resource: H5Reader, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """
        Slice the data reference's dataset directly.

        Only the ``dDAS`` record, which names the reference, is read
        besides the requested block. A file holds one patch, so the key
        `scan` reports is empty; any other is refused.
        """
        dataset, dims = _get_data_and_dims(resource)
        where = str(getattr(resource, "filename", "the resource"))
        resolve_keyed_source({"": dataset}, key, where=where)
        return slice_dataset(dataset, dims, windows)
