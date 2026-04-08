"""IO module for reading DASVader JLD2 data."""

from __future__ import annotations

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload
from dascore.utils.hdf5 import H5Reader

from .utils import (
    DATA_NAMES,
    _dereference,
    _get_attr_dict,
    _get_coord_manager,
    _get_reference_names,
    _is_dasvader_jld2,
    _read_dasvader,
)


class DASVaderV1(FiberIO):
    """
    Support for DASVader JLD2 files.

    Notes
    -----
    Legacy DASVader files may contain anonymous JLD2 object references. DASCore
    detects those files and raises `DASVaderCompatibilityError` with compatibility
    instructions instead of failing inside `h5py`. A known working stack for
    such legacy files is `h5py<3.16` with `HDF5 1.14.x`.
    """

    name = "DASVader"
    preferred_extensions = ("jld2",)
    version = "1"

    def get_format(self, resource: H5Reader, **kwargs) -> tuple[str, str] | bool:
        """
        Return format if file contains DASVader JLD2 data else False.

        Parameters
        ----------
        resource
            A path to the file which may contain DASVader data.
        """
        if _is_dasvader_jld2(resource):
            return self.name, self.version
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
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
        return [
            {
                "attrs": attrs,
                "coords": cm,
                "dims": cm.dims,
                "shape": cm.shape,
                "dtype": dtype,
            }
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read a DASVader spool of patches."""
        patches = _read_dasvader(resource, time=time, distance=distance)
        return dc.spool(patches)
