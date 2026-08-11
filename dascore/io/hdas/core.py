"""Core modules for reading Aragon Photonics HDAS data."""

from __future__ import annotations

from typing import Literal

import dascore as dc
from dascore.constants import opt_timeable_types
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.utils.hdf5 import H5Reader

from .utils import (
    V1_DATA_NAME,
    V1_HEADER_NAME,
    V2_DATA_NAME,
    V2_HEADER_NAME,
    _get_patches,
    _get_v1_coords_and_attrs,
    _get_v2_coords_and_attrs,
    _is_hdas,
)


class HDASV1(FiberIO):
    """
    Support for the Aragon Photonics HDAS format (vendor layout).

    Files hold a ``File_Header`` (1, 200) float array whose fixed
    positions carry the acquisition metadata, and an ``HDAS_DATA``
    dataset of shape (time, distance) holding strain rate. Used e.g. by
    the IGN-ADIF deployment in the PubDAS Global DAS Month dataset.
    """

    name = "HDAS"
    version = "1"
    preferred_extensions = ("hdf5", "h5")
    # The variant's dataset names, header shape, and coord/attr getter.
    _header_name = V1_HEADER_NAME
    _header_shape = (1, 200)
    _data_name = V1_DATA_NAME
    _required_attr: str | None = None
    _get_coords_and_attrs = staticmethod(_get_v1_coords_and_attrs)

    def get_format(
        self,
        resource: H5Reader,
        **kwargs,
    ) -> tuple[str, str] | Literal[False]:
        """
        Return format name and version if resource is an HDAS file.

        Parameters
        ----------
        resource
            An open h5 file which might contain HDAS data.
        """
        is_hdas = _is_hdas(
            resource,
            self._header_name,
            self._header_shape,
            self._data_name,
            self._required_attr,
        )
        if is_hdas:
            return self.name, self.version
        return False

    def scan(self, resource: H5Reader, **kwargs) -> list[ScanPayload]:
        """Scan an HDAS file, return summary info about the contents."""
        coords, attrs_dict = self._get_coords_and_attrs(resource)
        attrs = dc.PatchAttrs.model_validate(attrs_dict)
        return [
            make_scan_payload(
                attrs=attrs, coords=coords, dtype=str(resource[self._data_name].dtype)
            )
        ]

    def read(
        self,
        resource: H5Reader,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        distance: tuple[float | None, float | None] | None = None,
        **kwargs,
    ) -> dc.BaseSpool:
        """Read an HDAS file into a spool."""
        coords, attrs_dict = self._get_coords_and_attrs(resource)
        patches = _get_patches(
            resource, self._data_name, coords, attrs_dict, time=time, distance=distance
        )
        return dc.spool(patches)


class HDASV2(HDASV1):
    """
    Support for the Aragon Photonics HDAS format (attr-timed variant).

    Files hold an ``hdas_header`` (200,) float32 header and a ``data``
    dataset of shape (distance, time) with a ``start_time`` ISO attr,
    which is authoritative since the float32 header cannot represent an
    exact epoch. Used e.g. by the ICM-CSIC Canalink deployment in the
    PubDAS Global DAS Month dataset. The measurand is not recorded in
    the file, so data units are left unset.
    """

    version = "2"
    _header_name = V2_HEADER_NAME
    _header_shape = (200,)
    _data_name = V2_DATA_NAME
    _required_attr = "start_time"
    _get_coords_and_attrs = staticmethod(_get_v2_coords_and_attrs)
