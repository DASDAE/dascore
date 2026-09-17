"""XDAS NetCDF input mapped to DASCore Patches and Spools."""

from __future__ import annotations

from typing import Literal

import numpy as np

import dascore as dc
from dascore.io import FiberIO, ScanPayload, make_scan_payload
from dascore.io.utils import resolve_keyed_source, windows_to_slices
from dascore.io.xdas.utils import (
    get_attrs,
    get_coords,
    is_xdas_file,
    open_signals,
    require_filters,
)
from dascore.utils.hdf5 import H5Reader
from dascore.utils.io import _normalize_source_patch_keys
from dascore.utils.misc import raise_on_extra_kwargs


class XdasV1(FiberIO):
    """Read XDAS arrays and collections, with native paths as patch keys."""

    name = "xdas"
    version = "1"
    preferred_extensions = ("nc", "nc4", "netcdf")

    def get_format(
        self, resource: H5Reader, **kwargs
    ) -> tuple[str, str] | Literal[False]:
        """Detect XDAS metadata without reading signal data."""
        return (self.name, self.version) if is_xdas_file(resource) else False

    def scan(
        self, resource: H5Reader, snap: bool = True, **kwargs
    ) -> list[ScanPayload]:
        """Return one metadata payload per signal without loading its data."""
        with open_signals(resource) as signals:
            return [
                make_scan_payload(
                    attrs=get_attrs(variable, key),
                    coords=get_coords(dataset, variable, specs, snap=snap),
                    dtype=str(variable.dtype),
                    source_patch_key=key,
                )
                for key, (dataset, variable, specs, _) in signals.items()
            ]

    def read(
        self, resource: H5Reader, source_patch_key=(), snap: bool = True, **kwargs
    ) -> dc.Spool:
        """Load selected signals and coordinate ranges from an XDAS file."""
        patches = []
        with open_signals(resource) as signals:
            keys = _normalize_source_patch_keys(source_patch_key) or tuple(signals)
            for key in keys:
                dataset, variable, specs, node = resolve_keyed_source(signals, key)
                coords = get_coords(dataset, variable, specs, snap=snap)
                selection = {
                    name: value
                    for name, value in kwargs.items()
                    if name in coords.coord_map and value is not None
                }
                require_filters(node)
                coords, data = coords.select(array=variable, **selection)
                if data.size:
                    patches.append(
                        dc.Patch(
                            data=np.asarray(data),
                            coords=coords,
                            dims=variable.dims,
                            attrs=get_attrs(variable, key),
                        )
                    )
        return dc.spool(patches)

    def read_array(
        self,
        resource: H5Reader,
        windows: dict[str, tuple[int, int]],
        source_patch_key="",
        snap: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Read absolute sample windows for one signal without decoding coords."""
        raise_on_extra_kwargs(kwargs, "windows, source_patch_key and snap")
        with open_signals(resource) as signals:
            _, variable, _, node = resolve_keyed_source(signals, source_patch_key)
            slices = windows_to_slices(windows, variable.dims, variable.shape)
            require_filters(node)
            return variable[slices].to_numpy()
