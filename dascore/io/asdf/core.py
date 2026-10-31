"""IO module for reading seismic ASDF files."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.io import FiberIO
from dascore.io.mseed.utils import _allocate_rows
from dascore.io.utils import resolve_keyed_source
from dascore.utils.hdf5 import H5Reader

from .utils import VERSION, _scan, get_asdf_version, get_metadata, scan_groups


class ASDFV1(FiberIO):
    """Support the waveforms of seismic ASDF 1.0 files."""

    name = "ASDF"
    preferred_extensions = ("h5", "asdf")
    version = VERSION

    def get_version(self, resource: H5Reader, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return get_asdf_version(resource)

    def get_metadata(
        self, resource: H5Reader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Scan the waveforms' names and attrs."""
        groups, channel_map = scan_groups(resource)
        return get_metadata(groups, channel_map, snap)

    def _prepare_read(self, manager, snap, keys=frozenset()):
        """Scan once for the read, opening only the wanted keys' waveforms."""
        resource = manager.get_resource(H5Reader)
        groups, channel_map = _scan(resource, keys) if keys else scan_groups(resource)

        def load(requests):
            for windows, key in requests:
                yield self._reader(resource, groups, key)(windows)

        return get_metadata(groups, channel_map, snap), load

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Stack the keyed group's selected channels and time window."""
        return self._prepare_array_reader(resource, key=key)(windows)

    def _prepare_array_reader(self, resource: H5Reader, *, key: str = ""):
        """Resolve the keyed group from the handle's one shared scan."""
        return self._reader(resource, scan_groups(resource)[0], key)

    @staticmethod
    def _reader(resource, groups, key):
        """Return a reader of the keyed group's windows."""
        _, traces = resolve_keyed_source(groups, key, str(resource.filename))

        def read(windows):
            selected, time, array = _allocate_rows(traces, windows)
            for row, trace in enumerate(selected):
                array[row] = resource[trace.path][time]
            return array

        return read
