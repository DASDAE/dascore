"""IO module for reading seismic ASDF files."""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, windows_to_slices
from dascore.utils.hdf5 import H5Reader

from .utils import VERSION, get_asdf_version, get_metadata, group_traces, scan_traces


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
        return get_metadata(scan_traces(resource), snap)

    def _prepare_read(self, manager, snap, keys=frozenset()):
        """Share one waveform scan across every logical patch in this read."""
        resource = manager.get_resource(H5Reader)
        traces = scan_traces(resource)
        groups = group_traces(traces)

        def load(requests):
            for windows, key in requests:
                yield self._read_group(resource, groups, windows, key)

        return get_metadata(traces, snap), load

    def read_array(
        self, resource: H5Reader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Stack the keyed group's selected channels and time window."""
        groups = group_traces(scan_traces(resource))
        return self._read_group(resource, groups, windows, key)

    @staticmethod
    def _read_group(resource, groups, windows, key):
        """Read only the selected slice of each selected waveform."""
        traces = sorted(
            resolve_keyed_source(groups, key, str(resource.filename)),
            key=lambda item: (item.station, item.source_id),
        )
        dtype = np.result_type(*[item.dtype for item in traces])
        channels, time = windows_to_slices(
            windows, (len(traces), traces[0].sample_count)
        )
        selected = traces[channels]
        array = np.empty((len(selected), time.stop - time.start), dtype=dtype)
        for row, trace in enumerate(selected):
            array[row] = resource[trace.path][time]
        return array
