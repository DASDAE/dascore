"""IO support for MiniSEED files.

DASCore maps MiniSEED source IDs into 2D ``("channel", "time")`` patches.
MiniSEED source identity is preserved as per-channel coordinates instead of
scalar attrs because one DASCore patch can contain many MiniSEED sources.
The preserved source coordinates are ``source_id``, ``network``, ``station``,
``location``, and ``seed_channel``.

Compatible MiniSEED sources are merged into one patch when they share the same
format version, network, location, SEED channel, start time, sample rate, and
sample count. Sources with different sample rates, start times, or sample
counts are returned as separate patches. When sources split across multiple
patches, DASCore keeps stable integer ``channel`` coordinate values so
selections still refer to the same source.

The MiniSEED ``channel`` coordinate is an integer source index, not optical
distance. Optical distance is usually stored in companion DAS metadata rather
than MiniSEED record headers and should be attached from that metadata when
available.

The mapping follows the FDSN source identifier convention, where a SEED NSLC
code maps to ``FDSN:<network>_<station>_<location>_<band>_<source>_<subsource>``.
See https://docs.fdsn.org/projects/source-identifiers/en/latest/definition.html.

For DAS MiniSEED, GEOFON recommends representing each fiber sampling point as
a station because each sampling point has its own position; the channel code
``HSF`` is recommended for fiber optic DAS. See
https://geofon.gfz.de/redmine/projects/redmine/wiki/DAS.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, windows_to_slices
from dascore.utils.io import LocalPath
from dascore.utils.misc import optional_import

from .utils import (
    _detect_format,
    _group_segments,
    _read_segments,
    _scan_patches,
    _scan_segments,
    _source_patch_key,
    _SourceWindows,
    _trace_time_window,
)


class MSeedV2(FiberIO):
    """Support MiniSEED version 2 files."""

    name = "MSEED"
    preferred_extensions = ("mseed", "msd", "miniseed")
    version = "2"

    def get_version(self, resource: LocalPath, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return _match[1] if (_match := _detect_format(resource)) else None

    def get_metadata(self, resource: LocalPath, *, snap: bool = True) -> list[dc.Patch]:
        """Scan a MiniSEED file."""
        pymseed = optional_import("pymseed")
        return _scan_patches(resource, pymseed)

    def read_array(
        self, resource: LocalPath, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Decode one compatible MiniSEED group in channel/time order."""
        pymseed = optional_import("pymseed")
        groups = {
            _source_patch_key(group): segments
            for group, segments in _group_segments(_scan_segments(resource, pymseed))
        }
        summaries = sorted(
            resolve_keyed_source(groups, key),
            key=lambda item: (item.station, item.source_id),
        )
        shape = (len(summaries), summaries[0].sample_count)
        channels, time = windows_to_slices(windows, ("channel", "time"), shape)
        selected = summaries[channels]
        if not selected or time.start == time.stop:
            return np.empty(
                (len(selected), time.stop - time.start), dtype=summaries[0].dtype
            )
        source_windows: _SourceWindows = {
            item.source_id: [_trace_time_window(item)] for item in selected
        }
        segments = _read_segments(resource, pymseed, source_windows=source_windows)
        segments = sorted(segments, key=lambda item: (item.station, item.source_id))
        return np.stack([segment.data[time] for segment in segments])


class MSeedV3(MSeedV2):
    """Support MiniSEED version 3 files."""

    version = "3"
