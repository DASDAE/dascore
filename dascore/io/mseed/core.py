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

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import dascore as dc
from dascore.constants import snap_type
from dascore.exceptions import InvalidFiberFileError
from dascore.io import FiberIO
from dascore.io.utils import resolve_keyed_source, windows_to_slices
from dascore.utils.io import LocalPath
from dascore.utils.misc import optional_import

from .utils import (
    _can_merge_summaries,
    _coalesce_source_traces,
    _detect_format,
    _group_segments,
    _merge_summaries,
    _metadata_from_summaries,
    _record_to_summary,
    _scan_patches,
    _source_patch_key,
    _TraceSummary,
)


@dataclass(frozen=True)
class _ReadSummary(_TraceSummary):
    """A read-only header plus record ordinals retained for this read operation."""

    records: list[int]


def _scan_read_records(resource, pymseed):
    """Keep record membership while applying the scanner's grouping rules."""
    headers = [
        _ReadSummary(**_record_to_summary(record, pymseed).__dict__, records=[index])
        for index, record in enumerate(
            pymseed.MS3Record.from_file(str(resource), unpack_data=False)
        )
    ]

    def merge(pending, summary):
        pending.records.extend(summary.records)
        return _merge_summaries(pending, summary)

    summaries = _coalesce_source_traces(
        [header for header in headers if header.sample_count],
        _can_merge_summaries,
        merge,
    )
    groups = {
        _source_patch_key(group): segments
        for group, segments in _group_segments(summaries)
    }
    return headers, summaries, groups


class MSeedV2(FiberIO):
    """Support MiniSEED version 2 files."""

    name = "MSEED"
    preferred_extensions = ("mseed", "msd", "miniseed")
    version = "2"

    def get_version(self, resource: LocalPath, **kwargs) -> str | None:
        """Return the file version when the resource matches this family."""
        return _match[1] if (_match := _detect_format(resource)) else None

    def get_metadata(
        self, resource: LocalPath, *, snap: snap_type = True
    ) -> list[dc.Patch]:
        """Scan a MiniSEED file."""
        pymseed = optional_import("pymseed")
        return _scan_patches(resource, pymseed)

    def _prepare_read(self, manager, snap):
        """Share one header scan across every logical patch in this read."""
        resource = manager.get_resource(LocalPath)
        pymseed = optional_import("pymseed")
        headers, summaries, groups = _scan_read_records(resource, pymseed)

        def load(requests):
            return self._read_arrays(resource, pymseed, headers, groups, requests)

        return _metadata_from_summaries(summaries), load

    def read_array(
        self, resource: LocalPath, windows: dict[str, tuple[int, int]], key: str = ""
    ) -> np.ndarray:
        """Decode one compatible MiniSEED group in channel/time order."""
        pymseed = optional_import("pymseed")
        headers, _, groups = _scan_read_records(resource, pymseed)
        (array,) = self._read_arrays(
            resource, pymseed, headers, groups, [(windows, key)]
        )
        return array

    @staticmethod
    def _read_arrays(resource, pymseed, headers, groups, requests):
        """Copy selected record slices directly into their final output arrays."""
        arrays, copies = [], defaultdict(list)
        for windows, key in requests:
            summaries = sorted(
                resolve_keyed_source(groups, key),
                key=lambda item: (item.station, item.source_id),
            )
            dtype = np.result_type(*[item.dtype for item in summaries])
            shape = (len(summaries), summaries[0].sample_count)
            channels, time = windows_to_slices(windows, ("channel", "time"), shape)
            selected = summaries[channels]
            array = np.empty((len(selected), time.stop - time.start), dtype=dtype)
            arrays.append(array)
            for row, summary in enumerate(selected):
                offset = 0
                for index in summary.records:
                    count = headers[index].sample_count
                    start, stop = (
                        max(time.start, offset),
                        min(time.stop, offset + count),
                    )
                    if start < stop:
                        target = array[row, start - time.start : stop - time.start]
                        copies[index].append(
                            (slice(start - offset, stop - offset), target)
                        )
                    offset += count
        if copies:
            record_count = 0
            for index, record in enumerate(
                pymseed.MS3Record.from_file(str(resource), unpack_data=False)
            ):
                record_count += 1
                targets = copies.pop(index, ())
                if not targets:
                    continue
                header = headers[index]
                if (
                    str(record.sourceid) != header.source_id
                    or int(record.starttime) != header.start_ns
                    or float(record.samprate) != header.sample_rate
                    or str(record.formatversion) != header.format_version
                    or int(record.samplecnt) != header.sample_count
                    or str(record.encoding) != header.encoding
                ):
                    raise InvalidFiberFileError(
                        "MiniSEED decoded segments disagree with scanned headers."
                    )
                record.unpack_data()
                data = np.asarray(record.np_datasamples)
                if data.shape != (header.sample_count,) or data.dtype != np.dtype(
                    header.dtype
                ):
                    raise InvalidFiberFileError(
                        "MiniSEED decoded segments have the wrong shape or dtype."
                    )
                for selection, target in targets:
                    target[:] = data[selection]
            if copies or record_count != len(headers):
                raise InvalidFiberFileError(
                    "MiniSEED decoded segments do not match the scanned record count."
                )
        yield from arrays


class MSeedV3(MSeedV2):
    """Support MiniSEED version 3 files."""

    version = "3"
