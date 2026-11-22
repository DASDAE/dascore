"""
IO support for SAC binary files.

A SAC file holds one trace: a 632-byte header of 70 floats, 40 integers and
24 strings, then float32 samples, in either byte order. DASCore reads it as
a 2D ``("channel", "time")`` patch with one channel, as the MiniSEED reader
maps a trace. The trace's identity is kept as per-channel coordinates:
``network`` (KNETWK), ``station`` (KSTNM), ``location`` (KHOLE),
``seed_channel`` (KCMPNM), and ``source_id``, the FDSN source identifier
the MiniSEED reader builds from those codes
(``FDSN:<network>_<station>_<location>_<band>_<source>_<subsource>`` for a
three-letter channel code). The ``tag`` attr is ``NET.STA.LOC.CHA``.

``channel`` is the trace's index in its file, so it is 0 in every file. The
tag keeps traces apart, so ``chunk`` along time joins each trace's files and
never mixes traces. To join traces along ``channel`` where their times agree,
give each source one channel number, concatenate, grouping by nothing and
dropping the differing tags, then join along time:

```python
import dascore as dc

patches = list(dc.spool("path/to/sac_directory"))
sources = sorted({x.coords.get_array("source_id")[0] for x in patches})

def numbered(patch):
    channel = sources.index(patch.coords.get_array("source_id")[0])
    return patch.replace(coords=patch.coords.update(channel=[channel]))

joined = dc.spool([numbered(x) for x in patches]).concatenate(
    channel=None, group=(), conflict="drop"
)
merged = joined.chunk(time=None)
```

Time starts at the reference time (NZYEAR, NZJDAY, NZHOUR, NZMIN, NZSEC,
NZMSEC) plus B, with step DELTA. DELTA and B are read as the decimals they
were most likely written as: the shortest decimal that rounds to the stored
float, or a simple fraction a footer double holds exactly (1/3). So a float32
DELTA of 0.05 is an exact 1/20 s, where its binary value would add about a
nanosecond a sample. Header values set to SAC's undefined value (-12345) are
missing: an undefined B is zero, and without a reference time, time is a
timedelta starting at B. The version is NVHDR. Version 7 appends a footer of
doubles, from which DELTA and B are read in double precision.

Only evenly sampled time series (IFTYPE ITIME, LEVEN true) are read;
spectral, x-y and unevenly sampled files raise an `InvalidFiberFileError`.
Alphanumeric SAC files are not detected as SAC.

The SAC format is described in the SAC manual:
https://ds.iris.edu/files/sac-manual/manual/file_format.html.
"""

from __future__ import annotations

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.io import FiberIO
from dascore.utils.io import BinaryReader

from .utils import _get_metadata, _read_header, _read_samples, _read_trace_header


class SacV6(FiberIO):
    """Support SAC binary files of header version 6."""

    name = "SAC"
    preferred_extensions = ("sac",)
    version = "6"

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the header version when the resource is a SAC binary file."""
        return str(header.version) if (header := _read_header(resource)) else None

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Describe the trace from its header; a trace of no samples is no patch."""
        header = _read_trace_header(resource)
        return [_get_metadata(header, snap)] if header.npts else []

    def read_array(
        self,
        resource: BinaryReader,
        windows: windows_type = (),
        key: str = "",
    ) -> np.ndarray:
        """Read the samples in the windows."""
        return _read_samples(resource, _read_trace_header(resource), windows)


class SacV7(SacV6):
    """Support SAC binary files of header version 7, which add a footer."""

    version = "7"
