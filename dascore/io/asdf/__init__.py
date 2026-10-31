"""
Read the waveforms of seismic ASDF files.

The Adaptable Seismic Data Format (https://asdf-definition.readthedocs.io)
is HDF5 holding one dataset per trace under ``Waveforms/<NET.STA>/``, named
``NET.STA.LOC.CHA__<start>__<end>__<tag>``. It is not the astronomy
Advanced Scientific Data Format. Only version 1.0 waveforms are read;
StationXML, events, provenance and auxiliary data are ignored.

Traces map to patches as in the MiniSEED reader: 2D ``("channel", "time")``
patches with per-channel ``source_id`` (``NET.STA.LOC.CHA``), ``network``,
``station``, ``location`` and ``seed_channel`` coords, where ``channel`` is
a stable integer source index, not distance. Traces join one patch only
when they share network, location, SEED channel, start time, sample rate,
sample count and waveform tag. The ``tag`` attr is
``NET.LOC.CHA.<waveform tag>``, so groups stay apart under ``chunk``.

Some DAS archives give every fiber point its own location code, which
reads as one single-channel patch per point. Join them along ``channel``
with ``concatenate``, grouping by nothing and dropping the differing tags:

```python
import dascore as dc
spool = dc.spool("path/to/file.h5")
patch = spool.concatenate(channel=None, group=(), conflict="drop")[0]
```
"""

from __future__ import annotations

from .core import ASDFV1 as ASDFV1
