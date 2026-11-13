"""
Support for the Nokia Bell Labs multi-span (OFDR) DAS formats.

These are written by the Nokia Bell Labs interrogator used on the Ocean
Observatories Initiative Regional Cabled Array (2025-2026). There are two
variants: an extension-less binary file framed by ``OFDR1MAGIC`` holding
chunks of 32 time samples for every channel, and ``downsampled_*.h5`` files
holding a single ``data`` dataset ordered (distance, time).

When the recorder masked parts of the fiber (its ``masked`` spans add up
to the stored channel count), each span is its own patch, keyed by its
position ("0", "1", ...); otherwise the file is one patch starting at 0 m.

Data are raw phase in radians. Per the dataset's readme, absolute times
lag real time by several seconds, by an amount the metadata cannot recover;
the reader does not correct this. The dataset's reference code converts
the phase to strain rate by multiplying it by
``1.55e-6 / (4 * pi * 1.4677 * 0.78) / (Tp / Npulses) / gauge_length``,
where ``Tp`` and ``Npulses`` are the file's acquisition parameters.

Examples
--------
>>> import dascore as dc
>>> from dascore.utils.downloader import fetch
>>> spool = dc.spool(fetch("nokia_ofdr_1"))
>>> len(spool)
3
"""
from __future__ import annotations
from .core import NokiaOFDRV1, NokiaOFDRH5V1
