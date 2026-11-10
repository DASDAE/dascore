"""
Support for the DASCore Arrow patch format.

One patch per Arrow IPC file (``.arrow``) or stream (``.arrows``): the
values are one column, ``data``, in C order, and the dims, coordinates and
attrs are JSON in the schema metadata.
"""

from __future__ import annotations

from .core import ArrowV1
