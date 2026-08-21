"""
This module adds IO support for the files written by the Julia package DASVader:
https://github.com/marianoarnaiz/DASvader.jl

Interrogator identity
---------------------
The file's ``Hostname`` is reported as ``interrogator.name``. It was
previously reported as ``host_name``, which no longer appears.
"""
from __future__ import annotations

from .core import DASVaderV1
