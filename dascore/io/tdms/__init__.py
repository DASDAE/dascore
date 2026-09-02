"""
Module for reading and writing TDMS fiber data recorded by Silixa.

Interrogator identity
---------------------
``SystemInfomation.OS.HostName`` is reported as ``interrogator.name`` (eg
"iDAS005"). No ``interrogator.serial_number`` is set, for the reasons given
in :mod:`dascore.io.silixah5`. Earlier versions reported
``Devices0.SerialNum`` as ``interrogator.serial_number``.
"""
from __future__ import annotations
from .core import TDMSFormatterV4713
