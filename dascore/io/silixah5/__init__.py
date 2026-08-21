"""
IO support for Silixa HDF5 files.

Website: https://silixa.com/

Interrogator identity
---------------------
``SystemInfomation.OS.HostName`` is reported as ``interrogator.name`` (eg
"iDAS20110", "Carina-P52"). No ``interrogator.serial_number`` is set: the
``Chassis`` and ``Devices<N>`` serials name parts inside the unit -- a PXI
crate, the cards in its slots -- rather than the interrogator, and these
files state no serial for the unit itself. Earlier versions reported
``Devices1.SerialNum`` as ``interrogator.serial_number``.
"""

from .core import SilixaH5V1, SilixaH5V2
