"""IO support for PUNIU HiFi-DAS HDF5 files.

HiFi-DAS is a distributed acoustic sensing interrogator developed by PUNIU
using TGD-OFDR sensing.

Website: https://puniutech.com/en/products/hifi-das-v2-0/

Version 1 stores a two-dimensional differential signal in ``/default``.
Distance is measured from the first recorded channel in each file. Original
channel numbers are retained as an auxiliary coordinate. The acquisition
clock is preserved without a timezone adjustment; signal values are returned
without calibration or an assigned physical unit.
"""

from .core import PuniuV1
