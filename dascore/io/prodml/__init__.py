"""Support for the PRODML format.

``dc.write`` supports one raw time-by-distance patch as a standalone PRODML
2.1-compatible HDF5 file. Full PRODML 2.3 conformance requires EPC/XML, which
DASCore does not write.

Reading accepts a ``SchemaVersion`` attr in place of ``schemaVersion`` and
integer times in nanoseconds as well as microseconds. A node with a single
time sample takes its step from the node's ``OutputDataRate``, or has no step
when that rate is missing, zero or not finite.

More info about ProdML can be found here:
https://www.energistics.org/prodml-developers-users
"""
from __future__ import annotations
from .core import ProdMLV2_0, ProdMLV2_1
