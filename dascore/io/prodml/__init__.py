"""Support for the PRODML format.

``dc.write`` supports one raw time-by-distance patch as a standalone PRODML
2.1-compatible HDF5 file. Full PRODML 2.3 conformance requires EPC/XML, which
DASCore does not write.

Reading accepts a ``SchemaVersion`` attr in place of ``schemaVersion`` and
integer times in nanoseconds as well as microseconds, taking the unit from
the time array's ``Uom`` or, without one, from which unit puts the first time
on ``PartStartTime``. A node with a single
time sample takes its step from the node's ``OutputDataRate``, or has no step
when that rate is missing, zero or not finite.

``PRODML_FLAT`` reads files holding one root ``DAS`` (time, locus) dataset
whose attrs use PRODML raw-node and user-setting names (``PartStartTime``,
``OutputDataRate``, ``StartDistance``, ``SpatialResolution``,
``FibreLengthMultiplier``, ...). Distances are taken as metres, which these
files do not state. ``AmpScaling`` is the factor already applied to the stored
samples, so ``RawDataUnit`` describes them as stored; it is kept as the
``amp_scaling`` attr.

More info about ProdML can be found here:
https://www.energistics.org/prodml-developers-users
"""
from __future__ import annotations
from .core import ProdMLFlatV1, ProdMLV2_0, ProdMLV2_1
