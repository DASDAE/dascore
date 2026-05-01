"""
Sintela protobuf reader.

This module supports Sintela's MTLV-wrapped protobuf recordings. Format
detection only inspects the MTLV envelope and does not require protobuf to be
installed. Reading and scanning lazily import protobuf support when needed.
"""

from .core import SintelaProtobufV1
