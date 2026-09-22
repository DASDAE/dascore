"""
Sintela readers.

Standard TS05 protobuf selections allocate only the requested window and decode overlapping packets. All headers are checked and metadata collected; unselected payloads are streamed past or skipped. Other layouts use full decoding. Use absolute time bounds for spool selections; `samples=True` or `relative=True` may load the full patch.
"""
from .core import SintelaBinaryV3
from .core import SintelaProtobufV1
