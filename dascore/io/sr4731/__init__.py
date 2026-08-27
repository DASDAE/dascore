"""
SR-4731 OTDR SOR file support.

SR-4731 is the Telcordia OTDR Data Format, formerly known as the Bellcore
OTDR Data Format. The current reader supports the OFL100/FIBERCLOUD subset
represented by DASCore's test data. It intentionally does not implement every
SR-4731 block or vendor extension.
"""

from __future__ import annotations

from .core import SR4731V200 as SR4731V200
