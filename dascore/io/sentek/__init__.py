"""
Module for reading DAS data recorded by Sentek interrogator

Examples
--------

import dascore as dc

sentek_patch = dc.spool("examples://DASDMSShot00_20230328155653619.das")[0]
"""
from .core import SentekV5
