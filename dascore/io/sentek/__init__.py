"""
Module for reading DAS data recorded by Sentek interrogator

Examples
--------

import dascore as dc

data_sentek = dc.spool('path_to_file.das')
"""
from .core import SentekV5
