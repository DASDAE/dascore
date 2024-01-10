"""
Module for reading DAS data recorded by Sentek interrogator

Examples
--------

import dascore as dc
from dascore.utils.downloader import fetch

path_to_sentek_file = fetch("DASDMSShot00_20230328155653619.das")
sentek_patch = dc.spool(path_to_sentek_file)[0]
"""
from .core import SentekV5
