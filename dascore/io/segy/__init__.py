"""
SEGY format support module.

Notes
-----
- Distance information is not found in most SEGY DAS files so returned
  dimensions are "channel" and "time" rather than "distance" and "time".

Examples
--------
import dascore as dc
from dascore.utils.downloader import fetch

# get the path to a segy file.
path = fetch("conoco_segy_1.sgy")

segy_patch = dc.spool(path)[0]
"""

from .core import SegyV2
