"""
RSF format support module.

Notes
-----
-


Examples
--------
import dascore as dc
from dascore.utils.downloader import fetch

# get the path to a segy file.
path = fetch("conoco_segy_1.sgy")

segy_patch = dc.spool(path)[0]
"""

from .core import RSFV1
