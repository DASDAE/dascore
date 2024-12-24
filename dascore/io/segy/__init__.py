"""
SEGY format support module.

Notes
-----
- Distance information is not found in most SEGY DAS files so returned
  dimensions are "channel" and "time" rather than "distance" and "time".
- Segy standards found at: https://library.seg.org/pb-assets/technical-standards

segy v1 spec: seg_y_rev1-1686080991247.pdf

segy v2 spec: seg_y_rev2_0-mar2017-1686080998003.pdf

segy v2.1 spec: seg_y_rev2_1-oct2023-1701361639333.pdf

Examples
--------
import dascore as dc
from dascore.utils.downloader import fetch

# get the path to a segy file.
path = fetch("conoco_segy_1.sgy")

segy_patch = dc.spool(path)[0]
"""

from .core import SegyV1_0, SegyV2_0, SegyV2_1
