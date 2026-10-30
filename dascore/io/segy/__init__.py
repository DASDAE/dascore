"""
SEGY format support module.

Notes
-----
- Distance information is not found in most SEGY DAS files so returned
  dimensions are "channel" and "time" rather than "distance" and "time".
- Big- and little-endian files are both read. A non-standard minor revision
  (eg 1.80) is read as its major revision when whole traces fill the file.
- A trace header second of 60 rolls into the next minute. A header without a
  valid date gives a time coordinate relative to the first sample, with a
  warning; set the start with `patch.update_coords(time_min=...)`.
- Segy standards found at: https://library.seg.org/pb-assets/technical-standards

segy v1 spec: seg_y_rev1-1686080991247.pdf

segy v2 spec: seg_y_rev2_0-mar2017-1686080998003.pdf

segy v2.1 spec: seg_y_rev2_1-oct2023-1701361639333.pdf

Examples
--------
import dascore as dc

segy_patch = dc.spool("examples://conoco_segy_1.sgy")[0]
"""

from .core import SegyV1_0, SegyV2_0, SegyV2_1
