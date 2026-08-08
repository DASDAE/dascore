"""
Support for the Aragon Photonics HDAS HDF5 format.

HDAS interrogators store a 200-value float header whose fixed positions
carry the acquisition metadata (spatial sampling, trigger frequency and
downsampling factors, fiber position offset, and epoch start time). The
header spec here follows the reader script Aragon/IGN distribute with
their data (see e.g. the IGN-ADIF contribution to PubDAS Global DAS
Month, February 2023).

Two container variants exist in the wild:

- V1: ``File_Header`` (1, 200) float64 + ``HDAS_DATA`` (time, distance),
  holding strain rate (the vendor's own layout, e.g. IGN-ADIF).
- V2: ``hdas_header`` (200,) float32 + ``data`` (distance, time) with a
  ``start_time`` ISO attr (e.g. the ICM-CSIC Canalink deployment). The
  float32 header cannot hold an exact epoch, so the attr provides the
  start time, and the measurand is not recorded in the file.
"""

from .core import HDASV1, HDASV2
