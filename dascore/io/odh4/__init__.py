"""
Support for the OptaSense ODH4 HDF5 format.

This layout is produced by OptaSense ODH-4 interrogators and is used, for
example, by the University of Wisconsin-Madison SURF (Sanford Underground
Research Facility) deployment contributed to the PubDAS Global DAS Month
(February 2023) dataset.

Each file holds a single "raw_data" dataset of shape (channel, time) whose
root attributes carry the acquisition metadata: start/end times, sampling
rate, channel range and spacing, gauge length, raw data units (phase shift
in radians), and the scale factor to strain.
"""

from .core import ODH4V1
