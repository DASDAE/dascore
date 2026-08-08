"""
Support for the AI4EPS event HDF5 format.

This format is used by the AI4EPS group's earthquake DAS datasets, such as
the quakeflow_das dataset of Ridgecrest and Long Valley events:
https://huggingface.co/datasets/AI4EPS/quakeflow_das

Each file contains one event recording in a single "data" dataset whose
attributes carry the acquisition metadata (begin_time, dt_s, dx_m, unit)
and the event metadata (event_id, magnitude, hypocenter, ...).
"""

from .core import AI4EPSV1
