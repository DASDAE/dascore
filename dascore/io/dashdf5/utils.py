"""Utilities for terra15."""
from __future__ import annotations

# --- Getting format/version


def _get_das_hdf_version(resource):
    """Return version string if resource is a DAS hdf5 v1."""
