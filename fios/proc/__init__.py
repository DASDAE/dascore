"""
Module to Patch Processing.
"""
from fios.utils.misc import MethodNameSpace

from .decimate import decimate
from .detrend import detrend

# from .filter import pass_filter


class ProcessingPatchNamespace(MethodNameSpace):
    """Processing name space."""

    decimate = decimate
    detrend = detrend
    # pass_filter = pass_filter
