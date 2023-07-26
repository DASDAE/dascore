"""
Module for static, matplotlib-based visualizations and figure generation.
"""
from __future__ import annotations
from dascore.utils.misc import MethodNameSpace

from .spectrogram import spectrogram
from .waterfall import waterfall
from .wiggle import wiggle


class VizPatchNameSpace(MethodNameSpace):
    """A class for storing visualization namespace."""

    waterfall = waterfall
    spectrogram = spectrogram
    wiggle = wiggle
