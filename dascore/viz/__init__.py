"""
Module for static, matplotlib-based visualizations and figure generation.
"""
from __future__ import annotations
from dascore.utils.namespace import PatchNameSpace

from .spectrogram import spectrogram
from .waterfall import waterfall
from .wiggle import wiggle
from .map_fiber import map_fiber


class VizPatchNameSpace(PatchNameSpace):
    """A class for storing visualization namespace."""

    name = "viz"

    waterfall = waterfall
    spectrogram = spectrogram
    wiggle = wiggle
    map_fiber = map_fiber
