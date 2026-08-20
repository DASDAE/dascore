"""
Module for static, matplotlib-based visualizations and figure generation.
"""
from __future__ import annotations
from dascore.utils.namespace import InventoryNameSpace, PatchNameSpace

from .spectrogram import spectrogram
from .specplot import specplot
from .waterfall import waterfall
from .wiggle import wiggle
from .map_fiber import map_fiber
from .inventory import map_path, path, timeline


class VizPatchNameSpace(PatchNameSpace):
    """A class for storing visualization namespace."""

    name = "viz"

    waterfall = waterfall
    spectrogram = spectrogram
    specplot = specplot
    wiggle = wiggle
    map_fiber = map_fiber


class VizInventoryNameSpace(InventoryNameSpace):
    """The plots an inventory can draw of itself."""

    name = "viz"

    path = path
    map = map_path
    timeline = timeline
