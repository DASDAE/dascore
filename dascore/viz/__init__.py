"""
Module for static, matplotlib-based visualizations and figure generation.
"""
from __future__ import annotations
from dascore.utils.namespace import (
    InventoryNameSpace,
    PatchNameSpace,
    SpoolNameSpace,
)

from .spectrogram import spectrogram
from .specplot import specplot
from .waterfall import waterfall
from .wiggle import wiggle
from .map_fiber import map_fiber
from .inventory import map_path, path, timeline
from .spool import calendar, coverage


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


class VizSpoolNameSpace(SpoolNameSpace):
    """The plots a spool can draw of itself."""

    name = "viz"

    coverage = coverage
    calendar = calendar

    def __getattr__(self, item):
        """Point a patch plot at the patch it needs."""
        if item in VizPatchNameSpace.__dict__:
            msg = (
                f"{item!r} draws a patch, and a spool is many of them. Merge "
                f"the ones you mean into one patch first, then draw it: "
                f"spool.chunk(time=None)[0].viz.{item}()"
            )
            raise AttributeError(msg)
        msg = f"{type(self).__name__!r} object has no attribute {item!r}"
        raise AttributeError(msg)
