"""
Module for static, matplotlib-based visualizations and figure generation.
"""
from __future__ import annotations

from dascore.core.accessor import register_patch_accessor

from .map_fiber import map_fiber
from .spectrogram import spectrogram
from .waterfall import waterfall
from .wiggle import wiggle


@register_patch_accessor("viz")
class VizPatchAccessor:
    """Visualization namespace for Patch."""

    def __init__(self, patch):
        self._patch = patch

    def waterfall(self, *args, **kwargs):
        """See [`waterfall`](`dascore.viz.waterfall.waterfall`)."""
        return waterfall(self._patch, *args, **kwargs)

    def spectrogram(self, *args, **kwargs):
        """See [`spectrogram`](`dascore.viz.spectrogram.spectrogram`)."""
        return spectrogram(self._patch, *args, **kwargs)

    def wiggle(self, *args, **kwargs):
        """See [`wiggle`](`dascore.viz.wiggle.wiggle`)."""
        return wiggle(self._patch, *args, **kwargs)

    def map_fiber(self, *args, **kwargs):
        """See [`map_fiber`](`dascore.viz.map_fiber.map_fiber`)."""
        return map_fiber(self._patch, *args, **kwargs)
