"""
Module for static visualizations and figure generation.

For interactive visualizations see :module:`dascore.workbench`
"""
from dascore.utils.misc import MethodNameSpace

from .spectrogram import spectrogram
from .waterfall import waterfall


class VizPatchNameSpace(MethodNameSpace):
    """A class for storing visualization namespace."""

    waterfall = waterfall
    spectrogram = spectrogram
