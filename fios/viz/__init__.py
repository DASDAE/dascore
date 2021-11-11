"""
Module for static visualizations and figure generation.

For interactive visualizations see :module:`fios.workbench`
"""
from fios.utils.misc import MethodNameSpace

from .waterfall import waterfall


class VizPatchNameSpace(MethodNameSpace):
    """A class for storing visualization namespace."""

    waterfall = waterfall
