"""
Core module for visualization.
"""
import fios
from fios.utils.misc import pass_through_method

from .waterfall import waterfall


class TraceViz:
    """A class for storing visualization namespace."""

    def __init__(self, trace: "fios.Patch"):
        self.trace = trace

    waterfall = pass_through_method("trace")(waterfall)
