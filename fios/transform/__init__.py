"""
A module for applying transformation to Patches.
"""

from fios.utils.misc import MethodNameSpace
from .strain import velocity_to_strain_rate


class TransformPatchNameSpace(MethodNameSpace):
    """Patch namesapce for transformations."""

    velocity_to_strain_rate = velocity_to_strain_rate
