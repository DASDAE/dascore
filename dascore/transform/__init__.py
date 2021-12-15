"""
A module for applying transformation to Patches.
"""

from dascore.utils.misc import MethodNameSpace
from .strain import velocity_to_strain_rate
from .spectro import spectrogram
from .fft import rfft


class TransformPatchNameSpace(MethodNameSpace):
    """Patch namesapce for transformations."""

    velocity_to_strain_rate = velocity_to_strain_rate
    spectrogram = spectrogram
    rfft = rfft
