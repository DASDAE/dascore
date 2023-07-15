"""
A module for applying transformation to Patches.

Transforms are defined as
"""

from dascore.utils.misc import MethodNameSpace

from .differentiate import differentiate
from .fft import rfft
from .fourier import dft, idft
from .integrate import integrate
from .spectro import spectrogram
from .strain import velocity_to_strain_rate


class TransformPatchNameSpace(MethodNameSpace):
    """Patch namesapce for transformations."""

    velocity_to_strain_rate = velocity_to_strain_rate
    spectrogram = spectrogram

    rfft = rfft
    dft = dft
    idft = idft

    differentiate = differentiate
    integrate = integrate
