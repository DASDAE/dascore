"""
A module for applying transformation to Patches.

Transforms are defined as
"""
from __future__ import annotations

from .differentiate import differentiate
from .fft import rfft
from .fourier import dft, idft
from .integrate import integrate
from .spectro import spectrogram
from .strain import velocity_to_strain_rate
from .dispersion import dispersion_phase_shift
