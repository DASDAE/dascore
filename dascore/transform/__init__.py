"""
A module for applying transformation to Patches.

Transforms are defined as
"""
from __future__ import annotations

from .differentiate import differentiate
from .fft import rfft
from .fourier import dft, idft, stft, istft
from .integrate import integrate
from .hilbert import hilbert, envelope, phase_weighted_stack
from .spectro import spectrogram
from .strain import velocity_to_strain_rate, velocity_to_strain_rate_edgeless, radians_to_strain
from .dispersion import dispersion_phase_shift
from .taup import tau_p
