"""
A module for applying transformation to Patches.

Transforms are defined as
"""
from __future__ import annotations

from .differentiate import Differentiate
from .fourier import Dft, Idft, Stft, Istft
from .integrate import Integrate
from .hilbert import Hilbert, Envelope, PhaseWeightedStack
from .strain import VelocityToStrainRate, VelocityToStrainRateEdgeless, RadiansToStrain
from .dispersion import dispersion_phase_shift
from .taup import tau_p
from .stalta import stalta
from .fbe import fbe
from .kurtosis import Kurtosis
from .spectral_descriptors import (
    MedianFrequency,
    SpectralCentroid,
    SpectralEntropy,
    SpectralFlatness,
    SpectralKurtosis,
    SpectralPeakFrequency,
    SpectralPeakAmplitude,
)
