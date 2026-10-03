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
from .dispersion import DispersionPhaseShift
from .taup import TauP
from .stalta import Stalta
from .fbe import Fbe
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
