"""
Modules for reading and writing fiber data.
"""
from dascore.utils.misc import MethodNameSpace

from .wav.core import _write_wavfolder


class PatchIO(MethodNameSpace):
    to_wav = _write_wavfolder
