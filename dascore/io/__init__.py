"""
Modules for reading and writing fiber data.
"""
from dascore.io.core import write
from dascore.utils.io import BinaryReader, BinaryWriter, HDF5Reader, HDF5Writer
from dascore.utils.misc import MethodNameSpace


class PatchIO(MethodNameSpace):
    write = write
