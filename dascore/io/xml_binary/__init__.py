"""
Module for reading data stored in xml_binary format.

This format is a directory which contains metadata as a single xml file
as well as any number of binary files (raw numeric buffers) which contain
information about their start time in the file name. An example directory
might look like this:

data_folder
   metadata.xml
   DAS_20240530T011500_000000Z.raw
   DAS_20240530T011501_000000Z.raw
"""
from __future__ import annotations
from .core import XMLBinaryV1
