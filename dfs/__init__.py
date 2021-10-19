# -*- coding: utf-8 -*-
from dfs.version import __version__

from dfs.core.stream import Stream
from dfs.io.base import read, scan, get_format, write


import dfs.utils.accessor  # NOQA this needs to be at the last of the imports.
