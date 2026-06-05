"""
Support for Febus format.

This is used by the Febus DAS interrogator.

More info about febus can be found here: https://www.febus-optics.com/en/
"""

from __future__ import annotations

from .core import Febus1 as Febus1
from .core import Febus2 as Febus2
from .core import FebusBSLH5V1 as FebusBSLH5V1
from .core import FebusMTXH5V1 as FebusMTXH5V1
from .core import FebusT1V1 as FebusT1V1
