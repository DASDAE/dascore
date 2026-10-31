"""
Read WITSML 1.4 distributed temperature sensing (DTS) logs.

WITSML is the Energistics XML standard for well data. A DTS interrogator
writes one reading as a log (1.4.1.1 ``logs``/``log``, or 1.4.0.0
``wellLogs``/``wellLog``) with one comma-separated row per distance along
the fiber. Each data curve (for example TMP, ST, AST) becomes its own
patch, keyed and tagged by its mnemonic, with dims ``("distance", "time")``.
``time`` holds the reading's start and the non-dimensional ``sample_span``
how long it ran.

Distance sampling
-----------------
The interrogator fixes the spatial sampling, but the header's step and the
stored index values are both rounded. By default the distance grid runs
from the header's start to end index over the row count;
``snap=False`` returns the stored index values exactly.
"""

from __future__ import annotations

from .core import WITSMLV1400 as WITSMLV1400
from .core import WITSMLV1411 as WITSMLV1411
