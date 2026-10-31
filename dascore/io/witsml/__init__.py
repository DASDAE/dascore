"""
Read WITSML 1.4 distributed temperature sensing (DTS) logs.

WITSML is the Energistics XML standard for well data. A DTS interrogator
writes one reading as a log (1.4.1.1 ``logs``/``log``, or 1.4.0.0
``wellLogs``/``wellLog``) with one comma-separated row per distance along
the fiber. Each data curve (for example TMP, ST, AST) becomes its own
patch, keyed and tagged by its mnemonic, with dims ``("distance", "time")``.
``time`` holds the reading's start; the non-dimensional ``sample_span``
coord, mapped to ``time``, holds how long it ran.

Distance sampling
-----------------
The interrogator samples distance at a constant step, but the header's step
and the stored index values are both rounded. By default the distance grid
runs from the header's start to end index (else the stored ends) over the
row count; ``snap=False`` returns the stored index values exactly. Only
logs indexed by length along the fiber, with one block of data, are read.
"""

from __future__ import annotations

from .core import WITSMLV1400 as WITSMLV1400
from .core import WITSMLV1411 as WITSMLV1411
