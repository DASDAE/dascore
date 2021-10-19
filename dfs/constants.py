"""Constants used throughout obsplus."""
from typing import (
    Union,
)

import numpy as np

# Bump this to force re-downloading of all data file
DATA_VERSION = "0.0.0"

# Types DFS can convert into time representations
timeable_types = Union[int, float, str]

# expected fiber attributes
REQUIRED_FIBER_DIMS = ("time", "distance")

# expected DAS attributes
REQUIRED_DAS_ATTRS = ("sample_time", "sample_length")

# The smallest value an int64 can rep. (used as NaT by datetime64)
MININT64 = np.iinfo(np.int64).min

# The largest value an int64 can rep
MAXINT64 = np.iinfo(np.int64).max

# Large and small np.datetime64[ns] (used when defaults are needed)
SMALLDT64 = np.datetime64(MININT64 + 5_000_000_000, "ns")
LARGEDT64 = np.datetime64(MAXINT64 - 5_000_000_000, "ns")
