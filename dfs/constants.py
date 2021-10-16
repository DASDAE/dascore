"""Constants used throughout obsplus."""
from typing import (
    Union,
)

# Bump this to force re-downloading of all data file
DATA_VERSION = "0.0.0"

timeable_types = Union[int, float, str]

# expected DAS attributes
REQUIRED_DAS_ATTRS = ("sampling_time", "sampling_length")
