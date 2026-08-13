"""
Deprecated home of DASCore's models; import from [dascore.models](`dascore.models`).

Everything here is re-exported from its new home so out-of-tree readers which
import from this path keep working. Re-exported wholesale rather than listed,
since a second hand-maintained list is a second thing to forget.
"""

from __future__ import annotations

from pydantic import BaseModel

from dascore.models import *  # noqa: F403
from dascore.models import __all__ as _models_all

__all__ = [*_models_all, "BaseModel"]
