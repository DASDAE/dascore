"""A deprecated module."""

from __future__ import annotations

import warnings

from dascore.core import attrs


def __getattr__(name):
    out = getattr(attrs, name)
    msg = (
        "The DASCore.core.schema module is deprecated. Use DAScore.core.attrs "
        "instead."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return out
