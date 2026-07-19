"""
Backward compatibility for legacy DASDAE metadata.

Older DASDAE files mixed coordinate metadata into the patch attr namespace:
coord summaries were stored as a (sometimes pickled) ``coords`` attr, and
flat keys such as ``time_min`` or ``d_time`` lived alongside true patch
attrs. Files written after attrs and coords were fully separated carry the
``__attrs_coords_separate__`` root marker and store only true attrs.

This module is private to ``dascore.io.dasdae``; nothing else in DASCore may
import it. Attrs and coords are independent everywhere else, so all knowledge
of the old mixed shapes is quarantined here.
"""

from __future__ import annotations

import contextlib
import pickle
from collections.abc import Iterable

from dascore.config import get_config
from dascore.core.coords import CoordSummary
from dascore.exceptions import InvalidFiberFileError

# Every flat coord-summary key an old file may contain ({name}_{field}).
_LEGACY_COORD_FIELDS = tuple(CoordSummary.model_fields)
# The subset legacy writers actually flattened into attrs; dims/fingerprint
# never appeared as flat keys, so translate re-emits only these while the
# strip above removes the full (superset) field family.
_LEGACY_FLAT_FIELDS = ("min", "max", "step", "units", "dtype", "len")


def strip_legacy_coord_fields(attrs: dict, coord_names: Iterable[str]) -> dict:
    """
    Remove legacy flat coordinate metadata from an attr mapping.

    Only exact ``{name}_{field}`` compositions for the provided coord names
    are removed (plus the deprecated ``d_{name}`` spelling and the structural
    ``coords``/``dims`` keys). Nothing is inferred from key shape, so a true
    attr like ``pulse_len`` survives unless the file really stores a ``pulse``
    coordinate.
    """
    out = dict(attrs)
    out.pop("coords", None)
    out.pop("dims", None)
    for name in coord_names:
        for field in _LEGACY_COORD_FIELDS:
            out.pop(f"{name}_{field}", None)
        out.pop(f"d_{name}", None)
    return out


def translate_legacy_attrs(attrs):
    """Normalize legacy DASDAE attr payloads to flat coord metadata."""
    out = dict(attrs)
    coords = out.pop("coords", {})
    if isinstance(coords, str):
        # Older DASDAE files stored the coord-summary payload as a pickled
        # string attr. Unpickling runs arbitrary code, so the opt-in gate
        # must come before any decode attempt — a malicious payload executes
        # during pickle.loads itself, not when the result is used.
        if not get_config().allow_dasdae_format_unpickle:
            msg = (
                "This DASDAE file contains legacy pickled coordinate metadata. "
                "Unpickling DASDAE format metadata is disabled by default for "
                "security. If you trust this file, enable legacy compatibility "
                "with dc.set_config(allow_dasdae_format_unpickle=True)."
            )
            raise InvalidFiberFileError(msg)
        with contextlib.suppress(
            AttributeError,
            EOFError,
            KeyError,
            pickle.PickleError,
            TypeError,
            UnicodeError,
            ValueError,
        ):
            coords = pickle.loads(coords.encode("latin1"))
    if hasattr(coords, "to_summary_dict"):
        coords = coords.to_summary_dict()
    if not hasattr(coords, "items"):
        coords = {}
    for name, summary in coords.items():
        if hasattr(summary, "to_summary"):
            summary = summary.to_summary()
        if hasattr(summary, "model_dump"):
            summary = summary.model_dump()
        if not isinstance(summary, dict):
            continue
        for field in _LEGACY_FLAT_FIELDS:
            key = f"{name}_{field}"
            value = summary.get(field)
            if key not in out and value not in (None, ""):
                out[key] = value
    dims = out.get("dims", "")
    dims = tuple(dims.split(",")) if isinstance(dims, str) else tuple(dims or ())
    for name in dims:
        old_name = f"d_{name}"
        new_name = f"{name}_step"
        if new_name not in out and old_name in out:
            out[new_name] = out.pop(old_name)
    return out
