"""
Backward compatibility for legacy DASDAE metadata.

Older DASDAE files mixed coordinate metadata into the patch attr namespace:
coord summaries were stored as a (sometimes pickled) ``coords`` attr, and
flat keys such as ``time_min`` or ``d_time`` lived alongside true patch
attrs. They also stored unit companions (``gauge_length_units``) beside
values that patch attrs now record in the inventory's fixed units. Files
written after attrs and coords were fully separated carry the
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
from dascore.exceptions import InvalidFiberFileError, UnitError
from dascore.io.utils import convert_attr_units
from dascore.units import convert_units, get_quantity_str
from dascore.utils.misc import unbyte

# Every flat coord-summary key an old file may contain ({name}_{field}).
_LEGACY_COORD_FIELDS = tuple(CoordSummary.model_fields)
# The subset legacy writers actually flattened into attrs; dims/fingerprint
# never appeared as flat keys, so translate re-emits only these while the
# strip above removes the full (superset) field family.
_LEGACY_FLAT_FIELDS = ("min", "max", "step", "units", "dtype", "len")
# Attr names whose released reader models stored a ``{name}_units``
# companion, with the fixed unit each now records. The companion is spent
# at the parse boundary exactly as readers spend a foreign format's unit
# declaration, so a patch loaded from an old cache matches one freshly
# read from its source.
_LEGACY_UNIT_COMPANIONS = (
    ("gauge_length", "m"),
    ("pulse_width", "s"),
    ("pulse_rate", "Hz"),
)


def _units_are_length(units) -> bool:
    """True when a stated unit converts to meters."""
    quantity = get_quantity_str(unbyte(units))
    # A null/empty companion means unstated, and convert_units treats a
    # None from_units as a no-op rather than a mismatch, so it must be
    # rejected before the conversion probe or nothing would look wrong.
    if quantity is None:
        return False
    try:
        convert_units(1.0, to_units="m", from_units=quantity)
    except (TypeError, ValueError, UnitError):
        return False
    return True


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


def translate_legacy_attrs(attrs, coord_names: Iterable[str] = ()):
    """
    Normalize legacy DASDAE attr payloads to flat coord metadata.

    ``coord_names`` are the coordinates actually stored in the patch group;
    flat unit keys belonging to them are coordinate metadata rather than
    value companions, whatever the attr payload itself claims.
    """
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
                "security. If you trust this file, read it inside a "
                "'with dc.config_context(allow_dasdae_format_unpickle=True):' "
                "block, or enable it permanently with "
                "dc.set_config(allow_dasdae_format_unpickle=True)."
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
    # A coordinate sharing one of these names owns its ``{name}_units``
    # key — whether it was just flattened from the coords payload or is
    # the only surviving record of a flat-key-era coordinate's units — so
    # coord reconstruction still needs it and it is not a value companion.
    protected = set(coords) | set(dims) | set(coord_names)
    # xml_binary's released reader spelled the pulse units into the key
    # name; restated as a companion, the loop below converts it like every
    # other released spelling.
    ns_pulse = (
        "pulse_width_ns" in out
        and "pulse_width" not in out
        and not {"pulse_width", "pulse_width_ns"} & protected
    )
    if ns_pulse:
        out["pulse_width"] = out.pop("pulse_width_ns")
        out["pulse_width_units"] = "ns"
    for name, to_units in _LEGACY_UNIT_COMPANIONS:
        companion = f"{name}_units"
        if name in protected or companion not in out:
            continue
        length_pulse = (
            name == "pulse_width"
            and name in out
            and "pulse_length" not in out
            and "pulse_length" not in protected
            and _units_are_length(out[companion])
        )
        if length_pulse:
            # A length-dimensioned companion (a ProdML uom in meters, or a
            # hand-set attr) states the quantity the vocabulary names
            # pulse_length, so the name migrates along with the units.
            out["pulse_length"] = out.pop(name)
            out["pulse_length_units"] = out.pop(companion)
            convert_attr_units(out, "pulse_length", "m")
            continue
        convert_attr_units(out, name, to_units)
    return out
