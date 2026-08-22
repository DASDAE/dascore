"""Utils for displaying dascore objects."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sized
from contextlib import suppress
from functools import singledispatch

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from rich.style import Style
from rich.text import Text

import dascore as dc
from dascore.config import get_config
from dascore.constants import dascore_styles
from dascore.units import get_quantity_str

# How wide one value may print before it is elided. A repr is a glance at
# an object, and a single long field should not push the rest off screen.
_MAX_VALUE_CELLS = 80

# Fields which name a record rather than describe it. Hidden for the same
# reason a patch's lineage ids are: they would put a line of hex in every
# repr of every object without telling a reader anything.
_IDENTITY_FIELDS = frozenset({"object_type", "resource_id"})

# What is unrolled into its items rather than printed as itself. An array,
# a Series or a frame is a table, and has a string form of its own which
# already knows how much of itself to show.
_SEQUENCE_TYPES = (tuple, list, set, frozenset)

# The scalar types a model field is left unset as.
_NULLABLE_TYPES = (
    float,
    np.floating,
    np.datetime64,
    np.timedelta64,
    pd.Timestamp,
    pd.Timedelta,
    type(pd.NaT),
)


@singledispatch
def get_nice_text(value, style=None) -> Text:
    """
    Get a rich Text object for formatting nice display for various datatypes.

    Parameters
    ----------
    value
        The value which should be stylized.
    style
        A string which is either an entry in dascore.constants.dascore_styles
        or a valid rich style string.
    """
    txt = value if isinstance(value, Text) else Text(str(value))
    if style is not None:
        style = dascore_styles.get(style, style)
        txt.stylize(style)
    return txt


@get_nice_text.register(float)
@get_nice_text.register(np.float64)
def _nice_float_string(value, style=None):
    """Nice print value for floats."""
    fmt_str = f".{get_config().display_float_precision}f"
    return get_nice_text(Text(f"{float(value):{fmt_str}}"), style)


@get_nice_text.register(np.timedelta64)
@get_nice_text.register(pd.Timedelta)
def _nice_timedelta(value, style=None):
    """Get a nice timedelta value."""
    sec = dc.to_timedelta64(value) / np.timedelta64(1, "s")
    return get_nice_text(Text(f"{sec:.9}s"), style)


@get_nice_text.register(np.datetime64)
@get_nice_text.register(pd.Timestamp)
def _nice_datetime(value, style=None):
    """Get a nice timedelta value."""

    def simplify_str(dt_str):
        """Simplify the string to only show needed parts."""
        empty = str(dc.to_datetime64(0))
        original_str = str(dt_str)
        trimmed_str = original_str
        # strip off YEAR-MONTH-DAY if they aren't used.
        if empty.split("T")[0] == trimmed_str.split("T")[0]:
            trimmed_str = trimmed_str.split("T")[-1]
        # strip off HOUR-MIN-SEC if it isnt used
        elif empty.split("T")[-1] == trimmed_str.split("T")[-1]:
            trimmed_str = trimmed_str.split("T")[0]
        if "." in trimmed_str:  # strip trailing 0s.
            trimmed_str = trimmed_str.rstrip("0").rstrip(".")
        ind = original_str.find(trimmed_str)
        return dt_str[ind : ind + len(trimmed_str)]

    def stylize_str(dt_str):
        """
        Apply color/style to strings. This assumes the string is formatted
        in the standard ISO 8601 format.
        """
        # get relevant styles.
        ymd = dascore_styles["ymd"]
        hms = dascore_styles["hms"]
        dec = dascore_styles["dec"]
        if len(dt_str) < 20:  # this might be a timestamp string
            dt_str = str(dc.to_datetime64(dt_str))
        # parse out string components
        assert len(dt_str) >= 20  # need chars at least up to decimal
        year, month, day = dt_str[:4], dt_str[5:7], dt_str[8:10]
        hour, minute, second = dt_str[11:13], dt_str[14:16], dt_str[17:19]
        decimal_bit = dt_str[20:]
        # assemble text with styling
        out = Text("")
        out += Text("-").join([Text(x, ymd) for x in [year, month, day]])
        out += Text("T")
        out += Text(":").join([Text(x, hms) for x in [hour, minute, second]])
        out += Text(".") + Text(decimal_bit, dec)
        return out

    if pd.isnull(value):
        return get_nice_text(Text(str(value)), style)

    stylized_text = stylize_str(str(value))
    simplified_text = simplify_str(stylized_text)
    return get_nice_text(simplified_text, style)


def get_dascore_text():
    """Get stylized dascore text."""
    das_style = Style(color=dascore_styles["dc_blue"], bold=True)
    c_style = Style(color=dascore_styles["dc_red"], bold=True)
    ore_style = Style(color=dascore_styles["dc_yellow"], bold=True)
    das = Text("DAS", style=das_style)
    c = Text("C", style=c_style)
    ore = Text("ore", style=ore_style)
    return Text.assemble(das, c, ore)


def get_header_text(name: str, style: str = "bold") -> Text:
    """
    Get the banner which opens the repr of a top-level dascore object.

    Parameters
    ----------
    name
        What the object is called, including any icon (e.g. "Patch ⚡").
    style
        A rich style, or a key of dascore.constants.dascore_styles.
    """
    header = Text.assemble(get_dascore_text(), " ", Text(name, style=style))
    # cell_len, not len: an emoji is one character and two columns wide,
    # and the underline is drawn in columns.
    return header + Text("\n") + Text("-" * header.cell_len)


def indent_text(text: Text, prefix: str = "    ") -> Text:
    """
    Indent every line of a rich Text, keeping its styles.

    This is what lets a container drop a child's ``__rich__`` into place
    rather than describing the child itself.
    """
    return Text("\n").join(Text(prefix) + line for line in text.split("\n"))


def limit_reprs(items, limit: int | None = None) -> list[Text]:
    """
    Render at most ``limit`` items, with a line naming what was left out.

    Only the items which are shown are rendered, so the cost of a repr is
    what it prints rather than what the object holds. A repr also says how
    much it is not showing; silently stopping at ten reads as though ten
    is all there were.
    """
    limit = get_config().display_max_items if limit is None else limit
    items = list(items)
    texts = [x.__rich__() for x in items[:limit]]
    if left_out := len(items) - len(texts):
        texts.append(Text(f"... {left_out} more", style=dascore_styles["keys"]))
    return texts


def _length(value) -> int | None:
    """
    Return how many items a value holds, or None where it holds no items.

    A pint Quantity claims to be Sized and then refuses ``len`` when it
    wraps a scalar, so asking is the only way to know.
    """
    if not isinstance(value, Sized) or isinstance(value, str):
        return None
    with suppress(TypeError):
        return len(value)
    return None


def counts_to_text(counts, limit: int | None = None) -> Text:
    """
    Render a mapping of name to count as ``a: 2, b: 1``.

    Names past ``limit`` (config's ``display_max_items``) are summarized
    rather than dropped silently.
    """
    limit = get_config().display_max_items if limit is None else limit
    items = list(dict(counts).items())
    shown = ", ".join(f"{name}: {count}" for name, count in items[:limit])
    if len(items) <= limit:
        return Text(shown)
    return Text(shown) + Text(
        f", ... {len(items) - limit} more", dascore_styles["keys"]
    )


def _stated(value) -> bool:
    """Whether a value says anything: unset, blank and empty all do not."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, BaseModel):
        # A model which states nothing says nothing, however present it is.
        return bool(stated_fields(value))
    if (length := _length(value)) is not None:
        return length > 0
    # Only the scalar types a field is left unset as are asked; pandas
    # answers for anything, and an object it cannot judge is not null.
    if isinstance(value, _NULLABLE_TYPES):
        return not bool(pd.isnull(value))
    return True


def _is_default(info, value) -> bool:
    """
    Whether a field still holds the default it was given.

    Errs toward saying no: a default which cannot be compared (an array)
    or which is built fresh each time is treated as stated, so a doubtful
    field is shown rather than quietly dropped.
    """
    if info.default_factory is not None or info.default is PydanticUndefined:
        return False
    with suppress(Exception):
        return bool(info.default == value)
    return False


def stated_fields(model, skip=()) -> dict:
    """
    Return the fields of a pydantic model which state something.

    Unset, blank, empty and still-default fields are dropped, as are the
    identity fields and anything named in ``skip``. What is left is what
    the object actually says about itself.
    """
    skip = _IDENTITY_FIELDS | set(skip)
    out = {}
    for name, info in type(model).model_fields.items():
        if name in skip or name.startswith("_"):
            continue
        value = getattr(model, name, None)
        if not _stated(value) or _is_default(info, value):
            continue
        out[name] = value
    return out


def value_to_text(name: str, value, style=None, truncate: bool = True) -> Text:
    """
    Get the text one value prints as, given the name it is stated under.

    A value which knows how to display itself is asked to; a collection of
    such values is counted rather than unrolled.

    Parameters
    ----------
    name
        The name the value is stated under; units are read off it.
    value
        What to render.
    style
        A rich style for the rendered value.
    truncate
        Whether to elide a long value. True where many values share a
        line, False where the value has a line to itself.
    """
    if isinstance(value, Text):  # already rendered by the caller
        return value.copy()
    if isinstance(value, Mapping):
        if any(isinstance(x, BaseModel) for x in value.values()):
            text = Text(f"{len(value)}")
        else:
            text = Text(", ".join(f"{k}={v}" for k, v in value.items()))
    elif isinstance(value, _SEQUENCE_TYPES):
        values = list(value)
        if any(isinstance(x, BaseModel) for x in values):
            text = Text(f"{len(values)}")
        else:
            text = Text(", ".join(str(x) for x in values))
    elif name.endswith("units"):
        text = get_nice_text(get_quantity_str(value), style="units")
    elif hasattr(value, "__rich__"):
        text = value.__rich__()
    else:
        text = get_nice_text(value, style=style)
    if truncate:
        text.truncate(_MAX_VALUE_CELLS, overflow="ellipsis")
    return text


def model_to_line(model, skip=(), style=None, extra=None) -> Text:
    """
    Get a one-line summary of a pydantic model.

    Shaped like a coordinate's repr: the class name, then each field the
    model states. Containers of other models show how many they hold.

    Parameters
    ----------
    model
        The pydantic model to summarize.
    skip
        Field names to leave out, e.g. children the caller lists itself.
    style
        A rich style for the class name.
    extra
        Derived values to state after the fields, such as an interval a
        model computes rather than stores.
    """
    key_style = dascore_styles["keys"]
    base = Text(model.__class__.__name__, style=style or "bold")
    base += Text("(")
    fields = {**stated_fields(model, skip=skip), **dict(extra or {})}
    for name, value in fields.items():
        base += Text(f" {name}: ", key_style)
        base += value_to_text(name, value)
    base += Text(" )")
    return base


def mapping_to_text(mapping, header: str, style: str = "dc_yellow") -> Text:
    """
    Get the "➤ Header" block a mapping of names to values prints as.

    Parameters
    ----------
    mapping
        The names and values to list, one per line.
    header
        What to call the block.
    style
        A rich style, or a key of dascore.constants.dascore_styles.
    """
    txt = Text("➤ ") + Text(header, style=dascore_styles.get(style, style))
    for name, value in dict(mapping).items():
        # skip private entries for display
        if str(name).startswith("_"):
            continue
        txt += Text("\n    ")
        txt += Text(f"{name}: ", dascore_styles["keys"])
        # A block gives each value its own line, so it is not elided; a
        # one-line summary packs many values and has to bound each.
        txt += value_to_text(
            name, value, style=dascore_styles.get(name, None), truncate=False
        )
    return txt


def array_to_text(data, units=None) -> Text:
    """Convert a coordinate to string."""
    header = Text("➤ ") + Text("Data", style=dascore_styles["dc_red"])
    unitstr = Text("") if units is None else Text(f", units: {units}")
    header += Text(f" ({data.dtype}") + unitstr + Text(")")
    config = get_config()
    np_str = np.array2string(
        data,
        precision=config.display_float_precision,
        threshold=config.display_array_threshold,
    )
    numpy_format = textwrap.indent(np_str, "   ")
    return header + Text("\n") + Text(numpy_format)


def attrs_to_text(attrs) -> Text:
    """Convert pydantic model to text."""
    attrs = dc.PatchAttrs.from_dict(attrs).model_dump(exclude_defaults=True)
    # pop coords and dims since they show up in other places.
    attrs.pop("coords", None), attrs.pop("dims", None)
    # The lineage ids are not metadata a reader is looking at a patch to
    # read; they would put a line of hex on every patch anyone prints.
    attrs.pop("patch_id", None), attrs.pop("processing_id", None)
    return mapping_to_text(attrs, "Attributes") + Text("\n")
