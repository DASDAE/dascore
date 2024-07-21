"""Utils for displaying dascore objects."""

from __future__ import annotations

import textwrap
from functools import singledispatch

import numpy as np
import pandas as pd
from rich.style import Style
from rich.text import Text

import dascore as dc
from dascore.constants import FLOAT_PRECISION, dascore_styles
from dascore.units import get_quantity_str


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
    fmt_str = f".{FLOAT_PRECISION}"
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
        return str(value)

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


def array_to_text(data, units=None) -> Text:
    """Convert a coordinate to string."""
    header = Text("➤ ") + Text("Data", style=dascore_styles["dc_red"])
    unitstr = Text("") if units is None else Text(f", units: {units}")
    header += Text(f" ({data.dtype}") + unitstr + Text(")")
    threshold = dascore_styles["np_array_threshold"]
    np_str = np.array2string(
        data,
        precision=FLOAT_PRECISION,
        threshold=threshold,
    )
    numpy_format = textwrap.indent(np_str, "   ")
    return header + Text("\n") + Text(numpy_format)


def attrs_to_text(attrs) -> Text:
    """Convert pydantic model to text."""
    attrs = dc.PatchAttrs.from_dict(attrs).model_dump(exclude_defaults=True)
    # pop coords and dims since they show up in other places.
    attrs.pop("coords", None), attrs.pop("dims", None)
    txt = Text("➤ ") + Text("Attributes", style=dascore_styles["dc_yellow"])
    txt += Text("\n")
    for name, attr in dict(attrs).items():
        # skip private coords for display
        if name.startswith("_"):
            continue
        # determine styles here based on keys
        style = dascore_styles.get(name, None)
        if name.endswith("units"):
            attr = get_quantity_str(attr)
        # assemble text
        txt += Text("    ")
        txt += Text(f"{name}: ", dascore_styles["keys"])
        txt += get_nice_text(attr, style=style)
        txt += Text("\n")
    return txt
