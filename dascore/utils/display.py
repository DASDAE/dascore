"""
Utils for displaying dascore objects.
"""
import textwrap
from functools import singledispatch

import numpy as np
import pandas as pd
from rich.style import Style
from rich.text import Text

import dascore as dc
from dascore.constants import FLOAT_PRECISION, dascore_styles


@singledispatch
def get_nice_style(value) -> Text:
    """
    Get a rich Text object for formatting nice display for various datatypes.
    """
    return Text(str(value), "whiteq")


@get_nice_style.register(float)
@get_nice_style.register(np.float_)
def _nice_float_string(value):
    """Nice print value for floats."""
    fmt_str = f".{FLOAT_PRECISION}"
    return Text(f"{float(value):{fmt_str}}")


@get_nice_style.register(np.timedelta64)
@get_nice_style.register(pd.Timedelta)
def _nice_timedelta(value):
    """Get a nice timedelta value."""
    sec = dc.to_timedelta64(value) / np.timedelta64(1, "s")
    return Text(f"{sec:.9}s")


@get_nice_style.register(np.datetime64)
@get_nice_style.register(pd.Timestamp)
def _nice_datetime(value):
    """Get a nice timedelta value."""

    def simplify_str(dt_str):
        """Simplify the string to only show needed parts."""
        empty = str(dc.to_datetime64(0))
        out = str(dc.to_datetime64(value))
        # strip off YEAR-MONTH-DAY if they aren't used.
        if empty.split("T")[0] == out.split("T")[0]:
            out = out.split("T")[1]
        out = out.rstrip("0").rstrip(".")  # strip trailing 0s.
        return out

    def stylize_str(dt_str):
        """Apply color/style to strings"""
        ymd = dascore_styles["ymd"]
        hms = dascore_styles["hms"]
        dec = dascore_styles["dec"]
        out = Text()
        ymd_split = dt_str.split("T")
        if len(ymd_split) > 1:
            for char in ymd_split[0].split("-"):
                out += Text(char, style=ymd)
                out += Text("-")
            out = out[:-1] + Text("T")
        current = ymd_split[-1]
        dec_split = current.split(".")
        if len(dec_split) > 1 or "." not in current:
            for char in dec_split[0].split(":"):
                out += Text(char, hms)
                out += Text(":")
            out = out[:-1]
            if "." in current:
                out += Text(".")
        if "." in current:
            out += Text(dec_split[-1], dec)
        return out

    simplfied_str = simplify_str(str(value))
    return stylize_str(simplfied_str)


def get_dascore_text():
    """Get stylized dascore text."""
    das_style = Style(color=dascore_styles["dc_blue"], bold=True)
    c_style = Style(color=dascore_styles["dc_red"], bold=True)
    ore_style = Style(color=dascore_styles["dc_yellow"], bold=True)
    das = Text("DAS", style=das_style)
    c = Text("C", style=c_style)
    ore = Text("ore", style=ore_style)
    return Text.assemble(das, c, ore)


def array_to_text(data) -> Text:
    """Convert a coordinate to string."""
    header = Text("➤ ") + Text("Data", style=dascore_styles["dc_red"])
    header += Text(f" ({data.dtype})")
    np_str = np.array_str(data, precision=FLOAT_PRECISION, suppress_small=True)
    numpy_format = textwrap.indent(np_str, "   ")
    return header + Text("\n") + Text(numpy_format)


def attrs_to_text(attrs) -> Text:
    """
    Convert pydantic model to text.
    """
    default = dc.PatchAttrs()
    txt = Text("➤ ") + Text("Attributes", style=dascore_styles["dc_yellow"])
    txt += Text("\n")
    for name, attr in dict(attrs).items():
        if default.get(name, None) == attr or not (attr):
            continue
        txt += Text("    ")
        txt += Text(f"{name}: ", dascore_styles["keys"])
        txt += get_nice_style(attr)
        txt += Text("\n")
    return txt
