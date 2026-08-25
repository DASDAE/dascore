"""Utils for displaying dascore objects."""

from __future__ import annotations

import textwrap
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sized
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cache, singledispatch
from html import escape
from importlib.resources import files
from itertools import pairwise

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from rich.style import Style
from rich.text import Text

import dascore as dc
from dascore.config import get_config
from dascore.constants import dascore_styles
from dascore.units import get_quantity, get_quantity_str
from dascore.utils.time import to_float

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

_SECONDS_IN_DAY = 86_400.0

# What a step is stored to when it does not say, as a pandas Timedelta
# does not. Two steps closer together than one of these are the same
# sampling as far as anything can tell.
_NANOSECOND = 1e-9

# What counts as a time, and so what has a duration between two of it.
_TIME_TYPES = (pd.Timestamp, np.datetime64, pd.Timedelta, np.timedelta64)

# The most figures a rate is quoted to when it cannot state its step
# exactly. A rate which can gets as many as that takes, up to the point
# where it has stopped being a rate and become the step in other units.
_RATE_FIGURES = 4
_RATE_EXACT_FIGURES = 9

# What a byte count is read in, smallest first.
_SIZE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


# Steps a duration is worth reading in, largest first. A year is the
# Gregorian mean, the same one numpy converts a timedelta64 with, since
# a multi-year outage reads as "40.7 y" rather than "14852 d".
_UNITS = (
    ("y", 365.2425 * _SECONDS_IN_DAY),
    ("d", _SECONDS_IN_DAY),
    ("h", 3_600.0),
    ("m", 60.0),
    ("s", 1.0),
    ("ms", 1e-3),
    ("µs", 1e-6),
)

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


@dataclass(frozen=True, slots=True)
class Raw:
    """
    Text whose producer has already laid it out.

    A renderer emits it as it stands. This is what lets a class which
    states no nodes of its own still render correctly inside one which
    does, and it stays the home of anything whose spacing is load
    bearing, such as an array printed in columns.
    """

    text: Text


@dataclass(frozen=True, slots=True)
class Section:
    """
    A titled block: the line which names it, and what sits under it.

    A section with no body is a statement rather than a container.
    """

    title: Text
    body: tuple[Raw, ...] = ()


@dataclass(frozen=True, slots=True)
class Repr:
    """A whole repr: the banner which names the object, then its sections."""

    header: Text
    body: tuple[Section, ...] = field(default_factory=tuple)


@singledispatch
def render_text(node) -> Text:
    """Render a repr node as rich text."""
    msg = f"cannot render {type(node).__name__} as text"
    raise NotImplementedError(msg)


@render_text.register
def _render_raw(node: Raw) -> Text:
    # A copy, not the node's own Text: a caller which appends to what it
    # rendered would otherwise rewrite the node it just read.
    return node.text.copy()


@render_text.register
def _render_section(node: Section) -> Text:
    out = node.title.copy()
    for child in node.body:
        out += render_text(child)
    return out


@render_text.register
def _render_repr(node: Repr) -> Text:
    blocks = [node.header, *(render_text(x) for x in node.body)]
    return Text("\n").join(blocks)


def split_block(text: Text) -> Section:
    """
    Turn a block of rendered text into a section.

    The first line names the block and the rest is its body, which is
    how every block a dascore repr is built from already reads. Slicing
    rather than splitting is deliberate: ``Text.split`` drops a trailing
    blank line, and the attributes block ends on one.

    The body keeps the newline which separated it, so a span straddling
    the break still covers what it covered. Rendering the section gives
    back the same characters drawn the same way, though not the same
    spans: slicing restates a base style as a span, and a span across
    the break comes back as two touching ones.
    """
    index = text.plain.find("\n")
    if index == -1:
        return Section(text)
    return Section(text[:index], (Raw(text[index:]),))


# --- Rendering a repr as HTML -------------------------------------------

# The one class every fragment is wrapped in. Every rule in repr.css is
# scoped under it, so a stylesheet which lands in a notebook output --
# where it applies to the whole document -- cannot restyle anything else.
_HTML_ROOT = "dc-repr"

# Style words a class exists for. Anything else is dropped, which is what
# rich does with a word it cannot parse, so an unmappable style is
# uncolored in a browser and in a terminal alike.
_STYLE_WORDS = frozenset(
    {
        "bold",
        "underline",
        "blue",
        "bright_blue",
        "red",
        "yellow",
        "green",
        "cyan",
        "dark_orange",
        "grey50",
    }
)

# Words which change what follows rather than describing it: "not bold"
# is not bold, and "red on blue" paints a background. Read word by word
# the first would invert and the second would be mistaken, so a style
# using either is left unstyled.
_STYLE_QUALIFIERS = frozenset({"not", "on"})


def style_classes(style: str | Style | None) -> tuple[str, ...]:
    """Return the CSS classes a rich style is drawn with."""
    if style is None:
        return ()
    if isinstance(style, Style):
        words = [
            x for x, on in (("bold", style.bold), ("underline", style.underline)) if on
        ]
        if style.color is not None:
            words.append(style.color.name)
    else:
        words = str(style).split()
        if _STYLE_QUALIFIERS.intersection(words):
            return ()
    return tuple(f"dc-{x}" for x in words if x in _STYLE_WORDS)


def text_to_html(text: Text) -> str:
    """Render a rich Text as an inline HTML fragment."""
    plain = text.plain
    cuts = {0, len(plain)}
    for span in text.spans:
        cuts.update((span.start, span.end))
    base = style_classes(text.style)
    out = []
    for start, end in pairwise(sorted(cuts)):
        classes = list(base)
        for span in text.spans:
            if span.start <= start and end <= span.end:
                classes.extend(x for x in style_classes(span.style) if x not in classes)
        chunk = escape(plain[start:end], quote=False)
        out.append(
            f'<span class="{" ".join(classes)}">{chunk}</span>' if classes else chunk
        )
    return "".join(out)


@cache
def get_stylesheet() -> str:
    """Return the CSS every HTML repr carries."""
    return files("dascore").joinpath("repr.css").read_text(encoding="utf-8")


@singledispatch
def render_html(node) -> str:
    """Render a repr node as an HTML fragment."""
    msg = f"cannot render {type(node).__name__} as html"
    raise NotImplementedError(msg)


@render_html.register
def _html_raw(node: Raw) -> str:
    # A `pre`, because what a producer laid out is laid out in columns:
    # an array printed by numpy, a track list padded to line up. HTML
    # would collapse the runs of spaces which do that work.
    text = node.text
    # The newline which separated this from its title belongs to neither.
    if text.plain.startswith("\n"):
        text = text[1:]
    return f'<pre class="dc-body">{text_to_html(text)}</pre>'


@render_html.register
def _html_section(node: Section) -> str:
    title = text_to_html(node.title)
    if not node.body:
        # Nothing to fold, so nothing to offer folding.
        return f'<div class="dc-line">{title}</div>'
    lines = sum(x.text.plain.count("\n") for x in node.body)
    state = " open" if lines <= get_config().display_html_open_lines else ""
    body = "".join(render_html(x) for x in node.body)
    return f"<details{state}><summary>{title}</summary>{body}</details>"


@render_html.register
def _html_repr(node: Repr) -> str:
    # The banner underlines itself with dashes in a terminal, drawn in
    # columns; here that is a border, and an emoji is not two columns
    # wide in every font a browser might choose.
    banner = text_to_html(node.header.split("\n")[0])
    body = "".join(render_html(x) for x in node.body)
    return (
        f'<div class="{_HTML_ROOT}"><style>{get_stylesheet()}</style>'
        f'<div class="dc-banner">{banner}</div>{body}</div>'
    )


class RichRepr:
    """
    Print an object the way its ``__rich__`` renders it.

    One definition for every class which has a rich rendering, so a plain
    terminal and a rich one say the same thing and neither drifts from the
    other. A host states ``__rich__``; this states the rest.

    List this before any base which defines its own ``__str__``, as a
    pydantic model does. The one the MRO reaches first is the only one
    called, and a host which lists this last prints a field dump.
    """

    # No ``__rich__`` stub here on purpose: every host states one, so a
    # guard for its absence is a branch no test can reach.
    __rich__: Callable[[], Text]

    def __str__(self) -> str:
        return str(self.__rich__())

    __repr__ = __str__


class NodeRepr(RichRepr):
    """
    Print an object from the repr nodes it states.

    A host states ``_repr_node``; rendering it is the same call every
    time, so it is made once here -- as text for a terminal, and as
    HTML for a display which draws it.
    """

    _repr_node: Callable[[], Repr]

    def __rich__(self) -> Text:
        return render_text(self._repr_node())

    def _repr_html_(self) -> str | None:
        """
        The panel a notebook draws, or None to fall back to the text.

        None is how the display protocol says "not this time", and it is
        what a repr should say when it cannot draw itself: a traceback
        out of a formatter is printed into the cell on every echo of the
        object, which makes it undebuggable at the one moment someone is
        looking at it. Debug mode wants the traceback instead, which is
        what the test suite runs with, so a panel which cannot be drawn
        fails there and stays quiet for a reader.
        """
        if not get_config().display_html:
            return None
        try:
            return render_html(self._repr_node())
        except Exception:
            if get_config().debug:
                raise
            return None


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
    style = dascore_styles.get(style, style)
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


def elision_text(left_out: int) -> Text:
    """
    Say how many of something a repr did not show.

    One wording for every repr which stops early, so a spool's tracks
    and an inventory's networks say it the same way in one terminal.
    """
    return Text(f"... {left_out} more", style=dascore_styles["keys"])


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
        texts.append(elision_text(left_out))
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


def human_duration(value: np.timedelta64 | pd.Timedelta | float) -> str:
    """Say how long something lasted, in the largest unit which fits."""
    # to_float reads a duration in seconds, whichever time type states
    # it, and passes a plain number through as itself.
    seconds = to_float(value)
    if not np.isfinite(seconds) or seconds == 0:
        return ""
    size = abs(seconds)
    for name, scale in _UNITS:
        if size >= scale:
            return f"{size / scale:.1f} {name}".replace(".0 ", " ")
    return f"{size:.3g} s"


def duration_text(low, high) -> Text | None:
    """
    How long an extent lasted, as a repr states it.

    None where it lasted no time: `human_duration` says nothing of a
    zero, which reads as a label on a gap and as an empty pair of
    brackets here.

    Asked wherever a repr states two instants. A time is stated as an
    instant, so how far apart two of them are is a fact the line does
    not otherwise carry; every other kind of dimension states its own
    magnitude already, and saying it twice is not saying more.

    Only of two times. A duration is read in seconds, so a distance of
    299 handed to this would come back as "5 m" -- five minutes, of a
    span measured in metres.
    """
    if not isinstance(low, _TIME_TYPES) or not isinstance(high, _TIME_TYPES):
        return None
    try:
        span = high - low
    except (OutOfBoundsDatetime, OutOfBoundsTimedelta):
        # Two instants can lie further apart than a Timedelta holds. How
        # long that is matters less than the extents it would otherwise
        # take down with it.
        return None
    # Divided by a second rather than read with `to_float`, which counts
    # nanoseconds: a span of centuries is more of those than an int64
    # holds, and the wrap is silent. Ten thousand years read that way
    # came back as sixty one.
    try:
        seconds = span / np.timedelta64(1, "s")
    except TypeError:
        # A span held in years or months is not a fixed number of
        # seconds, so numpy refuses to say how many. Neither will this.
        return None
    if not (said := human_duration(seconds)):
        return None
    return Text(f"<{said}>", dascore_styles["keys"])


def _fewest_figures(value: float, accept) -> int | None:
    """The fewest significant figures of a value which `accept` allows."""
    for figures in range(1, _RATE_EXACT_FIGURES + 1):
        if accept(float(f"{value:.{figures}g}")):
            return figures
    return None


def _storage_quantum(step) -> float:
    """
    How finely the step is stored, in seconds.

    What counts as the same sampling: a step held to nanoseconds and a
    rate which inverts to within half of one cannot be told apart, while
    the same slack on a step held to picoseconds would hide a real
    difference.
    """
    dtype = getattr(step, "dtype", None)
    if dtype is None:
        return _NANOSECOND
    unit, count = np.datetime_data(dtype)
    # Divided, not read with `to_float`: that counts whole nanoseconds,
    # so the quantum of a picosecond step came back as zero and no rate
    # could ever land inside it.
    quantum = np.timedelta64(count, unit)  # ty: ignore[no-matching-overload]
    return float(quantum / np.timedelta64(1, "s"))


def rate_text(step) -> Text | None:
    """
    A sampling step said the way acquisition is quoted, or nothing.

    Only a step measured in time gets one: a rate is the reciprocal of a
    duration, and one over a distance is not how anyone states channel
    spacing. Only an exact one, too -- a step of 0.0039999998 s is
    250.0000125 Hz, and rounding that to 250 Hz claims a precision the
    step does not have.
    """
    if not isinstance(step, np.timedelta64 | pd.Timedelta):
        return None
    # Divided rather than read with `to_float`, which counts whole
    # nanoseconds: a step of 1500 ps truncates to 1 ns there, and the
    # repr would state 1 GHz of sampling which happens at 666.7 MHz.
    try:
        seconds = abs(step / np.timedelta64(1, "s"))
    except TypeError:
        # A month is not a fixed number of seconds, so it is not a fixed
        # number of samples per second either.
        return None
    if not np.isfinite(seconds) or seconds == 0:
        return None
    # A descending axis samples at the rate an ascending one does; the
    # direction is the sign of the step, not of the frequency.
    quantum = _storage_quantum(step)
    # Say it only if what is said gives the step back, in as few figures
    # as that takes. An exact rate is preferred over a shorter one which
    # merely lands inside the step's own resolution: an 8 ms step held
    # to milliseconds is exactly 125 Hz, and 120 Hz also inverts to
    # within half a millisecond of it, so taking the shortest first
    # states 120 Hz for sampling which happens at 125.
    exact = 1.0 / seconds
    figures = _fewest_figures(exact, lambda x: x == exact)
    if figures is None:
        # Not a rate any short number states exactly, so it is quoted to
        # the figures a rate is quoted to and has to land inside the
        # step's own resolution. Taking the fewest figures here instead
        # would say 300 Hz of a 3 ms step, which samples at 333.3.
        figures = _RATE_FIGURES
        if abs(1 / float(f"{exact:.{figures}g}") - seconds) > quantum / 2:
            return None
    hertz = float(f"{exact:.{figures}g}")
    # pint picks the prefix, so this reaches GHz and past it without a
    # table of its own to keep in step with.
    quantity = get_quantity(f"{hertz} Hz")
    assert quantity is not None  # a literal frequency always parses
    quantity = quantity.to_compact()
    # Positional, and only as many figures as the value has: `g` would
    # print 250 Hz as 2.5e+02 at the two figures it needs, and 1953.125
    # as 1.95312 at pint's default six.
    magnitude = np.format_float_positional(
        quantity.magnitude, precision=figures, fractional=False, trim="-"
    )
    said = f"{magnitude} {quantity.units:~P}"
    return Text(" · ", dascore_styles["keys"]) + Text(said, dascore_styles["units"])


def percent(value: float) -> str:
    """Say a fraction as a percentage, without rounding a hole away."""
    for places in range(4):
        text = f"{value:.{places}%}"
        # 100% is a claim about the whole span, so only a whole span earns it.
        if value >= 1.0 or float(text.rstrip("%")) < 100.0:
            return text
    return "<100%"


# What names a group which shares every attribute it states with the
# others, or states none: the key the acquisition is filed under. It is
# the first of the config's `patch_kind_attrs`, and the one of them
# which names a place rather than describing it.
ACQUISITION_ATTR = "acquisition_key"


def group_names(
    frame: pd.DataFrame,
    ignore: Iterable[str] = (),
    ordinals: Iterable | None = None,
    fallback: str | None = None,
) -> list[str]:
    """
    Name each row of a group frame by what tells it apart from the others.

    The one naming rule a spool has, kept apart from the drawing of it
    so a coverage plot's lanes and a spool repr's tracks are named the
    same way. The same way, not always the same name: both callers fall
    back on the same attribute, but they partition their rows
    differently and number an unnameable group differently -- the plot
    by its ``group_id``, a repr by row position.

    Parameters
    ----------
    frame
        One row per group.
    ignore
        Columns which describe the groups rather than tell them apart,
        such as the extent each is measured over. A column whose name
        begins with an underscore is skipped whether it is named here
        or not.
    ordinals
        What to call a group its own values cannot name. The row's
        position by default; a report passes its ``group_id``.
    fallback
        A column to name a group by when nothing tells it apart, asked
        before the ordinal is.
    """
    ignore = set(ignore)
    ordinals = range(len(frame)) if ordinals is None else list(ordinals)
    stated = [
        x for x in frame.columns if x not in ignore and not str(x).startswith("_")
    ]
    telling = [x for x in stated if frame[x].astype(str).nunique() > 1]
    described = []
    for _, row in frame.iterrows():
        stated_values = (str(row[x]) for x in telling if pd.notnull(row[x]))
        parts = [x for x in stated_values if x]
        # Nothing tells a lone group apart, since there is nothing to
        # tell it apart from, and every group of a spool of one
        # acquisition is in that position. It is still a named thing,
        # and the fallback says what its ordinal cannot.
        if not parts and fallback is not None:
            named = row.get(fallback)
            if pd.notnull(named) and str(named):
                parts = [str(named)]
        # A group which states neither is left blank here and named by
        # its ordinal below; the blank is a value it states, but drawing
        # it as an empty label would read as a rendering gap.
        described.append(" · ".join(parts))
    # Two groups can state the same attributes and still be two groups --
    # sampling rate and coordinate structure part them without being
    # shown -- so where a description is shared its ordinal tells them
    # apart. A lane which named two groups would silently draw one.
    shared = Counter(described)
    names = []
    for description, ordinal in zip(described, ordinals, strict=True):
        if not description:
            names.append(f"group {ordinal}")
        elif shared[description] > 1:
            names.append(f"{description} ({ordinal})")
        else:
            names.append(description)
    return names


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
    left_out = elision_text(len(items) - limit)
    # A zero limit shows nothing, and nothing needs no comma after it.
    return Text(f"{shown}, ") + left_out if shown else left_out


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
    # Started empty and appended to: a Text built as Text(x, style=...) makes
    # that style the base of everything appended after it, so the class name's
    # style would bleed onto every value.
    base = Text("")
    style = dascore_styles.get(style, style or "bold")
    base += Text(model.__class__.__name__, style=style)
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


def human_size(byte_count: int) -> str:
    """How much room something takes up, in the largest unit which fits."""
    size = float(byte_count)
    if not np.isfinite(size):
        # A dask array of unknown chunks reports nan bytes, and
        # "nan TiB" is a worse answer than not saying.
        return ""
    # The largest name takes whatever is left however big, so it is the
    # one the loop never has to decide about -- and the line after a
    # loop which always returns is a line no test can reach.
    *smaller, largest = _SIZE_UNITS
    for name in smaller:
        # Rounded before it is compared: a count a hair under a boundary
        # prints as 1024 of the smaller unit otherwise, which is the one
        # answer "the largest unit which fits" rules out.
        if round(size, 1) < 1024:
            return f"{size:.1f} {name}".replace(".0 ", " ")
        size /= 1024
    return f"{size:.1f} {largest}".replace(".0 ", " ")


def array_to_text(data, units=None) -> Text:
    """Convert a coordinate to string."""
    header = Text("➤ ") + Text("Data", style=dascore_styles["dc_red"])
    unitstr = Text("") if units is None else Text(f", units: {units}")
    header += Text(f" ({data.dtype}") + unitstr
    # How much room it takes up. A repr states the dtype and the shape,
    # which is the size in pieces; whether it fits in memory is the
    # question those two are usually being multiplied to answer.
    byte_count = getattr(data, "nbytes", None)
    if byte_count is not None and (size := human_size(byte_count)):
        header += Text(", ") + Text(size, dascore_styles["keys"])
    header += Text(")")
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
