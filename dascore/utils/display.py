"""Utils for displaying dascore objects."""

from __future__ import annotations

import numbers
import re
import textwrap
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cache, singledispatch
from graphlib import CycleError, TopologicalSorter
from html import escape
from importlib.resources import files
from itertools import groupby, pairwise
from typing import TypeVar

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from rich.console import Console
from rich.style import Style
from rich.text import Text

import dascore as dc
from dascore.config import get_config
from dascore.constants import dascore_styles
from dascore.units import get_quantity, get_quantity_str
from dascore.utils.time import to_float

# What a container holds, whatever that is: `_limit_items` takes any of
# them and hands back the same kind.
_Item = TypeVar("_Item")

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

# What an instant is, and so what is stated in calendar fields a range
# can share. An offset is not one: 1.5s and 1.25s have a leading "1" in
# common and it is not a field either of them states.
_INSTANT_TYPES = (pd.Timestamp, np.datetime64)

# What stands in a range's far end for the fields its near end already
# stated. Read as "and the rest of it is what you just read".
_REPEAT_MARK = "…"

# What divides one field of a rendered time from the next, kept so a
# head can be measured in whole fields. The date is not split here: it
# is one field, elided whole or not at all.
_TIME_FIELDS = re.compile(r"([:.])")

# What an instant has to say to be taken apart by position: a four digit
# year, a divider numpy writes as T and pandas as a space, and nothing
# after the fraction. One which says more -- a timezone offset, or a
# year of five digits -- is drawn as it states itself rather than sliced
# into fields which would then be wrong.
_INSTANT_LAYOUT = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?")

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
class Row:
    """
    One record, as the cells a table draws it in.

    The name is what the row is called; the fields are what it states,
    each one already rendered. A field a record has nothing to say for
    is absent rather than blank, so two rows need not state the same
    ones.

    Every field has a label, which is the column it belongs in, and
    says whether it wants that label printed as well. A value which
    already says what it is -- a span, in brackets -- does not, since a
    line has no heading to say it and a column does.
    """

    name: Text
    kind: Text
    fields: tuple[tuple[str, Text, bool], ...] = ()


@dataclass(frozen=True, slots=True)
class Table:
    """
    Records of one sort, drawn together.

    They need not state the same fields -- a coordinate of one sample
    has no span, and one read off a file may have no step -- so a
    column exists for every field any row states and a row with
    nothing for one leaves it empty.

    A terminal draws each record on its own line, which is what keeps
    `str()` unchanged; a panel draws them in columns, where each label
    is a heading said once.
    """

    rows: tuple[Row, ...] = ()
    numeric: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Section:
    """
    A titled block: the line which names it, and what sits under it.

    A section with no body is a statement rather than a container.

    Sections nest, which is what an inventory is: its networks hold
    fiber arrays and stations, and its fiber arrays hold acquisitions
    and optical paths. ``depth`` says how far down one sits. The title
    is the bare line either way; a terminal sets it in by its depth,
    since indentation is the only nesting it has, and a panel puts it
    in a block inside its parent's.
    """

    title: Text
    body: tuple[Raw | Table | Section, ...] = ()
    depth: int = 0


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
def _render_row(node: Row) -> Text:
    key_style = dascore_styles["keys"]
    out = Text("\n    ") + node.name + Text(": ") + node.kind + Text("(")
    for label, value, labelled in node.fields:
        out += Text(f" {label}: ", key_style) if labelled else Text(" ")
        out += value
    return out + Text(" )")


@render_text.register
def _render_table(node: Table) -> Text:
    # Joined on nothing: a row states the newline which puts it on its
    # own line, the same way a section body carries the one which
    # separated it from its title.
    return Text("").join(render_text(x) for x in node.rows)


@render_text.register
def _render_section(node: Section) -> Text:
    out = _section_title(node.title, node.depth)
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

# The one class every fragment is wrapped in, and what every rule in
# repr.css is scoped under. See that file for why.
# What a block title opens with in a terminal, where there is no
# disclosure triangle to draw one. A panel draws one, and both is two
# markers for one thing.
_SECTION_MARKER = "\u27a4 "
_HTML_ROOT = "dc-repr"

# How many colors the nesting ramp holds before it repeats. An inventory
# nests three levels -- network, fiber array, then what an array holds --
# and one spare covers a tree which grows a fourth; past that it reads by
# its rails rather than by a hue nobody could name.
_NEST_COLORS = 4

# Style words a class exists for. A color outside this list still draws
# in a terminal, where rich knows every color it can parse; here it
# draws as the host's own ink, which is the safe way to be wrong.
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


def _style_classes(style: Style) -> tuple[str, ...]:
    """Return the CSS classes a resolved rich style is drawn with."""
    words = [
        x for x, on in (("bold", style.bold), ("underline", style.underline)) if on
    ]
    if style.color is not None:
        words.append(style.color.name)
    return tuple(f"dc-{x}" for x in words if x in _STYLE_WORDS)


def _text_to_html(text: Text) -> str:
    """Render a rich Text as an inline HTML fragment."""
    plain = text.plain
    bounds = sorted({0, len(plain)}.union(*((x.start, x.end) for x in text.spans)))
    console = Console()
    # One style per run, built by adding each span to the runs it covers
    # rather than by asking every span about every run. A repr with a
    # few thousand spans -- a patch of many coordinates, a spool of many
    # tracks -- took over a second the other way round.
    at = {position: index for index, position in enumerate(bounds)}
    styles = [console.get_style(text.style, default="")] * max(len(bounds) - 1, 0)
    for span in text.spans:
        style = console.get_style(span.style, default="")
        for index in range(at[span.start], at[span.end]):
            # Added, not gathered: two spans can both state a color, and
            # which one wins is rich's arithmetic. Stacking both classes
            # and letting the stylesheet's source order pick made a
            # units string grey in a panel and blue in a terminal.
            styles[index] = styles[index] + style
    out = []
    for index, (start, end) in enumerate(pairwise(bounds)):
        chunk = escape(plain[start:end], quote=False)
        classes = _style_classes(styles[index])
        out.append(
            f'<span class="{" ".join(classes)}">{chunk}</span>' if classes else chunk
        )
    return "".join(out)


# A quoted string, or a comment. A string is matched first on purpose:
# `content: "/*"` is a value, and a stripper which read it as the start
# of a comment would swallow every declaration up to the next `*/`. A
# string runs over an escaped quote (`content: "\""`), which otherwise
# ends it early and leaves the rest of the value read as CSS.
_CSS_STRING_OR_COMMENT = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|/\*.*?\*/",
    re.DOTALL,
)


def _strip_css_comments(css: str) -> str:
    """Return the CSS with its comments, and the blank lines they leave, gone."""
    kept = _CSS_STRING_OR_COMMENT.sub(
        lambda m: "" if m.group().startswith("/*") else m.group(), css
    )
    return re.sub(r"\n{2,}", "\n", kept).strip() + "\n"


@cache
def _get_stylesheet() -> str:
    """
    Return the CSS every HTML repr carries, without its comments.

    They are near half the sheet by weight, and they are written for
    whoever edits it. A panel carries the whole sheet and a notebook
    carries one panel per cell, so they are dropped on the way out
    rather than sent to every reader of every cell.
    """
    css = files("dascore").joinpath("repr.css").read_text(encoding="utf-8")
    return _strip_css_comments(css)


@singledispatch
def _render_html(node) -> str:
    """Render a repr node as an HTML fragment."""
    msg = f"cannot render {type(node).__name__} as html"
    raise NotImplementedError(msg)


def _body_text(text: Text) -> Text:
    """
    What sits under a title, without the whitespace which framed it.

    The leading newline separated the body from its title and belongs to
    neither, and a trailing one drew a blank line inside the block --
    `attrs_to_text` ends on one so a printed patch ends on one.
    """
    plain = text.plain
    start = 1 if plain.startswith("\n") else 0
    return text[start : len(plain.rstrip("\n"))]


@_render_html.register
def _html_raw(node: Raw) -> str:
    # A `pre`, because what a producer laid out is laid out in columns:
    # an array printed by numpy, a track list padded to line up. HTML
    # would collapse the runs of spaces which do that work.
    text = _body_text(node.text)
    return f'<pre class="dc-body">{_text_to_html(text)}</pre>'


def _merge_columns(rows: Sequence[Row]) -> list[str]:
    """
    One column order which every row's own order agrees with.

    Rows state different fields -- a coordinate of one sample has no
    span, and a coordinate which selected nothing states neither a min
    nor a max -- so the columns are the union. Sorted rather than
    merged by hand: what each row states is an ordering constraint on
    part of the whole, and reading them in as edges is what keeps a
    field which appears late in one row and early in another from
    landing where no row puts it.

    Two rows can disagree outright, which is a cycle and has no answer;
    the order they were first stated in is the one taken then.
    """
    graph: dict[str, set[str]] = {}
    for row in rows:
        previous = None
        for label, _, _ in row.fields:
            graph.setdefault(label, set())
            if previous is not None:
                graph[label].add(previous)
            previous = label
    first = {label: index for index, label in enumerate(graph)}
    sorter = TopologicalSorter(graph)
    try:
        sorter.prepare()
    except CycleError:
        return list(graph)
    # Taken one at a time, earliest-stated first, so fields which
    # constrain nothing in each other -- two records sharing none --
    # stay in the order they were stated rather than interleaving.
    ready: list[str] = []
    out: list[str] = []
    while sorter.is_active():
        ready.extend(sorter.get_ready())
        ready.sort(key=first.__getitem__)
        label = ready.pop(0)
        out.append(label)
        sorter.done(label)
    return out


@_render_html.register
def _html_table(node: Table) -> str:
    # Every label any row states, in the order they are first stated, so
    # a row which says nothing for one leaves that cell empty rather
    # than shifting the ones after it.
    labels = _merge_columns(node.rows)
    # What each record is, which a terminal states in front of its
    # fields. A column of its own here rather than nothing at all.
    head = "<th>kind</th>" + "".join(
        f"<th>{escape(x, quote=False)}</th>" for x in labels
    )
    body = []
    for row in node.rows:
        stated = {label: value for label, value, _ in row.fields}
        name = _text_to_html(row.name)
        cells = [f"<td>{_text_to_html(row.kind)}</td>"]
        for label in labels:
            css = ' class="dc-num"' if label in node.numeric else ""
            value = stated.get(label)
            cells.append(f"<td{css}>{_text_to_html(value) if value else ''}</td>")
        body.append(f'<tr><th scope="row">{name}</th>{"".join(cells)}</tr>')
    return (
        f'<div class="dc-scroll"><table class="dc-table">'
        f"<thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


@singledispatch
def _visible_lines(node) -> int:
    """
    How many lines a reader sees of a node as it is drawn.

    Zero where a node draws nothing, which is what says a block has no
    body to fold: a section whose only child is the newline which
    separated it from its title is a statement, not a container.
    """
    msg = f"cannot count {type(node).__name__}"
    raise NotImplementedError(msg)


@_visible_lines.register
def _raw_lines(node: Raw) -> int:
    # Counted on what is drawn, not on what was handed over: the body
    # loses the newline which separated it from its title and any it
    # ends on, and a block counted before that folds one line early.
    plain = _body_text(node.text).plain
    return plain.count("\n") + 1 if plain else 0


@_visible_lines.register
def _table_lines(node: Table) -> int:
    # Its heading row is a line a reader sees, so a table of two records
    # draws three.
    return len(node.rows) + 1 if node.rows else 0


def _title_lines(node: Section) -> int:
    """How many lines the line which names a block runs to.

    Usually one. A field value may hold a newline -- a description
    often does -- and both a `summary` and a `.dc-line` keep it, so a
    block which counted its title as one line would fold a level late.
    """
    return node.title.plain.count("\n") + 1


@_visible_lines.register
def _section_lines(node: Section) -> int:
    # A folded section is one line, whatever it holds. That is what lets
    # a reader open a large tree a level at a time: an inventory of
    # twenty networks counts twenty lines here rather than every
    # acquisition under all of them, so its own block still opens.
    #
    # Counted once and held: asking again after deciding would walk the
    # same subtree a second time at every level, which is exponential in
    # how deep the tree goes.
    lines = _body_lines(node)
    shown = lines if lines <= get_config().display_html_open_lines else 0
    return _title_lines(node) + shown


def _body_lines(node: Section) -> int:
    """How many lines opening a section would show."""
    return sum(_visible_lines(x) for x in node.body)


def _nest_classes(depth: int) -> tuple[str, ...]:
    """
    What says how deep in a nesting something sits.

    A top-level block sits in no nesting and is given nothing. The color
    ramp wraps rather than deepening forever: past a few levels it is the
    rails which tell them apart, not which color they are.
    """
    if not depth:
        return ()
    return ("dc-nest", f"dc-d{(depth - 1) % _NEST_COLORS}")


@_render_html.register
def _html_section(node: Section) -> str:
    # A title is the bare line: the indentation a terminal draws nesting
    # with is added when a terminal draws it, and here the nesting is
    # the markup.
    title = node.title
    if title.plain.startswith(_SECTION_MARKER):
        title = title[len(_SECTION_MARKER) :]
    title = _text_to_html(title)
    nest = _nest_classes(node.depth)
    lines = _body_lines(node)
    if not lines:
        # Nothing to fold, so nothing to offer folding.
        return f'<div class="{" ".join(("dc-line", *nest))}">{title}</div>'
    state = " open" if lines <= get_config().display_html_open_lines else ""
    body = "".join(_render_html(x) for x in node.body)
    css = f' class="{" ".join(nest)}"' if nest else ""
    return f"<details{css}{state}><summary>{title}</summary>{body}</details>"


@_render_html.register
def _html_repr(node: Repr) -> str:
    # The banner underlines itself with dashes in a terminal, drawn in
    # columns; here that is a border, and an emoji is not two columns
    # wide in every font a browser might choose.
    banner = _text_to_html(node.header.split("\n")[0])
    body = "".join(_render_html(x) for x in node.body)
    return (
        f'<div class="{_HTML_ROOT}"><style>{_get_stylesheet()}</style>'
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
    time, so it is written once here -- as text for a terminal, and as
    HTML for a display which draws one. A notebook asks for both, so a
    host whose nodes are expensive to build should cache them.
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
            return _render_html(self._repr_node())
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


def _fixed_point_digits(text: str) -> int:
    """
    How many digits a fixed-point rendering says the value in.

    Only of a fixed-point rendering: the sign and the zeros standing for
    the scale come off the front as characters, which is exactly what
    they are there, and nowhere else.
    """
    return len(text.lstrip("-0.").replace(".", ""))


@get_nice_text.register(float)
@get_nice_text.register(np.float64)
def _nice_float_string(value, style=None):
    """
    Nice print value for floats.

    The configured precision counts decimals, which a value smaller than
    the last of them rounds away entirely: a pulse width of 8e-08 s drew
    as 0.000. Where fixed decimals have that little of the value left,
    it is said in as many figures instead, so what is printed is the
    number rather than the scale it is under. At least one figure: a
    precision of zero asks for a whole number, not for no number.
    """
    value = float(value)
    precision = get_config().display_float_precision
    figures = max(precision, 1)
    text = f"{value:.{precision}f}"
    if float(text) != value and _fixed_point_digits(text) < figures:
        text = f"{value:.{figures}g}"
    return get_nice_text(Text(text), style)


@get_nice_text.register(np.timedelta64)
@get_nice_text.register(pd.Timedelta)
def _nice_timedelta(value, style=None):
    """Get a nice timedelta value."""
    sec = dc.to_timedelta64(value) / np.timedelta64(1, "s")
    return get_nice_text(Text(f"{sec:.9}s"), style)


def _instant_string(value) -> str | None:
    """
    An instant as the ISO string its blocks are read out of.

    A value of coarser precision states less than that -- a date has no
    time on it at all -- so it is read back at the precision everything
    else here assumes.

    None where what it states cannot be taken apart by position at all.
    A timezone offset and a five digit year both push every field along
    by a character or two, and slicing them anyway draws an instant
    which is not the one handed over.
    """
    stated = str(value)
    if len(stated) < 20:  # no room for a time, so it states none
        stated = str(dc.to_datetime64(stated))
    return stated if _INSTANT_LAYOUT.fullmatch(stated) else None


@cache
def _empty_instant() -> str:
    """The epoch, which is what a block saying nothing looks like."""
    stated = _instant_string(dc.to_datetime64(0))
    # The epoch is an instant of this library's own making, and states
    # itself in the layout everything here is measured against.
    assert stated is not None
    return stated


def _instant_blocks(*stated: str) -> tuple[bool, bool]:
    """
    Which blocks instants are drawn in: the date, and the time.

    A block is drawn where any of them needs it, so the two ends of one
    range come out in one shape and can be read against each other. A
    midnight start drawn beside a stop seconds later would otherwise be
    a bare date beside a full instant, and the fields they have in
    common could not be seen to be in common.

    An instant of the epoch needs neither block, and is drawn as a time.
    """
    empty = _empty_instant()
    date = any(x[:10] != empty[:10] for x in stated)
    time = any(x[11:] != empty[11:] for x in stated)
    return date, time or not date


def _instant_text(stated: str, date: bool, time: bool) -> Text:
    """
    An instant drawn in the blocks asked for, each field styled.

    The string is ISO 8601 but for its divider, which a pandas timestamp
    states as a space; every field is taken by position, and the divider
    is written rather than copied.
    """
    ymd = dascore_styles["ymd"]
    hms = dascore_styles["hms"]
    dec = dascore_styles["dec"]
    out = Text("")
    if date:
        parts = (stated[:4], stated[5:7], stated[8:10])
        out += Text("-").join([Text(x, ymd) for x in parts])
    if time:
        if date:
            out += Text("T")
        parts = (stated[11:13], stated[14:16], stated[17:19])
        out += Text(":").join([Text(x, hms) for x in parts])
        # Only what the fraction says. A step of a millisecond leaves
        # six digits of nothing behind it at nanosecond precision.
        if fraction := stated[20:].rstrip("0"):
            out += Text(".") + Text(fraction, dec)
    return out


@get_nice_text.register(np.datetime64)
@get_nice_text.register(pd.Timestamp)
def _nice_datetime(value, style=None):
    """Get a nice datetime value, in the blocks which say something."""
    if pd.isnull(value) or (stated := _instant_string(value)) is None:
        return get_nice_text(Text(str(value)), style)
    return get_nice_text(_instant_text(stated, *_instant_blocks(stated)), style)


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


# What one level of a containment tree is set in from the one above, in
# a terminal. A panel nests instead, and drops it.
_INDENT = "    "


def _section_title(line: Text, depth: int) -> Text:
    """
    How a terminal draws the line which names a block.

    A block below the top starts on a line of its own, set in by how deep
    it sits; the top of a tree opens wherever its container left off, so
    it is drawn as it stands. Every line is set in, not just the first: a
    field value may hold a newline -- a description usually does -- and a
    continuation left at column zero reads as a block of its own.
    """
    if not depth:
        # A copy, not the node's own line: a renderer which appends to
        # what it drew would otherwise rewrite the node it read.
        return line.copy()
    return Text("\n") + _indent_text(line, _INDENT * depth)


def child_sections(items, depth: int) -> tuple[Section, ...]:
    """
    The blocks a container's children draw, and a line for any left out.

    Only what is shown is rendered, so the cost of a repr is what it
    prints rather than what the tree holds.
    """
    shown, left_out = _limit_items(items)
    out = [x._repr_section(depth) for x in shown]
    if left_out:
        out.append(Section(elision_text(left_out), depth=depth))
    return tuple(out)


def _indent_text(text: Text, prefix: str = _INDENT) -> Text:
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


def _limit_items(
    items: Iterable[_Item], limit: int | None = None
) -> tuple[list[_Item], int]:
    """
    Take at most ``limit`` items, and say how many were left behind.

    Where a repr stops early it says how much it is not showing, so the
    count is as much a part of the answer as the items are; silently
    stopping at ten reads as though ten is all there were.
    """
    limit = get_config().display_max_items if limit is None else limit
    # Config forbids a negative one, so only a caller can state it, and
    # `items[:-1]` would quietly show all but the last rather than none.
    assert limit >= 0, f"a display limit is a count, not {limit}"
    items = list(items)
    shown = items[:limit]
    return shown, len(items) - len(shown)


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

    Only of two times. A duration is read in seconds, so a distance of
    299 handed to this would come back as "5 m" -- five minutes, of a
    span measured in metres. `span_text` is what an extent of any kind
    is asked through; this is the arm of it which reads a clock.
    """
    if not isinstance(low, _TIME_TYPES) or not isinstance(high, _TIME_TYPES):
        return None
    # Both the same kind of time. An instant and a duration are not two
    # ends of one extent, and subtracting them raises out of the middle
    # of a repr in one order and gives an instant back in the other.
    if isinstance(low, _INSTANT_TYPES) != isinstance(high, _INSTANT_TYPES):
        return None
    # Read from the two ends together, which is exact, and from each of
    # them on its own where that cannot be done. An int64 of
    # nanoseconds holds about 584 years and two instants can lie
    # further apart than that: pandas raises over it, and numpy raises
    # over it on some versions and wraps on others -- five centuries
    # came back as 84.6 of them, pointing the other way from the ends
    # they were read off.
    seconds = None
    try:
        span = high - low
    except (OutOfBoundsDatetime, OutOfBoundsTimedelta, OverflowError):
        pass
    except TypeError:
        # Two times which do not agree on a timezone do not subtract at
        # all, and neither end can be read as the other's clock.
        return None
    else:
        # Divided by a second rather than read with `to_float`, which
        # counts nanoseconds: the same int64 the ends were held in, and
        # ten thousand years read that way came back as sixty one.
        try:
            seconds = span / np.timedelta64(1, "s")
        except TypeError:
            # A span held in years or months is not a fixed number of
            # seconds, so numpy refuses to say how many. Neither will
            # this.
            return None
        if (seconds < 0) != (high < low):
            seconds = None
    if seconds is None:
        # Each end on its own, which fits where the span between them
        # does -- to a precision only a far shorter span would miss.
        seconds = to_float(high) - to_float(low)
    if not (said := human_duration(seconds)):
        return None
    return Text(f"<{said}>", dascore_styles["keys"])


def span_text(low, high, units: str | None = None) -> Text | None:
    """
    How wide an extent is, as a repr states it.

    Asked wherever a repr states two ends. How far apart they lie is a
    fact those two numbers do not carry: fiber running from 1212.4 m to
    1636.7 m is 424.3 m of it, and the subtraction which says so is not
    work a reader should be left to do.

    A time span is a duration, said in the largest unit which fits.
    Every other extent is as wide as its own numbers say, since reading
    one in seconds would state a different quantity than the one
    measured.

    ``units`` is what to say that width in, and is for a line which
    states them nowhere else. A line which already names its units --
    a coordinate row with a units field, a dimension with its unit
    after it -- passes nothing rather than say them twice.

    None where there is no width to state: two ends which meet, which
    is one sample and not a span of nothing, and a dimension of labels,
    which has two ends and nothing measurable between them.
    """
    if isinstance(low, _TIME_TYPES) or isinstance(high, _TIME_TYPES):
        return duration_text(low, high)
    # Real, not Number: a complex pair has no width along one axis.
    # Nor is a bool a quantity, though Python counts one as an integer;
    # a true which is one more than a false is arithmetic, not a width.
    if isinstance(low, bool) or isinstance(high, bool):
        return None
    if not (isinstance(low, numbers.Real) and isinstance(high, numbers.Real)):
        return None
    try:
        if isinstance(low, numbers.Integral) and isinstance(high, numbers.Integral):
            # Subtracted as integers, which is exact and does not wrap:
            # two int64 ends one apart up near 2**60 are the same float,
            # so a width of one would come back as no width at all.
            width = float(int(high) - int(low))
        else:
            width = float(high) - float(low)
    except (OverflowError, ValueError, TypeError):
        # An end no float holds, a Python int of four hundred digits
        # among them. A repr states nothing rather than raising out of
        # the middle of itself.
        return None
    # A width, like a duration, is how far apart the two ends lie and
    # not which of them was handed over first.
    width = abs(width)
    if not np.isfinite(width) or width == 0:
        return None
    stated = f" {units}" if units else ""
    return Text(f"<{width:g}{stated}>", dascore_styles["keys"])


def _split_instant(rendered: str) -> tuple[str, str]:
    """
    A rendered instant, as the date it states and the time it states.

    One of the two is empty where the value did not need it: a repr
    drops a date of the epoch and a time of midnight, so the two ends
    of one range can arrive in different shapes.
    """
    date, divider, time = rendered.partition("T")
    if divider:
        return date, time
    # No divider, so it is one or the other. A date has dashes in it, a
    # time has colons, and neither has the other's.
    return (date, "") if "-" in date else ("", date)


def _shared_head(low: str, high: str) -> int:
    """
    How many leading characters of `high` repeat what `low` already says.

    Measured in whole fields, so a mark never stands for the first digit
    of one. The date is a single field: two instants on different days
    share nothing, since eliding the year of 2023-06-01 and 2023-07-01
    would leave the month to be noticed on its own.
    """
    low_date, low_time = _split_instant(low)
    high_date, high_time = _split_instant(high)
    # A date one end states and the other does not is not a date they
    # share, whatever the characters under it say.
    if bool(low_date) != bool(high_date) or low_date != high_date:
        return 0
    shared = len(high_date)
    # The divider sits between the date and the fields below it, and is
    # part of neither.
    at = shared + 1 if high_date and high_time else shared
    low_fields = _TIME_FIELDS.split(low_time)
    high_fields = _TIME_FIELDS.split(high_time)
    # Field, separator, field, at even indices and odd: a separator
    # counts toward the head like anything else, but the head only ends
    # at a field, so what is left starts with the separator before it.
    for index, (stated, repeated) in enumerate(zip(low_fields, high_fields)):
        if stated != repeated:
            break
        at += len(repeated)
        if index % 2 == 0:
            shared = at
    return shared


def range_texts(low, high) -> tuple[Text, Text]:
    """
    The two ends of a range, drawn as one range rather than as two values.

    Two things happen here which drawing each end on its own cannot do.

    They are drawn in one shape: a block either end needs is drawn on
    both, so a start at midnight comes out as ``2017-09-18T00:00:00``
    beside its stop rather than as a bare date. What each states is then
    in the same place, which is what lets the second be read against the
    first.

    And the far end states only what differs. The fields it repeats
    stand next to it already, so they are replaced by one mark: a range
    inside one day states that day once. A leading ``T`` goes with them,
    since with the date gone it divides nothing, while a leading ``:``
    or ``.`` stays -- it says which field the number after it is, so
    41.5 cannot be read as an hour.

    Only of two instants. An offset of 1.25s repeats nothing of one of
    1.5s -- the digit they lead with is not a field either of them
    states -- and any other pair is drawn as it would have been anyway.
    """
    drawn = (get_nice_text(low), get_nice_text(high))
    if not isinstance(low, _INSTANT_TYPES) or not isinstance(high, _INSTANT_TYPES):
        return drawn
    if pd.isnull(low) or pd.isnull(high):
        return drawn
    near_stated, far_stated = _instant_string(low), _instant_string(high)
    if near_stated is None or far_stated is None:
        return drawn
    stated = (near_stated, far_stated)
    blocks = _instant_blocks(*stated)
    near, far = (_instant_text(x, *blocks) for x in stated)
    if not (shared := _shared_head(near.plain, far.plain)):
        return near, far
    rest = far[shared:]
    if rest.plain.startswith("T"):
        rest = rest[1:]
    # Started empty and appended to: a Text built as Text(x, style=...)
    # makes that style the base of everything appended after it, so the
    # mark's grey would bleed onto the fields which survived it.
    said = Text("")
    said += Text(_REPEAT_MARK, dascore_styles["keys"])
    return near, said + rest


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
        # A part repeating the one before it says the same thing twice:
        # DSS data tagged "DSS" named its track "XM.MINE1.03.WSF · DSS · DSS".
        parts = [value for value, _ in groupby(parts)]
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


def _value_to_text(name: str, value, style=None, truncate: bool = True) -> Text:
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
        base += _value_to_text(name, value)
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
        txt += _value_to_text(
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
