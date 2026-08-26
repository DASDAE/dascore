"""Tests for displaying dascore objects."""

from __future__ import annotations

import re
import time
from html import unescape
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.style import Style
from rich.text import Text

import dascore as dc
from dascore.config import config_context
from dascore.constants import dascore_styles
from dascore.core.annotations import (
    AnnotationColumn,
    AnnotationSet,
    AnnotationSetAttrs,
)
from dascore.core.inventory import Acquisition, Cable, Network
from dascore.utils import display
from dascore.utils.display import (
    _NEST_COLORS,
    _STYLE_WORDS,
    Raw,
    Repr,
    Row,
    Section,
    Table,
    _body_lines,
    _get_stylesheet,
    _indent_text,
    _limit_items,
    _render_html,
    _section_title,
    _storage_quantum,
    _strip_css_comments,
    _style_classes,
    _text_to_html,
    _value_to_text,
    _visible_lines,
    array_to_text,
    attrs_to_text,
    child_sections,
    counts_to_text,
    duration_text,
    get_header_text,
    get_nice_text,
    group_names,
    human_duration,
    human_size,
    mapping_to_text,
    model_to_line,
    percent,
    rate_text,
    render_text,
    split_block,
    stated_fields,
)
from dascore.utils.patch import _format_values


class TestGetNiceText:
    """Tests for converting coordinate to nice looking rich Text."""

    def test_simple_datetime(self):
        """Ensure the process works for datetime objects."""
        dt = dc.to_datetime64("2023-10-01")
        # YMD should just show YMD
        txt1 = get_nice_text(dt)
        assert str(txt1) == "2023-10-01"
        # Unless YMD is 1970-01-01
        txt2 = get_nice_text(dc.to_datetime64(0))
        assert str(txt2) == "00:00:00"
        # Decimals are displayed if present
        txt3 = get_nice_text(dc.to_datetime64(1.111111111))
        assert str(txt3).endswith(".111111111")

    def test_nat(self):
        """Tests for NaT."""
        dt = np.datetime64("NaT")
        txt = get_nice_text(dt)
        assert str(txt) == "NaT"

    def test_timestamp(self):
        """Tests for pandas timestamps."""
        ts = pd.Timestamp("2012-01-10")
        txt = get_nice_text(ts)
        assert str(txt) == "2012-01-10"

    def test_float_precision_config(self):
        """Float display precision should come from runtime config."""
        with config_context(display_float_precision=1):
            txt = get_nice_text(1.234)
        assert str(txt) == "1.2"


class TestArrayFormatting:
    """Tests for config-backed array formatting behavior."""

    def test_array_threshold_config(self):
        """Array display truncation threshold should be configurable."""
        data = np.arange(10)
        with config_context(display_array_threshold=3):
            txt = array_to_text(data)
        assert "..." in str(txt)

    def test_patch_history_threshold_config(self):
        """Patch history formatting should use the configured threshold."""
        data = np.arange(10)
        with config_context(
            display_float_precision=0,
            display_patch_history_array_threshold=3,
        ):
            out = _format_values(data)
        assert "..." in out


def styles_at(text, substring):
    """Return the styles covering the first character of a substring."""
    start = text.plain.index(substring)
    return {x.style for x in text.spans if x.start <= start < x.end}


class TestGetHeaderText:
    """Tests for the banner which opens a top-level object's repr."""

    def test_underline_matches_columns(self):
        """The underline is as wide as the header prints, not as long."""
        header, line = str(get_header_text("Patch ⚡")).split("\n")
        assert len(line) == Text(header).cell_len
        # An emoji is two columns wide, so the two differ in length.
        assert len(line) > len(header)

    def test_names_the_object(self):
        """The name given is in the banner, beside the DASCore text."""
        out = str(get_header_text("Inventory"))
        assert out.startswith("DASCore Inventory")

    def test_style_key_resolved(self):
        """A dascore_styles key is resolved, as the other helpers resolve it."""
        text = get_header_text("Thing", style="dc_blue")
        assert styles_at(text, "Thing") == {dascore_styles["dc_blue"]}


class TestIndentText:
    """Tests for indenting a rich Text."""

    def test_every_line_indented(self):
        """Each line gains the prefix, including the first."""
        out = str(_indent_text(Text("a\nb"), "  "))
        assert out == "  a\n  b"

    def test_styles_kept(self):
        """Indenting does not flatten the styles it wraps."""
        text = Text("red", style="red")
        assert _indent_text(text).spans


class TestCountsToText:
    """Tests for rendering value counts."""

    def test_pairs_rendered(self):
        """Names and counts are paired on one line."""
        assert str(counts_to_text({"a": 2, "b": 1})) == "a: 2, b: 1"

    def test_tail_summarized(self):
        """Names past the limit are counted rather than listed."""
        out = str(counts_to_text({"a": 1, "b": 1, "c": 1}, limit=1))
        assert out == "a: 1, ... 2 more"

    def test_zero_limit_has_no_leading_comma(self):
        """Nothing is shown, and nothing needs a comma after it."""
        assert str(counts_to_text({"a": 1, "b": 2}, limit=0)) == "... 2 more"


class TestStatedFields:
    """Tests for reading what a model states about itself."""

    def test_defaults_and_identity_dropped(self):
        """A default, a blank and an identity field all state nothing."""
        out = stated_fields(Cable(resource_id="c1", name="c"))
        assert out == {"name": "c"}

    def test_unset_times_dropped(self):
        """An unset (NaT) epoch is not a fact about the object."""
        assert "time_min" not in stated_fields(Network(code="XT"))

    def test_set_times_kept(self):
        """A stated epoch is."""
        network = Network(code="XT", time_min="2020-01-01")
        assert "time_min" in stated_fields(network)

    def test_uncomparable_default_is_stated(self):
        """A default which cannot be compared is shown rather than hidden."""

        class Model(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            data: np.ndarray = np.array([1, 2])

        assert "data" in stated_fields(Model())

    def test_mapping_of_models_counted(self):
        """A pool of models says how many it holds, not what each one is."""
        columns = {"pick": AnnotationColumn(description="a pick")}
        attrs = AnnotationSetAttrs(dims=("time",), columns=columns)
        assert "columns: 1" in str(model_to_line(attrs))


class TestModelToLine:
    """Tests for the one-line model summary."""

    def test_only_the_name_is_styled(self):
        """
        A style stays on what it marks.

        Text(x, style=...) makes that style the base of everything appended
        after it, so building a line that way paints every value with the
        class name's style.
        """
        line = model_to_line(Network(code="XT"))
        assert line.style == ""
        assert styles_at(line, "Network") == {"bold"}
        assert styles_at(line, " code: ") == {dascore_styles["keys"]}
        assert styles_at(line, "XT") == set()  # the value is not a key

    def test_names_class_and_fields(self):
        """The line names the class, then what the model states."""
        out = str(model_to_line(Network(code="XT")))
        assert out == "Network( code: XT )"

    def test_identity_field_hidden(self):
        """The class tag is not something a reader asked about."""
        assert "object_type" not in str(model_to_line(Cable(resource_id="c1")))

    def test_children_counted(self):
        """A container of models shows how many it holds, not each one."""
        network = Network(code="XT", stations=[{"code": "S1"}, {"code": "S2"}])
        assert "stations: 2" in str(model_to_line(network))

    def test_units_formatted(self):
        """A units field prints as a unit string."""
        acq = Acquisition(code="DAS", data_units="m/s")
        assert "data_units: m / s" in str(model_to_line(acq))

    def test_extra_values_appended(self):
        """Derived values a caller passes are stated after the fields."""
        out = str(model_to_line(Network(code="XT"), extra={"span": "wide"}))
        assert out.endswith("span: wide )")

    def test_skip(self):
        """Fields the caller lists itself can be left out."""
        network = Network(code="XT", stations=[{"code": "S1"}])
        assert "stations" not in str(model_to_line(network, skip=("stations",)))


class TestMappingToText:
    """Tests for the named block a mapping prints as."""

    def test_header_and_entries(self):
        """The block is a header and one line per entry."""
        out = str(mapping_to_text({"a": 1}, "Things"))
        assert out == "➤ Things\n    a: 1"

    def test_private_skipped(self):
        """Private names are not shown."""
        assert "_b" not in str(mapping_to_text({"a": 1, "_b": 2}, "Things"))

    def test_long_value_not_elided(self):
        """A value with a line to itself is shown whole."""
        long = "x" * 500
        assert long in str(mapping_to_text({"a": long}, "Things"))

    def test_sequence_joined(self):
        """A sequence of plain values is listed rather than counted."""
        out = str(mapping_to_text({"a": (1, 2, 3)}, "Things"))
        assert out.endswith("a: 1, 2, 3")


class TestValueToText:
    """Tests for rendering one value."""

    def test_long_value_elided(self):
        """A value sharing a line with others is bounded."""
        out = str(_value_to_text("a", "x" * 500))
        assert len(out) < 500
        assert out.endswith("…")

    def test_pre_rendered_text_kept(self):
        """Text the caller built is used as-is, not re-read as a sequence."""
        assert str(_value_to_text("a", Text("a: 2, b: 1"))) == "a: 2, b: 1"

    def test_array_keeps_its_own_repr(self):
        """An array states how much of itself to show; it is not unrolled."""
        out = str(_value_to_text("a", np.arange(10_000)))
        assert "..." in out and len(out) < 100

    def test_frame_keeps_its_own_repr(self):
        """Nor is a frame read as a sequence of its column names."""
        out = str(_value_to_text("a", pd.DataFrame({"a": [1, 2]}), truncate=False))
        assert "1" in out and "2" in out

    def test_nat_is_text(self):
        """An unset time renders like every other value, not as a str."""
        assert str(_value_to_text("a", np.datetime64("NaT"))) == "NaT"


class TestPercent:
    """How a fraction says itself."""

    @pytest.mark.parametrize(
        "value, text",
        [
            (1.0, "100%"),
            (0.9993334, "99.9%"),
            (0.92276, "92%"),
            (1.246e-08, "0%"),
            (0.0, "0%"),
        ],
    )
    def test_percent(self, value, text):
        """A percentage reads at the precision it needs."""
        assert percent(value) == text

    def test_a_hole_is_never_rounded_away(self):
        """Only a whole span reads as 100%."""
        assert percent(0.999999999) == "<100%"
        assert percent(1.0) == "100%"


class TestHumanDuration:
    """How long something lasted, in the unit which fits."""

    @pytest.mark.parametrize(
        "seconds, text",
        [
            (0.0, ""),
            (0.008, "8 ms"),
            (1.004, "1 s"),
            (90.0, "1.5 m"),
            (7200.0, "2 h"),
            (86_400.0 * 3, "3 d"),
            # A multi-year outage is not worth reading in days.
            (86_400.0 * 400, "1.1 y"),
            (86_400.0 * 14_852, "40.7 y"),
        ],
    )
    def test_human_duration(self, seconds, text):
        """A duration is stated in the largest unit which fits it."""
        assert human_duration(pd.Timedelta(seconds=seconds)) == text

    def test_duration_of_a_plain_number(self):
        """A plain number is read as a count of seconds, as the gap ticks are."""
        assert human_duration(12.0) == "12 s"

    def test_duration_smaller_than_any_unit(self):
        """A span under a microsecond still says how long it is."""
        assert human_duration(1e-9) == "1e-09 s"

    def test_an_unmeasurable_duration_says_nothing(self):
        """A duration nothing states is not a duration of zero."""
        assert human_duration(np.nan) == ""


class TestGroupNames:
    """How a group is named by what tells it apart."""

    def test_only_what_differs_names_a_group(self):
        """A value every group shares tells none of them apart."""
        frame = pd.DataFrame({"tag": ["a", "b"], "kind": ["das", "das"]})
        assert group_names(frame) == ["a", "b"]

    def test_several_values_join(self):
        """A group stating two telling values is named by both."""
        frame = pd.DataFrame({"tag": ["a", "b"], "kind": ["das", "dss"]})
        assert group_names(frame) == ["a · das", "b · dss"]

    def test_a_nameless_group_takes_its_position(self):
        """A group nothing tells apart falls back to its ordinal."""
        frame = pd.DataFrame({"tag": ["same", "same"]})
        assert group_names(frame) == ["group 0", "group 1"]

    def test_ordinals_may_be_given(self):
        """A caller with its own group ids names by those instead."""
        frame = pd.DataFrame({"tag": ["same", "same"]})
        assert group_names(frame, ordinals=[7, 9]) == ["group 7", "group 9"]

    def test_a_shared_description_is_told_apart(self):
        """Two groups describing themselves alike are still two groups."""
        frame = pd.DataFrame({"tag": ["a", "a", "b"], "n": [1, 2, 3]})
        assert group_names(frame, ignore=("n",)) == ["a (0)", "a (1)", "b"]

    def test_ignored_columns_never_name(self):
        """A column describing the group does not tell it apart."""
        frame = pd.DataFrame({"tag": ["a", "b"], "span": [1.0, 2.0]})
        assert group_names(frame, ignore=("span",)) == ["a", "b"]

    def test_a_private_column_never_names(self):
        """A column the index keeps to itself is not a value a group states."""
        frame = pd.DataFrame({"tag": ["a", "b"], "_key": ["x", "y"]})
        assert group_names(frame) == ["a", "b"]

    def test_a_fallback_names_what_nothing_tells_apart(self):
        """A lone group has nothing to be told apart from, but has a name."""
        frame = pd.DataFrame({"tag": ["das"], "acquisition_key": ["XM.A..HSF"]})
        assert group_names(frame, fallback="acquisition_key") == ["XM.A..HSF"]
        assert group_names(frame) == ["group 0"]

    def test_the_fallback_yields_to_what_does_tell_them_apart(self):
        """A fallback names a group nothing else can, and only that."""
        frame = pd.DataFrame(
            {"tag": ["alpha", "beta"], "acquisition_key": ["XM.A..HSF"] * 2}
        )
        assert group_names(frame, fallback="acquisition_key") == ["alpha", "beta"]

    def test_a_shared_fallback_still_takes_the_ordinal(self):
        """Two groups the fallback names alike are still two groups."""
        frame = pd.DataFrame({"tag": ["das"] * 2, "acquisition_key": ["XM.A..HSF"] * 2})
        assert group_names(frame, fallback="acquisition_key") == [
            "XM.A..HSF (0)",
            "XM.A..HSF (1)",
        ]

    @pytest.mark.parametrize(
        "frame", [{"tag": ["das"], "acquisition_key": [""]}, {"tag": ["das"]}]
    )
    def test_a_fallback_which_says_nothing_is_not_a_name(self, frame):
        """A blank fallback, or none at all, leaves the ordinal to name it."""
        assert group_names(pd.DataFrame(frame), fallback="acquisition_key") == [
            "group 0"
        ]

    def test_an_unstated_value_is_not_a_blank(self):
        """A group which recorded nothing is not named by an empty part."""
        frame = pd.DataFrame({"tag": ["a", ""], "kind": ["das", "dss"]})
        assert group_names(frame) == ["a · das", "dss"]


class TestDascoreStyles:
    """Tests for the style table every repr draws from."""

    @pytest.mark.parametrize("name", sorted(dascore_styles))
    def test_every_style_parses(self, name):
        """
        Rich must be able to read every style dascore states.

        It resolves a style it cannot parse to a blank one rather than
        raising, so a misspelling here does not fail a test, it silently
        stops coloring whatever states it.
        """
        assert Style.parse(dascore_styles[name])

    @pytest.mark.parametrize("name", sorted(dascore_styles))
    def test_every_style_is_used(self, name):
        """
        A style nothing asks for is a style no one maintains.

        Both of the styles removed with this test had gone unasked-for
        long enough that one of them had stopped parsing unnoticed.

        The name has to be matched where a style is *looked up*, not
        wherever it appears: `dtypes` is also an unrelated dict key in
        `workflow/serialize.py`, so a plain search for the word says it
        is used when nothing styles anything with it.
        """
        sources = [
            x.read_text(encoding="utf-8")
            for x in Path(dc.__file__).parent.rglob("*.py")
            if x.name != "constants.py"
        ]
        lookups = [
            rf"dascore_styles\[[\"']{name}[\"']\]",
            rf"style\s*=\s*[\"']{name}[\"']",
            rf"_rich_style\s*=\s*[\"']{name}[\"']",
            rf"[\"']{name}[\"']\s*\)",
        ]
        assert any(re.search(p, x) for p in lookups for x in sources)


def resolved_styles(text: Text) -> list:
    """The style each character of a Text actually renders with."""
    console = Console()
    return [text.get_style_at_offset(console, x) for x in range(len(text.plain))]


@pytest.fixture(scope="module")
def repr_blocks():
    """Every kind of block a dascore repr is built from."""
    patch = dc.get_example_patch()
    inventory = dc.get_example_inventory("tunnel")
    return {
        "coords": patch.coords.__rich__(),
        "coord": patch.coords.coord_map["time"].__rich__(),
        "data": array_to_text(patch.data),
        "attrs": attrs_to_text(patch.attrs),
        "inventory": inventory.__rich__(),
        "model": inventory.networks[0].__rich__(),
        "header": get_header_text("Patch ⚡"),
        "one_line": Text("no newline in this one", style="bold"),
        "empty": Text(""),
    }


class TestSplitBlock:
    """Tests for turning a rendered block into a section."""

    def test_round_trips(self, repr_blocks):
        """
        Splitting then rendering gives back what went in.

        A node holds the `Text` its producer made rather than a recipe
        for rebuilding it, which is what keeps the two reprs from
        drifting: `render_text` reassembles, it does not re-derive.

        Compared by what each character draws rather than with `==`,
        which reads `Text.style` and `Text.spans` -- slicing moves a
        base style into a span, so a block-level style would fail an
        equality check while drawing exactly the same.
        """
        for name, text in repr_blocks.items():
            trip = render_text(split_block(text))
            assert trip.plain == text.plain, name
            assert resolved_styles(trip) == resolved_styles(text), name

    @pytest.mark.parametrize(
        "text",
        [
            Text(""),
            Text("\n"),
            Text("\nbody"),
            Text("head\n"),
            Text("head\n\n"),
            Text("solo"),
            Text("a\n\nb"),
            Text("head\nbody", style="bold green"),
        ],
        ids=[
            "empty",
            "newline",
            "leading",
            "trailing",
            "two_trailing",
            "no_newline",
            "blank_middle",
            "base_style",
        ],
    )
    def test_round_trips_whatever_the_shape(self, text):
        """Blocks a producer has not written yet still have to survive."""
        trip = render_text(split_block(text))
        assert trip.plain == text.plain
        assert resolved_styles(trip) == resolved_styles(text)

    def test_a_span_across_the_break_still_draws_the_same(self):
        """
        A span which straddles the split comes back as two.

        The body carries the newline that split it, so the two spans
        touch and the break itself keeps its style. What a reader sees
        is unchanged; only the grouping differs.
        """
        text = Text("head\nbody")
        text.stylize("bold", 2, 7)
        trip = render_text(split_block(text))
        assert trip.plain == text.plain
        assert resolved_styles(trip) == resolved_styles(text)
        assert len(trip.spans) == 2

    def test_first_line_is_the_title(self, repr_blocks):
        """The line a reader is shown when the body is not."""
        section = split_block(repr_blocks["coords"])
        assert section.title.plain == "➤ Coordinates (distance: 300, time: 2000)"

    def test_a_single_line_has_no_body(self, repr_blocks):
        """A block with nothing under it is a statement, not a container."""
        assert split_block(repr_blocks["one_line"]).body == ()


class TestRenderText:
    """Tests for rendering repr nodes as text."""

    def test_raw_is_emitted_as_it_stands(self):
        """A producer which laid its own text out is not re-laid-out."""
        text = Text("   already    spaced")
        assert render_text(Raw(text)) == text

    def test_a_body_carries_its_own_separators(self):
        """
        A section concatenates its body rather than spacing it out.

        ``Raw`` means text its producer has already laid out, and that
        includes the newline which put it on the next line.
        """
        node = Section(Text("title"), (Raw(Text("\na")), Raw(Text("\nb"))))
        assert render_text(node).plain == "title\na\nb"

    def test_repr_joins_header_and_sections(self):
        """The banner, then each section under it."""
        node = Repr(Text("banner"), (Section(Text("one")), Section(Text("two"))))
        assert render_text(node).plain == "banner\none\ntwo"

    def test_a_repr_may_state_no_sections(self):
        """An object with nothing to show is still an object."""
        assert render_text(Repr(Text("banner"))).plain == "banner"

    def test_something_which_is_not_a_node(self):
        """A renderer says what it cannot draw rather than drawing it wrong."""
        with pytest.raises(NotImplementedError, match="cannot render int"):
            render_text(42)


class TestRichRepr:
    """Tests for the mixin every rich-rendered class prints through."""

    @pytest.fixture(scope="class")
    def rich_objects(self):
        """One of each class which prints through the mixin."""
        patch = dc.get_example_patch()
        inventory = dc.get_example_inventory("tunnel")
        frame = pd.DataFrame(
            {"time_min": [0.0, 1.0], "time_max": [0.5, 1.5], "group": ["a", "b"]}
        )
        return {
            "patch": patch,
            "coord_manager": patch.coords,
            "coord": patch.coords.coord_map["time"],
            "spool": dc.get_example_spool(),
            "inventory": inventory,
            "network": inventory.networks[0],
            "optical_path": inventory.networks[0].fiber_arrays[0].optical_paths[0],
            "annotation_set": AnnotationSet(frame, dims=("time",)),
            "annotation_column": AnnotationColumn(description="a pick", units="s"),
            "empty_annotation_set": AnnotationSet(frame.iloc[:0], dims=("time",)),
        }

    def test_str_is_the_rich_rendering(self, rich_objects):
        """
        What a plain terminal prints is what a rich one renders.

        This is the assertion which catches a pydantic host listing the
        mixin after ``BaseModel``: the field dump it would print instead
        is still a non-empty string, so a weaker test passes.
        """
        for name, obj in rich_objects.items():
            assert str(obj) == str(obj.__rich__()), name

    def test_repr_is_str(self, rich_objects):
        """Neither form drifts from the other."""
        for name, obj in rich_objects.items():
            assert repr(obj) == str(obj), name


class TestSpoolReprNode:
    """
    Tests for the sections a spool states.

    The spool is the one repr whose summary is allowed to fail without
    the repr failing, which means an assertion on what it says passes
    just as well when it says nothing. These pin the branches instead.
    """

    @pytest.fixture(scope="class")
    def directory_spool(self, tmp_path_factory):
        """A spool which knows the directory it was read from."""
        path = tmp_path_factory.mktemp("spool_repr")
        for index, patch in enumerate(dc.get_example_spool()):
            patch.io.write(path / f"patch_{index}.h5", "dasdae")
        return dc.spool(path).update()

    def test_a_path_is_its_own_section(self, directory_spool):
        """A spool read from disk says where it was read from."""
        titles = [str(x.title) for x in directory_spool._repr_node().body]
        assert any("Path:" in x for x in titles)

    def test_a_path_section_has_no_body(self, directory_spool):
        """One line is a statement, and a renderer must not fold it."""
        section = next(
            x for x in directory_spool._repr_node().body if "Path:" in str(x.title)
        )
        assert section.body == ()

    def test_too_many_patches_states_the_limit(self):
        """Past the limit a spool says what it is instead of summarising."""
        spool = dc.get_example_spool()
        with config_context(display_max_patches=1):
            rendered = str(spool)
        assert "Not summarized" in rendered
        assert "display_max_patches=1" in rendered
        assert "Dimensions" not in rendered

    def test_a_summary_which_raises_still_prints_the_header(self, monkeypatch):
        """
        A repr which raises makes an object undebuggable exactly when
        someone needs to look at it, so the summary is allowed to fail
        and the banner is not.
        """
        spool = dc.get_example_spool()

        def boom(self):
            raise ValueError("no summary for you")

        monkeypatch.setattr(type(spool), "_summary_blocks", boom)
        rendered = str(spool)
        assert "Spool" in rendered
        assert "Dimensions" not in rendered


class TestRateText:
    """Tests for stating a sampling step as the rate it is quoted in."""

    @pytest.mark.parametrize(
        ("step", "expected"),
        [
            (np.timedelta64(4_000_000, "ns"), "250 Hz"),
            (np.timedelta64(1, "s"), "1 Hz"),
            (np.timedelta64(1, "ms"), "1 kHz"),
            (np.timedelta64(976_562, "ns"), "1.024 kHz"),
            (np.timedelta64(10_417, "ns"), "96 kHz"),
            (np.timedelta64(1_000, "ns"), "1 MHz"),
            (np.timedelta64(1, "ns"), "1 GHz"),
            (np.timedelta64(6_400_000, "ns"), "156.25 Hz"),
            (np.timedelta64(512_000, "ns"), "1.953125 kHz"),
            (np.timedelta64(2_000_000, "ns"), "500 Hz"),
            # Slow acquisition: a rate under a hertz still reads as one.
            (np.timedelta64(10, "s"), "100 mHz"),
        ],
    )
    def test_a_time_step_states_its_rate(self, step, expected):
        """A step in time is quoted in Hz, which is what a rate is."""
        assert expected in str(rate_text(step))

    @pytest.mark.parametrize(
        "step",
        [
            np.timedelta64(4_000_000, "ns"),
            np.timedelta64(6_400_000, "ns"),
            np.timedelta64(512_000, "ns"),
            np.timedelta64(10_417, "ns"),
            np.timedelta64(1, "ns"),
            np.timedelta64(8, "ms"),
            np.timedelta64(3, "ms"),
            np.timedelta64(7, "s"),
            pd.Timedelta(seconds=0.004),
        ],
    )
    def test_the_rate_said_describes_the_step(self, step):
        """
        Whatever is printed has to describe the step beside it.

        Held over the printed characters rather than the number they
        came from, since a rate rounded for the check and then formatted
        to fewer figures, or into exponent notation, states something
        the step does not. A rate is quoted to four figures, so four
        figures is how closely it has to agree -- not to a fixed
        nanosecond, which only matches the code for steps stored in
        them.
        """
        magnitude, unit = str(rate_text(step)).strip().removeprefix("· ").split(" ")
        prefixes = {"mHz": 1e-3, "Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        said = float(magnitude) * prefixes[unit]
        true_rate = 1.0 / (step / np.timedelta64(1, "s"))
        assert abs(said - true_rate) <= abs(true_rate) * 5e-4

    @pytest.mark.parametrize(
        ("step", "expected"),
        [
            (np.timedelta64(8, "ms"), "125 Hz"),
            (np.timedelta64(4, "ms"), "250 Hz"),
            (np.timedelta64(3, "ms"), "333.3 Hz"),
            (np.timedelta64(13, "ms"), "76.92 Hz"),
        ],
    )
    def test_a_step_stored_coarser_than_a_nanosecond(self, step, expected):
        """
        A step held to milliseconds is exact, not rounded.

        An 8 ms step is exactly 125 Hz, and 120 Hz also inverts to
        within half a millisecond of it. Taking the shortest rate which
        lands inside the step's resolution states 120 Hz of sampling
        which happens at 125, so an exact rate is preferred.
        """
        assert expected in str(rate_text(step))

    def test_a_descending_axis_samples_at_the_same_rate(self):
        """
        Direction is the sign of the step, not of the frequency.

        A time axis which counts down samples exactly as often as the
        one which counts up.
        """
        down = rate_text(np.timedelta64(-4, "ms"))
        up = rate_text(np.timedelta64(4, "ms"))
        assert str(down) == str(up)
        assert "250 Hz" in str(down)

    def test_a_step_finer_than_a_nanosecond(self):
        """
        A step is read at the resolution it is stored at.

        1500 ps counted in whole nanoseconds is 1 ns, which would state
        1 GHz of sampling that happens at 666.7 MHz.
        """
        assert "666.7 MHz" in str(rate_text(np.timedelta64(1500, "ps")))
        assert "1 GHz" in str(rate_text(np.timedelta64(1000, "ps")))

    def test_a_quantum_finer_than_a_nanosecond_is_not_zero(self):
        """
        The resolution a step is held at is read the same way it is.

        Counted in whole nanoseconds a picosecond quantum is zero, and
        no rate at all can land inside a tolerance of nothing.
        """
        assert _storage_quantum(np.timedelta64(1, "ps")) == pytest.approx(1e-12)

    @pytest.mark.parametrize("unit", ["M", "Y"])
    def test_a_step_held_in_months_or_years(self, unit):
        """
        Neither is a fixed number of seconds.

        So neither is a fixed number of samples per second, and asking
        numpy raises out of the middle of a repr.
        """
        assert rate_text(np.timedelta64(1, unit)) is None

    def test_a_rate_carries_no_float_noise(self):
        """
        What is printed is rounded to the figures it was chosen at.

        A day step is 11.57 µHz; the shortest exact form of the float
        behind it is 11.569999999999999.
        """
        assert "11.57 µHz" in str(rate_text(np.timedelta64(1, "D")))

    def test_a_rate_never_reads_in_exponent_notation(self):
        """250 Hz needs two figures, and `g` prints those as 2.5e+02."""
        assert "e+" not in str(rate_text(np.timedelta64(4_000_000, "ns")))

    def test_a_rate_which_would_not_give_the_step_back(self):
        """
        A step which is not a round rate says no rate at all.

        3999999 ns is 250.0000625 Hz. Saying "250 Hz" would claim a
        precision the step does not have, and saying every figure of it
        is the step again in different units.
        """
        assert rate_text(np.timedelta64(3_999_999, "ns")) is None

    @pytest.mark.parametrize(
        "step",
        [1.0, 300, "not a step", None, np.timedelta64(0, "ns"), np.timedelta64("NaT")],
    )
    def test_only_a_step_measured_in_time(self, step):
        """
        A rate is the reciprocal of a duration.

        One over a distance is not how anyone states channel spacing,
        and one over zero is not a rate at all.
        """
        assert rate_text(step) is None


class TestHumanSize:
    """Tests for saying how much room something takes up."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            # A byte count is read in binary, so a thousand of them is
            # still a thousand bytes and not a kibibyte.
            (1000, "1000 B"),
            (1023, "1023 B"),
            (1024, "1 KiB"),
            # A hair under a boundary belongs in the unit above it:
            # "1024 KiB" is the one answer "largest which fits" rules out.
            (1024**2 - 1, "1 MiB"),
            (1024**3 - 1, "1 GiB"),
            (4_800_000, "4.6 MiB"),
            (2**30, "1 GiB"),
            (2**40, "1 TiB"),
            (2**51, "2048 TiB"),
        ],
    )
    def test_size_in_the_largest_unit_which_fits(self, count, expected):
        """A byte count is read in whatever unit keeps it short."""
        assert human_size(count) == expected

    def test_an_unknown_size_draws_no_comma(self):
        """The comma introduces a size, so it goes when there is none."""

        class UnknownChunks(np.ndarray):
            """An array which cannot say how much room it takes up."""

            nbytes = float("nan")

        data = np.zeros((2, 2)).view(UnknownChunks)
        rendered = str(array_to_text(data))
        assert "float64)" in rendered
        assert ", )" not in rendered

    def test_a_size_which_is_not_a_number(self):
        """
        A dask array of unknown chunks reports nan bytes.

        "nan TiB" is a worse answer than not saying.
        """
        assert human_size(float("nan")) == ""


class TestDurationText:
    """Tests for how long an extent lasted."""

    def test_two_instants_state_their_distance(self):
        """The fact two times do not carry on their own."""
        low = np.datetime64("2020-01-01T00:00:00")
        high = np.datetime64("2020-01-01T00:00:24")
        assert "24 s" in str(duration_text(low, high))

    @pytest.mark.parametrize("unit", ["Y", "M"])
    def test_a_span_of_years_or_months(self, unit):
        """
        Neither is a fixed number of seconds.

        numpy refuses to divide one into seconds rather than guessing a
        calendar, and so does this rather than raising out of a repr.
        """
        low, high = np.datetime64("2000", unit), np.datetime64("2010", unit)
        assert duration_text(low, high) is None

    def test_no_time_at_all_says_nothing(self):
        """A zero would read as a label on a gap."""
        instant = np.datetime64("2020-01-01T00:00:00")
        assert duration_text(instant, instant) is None

    @pytest.mark.parametrize(
        ("low", "high"),
        [
            (0.0, 299.0),
            (0, 299),
            ("a", "b"),
            (None, None),
            (1, np.datetime64("2020-01-01")),
            # The other way round: `low` is a time and `high` is not, so
            # only the second half of the guard can refuse it. Without
            # that half numpy raises out of the middle of a repr.
            (np.datetime64("2020-01-01"), 1),
        ],
        ids=["floats", "ints", "strings", "none", "mixed_low", "mixed_high"],
    )
    def test_only_two_times_have_a_duration(self, low, high):
        """
        A duration is read in seconds.

        A distance of 299 handed to this would come back as "5 m" --
        five minutes, of a span measured in metres.
        """
        assert duration_text(low, high) is None

    def test_further_apart_than_a_timedelta_holds(self):
        """
        Two instants can lie further apart than a Timedelta holds.

        How long that is matters less than the extents it would
        otherwise take down with it.
        """
        assert duration_text(pd.Timestamp.min, pd.Timestamp.max) is None

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            ("1677-09-21T00:12:44", "2262-04-11T23:47:16", "584.6 y"),
            ("0001-01-01", "9999-12-31", "9999 y"),
        ],
        ids=["centuries", "millennia"],
    )
    def test_a_span_too_long_to_count_in_nanoseconds(self, low, high, expected):
        """
        A long span is still said, and said correctly.

        Read as nanoseconds these overflow an int64 silently: the first
        of them came back as 1.7 seconds, and the second as 61.6 years.
        """
        said = duration_text(np.datetime64(low), np.datetime64(high))
        assert expected in str(said)


class TestValuesAReaderCanRead:
    """Tests for the facts a repr states about the things it shows."""

    def test_a_time_coord_states_its_span(self):
        """Two instants say nothing about how far apart they are."""
        coord = dc.get_example_patch().coords.coord_map["time"]
        assert "<8 s>" in str(coord)

    def test_a_time_coord_states_its_rate(self):
        """DAS acquisition is quoted in Hz, not in seconds per sample."""
        coord = dc.get_example_patch().coords.coord_map["time"]
        assert "250 Hz" in str(coord)

    def test_a_distance_coord_states_neither(self):
        """
        A distance from 0 to 299 m already says how wide it is, and a
        rate over it is not a quantity anyone quotes.
        """
        rendered = str(dc.get_example_patch().coords.coord_map["distance"])
        assert "Hz" not in rendered
        assert "<" not in rendered

    def test_the_data_states_how_much_room_it_takes(self):
        """Whether it fits in memory is what dtype times shape is for."""
        assert "4.6 MiB" in str(dc.get_example_patch())

    def test_an_annotation_set_over_distance_states_no_span(self):
        """
        A distance annotation is not measured in time.

        299 metres read as a duration is "5 m", which is five minutes.
        """
        frame = pd.DataFrame({"distance_min": [0.0], "distance_max": [299.0]})
        rendered = str(AnnotationSet(frame, dims=("distance",)))
        assert "299" in rendered
        assert "<" not in rendered

    def test_an_annotation_set_states_its_span(self):
        """The same fact a spool and a patch coordinate state."""
        frame = pd.DataFrame(
            {
                "time_min": pd.to_datetime(["2020-01-01T00:00:00"]),
                "time_max": pd.to_datetime(["2020-01-01T00:00:09"]),
            }
        )
        assert "<9 s>" in str(AnnotationSet(frame, dims=("time",)))


class TestStyleClasses:
    """Tests for the classes a resolved rich style is drawn with."""

    @pytest.mark.parametrize(
        ("style", "expected"),
        [
            ("blue", ("dc-blue",)),
            ("bold green", ("dc-bold", "dc-green")),
            ("bright_blue", ("dc-bright_blue",)),
            ("bold dark_orange", ("dc-bold", "dc-dark_orange")),
            ("underline", ("dc-underline",)),
            ("", ()),
        ],
    )
    def test_the_words_of_a_style(self, style, expected):
        """Each word a class exists for becomes one."""
        assert _style_classes(Style.parse(style)) == expected

    def test_a_background_is_not_a_foreground(self):
        """
        "red on blue" paints blue behind red text.

        Rich resolves that; reading the words would take the background
        for the color of the text.
        """
        assert _style_classes(Style.parse("red on blue")) == ("dc-red",)

    def test_a_style_which_turns_something_off(self):
        """Rich resolves "not bold" to not bold."""
        assert _style_classes(Style.parse("not bold")) == ()

    @pytest.mark.parametrize("color", ["#ff0000", "purple4", "rgb(1,2,3)"])
    def test_a_color_no_class_exists_for(self, color):
        """
        Drawn as the host's own ink rather than drawn wrong.

        Nothing an object states reaches a CSS attribute this way, so
        the stylesheet stays the only thing which says what a color is.
        """
        assert _style_classes(Style.parse(color)) == ()


class TestTextToHtml:
    """Tests for rendering a rich Text as an HTML fragment."""

    @pytest.mark.parametrize(
        ("plain", "expected"),
        [("a & b", "a &amp; b"), ("<script>", "&lt;script&gt;"), ("x > y", "x &gt; y")],
    )
    def test_content_is_escaped(self, plain, expected):
        """
        A tag, a unit or a path is a value someone else chose.

        A trusted notebook does not sanitize what a repr emits, so what
        the repr emits has to be safe on its own.
        """
        assert _text_to_html(Text(plain)) == expected

    def test_quotes_are_left_alone(self):
        """Nothing is written into an attribute, so nothing needs it."""
        assert _text_to_html(Text('say "hi"')) == 'say "hi"'

    def test_unstyled_text_emits_no_span(self):
        """A span which says nothing is bytes in every repr forever."""
        assert "<span" not in _text_to_html(Text("plain"))

    def test_a_later_span_wins_a_color(self):
        """
        Two spans can both state a color, and rich says which wins.

        Stacking both classes and letting the stylesheet's source order
        pick made the sampling rate grey in a panel and blue in a
        terminal.
        """
        text = Text("250 Hz")
        text.stylize("grey50", 0, 6)
        text.stylize("bright_blue", 0, 6)
        assert 'class="dc-bright_blue"' in _text_to_html(text)
        assert "dc-grey50" not in _text_to_html(text)

    def test_a_repr_with_many_spans_is_not_slow(self):
        """
        A style is added to the runs it covers, not asked about each.

        The other way round a few thousand spans -- a patch of many
        coordinates -- took over a second to draw.
        """
        text = Text("")
        for index in range(2000):
            part = Text(f"field{index}: value{index} ")
            part.stylize("grey50", 0, 7)
            part.stylize("bright_blue", 8, 14)
            text += part
        start = time.perf_counter()
        _text_to_html(text)
        assert time.perf_counter() - start < 1.0

    def test_a_style_reaches_only_the_runs_it_covers(self):
        """
        A span states a style for its own characters and no others.

        Applied to every run instead, the banner smears: DAS, C and ore
        each take all three colors, and every value bleeds across its
        line.
        """
        text = Text("abcdef")
        text.stylize("bold", 0, 2)
        text.stylize("blue", 4, 6)
        html = _text_to_html(text)
        assert '<span class="dc-bold">ab</span>' in html
        assert "cd" in html.replace('<span class="dc-bold">ab</span>', "")
        assert '<span class="dc-blue">ef</span>' in html
        assert "dc-blue" not in html.split("cd")[0]

    def test_overlapping_spans_compose(self):
        """
        A styled value inside a styled field keeps both.

        `get_nice_text` stylizes a Text which already carries spans, so
        this is how a date inside a coordinate line is drawn.
        """
        text = Text("abcdef")
        text.stylize("bold", 0, 4)
        text.stylize("blue", 2, 6)
        html = _text_to_html(text)
        assert 'class="dc-bold dc-blue"' in html

    def test_an_empty_text(self):
        """An object may state a block with nothing in it."""
        assert _text_to_html(Text("")) == ""


class TestRenderHtml:
    """Tests for rendering repr nodes as HTML."""

    def test_a_section_folds(self, repr_blocks):
        """A block with a body is what a reader opens and closes."""
        html = _render_html(split_block(repr_blocks["coords"]))
        assert html.startswith("<details")
        assert "<summary>" in html

    def test_a_section_with_no_body_does_not(self, repr_blocks):
        """One line is a statement; offering to fold it says otherwise."""
        html = _render_html(split_block(repr_blocks["one_line"]))
        assert "<details" not in html
        assert 'class="dc-line"' in html

    def test_a_long_section_starts_closed(self, repr_blocks):
        """An array is not read at a glance, so it does not open at one."""
        with config_context(display_html_open_lines=0):
            assert "<details>" in _render_html(split_block(repr_blocks["coords"]))

    def test_a_short_section_starts_open(self, repr_blocks):
        """
        What a reader would have opened anyway is opened for them.

        The coordinates block is two lines; the limit is what says two
        is few enough, so it is asked for rather than assumed.
        """
        with config_context(display_html_open_lines=2):
            assert "<details open>" in _render_html(split_block(repr_blocks["coords"]))

    @pytest.mark.parametrize("block", ["coords", "attrs"])
    def test_the_limit_counts_body_lines(self, repr_blocks, block):
        """
        One line either side of the limit decides differently.

        A section is folded by how much it holds, so the count has to be
        of what is drawn rather than of what was handed over. Both
        blocks draw two lines; `attrs` is the one which also ends on a
        newline, and counting that as a third folded it a line early.
        """
        section = split_block(repr_blocks[block])
        with config_context(display_html_open_lines=1):
            assert "<details>" in _render_html(section)
        with config_context(display_html_open_lines=2):
            assert "<details open>" in _render_html(section)

    def test_the_count_is_the_lines_a_reader_sees(self, repr_blocks):
        """The limit means what a reader would count, not what a Text holds."""
        for name, block in repr_blocks.items():
            section = split_block(block)
            if not section.body:
                continue
            html = _render_html(section)
            body = re.search(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL).group(1)
            drawn = re.sub(r"<[^>]+>", "", body).count("\n") + 1
            with config_context(display_html_open_lines=drawn):
                assert "<details open>" in _render_html(section), name
            with config_context(display_html_open_lines=drawn - 1):
                assert "<details open>" not in _render_html(section), name

    def test_the_banner_drops_its_underline(self):
        """
        A terminal underlines the banner with dashes, drawn in columns.

        Here that is a border, and an emoji is not two columns wide in
        every font a browser might choose.
        """
        node = Repr(get_header_text("Patch ⚡"))
        assert "---" not in _render_html(node)

    def test_the_fragment_is_scoped(self):
        """
        Everything the stylesheet says is said about this class.

        A repr is emitted into a notebook output, where a `style`
        applies to the whole document around it.
        """
        assert _render_html(Repr(Text("x"))).startswith('<div class="dc-repr">')

    def test_something_which_is_not_a_node(self):
        """A renderer says what it cannot draw rather than drawing it wrong."""
        with pytest.raises(NotImplementedError, match="cannot render int"):
            _render_html(42)


class TestLimitItems:
    """Tests for taking only as much of a container as a repr shows."""

    def test_everything_fits(self):
        """Nothing is left behind while everything fits."""
        assert _limit_items([1, 2, 3], limit=3) == ([1, 2, 3], 0)

    def test_the_tail_is_counted(self):
        """What is not shown is counted rather than dropped silently."""
        assert _limit_items(range(10), limit=4) == ([0, 1, 2, 3], 6)

    def test_nothing_is_shown(self):
        """A limit of none still says how much there was."""
        assert _limit_items([1, 2], limit=0) == ([], 2)

    def test_the_limit_comes_from_config(self):
        """The default cap is a runtime setting, not a constant here."""
        with config_context(display_max_items=1):
            assert _limit_items("abc") == (["a"], 2)


class TestSectionTitle:
    """Tests for how a terminal sets a nested block's line in."""

    def test_the_top_is_drawn_as_it_stands(self):
        """A tree's root opens where its container left off."""
        assert _section_title(Text("a"), 0).plain == "a"

    def test_the_top_is_not_the_line_it_was_given(self):
        """A renderer appends to what it drew, and must not rewrite it."""
        line = Text("a")
        _section_title(line, 0).append("b")
        assert line.plain == "a"

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_a_level_starts_a_line(self, depth):
        """Every level below the top begins on its own line, set in."""
        assert _section_title(Text("a"), depth).plain == "\n" + "    " * depth + "a"

    @pytest.mark.parametrize("depth", [1, 2])
    def test_every_line_is_set_in(self, depth):
        """
        A field value may hold a newline -- a description usually does.

        A continuation left at column zero reads as a block of its own,
        which is the misreading the indentation exists to prevent.
        """
        indent = "    " * depth
        out = _section_title(Text("a\nb"), depth).plain
        assert out == f"\n{indent}a\n{indent}b"

    def test_the_styles_survive(self):
        """
        Setting a line in does not repaint it, or paint the indent.

        Read per character, since indenting splits a Text and rejoins
        it, and a span which came back one character wide of where it
        started would color the wrong word.
        """
        line = Text("ab")
        line.stylize("bold", 0, 1)
        drawn = _section_title(line, 1)
        styles = dict(zip(drawn.plain, resolved_styles(drawn), strict=True))
        assert styles["a"].bold
        assert not styles["b"].bold
        assert not styles[" "].bold


class _Leaf:
    """A model-like object which holds nothing."""

    def __init__(self, name):
        self.name = name

    def _repr_section(self, depth=0):
        return Section(Text(self.name), depth=depth)


class TestChildSections:
    """Tests for the blocks a container's children draw."""

    def test_one_block_per_child(self):
        """Each child draws itself, at the depth it was given."""
        out = child_sections([_Leaf("a"), _Leaf("b")], 2)
        assert [x.title.plain for x in out] == ["a", "b"]
        assert {x.depth for x in out} == {2}

    def test_the_tail_is_named(self):
        """A container which stops early says how much it stopped short of."""
        with config_context(display_max_items=2):
            out = child_sections([_Leaf(str(x)) for x in range(5)], 1)
        assert len(out) == 3
        assert out[-1].title.plain == "... 3 more"
        # The line sits with the children it stands for, not beside them.
        assert out[-1].depth == 1

    def test_a_hidden_child_is_not_drawn(self):
        """Cost is what a repr prints, not what the tree holds."""
        drawn = []

        class Counted(_Leaf):
            def _repr_section(self, depth=0):
                drawn.append(self.name)
                return super()._repr_section(depth)

        with config_context(display_max_items=2):
            child_sections([Counted(str(x)) for x in range(50)], 1)
        assert drawn == ["0", "1"]

    def test_no_children(self):
        """A container which holds nothing states nothing."""
        assert child_sections((), 1) == ()


class TestVisibleLines:
    """Tests for how many lines a node is counted as showing."""

    def test_a_raw_counts_what_is_drawn(self):
        """The newline which separated a body from its title is not a line."""
        assert _visible_lines(Raw(Text("\na\nb\n"))) == 2

    def test_an_empty_raw_draws_nothing(self):
        """A body which is only the separator is no body at all."""
        assert _visible_lines(Raw(Text("\n"))) == 0

    def test_a_table_counts_its_heading(self):
        """Its heading row is a line a reader sees."""
        row = Row(Text("x"), Text("k"), (("a", Text("1"), True),))
        assert _visible_lines(Table((row, row))) == 3

    def test_an_empty_table_draws_nothing(self):
        """A table of no records has no heading to draw either."""
        assert _visible_lines(Table()) == 0

    def test_an_open_section_counts_what_it_holds(self):
        """A parent counts everything below it while it is open."""
        leaf = Section(Text("leaf"), depth=2)
        mid = Section(Text("mid"), (leaf, leaf), depth=1)
        with config_context(display_html_open_lines=12):
            assert _visible_lines(mid) == 3
            assert _visible_lines(Section(Text("top"), (mid,))) == 4

    def test_a_folded_section_is_one_line(self):
        """
        What is folded is one line, whatever it holds.

        That is what lets a reader open a large tree a level at a time:
        counted in full, a container of twenty full networks would fold
        the block which lists them, and the panel would open on nothing.
        """
        leaf = Section(Text("leaf"), depth=2)
        mid = Section(Text("mid"), (leaf,) * 6, depth=1)
        top = Section(Text("top"), (mid,) * 3)
        with config_context(display_html_open_lines=3):
            assert _visible_lines(mid) == 1
            assert _visible_lines(top) == 4
            # And so the outer block still opens, on three folded lines.
            # Counted on " open>", since a nested block carries classes
            # between the tag and the attribute and "<details open>"
            # could only ever match the one at the top.
            html = _render_html(top)
            assert html.count(" open>") == 1
            assert html.startswith("<details open>")

    def test_a_title_which_runs_on_counts_every_line(self):
        """
        A folded block still draws all of its title.

        A value may hold a newline, and both a `summary` and a
        `.dc-line` keep it, so seven two-line blocks draw fourteen
        lines. Counted as seven, the block holding them opens under a
        limit it is really twice the size of.
        """
        leaf = Section(Text("first\nsecond"), depth=1)
        assert _visible_lines(leaf) == 2
        top = Section(Text("top"), (leaf,) * 7)
        assert _body_lines(top) == 14
        with config_context(display_html_open_lines=12):
            assert " open>" not in _render_html(top)
        with config_context(display_html_open_lines=14):
            assert " open>" in _render_html(top)

    def test_a_node_is_counted_once(self, monkeypatch):
        """
        Every node is visited once, however deep the tree goes.

        Asking a section how long it is and then asking again after
        deciding to open it walks the same subtree twice at every level,
        which doubles per level rather than adding one. Counted rather
        than timed: the growth is the defect, and a clock only shows it
        on a tree far larger than one anybody has.
        """
        seen = []
        real = display._visible_lines

        def counted(node):
            seen.append(node)
            return real(node)

        monkeypatch.setattr(display, "_visible_lines", counted)
        node = Section(Text("leaf"), depth=12)
        for depth in range(11, -1, -1):
            node = Section(Text(f"n{depth}"), (node,), depth=depth)
        with config_context(display_html_open_lines=100):
            counted(node)
        assert len(seen) == 13

    def test_something_which_is_not_a_node(self):
        """A counter says what it cannot count rather than guessing."""
        with pytest.raises(NotImplementedError, match="cannot count int"):
            _visible_lines(42)


class TestNestedSections:
    """Tests for a block drawn inside another block."""

    @pytest.fixture
    def tree(self):
        """A title, a child, and a grandchild under it."""
        leaf = Section(Text("leaf"), depth=2)
        mid = Section(Text("mid"), (leaf,), depth=1)
        return Section(Text("\u27a4 top"), (mid,))

    def test_text_is_concatenation(self, tree):
        """
        A terminal draws the nesting by the indentation in the titles.

        That is what keeps `str()` unchanged: the nodes state exactly
        the characters the old hand-built text stated.
        """
        assert render_text(tree).plain == "\u27a4 top\n    mid\n        leaf"

    def test_the_panel_nests(self, tree):
        """
        A child block is drawn inside its parent, not after it.

        Checked by where the parent closes: a child which came after
        would sit past the `</details>` which ends it, and the tree
        would read as three blocks side by side rather than one.
        """
        html = _render_html(tree)
        assert html.count("<details") == 2
        opened = html.index("<summary>top</summary>") + len("<summary>top</summary>")
        closed = html.index("</details>")
        assert opened < html.index("<summary>mid</summary>") < closed
        assert opened < html.index(">leaf<") < closed

    def test_a_nested_title_drops_its_indentation(self, tree):
        """
        The indentation is the terminal's way of showing nesting.

        A panel nests the markup instead, and `summary` keeps
        whitespace, so a title drawn as handed over would open on a
        blank line.
        """
        html = _render_html(tree)
        for title in re.findall(r"<summary>(.*?)</summary>", html, re.DOTALL):
            assert title == title.lstrip()
        assert '<div class="dc-line dc-nest dc-d1">leaf</div>' in html

    def test_a_level_is_said_in_the_markup(self, tree):
        """
        Which level a line belongs to is a class, not a color here.

        A host which drops the stylesheet still nests, so this says only
        that the depth reaches the markup at all.
        """
        html = _render_html(tree)
        assert 'class="dc-nest dc-d0"' in html
        assert "dc-d1" in html

    def test_the_top_is_in_no_nesting(self):
        """A block at the top of a repr sits inside nothing."""
        html = _render_html(Section(Text("top"), (Raw(Text("\nbody")),)))
        assert "dc-nest" not in html

    def test_the_ramp_wraps(self):
        """
        Past a few levels it is the rails which tell them apart.

        Checked at the wrap rather than at the first level, since a ramp
        which ran off the end would draw an undefined color there.
        """
        node = Section(Text("x"), depth=_NEST_COLORS + 1)
        assert f"dc-d{(_NEST_COLORS + 1 - 1) % _NEST_COLORS}" in _render_html(node)

    @pytest.mark.parametrize(("limit", "open_blocks"), [(2, 2), (1, 1), (0, 0)])
    def test_a_deep_tree_folds_from_the_outside_in(self, tree, limit, open_blocks):
        """
        Each level decides for itself, and an outer one counts a folded
        child as the one line it draws.

        So the block which holds the tree stays open while the levels
        under it close, and a reader opens as far down as they asked.
        """
        with config_context(display_html_open_lines=limit):
            assert _render_html(tree).count(" open>") == open_blocks


class TestBodyText:
    """Tests for the whitespace which frames a section body."""

    def test_a_body_which_is_only_the_separator(self):
        """
        A block whose body is the newline which split it has no body.

        Offering to fold nothing draws a triangle over an empty box.
        """
        html = _render_html(split_block(Text("title\n")))
        assert "<details" not in html
        assert 'class="dc-line"' in html

    def test_a_trailing_newline_draws_no_blank_line(self):
        """
        `attrs_to_text` ends on a newline so a printed patch does.

        In a panel that is a blank line inside the block instead.
        """
        html = dc.get_example_patch()._repr_html_()
        assert "\n</pre>" not in html

    def test_a_body_keeps_the_lines_between(self):
        """Only the framing goes; a blank line inside the body stays."""
        html = _render_html(split_block(Text("title\na\n\nb\n")))
        assert ">a\n\nb<" in html


class TestCoordinatesAreStated:
    """
    Tests for what the coordinates block says, rather than for the two
    renderings of it agreeing.

    Parity is a relative claim: it holds just as well when both reprs
    say nothing. Deleting every coordinate from the block left it
    passing, so these say what has to be there.
    """

    @pytest.fixture(scope="class")
    def patch(self):
        """A patch with a dimension coordinate and one riding on it."""
        return dc.get_example_patch().update_coords(quality=("distance", np.ones(300)))

    def test_the_text_states_every_coordinate(self, patch):
        """A block which names none of them is not a coordinates block."""
        rendered = str(patch.coords)
        for name in ("distance", "time", "quality"):
            assert name in rendered, name

    def test_the_panel_states_every_coordinate(self, patch):
        """The same, drawn."""
        html = patch._repr_html_()
        for name in ("distance", "time", "quality"):
            assert name in html, name

    def test_a_dimension_is_marked(self, patch):
        """
        The `*` is how a reader tells a dimension from a coordinate
        which merely rides on one.
        """
        assert "*distance" in str(patch.coords)
        assert "*<span" in patch._repr_html_()

    def test_a_coordinate_which_is_not_a_dimension_states_its_dims(self, patch):
        """Which axis it lies along is the thing it has to say."""
        assert "quality ('distance',)" in str(patch.coords)

    def test_a_private_coordinate_is_not_shown(self):
        """
        A name starting with an underscore is the manager's business.

        Left in, every patch would print the coordinates it keeps for
        bookkeeping beside the ones a reader asked for.
        """
        patch = dc.get_example_patch()
        coords = patch.coords.update(_hidden=("distance", np.ones(300)))
        assert "_hidden" not in str(coords)
        assert "_hidden" not in _render_html(coords._repr_section())

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("min", "0"),
            ("max", "299"),
            ("step", "1"),
            ("shape", "(300,)"),
            # Asked of the coordinate rather than stated: an integer is
            # 32 bits where the suite runs in WebAssembly and 64 here.
            ("dtype", None),
            ("units", "m"),
        ],
    )
    def test_the_facts_a_coordinate_states(self, patch, label, value):
        """
        Both the label and the value, in both renderings.

        Stripped from both sides by the parity check -- it compares what
        is said, not what it is called -- so renaming a label passed.
        """
        if value is None:
            value = str(patch.coords.coord_map["distance"].dtype)
        assert f"{label}: " in str(patch.coords)
        assert f"<th>{label}</th>" in patch._repr_html_()
        assert value in str(patch.coords)

    def test_the_kind_of_each_coordinate(self, patch):
        """A terminal states it in front of the fields; a panel columns it."""
        assert "CoordRange(" in str(patch.coords)
        assert "CoordRange" in patch._repr_html_()

    def test_a_name_which_looks_like_markup(self):
        """
        A coordinate name is a value someone else chose, and it is the
        one cell of the new markup a file can fill.
        """
        patch = dc.get_example_patch().update_coords(
            **{"<script>": ("distance", np.ones(300))}
        )
        html = patch._repr_html_()
        assert "<script>" not in html.replace("<script>alert", "X")
        assert "&lt;script&gt;" in html


class TestTableColumns:
    """Tests for which column a value is drawn in."""

    @staticmethod
    def _columns(rows):
        """The headings a table of these rows draws."""
        made = tuple(
            Row(Text(name), Text("K"), tuple((x, Text(x), True) for x in fields))
            for name, fields in rows
        )
        return re.findall(r"<th>(\w+)</th>", _render_html(Table(made)))

    def test_rows_which_state_the_same_fields(self):
        """The order one row states them in is the order they are in."""
        assert self._columns([("a", "xy"), ("b", "xy")]) == ["kind", "x", "y"]

    def test_a_field_only_one_row_states(self):
        """
        It belongs where the row stating it puts it.

        Only a time coordinate has a span, and it states it between its
        max and its step rather than after everything a distance says.
        """
        assert self._columns([("a", "xz"), ("b", "xyz")]) == ["kind", "x", "y", "z"]

    def test_a_row_which_states_part_of_what_a_later_row_does(self):
        """
        A record stating a subset which is not a prefix.

        Selecting nothing leaves a coordinate with only a shape and a
        dtype, and merging by index put the units of the row after it
        between them -- an order no row states.
        """
        assert self._columns([("a", "cd"), ("b", "abcde")]) == [
            "kind",
            "a",
            "b",
            "c",
            "d",
            "e",
        ]

    def test_rows_which_state_fields_in_conflicting_orders(self):
        """
        Two records can disagree outright, which has no answer.

        The order they were first stated in is taken, rather than
        raising out of the middle of a repr.
        """
        assert self._columns([("a", "ab"), ("b", "ba")]) == ["kind", "a", "b"]

    def test_rows_which_share_no_fields(self):
        """
        A record sharing nothing with the ones before it adds its fields
        after theirs, not in front of them.
        """
        assert self._columns([("a", "ab"), ("b", "cd")]) == ["kind", "a", "b", "c", "d"]

    def test_a_table_scrolls_inside_its_own_wrapper(self):
        """
        `overflow` does nothing on a `display: table`, so without a
        wrapper a wide coordinates block scrolls the whole panel.

        Asserted on the markup rather than on the panel, since the
        stylesheet names the class too and a search of the whole panel
        finds it whether or not anything is wrapped in it.
        """
        markup = dc.get_example_patch()._repr_html_().split("</style>", 1)[1]
        assert '<div class="dc-scroll"><table' in markup

    def test_the_heading_counts_as_a_line(self):
        """
        A table of two records draws three lines.

        The limit is what a reader would count, and they count the
        headings.
        """
        section = dc.get_example_patch().coords._repr_section()
        with config_context(display_html_open_lines=2):
            assert "<details open>" not in _render_html(section)
        with config_context(display_html_open_lines=3):
            assert "<details open>" in _render_html(section)

    def test_a_value_is_drawn_in_its_own_column(self):
        """
        Every row states a cell for every column, so a row which says
        nothing for one leaves it empty rather than shifting the rest.
        """
        rows = (
            Row(Text("first"), Text("K"), (("x", Text("1"), True),)),
            Row(
                Text("second"),
                Text("K"),
                (("x", Text("2"), True), ("y", Text("3"), True)),
            ),
        )
        html = _render_html(Table(rows))
        heads = re.findall(r"<th>(\w+)</th>", html)
        for row in re.findall(r"<tr><th scope=\"row\">.*?</tr>", html, re.DOTALL):
            assert len(re.findall(r"<td[^>]*>", row)) == len(heads)
        body = re.search(r"<tbody>(.*)</tbody>", html, re.DOTALL).group(1)
        assert "<td>1</td><td></td>" in body


def _css_body() -> str:
    """The stylesheet as it ships, which is to say with no comments in it.

    A rule and a comment about a rule read the same to a substring
    search, and a test which cannot tell them apart passes on a rule
    which was commented out. Nothing is stripped here because what
    ships is already stripped; ``test_the_sheet_ships_without_its_prose``
    is what holds that true.
    """
    return _get_stylesheet()


def _css_rule(selector: str) -> str:
    """
    What one selector declares, or "" where it declares nothing.

    Matched on the whole selector list a block states, so a rule found
    here is one a browser would apply rather than a substring of the
    name of another.
    """
    for block, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _css_body()):
        if any(x.strip() == selector for x in block.split(",")):
            return body
    return ""


class TestStylesheet:
    """Tests for the CSS every repr carries."""

    def test_every_selector_is_scoped(self):
        """
        A bare `pre` rule here restyles every code block on the page.

        The stylesheet travels inside a notebook output cell, where it
        applies to the whole document around it.
        """
        body = _css_body()
        selectors = [
            part.strip()
            for block in re.findall(r"([^{}]+)\{", body)
            for part in block.split(",")
        ]
        stated = [x for x in selectors if x and not x.startswith("@")]
        assert stated
        assert all(".dc-repr" in x for x in stated)

    def test_it_states_no_rule_which_reaches_outside(self):
        """
        `@font-face` and `@import` are scoped to nothing at all.

        Neither can be written under a class, so a stylesheet which
        travels inside someone else's document must not state one.
        """
        body = _css_body()
        for rule in ("@font-face", "@import", "@page"):
            assert rule not in body, rule

    def test_a_class_exists_for_every_style_a_repr_states(self):
        """
        A word which draws in a terminal and not in a browser is a
        difference between the two reprs that nobody chose.

        Walked from what the objects actually state, which is the
        direction that bites: a word dropped from the list stops
        coloring most of a panel and no test of the list notices.
        """
        console = Console()
        seen = set()
        for obj in (
            dc.get_example_patch(),
            dc.get_example_spool("diverse_das"),
            dc.get_example_inventory("tunnel"),
        ):
            texts = [obj._repr_node().header, *_node_texts(obj._repr_node().body)]
            for text in texts:
                for offset in range(len(text.plain)):
                    style = text.get_style_at_offset(console, offset)
                    if style.color or style.bold or style.underline:
                        seen.add(style)
        assert seen
        for style in seen:
            assert _style_classes(style), style

    def test_a_rule_exists_for_every_nesting_level(self):
        """A level the renderer can reach and the CSS cannot draws no rail."""
        for depth in range(_NEST_COLORS):
            rule = _css_rule(f".dc-repr .dc-d{depth}")
            assert f"--dc-rail: var(--dc-d{depth})" in rule, depth

    def test_a_nested_leaf_is_given_a_marker_of_room(self):
        """
        A leaf has no disclosure triangle, and is given the width of one.

        Without it a leaf's text starts where its parent's text starts,
        a marker to the left of the blocks beside it, and the tree reads
        as though the leaf sat a level higher than it does.
        """
        assert "padding-left" in _css_rule(".dc-repr .dc-line.dc-nest")

    def test_a_line_which_runs_on_stays_in_its_column(self):
        """
        A title can hold a newline, and both its lines start together.

        The triangle is pulled back into a gutter the whole title is
        set in by, rather than taking room from its first line only:
        given the latter a continuation starts under the triangle, a
        marker to the left of the line it continues.
        """
        assert "-1.1em" in _css_rule(".dc-repr summary::before")
        assert "0.1em 0 0.1em 1.1em" in _css_rule(".dc-repr summary")

    def test_the_sheet_ships_without_its_prose(self):
        """
        The comments explain the sheet to whoever edits it, not to a reader.

        They are near half the file by weight, and a notebook carries
        the whole sheet once per cell, so the copy which goes out has
        them taken off.
        """
        source = files("dascore").joinpath("repr.css").read_text(encoding="utf-8")
        # A control: what is stripped is something the file really holds.
        assert "/*" in source
        assert "/*" not in _get_stylesheet()
        # And nothing else went with them.
        assert ".dc-repr .dc-banner" in _get_stylesheet()
        assert len(_get_stylesheet()) < len(source) / 1.5

    def test_a_value_which_reads_like_a_comment_is_kept(self):
        """
        `content` takes a string, and a string may hold a comment marker.

        The stripper is told about quotes for that reason: reading
        `"/*"` as the start of a comment would swallow every declaration
        up to the next `*/`, which is a stylesheet that no longer
        parses rather than one which lost a note. And about escapes,
        since an escaped quote inside a string would otherwise end it
        early and leave the rest of the value read as CSS.
        """
        for value in ('"/*"', '"\\"/*"'):
            css = f"a {{ content: {value}; }}\n/* gone */\nb {{ color: red }}\n"
            stripped = _strip_css_comments(css)
            assert f"content: {value}" in stripped
            assert "b { color: red }" in stripped
            assert "gone" not in stripped

    def test_a_row_is_not_striped_by_the_host(self):
        """
        A host may restyle a table it did not write.

        Quarto hands a cell's output through pandoc, which re-emits the
        coordinates table with bootstrap's `table-striped` on it, and
        bootstrap stripes a row with an inset shadow rather than a
        background -- so clearing the background is not enough, and
        every other row of the docs site's panel comes out grey.
        """
        assert "box-shadow: none" in _css_rule(".dc-repr .dc-table td")

    def test_a_wide_line_scrolls_inside_its_own_block(self):
        """
        A container's line sits in a `summary` now, not in the array dump.

        Without an overflow of its own a long one pushes the whole
        panel sideways, so the banner and every block below it scroll
        along with it.
        """
        assert "overflow-x: auto" in _css_rule(".dc-repr summary")

    def test_every_theme_states_the_whole_ramp(self):
        """
        A token defined in one theme and not another draws nothing there.

        The rails are decoration, so what is lost is not information --
        but a rail which is there on a light page and gone on a dark one
        is the kind of difference nobody chose.
        """
        css = _css_body()
        # A block which defines any of the palette defines all of it:
        # those are the theme blocks, whatever host they are written for.
        blocks = re.findall(r"\{([^{}]*--dc-blue:[^{}]*)\}", css)
        assert len(blocks) >= 3  # the default, the reader's OS, the host
        for block in blocks:
            for depth in range(_NEST_COLORS):
                assert f"--dc-d{depth}:" in block, (depth, block[:40])

    def test_a_class_exists_for_every_style_word(self):
        """
        A word which draws in a terminal and not in a browser is a
        difference between the two reprs that nobody chose.
        """
        css = _get_stylesheet()
        for word in _STYLE_WORDS:
            assert f".dc-{word}" in css, word

    def test_a_title_draws_one_marker(self):
        """
        A terminal opens a title with an arrow because it has no
        triangle to draw; a panel draws the triangle, and both is two
        markers for one thing.
        """
        html = dc.get_example_patch()._repr_html_()
        assert "\u27a4" not in html
        assert "Coordinates" in html

    def test_a_line_and_a_summary_keep_their_spaces(self):
        """
        A producer's indent is content: the spool states its path
        indented, and HTML collapses runs of spaces by default.
        """
        css = _get_stylesheet()
        for selector in (".dc-repr .dc-line", ".dc-repr summary"):
            block = css[css.index(selector) : css.index("}", css.index(selector))]
            assert "white-space: pre" in block, selector

    def test_a_host_which_states_a_light_theme(self):
        """
        Every host which can say it is dark can say it is light.

        One named in the dark rule and not the light one keeps dark ink
        on a light page whenever the reader's system is dark.
        """
        css = _get_stylesheet()
        dark = css[css.index('data-jp-theme-light="false"') :]
        dark = dark[: dark.index("}")]
        light = css[css.index('data-jp-theme-light="true"') :]
        light = light[: light.index("}")]
        for host in ("vscode-theme-kind", "quarto-"):
            assert host in dark and host in light, host

    def test_it_is_read_once(self):
        """Every repr carries it, so every repr should not read it."""
        assert _get_stylesheet() is _get_stylesheet()


def _decompose(line: str) -> list[str]:
    """
    A stated line as the values in it, without the framing.

    Labels, class names and punctuation are how a printed line says
    which value is which; a table says that with a column heading, so
    both sides are read this way and compared on what they state.
    """
    said = line.strip().removeprefix("\u27a4 ")
    if not said or set(said) == {"-"}:
        return []
    # The name may hold a space: a coordinate which is not a dimension
    # states the dimensions it rides on, as `quality ('distance',)`.
    record = re.match(r"(.+?): (\w+)\((.*) \)$", said)
    if record is None:
        return [said]
    name, kind, fields = record.groups()
    out = [name, kind]
    for value in re.split(r"\s\w+: ", " " + fields):
        value = value.strip()
        if not value:
            continue
        # A span states itself in brackets rather than by a label, so a
        # label does not split it off; a column heading says which
        # column it is in and the panel gives it a cell of its own.
        span = re.search(r"\s(<[^>]*>)$", value)
        if span:
            out.extend([value[: span.start()].strip(), span.group(1)])
        else:
            out.append(value)
    return out


def _node_texts(nodes) -> list[Text]:
    """
    Every Text a tree of repr nodes states, top down.

    Recursive because sections nest: an inventory's networks hold fiber
    arrays which hold acquisitions, and a walk which stopped at the top
    would check the one level nothing much is drawn on.
    """
    out: list[Text] = []
    for node in nodes:
        if isinstance(node, Table):
            for row in node.rows:
                out.extend([row.name, row.kind])
                out.extend(v for _, v, _ in row.fields)
        elif isinstance(node, Section):
            out.append(node.title)
            out.extend(_node_texts(node.body))
        else:
            out.append(node.text)
    return out


def _drawn_values(html: str) -> list[str]:
    """
    What a reader sees in a panel, in the order it is drawn.

    Taken block by block rather than by stripping every tag, since the
    tags inside one are spans coloring part of a line and stripping them
    into newlines would make a line out of each.
    """
    out: list[str] = []
    for block in re.finditer(
        r'<div class="dc-banner">(?P<banner>.*?)</div>'
        r"|<summary>(?P<summary>.*?)</summary>"
        r'|<pre class="dc-body">(?P<pre>.*?)</pre>'
        r'|<div class="dc-line[^"]*">(?P<line>.*?)</div>'
        r'|<table class="dc-table">(?P<table>.*?)</table>',
        html,
        re.DOTALL,
    ):
        [(kind, content)] = [
            (k, v) for k, v in block.groupdict().items() if v is not None
        ]
        if kind == "table":
            # Already one value per cell, which is what a table is for.
            for row in re.findall(r"<tr>(.*?)</tr>", content, re.DOTALL):
                cells = [
                    _bare(x)
                    for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
                ]
                if cells and cells[0]:
                    out.extend(x for x in cells if x)
            continue
        for line in _bare(content).split("\n"):
            out.extend(_decompose(line))
    return out


def _bare(html: str) -> str:
    """The text of an HTML fragment, with its tags and entities read back."""
    return unescape(re.sub(r"<[^>]+>", "", html)).strip()


def _said_values(text: str) -> list[str]:
    """What a printed repr states, read the same way the panel is."""
    out: list[str] = []
    for line in text.split("\n"):
        out.extend(_decompose(line))
    return out


def _boom(node):
    """Stand in for a renderer which cannot draw what it is given."""
    raise ValueError("no panel for you")


class TestHtmlRepr:
    """Tests for the panel a notebook draws."""

    @pytest.fixture(scope="class")
    def html_objects(self):
        """One of each class which states a panel."""
        return {
            "patch": dc.get_example_patch(),
            "spool": dc.get_example_spool("diverse_das"),
            "inventory": dc.get_example_inventory("tunnel"),
            # A coordinate which is not a dimension, whose name carries
            # the dimensions it rides on.
            "non_dim": dc.get_example_patch().update_coords(
                quality=("distance", np.ones(300))
            ),
            # A coordinate which selected nothing states only a shape
            # and a dtype, which is a subset of what the one beside it
            # states and not a prefix of it.
            "partial": dc.get_example_patch().select(distance=(1e9, 2e9)),
        }

    def test_each_states_a_panel(self, html_objects):
        """The hook a display looks for, on the objects people echo."""
        for name, obj in html_objects.items():
            assert obj._repr_html_().startswith('<div class="dc-repr">'), name

    def test_the_panel_is_deterministic(self, html_objects):
        """
        No uuid, no counter, no timestamp.

        A notebook is a file in version control, and a repr which drew
        itself differently every time would rewrite it on every run.
        """
        for name, obj in html_objects.items():
            assert obj._repr_html_() == obj._repr_html_(), name
            assert "id=" not in obj._repr_html_(), name

    def test_the_panel_carries_no_inline_styles(self, html_objects):
        """
        Every color goes through a class, so the stylesheet is the only
        thing which says what one is and both themes stay reachable.
        """
        for name, obj in html_objects.items():
            assert 'style="' not in obj._repr_html_(), name

    def test_a_value_which_looks_like_markup(self):
        """A tag is a value someone else chose."""
        patch = dc.get_example_patch().update_attrs(
            tag="<script>alert(1)</script>", station="a & b"
        )
        html = patch._repr_html_()
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
        assert "a &amp; b" in html

    def test_no_panel_when_it_is_turned_off(self, html_objects):
        """
        None is how the protocol says "not this time".

        The display falls back to the text repr, which says the same
        words.
        """
        with config_context(display_html=False):
            for name, obj in html_objects.items():
                assert obj._repr_html_() is None, name

    def test_a_panel_which_cannot_be_drawn(self, monkeypatch):
        """
        A traceback out of a formatter is printed into the cell on every
        echo of the object, which makes it undebuggable exactly when
        someone is looking at it.

        Only the drawing is broken here, not the object: breaking
        `_repr_node` breaks the text repr too, and an object which
        cannot be printed cannot be reported on either.

        Debug is turned off here because the suite runs with it on
        (`tests/conftest.py`), which is the right way round: a panel
        which cannot be drawn should fail in CI and stay quiet for a
        reader. So this is a path only a reader takes.
        """
        monkeypatch.setattr(display, "_render_html", _boom)
        with config_context(debug=False):
            assert dc.get_example_patch()._repr_html_() is None

    def test_debug_mode_wants_the_traceback(self, monkeypatch):
        """Swallowing it in CI is how a broken repr ships unnoticed."""
        monkeypatch.setattr(display, "_render_html", _boom)
        with config_context(debug=True), pytest.raises(ValueError, match="no panel"):
            dc.get_example_patch()._repr_html_()

    def test_the_panel_says_what_the_text_says(self, html_objects):
        """
        The two reprs are one repr shown two ways.

        Tags stripped and entities read back, the panel holds every line
        the text holds. Two things it legitimately does not: the leading
        space, which it states with a margin rather than with
        characters, and the arrow a terminal opens a title with, which
        it draws as a disclosure triangle instead.
        """
        for name, obj in html_objects.items():
            # As a sequence, so a section drawn twice or drawn out of
            # order is a difference rather than a match.
            assert _drawn_values(obj._repr_html_()) == _said_values(str(obj)), name

    def test_the_stylesheet_stays_small(self):
        """
        Every repr carries it, so a notebook carries one copy per cell.

        Held on its own rather than on the panel, where growth in one
        hides growth in the other. The ceiling sits about a section's
        worth of rules above what ships, so a block of them trips it
        rather than a rule or two -- which means it has to be raised
        deliberately whenever a block is added. It counts the sheet as
        it goes out, so a comment is free and a rule is not.
        """
        assert len(_get_stylesheet().encode()) < 6_000

    def test_a_panel_is_mostly_the_object(self, html_objects):
        """
        What a panel adds to the stylesheet is what it says.

        Held apart from the sheet, so a section drawn twice is over the
        ceiling rather than lost inside the headroom.
        """
        overhead = len(_get_stylesheet().encode())
        for name, obj in html_objects.items():
            assert len(obj._repr_html_().encode()) - overhead < 4_000, name
