"""Tests for displaying dascore objects."""

from __future__ import annotations

import re
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
from dascore.utils.display import (
    Raw,
    Repr,
    Section,
    array_to_text,
    attrs_to_text,
    counts_to_text,
    duration_text,
    get_header_text,
    get_nice_text,
    group_names,
    human_duration,
    human_size,
    indent_text,
    limit_reprs,
    mapping_to_text,
    model_to_line,
    percent,
    rate_text,
    render_text,
    split_block,
    stated_fields,
    value_to_text,
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
        out = str(indent_text(Text("a\nb"), "  "))
        assert out == "  a\n  b"

    def test_styles_kept(self):
        """Indenting does not flatten the styles it wraps."""
        text = Text("red", style="red")
        assert indent_text(text).spans


class TestLimitReprs:
    """Tests for capping how many children a repr lists."""

    def test_under_limit_unchanged(self):
        """Nothing is added while everything fits."""
        items = [Network(code="A"), Network(code="B")]
        out = limit_reprs(items, limit=3)
        assert [str(x) for x in out] == [str(x) for x in items]

    def test_over_limit_says_what_is_missing(self):
        """The tail is named rather than dropped silently."""
        items = [Network(code=f"N{x}") for x in range(5)]
        out = limit_reprs(items, limit=2)
        assert len(out) == 3
        assert "3 more" in str(out[-1])

    def test_limit_from_config(self):
        """The default cap comes from runtime config."""
        with config_context(display_max_items=1):
            out = limit_reprs([Network(code="A"), Network(code="B")])
        assert len(out) == 2
        assert "1 more" in str(out[-1])

    def test_hidden_children_not_rendered(self):
        """A child which is not shown is not built, however many there are."""
        rendered = []

        class Counted:
            def __rich__(self):
                rendered.append(1)
                return Text("x")

        limit_reprs([Counted() for _ in range(50)], limit=2)
        assert len(rendered) == 2


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
        out = str(value_to_text("a", "x" * 500))
        assert len(out) < 500
        assert out.endswith("…")

    def test_pre_rendered_text_kept(self):
        """Text the caller built is used as-is, not re-read as a sequence."""
        assert str(value_to_text("a", Text("a: 2, b: 1"))) == "a: 2, b: 1"

    def test_array_keeps_its_own_repr(self):
        """An array states how much of itself to show; it is not unrolled."""
        out = str(value_to_text("a", np.arange(10_000)))
        assert "..." in out and len(out) < 100

    def test_frame_keeps_its_own_repr(self):
        """Nor is a frame read as a sequence of its column names."""
        out = str(value_to_text("a", pd.DataFrame({"a": [1, 2]}), truncate=False))
        assert "1" in out and "2" in out

    def test_nat_is_text(self):
        """An unset time renders like every other value, not as a str."""
        assert str(value_to_text("a", np.datetime64("NaT"))) == "NaT"


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
        1 GHz of sampling that happens at 666.7 MHz. The true rate is
        not a round one, so the honest answer is to say no rate.
        """
        assert rate_text(np.timedelta64(1500, "ps")) is None
        assert "1 GHz" in str(rate_text(np.timedelta64(1000, "ps")))

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
