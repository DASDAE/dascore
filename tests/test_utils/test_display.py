"""Tests for displaying dascore objects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict
from rich.text import Text

import dascore as dc
from dascore.config import config_context
from dascore.constants import dascore_styles
from dascore.core.annotations import AnnotationColumn, AnnotationSetAttrs
from dascore.core.inventory import Acquisition, Cable, Network
from dascore.utils.display import (
    array_to_text,
    counts_to_text,
    get_header_text,
    get_nice_text,
    group_names,
    human_duration,
    indent_text,
    limit_reprs,
    mapping_to_text,
    model_to_line,
    percent,
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
        assert "start_time" not in stated_fields(Network(code="XT"))

    def test_set_times_kept(self):
        """A stated epoch is."""
        network = Network(code="XT", start_time="2020-01-01")
        assert "start_time" in stated_fields(network)

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
