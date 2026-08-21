"""Tests for the plots a spool draws of itself."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PatchCollection

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.viz import VizSpoolNameSpace
from dascore.viz.spool import COVERAGE_COLORS, _human_duration, _percent, coverage


def _lanes(ax):
    """The lane names an axes shows, top to bottom."""
    return [x.get_text() for x in ax.get_yticklabels()]


def _boxes(ax):
    """Every drawn interval, as (x0, width) in axis units."""
    out = []
    for collection in ax.collections:
        if not isinstance(collection, PatchCollection):
            continue
        for path, color in zip(
            collection.get_paths(), collection.get_facecolors(), strict=False
        ):
            box = path.get_extents()
            out.append((float(box.x0), float(box.width), tuple(np.round(color, 4))))
    return out


def _kinds(ax):
    """How many boxes were drawn in each of the two colors."""
    gap = tuple(np.round(plt.matplotlib.colors.to_rgba(COVERAGE_COLORS["gap"]), 4))
    data = tuple(np.round(plt.matplotlib.colors.to_rgba(COVERAGE_COLORS["data"]), 4))
    drawn = [x[2] for x in _boxes(ax)]
    return {"gap": drawn.count(gap), "data": drawn.count(data)}


@pytest.fixture(scope="module")
def diverse():
    """A spool with several groups, gaps of several sizes, and an outlier."""
    return dc.get_example_spool("diverse_das")


@pytest.fixture(scope="module")
def whole():
    """A spool of one continuous group."""
    return dc.get_example_spool("random_das")


class TestNamespace:
    """The plots hang off spool.viz."""

    def test_registered(self, whole):
        """Spool.viz is the viz namespace, holding coverage."""
        assert isinstance(whole.viz, VizSpoolNameSpace)
        assert whole.viz.coverage.__name__ == "coverage"

    def test_declared_as_an_entry_point(self):
        """An install must carry the namespace, not just an import of it."""
        text = (Path(dc.__file__).parent.parent / "pyproject.toml").read_text()
        block = text.split('[project.entry-points."dascore.spool_namespace"]')[1]
        assert 'viz = "dascore.viz:VizSpoolNameSpace"' in block.split("[")[0]

    def test_patch_plot_points_at_the_patch(self, whole):
        """A patch plot asked of a spool says how to get the patch."""
        with pytest.raises(AttributeError, match="a spool is many of them") as info:
            whole.viz.waterfall()
        assert "chunk(time=None)[0].viz.waterfall()" in str(info.value)

    def test_unknown_name(self, whole):
        """A name which is no plot at all still says so plainly."""
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            whole.viz.nope()


class TestCoverage:
    """What the spool holds, and where it does not."""

    def test_one_lane_per_group(self, diverse):
        """Each group of patches which could combine gets its own lane."""
        ax = diverse.viz.coverage()
        report = diverse.get_coverage("time")
        assert len(_lanes(ax)) == len(report)
        # The lane says which group it is and how complete it is.
        assert any(x.startswith("big_gaps") for x in _lanes(ax))
        assert all("%" in x for x in _lanes(ax))

    def test_gaps_are_drawn_where_the_spool_says(self, diverse):
        """Every gap the spool reports is drawn, and nothing else is."""
        ax = diverse.viz.coverage()
        gaps = diverse.get_gaps("time")
        assert _kinds(ax)["gap"] == len(gaps)
        drawn = sorted(round(x, 6) for x, _, _ in _boxes(ax) if _is_gap(x, ax))
        expected = sorted(
            round(float(plt.matplotlib.dates.date2num(pd.Timestamp(x))), 6)
            for x in gaps["time_min"]
        )
        assert drawn == expected

    def test_runs_and_holes_tile_the_span(self, diverse):
        """Together the two kinds cover each group's extent exactly once."""
        ax = diverse.viz.coverage()
        report = diverse.get_coverage("time")
        spans = {
            float(
                plt.matplotlib.dates.date2num(pd.Timestamp(row["time_max"]))
                - plt.matplotlib.dates.date2num(pd.Timestamp(row["time_min"]))
            )
            for _, row in report.iterrows()
        }
        by_lane: dict = {}
        for collection in ax.collections:
            if isinstance(collection, PatchCollection):
                total = sum(x.get_extents().width for x in collection.get_paths())
                by_lane[round(float(total), 8)] = True
        for span in spans:
            assert any(abs(x - span) < 1e-6 for x in by_lane), span

    def test_a_whole_spool_has_no_gaps(self, whole):
        """A continuous spool draws one run and no holes."""
        ax = whole.viz.coverage()
        assert _kinds(ax) == {"gap": 0, "data": 1}
        assert _lanes(ax)[0].endswith("100%")

    def test_tolerance_changes_what_counts(self, diverse):
        """A larger tolerance closes the gaps it now spans."""
        tight = _kinds(diverse.viz.coverage(tolerance=1.5))["gap"]
        plt.close("all")
        loose = _kinds(diverse.viz.coverage(tolerance=1e6))["gap"]
        assert loose < tight

    def test_group_argument(self, diverse):
        """Grouping by nothing at all puts every patch in one lane."""
        ax = diverse.viz.coverage(group=[])
        assert len(_lanes(ax)) < len(diverse.get_coverage("time"))

    def test_window(self, diverse):
        """A window states the limits, and accepts times as text."""
        ax = diverse.viz.coverage(time=("2020-01-03", "2020-01-03T00:00:30"))
        low, high = ax.get_xlim()
        assert high - low == pytest.approx(30.0 / 86_400, rel=1e-3)

    def test_half_open_window(self, diverse):
        """Either end may be left to the data."""
        ax = diverse.viz.coverage(time=("2020-01-03", None))
        assert ax.get_xlim()[1] > ax.get_xlim()[0]
        plt.close("all")
        ax = diverse.viz.coverage(time=(None, "2020-01-03"))
        assert ax.get_xlim()[1] > ax.get_xlim()[0]

    def test_dimension_without_a_window(self, diverse):
        """An Ellipsis names the dimension and asks for all of it."""
        ax = diverse.viz.coverage(time=...)
        assert len(_lanes(ax)) == len(diverse.get_coverage("time"))

    def test_another_dimension(self, diverse):
        """A spool can be measured along any dimension it states."""
        ax = diverse.viz.coverage(distance=...)
        assert ax.get_xlabel() == "distance"
        assert len(_lanes(ax)) == len(diverse.get_coverage("distance"))

    def test_a_dimension_nothing_states(self, whole):
        """The spool's own refusal stands; it names the dimensions there are."""
        with pytest.raises(Exception, match="Cannot report on 'depth'") as info:
            whole.viz.coverage(depth=...)
        assert "time" in str(info.value)

    def test_every_group_gets_a_lane(self, diverse):
        """Groups which state the same attributes are still separate lanes."""
        ax = diverse.viz.coverage(distance=...)
        report = diverse.get_coverage("distance")
        lanes = _lanes(ax)
        # Along distance these groups share every shown attribute, and are
        # parted by sampling and structure, which the lane cannot show.
        assert len(lanes) == len(report) > 6
        assert len(set(lanes)) == len(lanes)

    @pytest.mark.parametrize(
        "bad, match",
        [
            (dict(time=5), "must be a .start, end. pair"),
            (dict(time=("2020-01-04", "2020-01-03")), "must be increasing"),
            (dict(time=..., distance=...), "names 2"),
        ],
    )
    def test_bad_selection(self, diverse, bad, match):
        """A selection which is not one is explained."""
        with pytest.raises(ParameterError, match=match):
            diverse.viz.coverage(**bad)

    def test_gap_labels(self, diverse):
        """A gap says how long it is, in a unit worth reading."""
        ax = diverse.viz.coverage(time=("2020-01-03", "2020-01-03T00:00:30"))
        assert any(x.get_text().endswith(" s") for x in ax.texts)

    def test_color_override(self, diverse):
        """The two colors can be replaced."""
        ax = diverse.viz.coverage(color={"data": "black", "gap": "white"})
        drawn = [x[2][:3] for x in _boxes(ax)]
        assert (0.0, 0.0, 0.0) in drawn

    def test_ax_and_show(self, whole, monkeypatch):
        """A given ax is drawn on; show calls plt.show."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        _, ax = plt.subplots()
        assert whole.viz.coverage(ax=ax, show=True) is ax
        assert called

    def test_figsize(self, whole):
        """Figsize sizes the figure built when no ax is given."""
        ax = whole.viz.coverage(figsize=(5, 2))
        assert tuple(ax.get_figure().get_size_inches()) == (5.0, 2.0)

    def test_module_function(self, whole):
        """The plot is callable without the namespace, as tests need."""
        assert coverage(whole) is not None


class TestNaming:
    """How a lane says which group it is and how complete."""

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
        assert _percent(value) == text

    def test_a_hole_is_never_rounded_away(self):
        """Only a whole span reads as 100%."""
        assert _percent(0.999999999) == "<100%"
        assert _percent(1.0) == "100%"

    @pytest.mark.parametrize(
        "seconds, text",
        [
            (0.0, ""),
            (0.008, "8 ms"),
            (1.004, "1 s"),
            (90.0, "1.5 m"),
            (7200.0, "2 h"),
            (86_400.0 * 3, "3 d"),
        ],
    )
    def test_human_duration(self, seconds, text):
        """A gap is stated in the largest unit which fits it."""
        assert _human_duration(pd.Timedelta(seconds=seconds)) == text

    def test_duration_of_a_plain_number(self):
        """A dimension which is not time still labels its gaps."""
        assert _human_duration(12.0) == "12 s"

    def test_group_id_when_nothing_tells_them_apart(self, monkeypatch):
        """Where no attribute varies, the group's ordinal names the lane."""
        from dascore.viz import spool as module

        report = pd.DataFrame(
            {
                "group_id": [0, 1],
                "coverage": [1.0, 1.0],
                "tag": ["same", "same"],
                "time_min": [0.0, 0.0],
                "time_max": [1.0, 1.0],
            }
        )
        assert module._lane_names(report) == ["group 0  100%", "group 1  100%"]


def _is_gap(x0, ax) -> bool:
    """Whether the box starting here was drawn in the gap color."""
    gap = tuple(np.round(plt.matplotlib.colors.to_rgba(COVERAGE_COLORS["gap"]), 4))
    return any(
        abs(start - x0) < 1e-9 and color == gap for start, _, color in _boxes(ax)
    )
