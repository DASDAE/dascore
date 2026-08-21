"""Tests for the plots a spool draws of itself."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PatchCollection

import dascore as dc
from dascore.exceptions import InvalidSpoolQueryError, ParameterError
from dascore.viz import VizSpoolNameSpace
from dascore.viz import spool as spool_viz
from dascore.viz.spool import (
    COVERAGE_COLORS,
    _human_duration,
    _percent,
    _union,
    calendar,
    coverage,
)


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


@pytest.fixture(scope="module")
def deployment():
    """Two months of a sparsely sampled deployment, with real outages."""
    return dc.get_example_spool("sparse_dss")


def _cells(ax):
    """The value drawn in each calendar cell, as a masked (months, 31) array."""
    return ax.collections[0].get_array().reshape(-1, 31)


def _cell(ax, day) -> float:
    """The value drawn for one day, named as text."""
    stamp = pd.Timestamp(day)
    cells = _cells(ax)
    first = pd.Timestamp(ax.get_yticklabels(minor=True)[0].get_text())
    row = (stamp.year - first.year) * 12 + stamp.month - first.month
    return float(cells[row, stamp.day - 1])


class TestNamespace:
    """The plots hang off spool.viz."""

    def test_registered(self, whole):
        """Spool.viz is the viz namespace, holding the spool's plots."""
        assert isinstance(whole.viz, VizSpoolNameSpace)
        assert whole.viz.coverage.__name__ == "coverage"
        assert whole.viz.calendar.__name__ == "calendar"

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
            # A lane needs a width, so its two ends may not be one point.
            (dict(time=("2020-01-03", "2020-01-03")), "must be increasing"),
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
            # A multi-year outage is not worth reading in days.
            (86_400.0 * 400, "1.1 y"),
            (86_400.0 * 14_852, "40.7 y"),
        ],
    )
    def test_human_duration(self, seconds, text):
        """A gap is stated in the largest unit which fits it."""
        assert _human_duration(pd.Timedelta(seconds=seconds)) == text

    def test_duration_of_a_plain_number(self):
        """A dimension which is not time still labels its gaps."""
        assert _human_duration(12.0) == "12 s"

    def test_duration_smaller_than_any_unit(self):
        """A gap under a microsecond still says how long it is."""
        assert _human_duration(1e-9) == "1e-09 s"

    def test_an_attr_spelled_like_an_envelope(self):
        """Only the measured dimension owns the envelope column names."""
        report = pd.DataFrame(
            {
                "group_id": [0, 1],
                "coverage": [1.0, 1.0],
                # An attr may be called this; it still names its group.
                "site_min": ["west", "east"],
                "time_min": [0.0, 0.0],
                "time_max": [1.0, 1.0],
                "time_step": [0.1, 0.1],
            }
        )
        names = spool_viz._lane_names(report, "time")
        assert [x.split()[0] for x in names] == ["west", "east"]

    def test_the_measured_dimension_never_names_a_lane(self):
        """The axis already says the extent, so the lane does not repeat it."""
        report = pd.DataFrame(
            {
                "group_id": [0, 1],
                "coverage": [1.0, 1.0],
                "tag": ["same", "same"],
                "time_min": [0.0, 5.0],
                "time_max": [1.0, 6.0],
            }
        )
        assert spool_viz._lane_names(report, "time") == [
            "group 0  100%",
            "group 1  100%",
        ]

    def test_a_key_is_shown(self):
        """An acquisition key tells two groups apart, and names them."""
        first = dc.get_example_spool("random_das", acquisition_key="DAS1.R1..RAW")
        second = dc.get_example_spool("random_das", acquisition_key="DAS2.R2..RAW")
        spool = dc.spool(list(first) + list(second))
        names = spool_viz._lane_names(spool.get_coverage("time"), "time")
        assert [x.split()[0] for x in names] == ["DAS1.R1..RAW", "DAS2.R2..RAW"]

    def test_an_unrecorded_attr_is_not_a_value(self, diverse):
        """A group which states no acquisition key is not named by a blank."""
        names = spool_viz._lane_names(diverse.get_coverage("time"), "time")
        assert not any(x.startswith("·") or x.startswith(" ") for x in names)


class TestCalendar:
    """How much of each day a spool holds."""

    def test_one_cell_per_day(self, deployment):
        """Every day between the first and the last gets a cell."""
        ax = deployment.viz.calendar()
        drawn = np.count_nonzero(~np.ma.getmaskarray(_cells(ax)))
        assert drawn == 60  # 2024-01-01 through 2024-02-29
        assert [x.get_text() for x in ax.get_yticklabels(minor=True)] == [
            "2024-Jan",
            "2024-Feb",
        ]

    def test_percent(self, deployment):
        """A day reads as the fraction of it the spool covers."""
        ax = deployment.viz.calendar()
        assert _cell(ax, "2024-01-05") == 100.0  # both acquisitions run
        assert _cell(ax, "2024-01-18") == 0.0  # the site outage
        assert _cell(ax, "2024-01-09") == 75.0  # one, and it stopped early
        assert _cell(ax, "2024-02-11") == 50.0  # both stopped early

    def test_never_over_a_whole_day(self, deployment):
        """Acquisitions which run at once cover one day between them."""
        assert _cells(deployment.viz.calendar()).max() == 100.0

    def test_overlap_is_measured_once(self):
        """Two copies of one interval cover what one of them does."""
        spool = dc.get_example_spool("random_das")
        doubled = dc.spool(list(spool) + list(spool))
        assert _cells(doubled.viz.calendar()).max() <= 100.0

    def test_gap_method(self, deployment):
        """The gap method says how much of the day is missing."""
        ax = deployment.viz.calendar(method="gap")
        assert _cell(ax, "2024-01-18") == 86_400.0
        assert _cell(ax, "2024-01-09") == 86_400.0 * 0.25
        assert _cell(ax, "2024-01-05") == 0.0

    def test_count_method(self, deployment):
        """The count method counts the patches which overlap the day."""
        ax = deployment.viz.calendar(method="count")
        assert _cell(ax, "2024-01-15") == 2  # both acquisitions run
        assert _cell(ax, "2024-01-05") == 1  # strain has not started
        assert _cell(ax, "2024-02-03") == 1  # temperature is down
        assert _cell(ax, "2024-01-18") == 0  # the site outage

    def test_a_run_reaching_into_the_next_day_gets_it(self):
        """A run covers a step past its last sample, and that day counts."""
        patch = dc.get_example_patch(
            "random_das",
            time_min=np.datetime64("2024-01-31T22:30"),
            time_step=np.timedelta64(1, "h"),
            shape=(2, 2),
        )
        ax = dc.spool([patch]).viz.calendar()
        assert _cell(ax, "2024-01-31") == pytest.approx(5_400 / 864.0)
        # The last sample is at 23:30 and covers the hour after it.
        assert _cell(ax, "2024-02-01") == pytest.approx(1_800 / 864.0)

    def test_a_run_ending_at_midnight_opens_no_day(self):
        """Covering up to a day is not reaching into it."""
        patch = dc.get_example_patch(
            "random_das",
            time_min=np.datetime64("2024-01-31T23:00"),
            time_step=np.timedelta64(1, "h"),
            shape=(2, 1),
        )
        ax = dc.spool([patch]).viz.calendar()
        assert np.count_nonzero(~np.ma.getmaskarray(_cells(ax))) == 1

    def test_a_descending_coordinate_covers_as_much(self):
        """A run is a step long whichever way its samples are ordered."""
        forward = dc.get_example_patch(
            "random_das",
            time_min=np.datetime64("2024-01-05"),
            time_step=np.timedelta64(1, "h"),
            shape=(2, 24),
        )
        backward = forward.update_coords(time=forward.coords.get_array("time")[::-1])
        ax = dc.spool([backward]).viz.calendar()
        assert _cell(ax, "2024-01-05") == 100.0

    def test_a_single_day_draws_a_cell(self):
        """The last day is a day with data, so a one-day spool is not empty."""
        ax = dc.get_example_spool("random_das").viz.calendar()
        assert np.count_nonzero(~np.ma.getmaskarray(_cells(ax))) == 1

    def test_days_outside_the_data_are_blank(self, deployment):
        """A month has no thirtieth of February, and no cell for one."""
        cells = _cells(deployment.viz.calendar())
        assert np.ma.getmaskarray(cells)[1, 29:].all()

    def test_window(self, deployment):
        """A window states the days to draw."""
        ax = deployment.viz.calendar(time=("2024-01-05", "2024-01-09"))
        assert np.count_nonzero(~np.ma.getmaskarray(_cells(ax))) == 5

    def test_half_open_window(self, deployment):
        """Either end may be left to the spool."""
        ax = deployment.viz.calendar(time=(None, "2024-01-31"))
        assert np.count_nonzero(~np.ma.getmaskarray(_cells(ax))) == 31

    def test_a_window_of_one_day(self, deployment):
        """Both ends may name the same day; a calendar cell is a day wide."""
        ax = deployment.viz.calendar(time=("2024-01-18", "2024-01-18"))
        assert np.count_nonzero(~np.ma.getmaskarray(_cells(ax))) == 1
        assert _cell(ax, "2024-01-18") == 0.0

    def test_tolerance_changes_what_counts(self, deployment):
        """A tolerance which closes the gaps fills the days they emptied."""
        assert _cell(deployment.viz.calendar(tolerance=200), "2024-01-18") == 100.0

    def test_group_reaches_the_reports(self, deployment):
        """Grouping is the spool's own argument, and its refusal stands."""
        with pytest.raises(InvalidSpoolQueryError, match="nope"):
            deployment.viz.calendar(group="nope")

    @pytest.mark.parametrize(
        "bad, match",
        [
            (dict(method="nope"), "not a calendar measure"),
            (dict(time=5), "must be a .start, end. pair"),
            (dict(time=("2024-02-01", "2024-01-01")), "must be increasing"),
        ],
    )
    def test_bad_arguments(self, deployment, bad, match):
        """An argument which states nothing drawable is explained."""
        with pytest.raises(ParameterError, match=match):
            deployment.viz.calendar(**bad)

    def test_an_empty_spool(self):
        """A spool with no time in it has no calendar."""
        with pytest.raises(ParameterError, match="no time to draw"):
            dc.spool([]).viz.calendar()

    def test_ax_and_show(self, whole, monkeypatch):
        """A given ax is drawn on; show calls plt.show."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        _, ax = plt.subplots()
        assert whole.viz.calendar(ax=ax, show=True) is ax
        assert called

    def test_figsize(self, whole):
        """Figsize sizes the figure built when no ax is given."""
        ax = whole.viz.calendar(figsize=(4, 3))
        assert tuple(ax.get_figure().get_size_inches()) == (4.0, 3.0)

    def test_module_function(self, whole):
        """The plot is callable without the namespace, as tests need."""
        assert calendar(whole) is not None


class TestUnion:
    """Intervals are measured once, however they overlap."""

    def test_disjoint(self):
        """Intervals which do not touch are left alone."""
        assert _union(np.array([0, 5]), np.array([2, 7])) == [(0, 2), (5, 7)]

    def test_overlapping(self):
        """Intervals which overlap become the one they span."""
        assert _union(np.array([0, 1]), np.array([3, 7])) == [(0, 7)]

    def test_nested(self):
        """An interval inside another adds nothing to it."""
        assert _union(np.array([0, 2]), np.array([9, 3])) == [(0, 9)]

    def test_touching(self):
        """Intervals which meet at a point are one interval."""
        assert _union(np.array([0, 4]), np.array([4, 8])) == [(0, 8)]

    def test_out_of_order(self):
        """The order they arrive in does not matter."""
        assert _union(np.array([5, 0]), np.array([7, 2])) == [(0, 2), (5, 7)]


class TestSparseDssExample:
    """The example the calendar is drawn from."""

    def test_two_acquisitions(self, deployment):
        """It holds a temperature and a strain acquisition."""
        report = deployment.get_coverage("time")
        assert set(report["tag"]) == {"temperature", "strain"}

    def test_stays_small(self, deployment):
        """Sampled once an hour, so the whole deployment is tiny."""
        contents = deployment.get_contents()
        total = sum(len(x.data.tobytes()) for x in deployment)
        assert total < 1_000_000
        assert (contents["time_step"] == np.timedelta64(1, "h")).all()

    def test_reports_the_outage(self, deployment):
        """The site outage is a gap in both acquisitions."""
        gaps = deployment.get_gaps("time")
        outage = gaps[gaps["gap_size"] > np.timedelta64(3, "D")]
        assert set(outage["tag"]) == {"temperature", "strain"}


def _is_gap(x0, ax) -> bool:
    """Whether the box starting here was drawn in the gap color."""
    gap = tuple(np.round(plt.matplotlib.colors.to_rgba(COVERAGE_COLORS["gap"]), 4))
    return any(
        abs(start - x0) < 1e-9 and color == gap for start, _, color in _boxes(ax)
    )
