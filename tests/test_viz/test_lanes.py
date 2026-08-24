"""Tests for the interval-lane renderer."""

from __future__ import annotations

import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba_array
from matplotlib.figure import Figure

from dascore.exceptions import ParameterError
from dascore.viz._lanes import (
    _LABEL_PAD,
    SEPARATOR_COLOR,
    UNCOVERED_COLOR,
    WHEEL_ORDER,
    _pack_rows,
    _text_points,
    estimate_legend_rows,
    legend_column_points,
    plot_lanes,
)


def _collections(ax):
    """The patch collections an axes holds, in drawing order."""
    return [x for x in ax.collections if isinstance(x, PatchCollection)]


def _extents(collection):
    """The (x0, width) of every path in a collection."""
    out = []
    for path in collection.get_paths():
        box = path.get_extents()
        out.append((float(box.x0), float(box.width)))
    return out


def _texts(ax):
    """Every label drawn on the axes."""
    return [x.get_text() for x in ax.texts]


def _legend_of(figure, ax):
    """The legend a figure grew, whichever of the two owns it."""
    return figure.legends[0] if figure.legends else ax.get_legend()


def _many_values(count=30):
    """A frame of one lane naming more values than an axes is tall."""
    names = [f"value {x:02d}" for x in range(count)]
    return names, pd.DataFrame(
        {
            "start": np.arange(float(count)),
            "end": np.arange(float(count)) + 1.0,
            "v": names,
        }
    )


def _overflowing(ax):
    """Labels drawn wider or taller, in pixels, than the box holding them."""
    figure = ax.get_figure()
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    boxes = [x.get_extents() for x in _collections(ax)[0].get_paths()]
    out = []
    for text in ax.texts:
        drawn = text.get_window_extent(renderer)
        middle = text.get_position()
        for box in boxes:
            if not (box.x0 <= middle[0] <= box.x1):
                continue
            corner = ax.transData.transform((box.x0, box.y0))
            far = ax.transData.transform((box.x1, box.y1))
            if drawn.width > abs(far[0] - corner[0]) or drawn.height > abs(
                far[1] - corner[1]
            ):
                out.append(text.get_text())
    return out


@pytest.fixture()
def string_frame():
    """Two lanes of named zones."""
    return pd.DataFrame(
        {
            "group": ["zone", "zone", "other", "other"],
            "start": [0.0, 10.0, 0.0, 30.0],
            "end": [10.0, 20.0, 20.0, 40.0],
            "value": ["north", "south", "north", "west"],
        }
    )


@pytest.fixture()
def kinds_frame():
    """One lane of every value kind, plus a point marker."""
    return pd.DataFrame(
        {
            "lane": ["text", "text", "flag", "flag", "count", "count", "tick"],
            "start": [0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 3.0],
            "end": [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 3.0],
            # A row belongs to its lane by stating no value at all.
            "value": ["a", "b", None, None, 1, 2, None],
        }
    )


class TestReadFrame:
    """The frame contract: named columns, kinds, and refusals."""

    def test_renamed_columns(self):
        """Bound columns are named by the caller, as a spool frame needs."""
        frame = pd.DataFrame({"time_min": [0.0, 5.0], "time_max": [4.0, 9.0]})
        ax = plot_lanes(frame, start="time_min", end="time_max")
        assert _extents(_collections(ax)[0]) == [(0.0, 4.0), (5.0, 4.0)]

    def test_mapping_input(self):
        """A plain mapping of columns is accepted as a frame."""
        ax = plot_lanes({"start": [0.0], "end": [1.0]})
        assert len(_collections(ax)) == 1

    def test_datetime_bounds(self):
        """Datetime bounds convert to matplotlib dates once, up front."""
        frame = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                "end": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            }
        )
        limits = pd.to_datetime(["2024-01-01", "2024-01-06"]).to_numpy()
        ax = plot_lanes(frame, x_limits=limits)
        widths = [w for _, w in _extents(_collections(ax)[0])]
        assert widths == pytest.approx([1.0, 2.0])

    def test_timezone_aware_bounds(self):
        """Zoned datetimes are drawn at their UTC instant."""
        frame = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01T00:00"]).tz_localize("UTC"),
                "end": pd.to_datetime(["2024-01-02T00:00"]).tz_localize("UTC"),
            }
        )
        ax = plot_lanes(frame)
        x0, width = _extents(_collections(ax)[0])[0]
        assert width == pytest.approx(1.0)
        assert x0 == pytest.approx(mdates.date2num(np.datetime64("2024-01-01")))

    def test_datetime_x_limits(self):
        """Limits given as plain datetimes land on the same axis as the bars."""
        frame = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-02"]),
                "end": pd.to_datetime(["2024-01-03"]),
            }
        )
        limits = (datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 4))
        ax = plot_lanes(frame, x_limits=limits)
        assert ax.get_xlim()[0] == pytest.approx(
            mdates.date2num(np.datetime64("2024-01-01"))
        )

    def test_datetime_axis_is_formatted(self):
        """Dated bounds get a date axis, not raw ordinals."""
        frame = pd.DataFrame(
            {
                "start": pd.to_datetime(["2024-01-01"]),
                "end": pd.to_datetime(["2024-01-05"]),
            }
        )
        ax = plot_lanes(frame)
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
        ticks = " ".join(x.get_text() for x in ax.get_xticklabels())
        assert "Jan" in ticks, f"expected dates, got {ticks}"

    def test_missing_bounds(self):
        """A frame without the bound columns names what it has."""
        with pytest.raises(ParameterError, match="needs the columns"):
            plot_lanes(pd.DataFrame({"a": [1]}))

    def test_missing_named_column(self):
        """A lane/value/label name not in the frame is refused."""
        frame = pd.DataFrame({"start": [0.0], "end": [1.0]})
        with pytest.raises(ParameterError, match="lane='group' is not a column"):
            plot_lanes(frame, lane="group")

    def test_empty_frame(self):
        """Nothing to draw is an error, not a blank figure."""
        with pytest.raises(ParameterError, match="no rows"):
            plot_lanes(pd.DataFrame({"start": [], "end": []}))

    def test_backwards_interval(self):
        """An interval ending before it starts is refused by lane."""
        frame = pd.DataFrame({"start": [5.0], "end": [1.0], "lane": ["x"]})
        with pytest.raises(ParameterError, match="lane 'x' ends before it starts"):
            plot_lanes(frame, lane="lane")

    def test_mixed_kinds(self):
        """A lane mixing strings and numbers has no one color scheme."""
        frame = pd.DataFrame({"start": [0.0, 1.0], "end": [1.0, 2.0], "v": ["a", 1]})
        with pytest.raises(ParameterError, match="mixes value kinds"):
            plot_lanes(frame, value="v")

    def test_duplicate_lanes(self):
        """Naming a lane twice in lanes is refused."""
        frame = pd.DataFrame({"start": [0.0], "end": [1.0], "lane": ["a"]})
        with pytest.raises(ParameterError, match="names a lane twice"):
            plot_lanes(frame, lane="lane", lanes=("a", "a"))


class TestLayout:
    """Lane order, packing, points, and open edges."""

    def test_lane_order_first_appearance(self, string_frame):
        """Lanes appear in the order the frame first names them."""
        ax = plot_lanes(string_frame, lane="group", value="value")
        assert [x.get_text() for x in ax.get_yticklabels()] == ["zone", "other"]

    def test_explicit_lanes_filter_and_pad(self, string_frame):
        """lanes= orders, filters, and keeps an empty lane for alignment."""
        ax = plot_lanes(
            string_frame, lane="group", value="value", lanes=("other", "empty")
        )
        assert [x.get_text() for x in ax.get_yticklabels()] == ["other", "empty"]
        # Only the one populated lane produced boxes.
        assert len(_collections(ax)) == 1

    def test_packing_overlaps(self):
        """Overlapping intervals take separate sub-rows."""
        frame = pd.DataFrame({"start": [0.0, 5.0, 20.0], "end": [10.0, 15.0, 30.0]})
        assert _pack_rows(frame).tolist() == [0, 1, 0]
        ax = plot_lanes(frame)
        heights = {
            round(float(p.get_extents().height), 3)
            for p in _collections(ax)[0].get_paths()
        }
        assert heights == {0.4}

    def test_packing_caps_sub_rows(self):
        """A pileup degrades to the last sub-row rather than growing forever."""
        n = 12
        frame = pd.DataFrame({"start": [0.0] * n, "end": [10.0] * n})
        assert _pack_rows(frame).max() == 7

    def test_no_packing(self):
        """pack=False draws everything in one row."""
        frame = pd.DataFrame({"start": [0.0, 5.0], "end": [10.0, 15.0]})
        ax = plot_lanes(frame, pack=False)
        heights = {
            round(float(p.get_extents().height), 3)
            for p in _collections(ax)[0].get_paths()
        }
        assert heights == {0.8}

    def test_point_marker(self):
        """An interval of zero width is drawn as a tick, not lost."""
        frame = pd.DataFrame({"start": [0.0, 5.0], "end": [10.0, 5.0]})
        ax = plot_lanes(frame)
        assert len(_collections(ax)[0].get_paths()) == 1
        xs = [line.get_xdata()[0] for line in ax.lines]
        assert xs == [5.0, 5.0]

    def test_open_edges(self):
        """Open bounds earn a hatched sliver at that end."""
        frame = pd.DataFrame(
            {
                "start": [0.0, 10.0],
                "end": [10.0, 20.0],
                "open_start": [True, False],
                "open_end": [False, True],
            }
        )
        ax = plot_lanes(frame)
        hatched = [c for c in _collections(ax) if c.get_hatch()]
        assert len(hatched) == 1
        starts = sorted(x for x, _ in _extents(hatched[0]))
        assert starts[0] == pytest.approx(0.0)
        assert starts[1] < 20.0

    def test_labels_fit_or_drop(self):
        """A label wider than its box is dropped; others are drawn."""
        frame = pd.DataFrame(
            {"start": [0.0, 50.0], "end": [50.0, 50.5], "v": ["wide", "narrow"]}
        )
        ax = plot_lanes(frame, value="v")
        assert _texts(ax) == ["wide"]

    def test_a_narrow_box_turns_its_label(self):
        """Text too wide for its box is stood on end rather than dropped."""
        # Boxes narrower than the text but far taller than it is tall.
        frame = pd.DataFrame(
            {
                "start": [0.0, 20.0],
                "end": [1.6, 21.6],
                "v": ["alpha zone", "beta zone"],
            }
        )
        _, ax = plt.subplots(figsize=(4, 4))
        plot_lanes(frame, ax=ax, value="v")
        assert sorted(_texts(ax)) == ["alpha zone", "beta zone"]
        assert {x.get_rotation() for x in ax.texts} == {90.0}

    @pytest.mark.parametrize("slack,rotation", [(1.0, 90.0), (_LABEL_PAD + 1.0, 0.0)])
    def test_a_label_needs_clearance(self, slack, rotation):
        """A label the exact width of its box would touch the next one.

        A box wider than the text but by less than the clearance is
        refused the flat label it would otherwise take, which is what
        keeps two labels in neighboring boxes from reading as one word.
        """
        text = "value"
        needed = _text_points(text, plt.rcParams["font.size"] * 0.8)[0]
        figure, ax = plt.subplots(figsize=(4, 2), dpi=100)
        figure.draw_without_rendering()
        # Scale the axes so one data unit is exactly the room to test.
        points = ax.get_window_extent().width * 72 / figure.dpi
        plt.close(figure)
        _, ax = plt.subplots(figsize=(4, 2), dpi=100)
        plot_lanes(
            pd.DataFrame({"start": [0.0], "end": [1.0], "v": [text]}),
            ax=ax,
            value="v",
            x_limits=(0.0, points / (needed + slack)),
        )
        assert _texts(ax) == [text]
        assert ax.texts[0].get_rotation() == rotation

    def test_no_drawn_label_overflows_its_box(self):
        """Every label kept is measured against the axes it lands in.

        The legend takes its room after the boxes are drawn, so a label
        judged before that is judged against an axes which no longer
        exists by the time it is rendered.
        """
        frame = pd.DataFrame(
            {
                "start": np.arange(20.0),
                "end": np.arange(20.0) + 0.9,
                "v": [f"value {x}" for x in range(20)],
            }
        )
        ax = plot_lanes(frame, value="v")
        # Every label kept sits inside its box -- and they were kept. The
        # first half of that is true of a figure which drew none at all.
        assert len(ax.texts) == len(frame)
        assert _overflowing(ax) == []

    @pytest.mark.parametrize("text", [" ", "  ", "two\nlines", "$x^2$"])
    def test_labels_matplotlib_lays_out_its_own_way(self, text):
        """Whitespace, several lines and mathtext are all measurable.

        A label is measured before it is drawn, so a string the measurer
        cannot read would take the whole figure down with it.
        """
        frame = pd.DataFrame({"start": [0.0], "end": [10.0], "v": [text]})
        ax = plot_lanes(frame, value="v")
        assert _overflowing(ax) == []

    def test_a_legend_naming_nothing_takes_no_rows(self):
        """A caller sizing a figure for no legend keeps no room for one."""
        assert estimate_legend_rows([], 720.0) == 0
        assert estimate_legend_rows(["one"], 720.0) == 1

    def test_a_legend_estimate_counts_the_lines_it_names(self):
        """A value written on two lines takes two lines of legend.

        Counting entries rather than lines keeps too little room, and
        the legend then goes below into space nobody reserved.
        """
        assert legend_column_points(["a\nb"]) == legend_column_points(["a", "b"])
        assert estimate_legend_rows(["a\nb"], 720.0) == 2

    def test_a_label_of_two_lines_is_two_lines_tall(self):
        """Height is what decides a rotated label, so lines must count."""
        size = plt.rcParams["font.size"] * 0.8
        one = _text_points("two", size)[1]
        assert _text_points("two\nlines", size)[1] > 2 * one

    def test_max_labels(self):
        """Past max_labels no text is drawn at all."""
        frame = pd.DataFrame({"start": [0.0, 50.0], "end": [50.0, 100.0]})
        frame["v"] = ["a", "b"]
        ax = plot_lanes(frame, value="v", max_labels=1)
        assert _texts(ax) == []

    def test_label_column(self):
        """label= overrides the default text."""
        frame = pd.DataFrame({"start": [0.0], "end": [100.0], "v": [3.5], "t": ["x"]})
        ax = plot_lanes(frame, value="v", label="t")
        assert _texts(ax) == ["x"]

    def test_default_labels(self, kinds_frame):
        """Numbers state themselves; a row stating no value draws no text."""
        ax = plot_lanes(kinds_frame, lane="lane", value="value")
        assert sorted(_texts(ax)) == ["1", "2", "a", "b"]

    def test_x_label_and_show(self, shown):
        """x_label is applied and show calls plt.show."""
        ax = plot_lanes({"start": [0.0], "end": [1.0]}, x_label="Time", show=True)
        assert ax.get_xlabel() == "Time"
        assert shown


class TestColors:
    """The color policy per value kind and the overrides."""

    def test_string_colors_frame_wide(self, string_frame):
        """One string value is one color in every lane."""
        ax = plot_lanes(string_frame, lane="group", value="value")
        zone, other = _collections(ax)
        assert np.allclose(zone.get_facecolors()[0], other.get_facecolors()[0])
        labels = [x.get_text() for x in ax.get_legend().get_texts()]
        assert labels == ["north", "south", "west"]

    def test_membership_lane(self, kinds_frame):
        """Rows which state no value take the lane's color, and name it."""
        ax = plot_lanes(kinds_frame, lane="lane", value="value", lanes=("flag",))
        colors = _collections(ax)[0].get_facecolors()
        assert len(colors) == 2
        assert colors[0][3] == pytest.approx(1.0)
        assert colors[1][3] == pytest.approx(1.0)
        assert np.allclose(colors[0], colors[1])
        assert [x.get_text() for x in ax.get_legend().get_texts()] == ["flag"]

    def test_nan_states_membership(self, kinds_frame):
        """A NaN value, as a mixed frame spells a missing one, is no value."""
        frame = kinds_frame.assign(value=["a", "b", np.nan, np.nan, 1, 2, None])
        ax = plot_lanes(frame, lane="lane", value="value")
        assert sorted(_texts(ax)) == ["1", "2", "a", "b"]

    def test_booleans_are_refused(self, kinds_frame):
        """True and false are not values; membership is a row with none."""
        frame = kinds_frame.assign(value=["a", "b", True, False, 1, 2, None])
        with pytest.raises(ParameterError, match="not values"):
            plot_lanes(frame, lane="lane", value="value")

    def test_numeric_few_values(self, kinds_frame):
        """A few numbers are colored continuously but earn no colorbar."""
        ax = plot_lanes(kinds_frame, lane="lane", value="value", lanes=("count",))
        colors = _collections(ax)[0].get_facecolors()
        assert not np.allclose(colors[0], colors[1])
        assert len(ax.get_figure().axes) == 1

    def test_numeric_many_values_colorbar(self):
        """Past a handful of distinct numbers a colorbar is drawn."""
        n = 10
        frame = pd.DataFrame(
            {"start": np.arange(n) * 1.0, "end": np.arange(n) + 1.0, "v": range(n)}
        )
        ax = plot_lanes(frame, value="v")
        assert len(ax.get_figure().axes) == 2

    def test_numeric_lanes_get_their_own_colorbar(self):
        """Each numeric lane is its own scale, so each names its own bar."""
        n = 8
        frame = pd.DataFrame(
            {
                "lane": ["a"] * n + ["b"] * n,
                "start": list(range(n)) * 2,
                "end": [x + 1 for x in range(n)] * 2,
                "value": list(range(n)) + [100 + x for x in range(n)],
            }
        )
        ax = plot_lanes(frame, lane="lane", value="value")
        bars = [x for x in ax.get_figure().axes if x is not ax]
        assert [x.get_ylabel() for x in bars] == ["a", "b"]
        assert bars[0].get_ylim() == pytest.approx((0.0, 7.0))
        assert bars[1].get_ylim() == pytest.approx((100.0, 107.0))

    def test_one_value_is_not_no_value(self):
        """A lone number colors its rows without swallowing the missing one."""
        frame = pd.DataFrame(
            {"start": [0.0, 1.0, 2.0], "end": [1.0, 2.0, 3.0], "v": [5.0, None, 5.0]}
        )
        ax = plot_lanes(frame, value="v")
        colors = _collections(ax)[0].get_facecolors()
        assert np.allclose(colors[0], colors[2])
        assert not np.allclose(colors[0], colors[1])
        assert np.allclose(colors[1][:3], plt.matplotlib.colors.to_rgb(UNCOVERED_COLOR))

    def test_membership_has_one_key_in_a_mapping(self):
        """A color keyed on None applies however the dtype spelled it."""
        frame = pd.DataFrame({"start": [0.0, 1.0], "end": [1.0, 2.0], "v": ["a", None]})
        ax = plot_lanes(frame, value="v", color={None: "red", "a": "blue"})
        colors = _collections(ax)[0].get_facecolors()
        assert np.allclose(colors[0][:3], [0, 0, 1])
        assert np.allclose(colors[1][:3], [1, 0, 0])

    def test_numeric_one_value(self):
        """One number is not a scale, so every box shares one color."""
        frame = pd.DataFrame({"start": [0.0, 1.0], "end": [1.0, 2.0], "v": [4, 4]})
        ax = plot_lanes(frame, value="v")
        colors = _collections(ax)[0].get_facecolors()
        assert np.allclose(colors[0], colors[1])

    def test_color_string(self, string_frame):
        """A single color string paints every box, legend and all."""
        ax = plot_lanes(string_frame, lane="group", value="value", color="red")
        for collection in _collections(ax):
            assert np.allclose(collection.get_facecolors()[:, :3], [1, 0, 0])
        assert ax.get_legend() is None

    def test_color_numeric_cmap(self, kinds_frame):
        """For a numeric lane a color string names the colormap."""
        ax = plot_lanes(
            kinds_frame, lane="lane", value="value", lanes=("count",), color="Greys"
        )
        colors = _collections(ax)[0].get_facecolors()
        assert np.allclose(colors[:, 0], colors[:, 1])

    def test_color_mapping(self, string_frame):
        """A value->color mapping applies, and unmapped values are grey."""
        ax = plot_lanes(
            string_frame, lane="group", value="value", color={"north": "blue"}
        )
        zone = _collections(ax)[0].get_facecolors()
        assert np.allclose(zone[0][:3], [0, 0, 1])
        assert np.allclose(zone[1][:3], plt.matplotlib.colors.to_rgb(UNCOVERED_COLOR))
        assert [x.get_text() for x in ax.get_legend().get_texts()] == ["north"]

    def test_color_by_lane_mapping(self, string_frame):
        """A lane->mapping mapping colors each lane its own way."""
        color = {"zone": {"north": "blue"}, "missing": "red"}
        ax = plot_lanes(string_frame, lane="group", value="value", color=color)
        zone, other = _collections(ax)
        assert np.allclose(zone.get_facecolors()[0][:3], [0, 0, 1])
        # The lane the mapping does not name takes the default string colors.
        assert not np.allclose(other.get_facecolors()[0][:3], [0, 0, 1])

    def test_color_name_on_a_numeric_lane(self, kinds_frame):
        """A color which names no colormap is a color, not an error."""
        ax = plot_lanes(
            kinds_frame, lane="lane", value="value", lanes=("count",), color="red"
        )
        colors = _collections(ax)[0].get_facecolors()
        assert np.allclose(colors[:, :3], [1, 0, 0])

    def test_a_number_nobody_stated(self):
        """A missing number in a numeric lane is drawn, not made invisible."""
        n = 10
        values = [float(x) for x in range(n)]
        values[3] = float("nan")
        frame = pd.DataFrame(
            {"start": np.arange(n) * 1.0, "end": np.arange(n) + 1.0, "v": values}
        )
        ax = plot_lanes(frame, value="v")
        colors = _collections(ax)[0].get_facecolors()
        # It states no value, so it takes the color which says so rather
        # than the transparent one a colormap gives a NaN.
        assert colors[3][3] == pytest.approx(1.0)
        assert not np.allclose(colors[3], colors[0])

    def test_legend_off_suppresses_the_colorbar(self):
        """legend='off' means no colorbar either."""
        n = 10
        frame = pd.DataFrame(
            {"start": np.arange(n) * 1.0, "end": np.arange(n) + 1.0, "v": range(n)}
        )
        ax = plot_lanes(frame, value="v", legend="off")
        assert len(ax.get_figure().axes) == 1

    def test_vocabulary_widens_the_palette(self, string_frame):
        """A value the frame lacks still reserves its color."""
        partial = string_frame[string_frame["group"] == "zone"]
        alone = plot_lanes(partial, lane="group", value="value")
        plt.close("all")
        together = plot_lanes(partial, lane="group", value="value", vocabulary=["west"])
        # 'west' sorts after 'south', so reserving it must not move north.
        assert np.allclose(
            _collections(alone)[0].get_facecolors()[0],
            _collections(together)[0].get_facecolors()[0],
        )
        shifted = plot_lanes(partial, lane="group", value="value", vocabulary=["a"])
        assert not np.allclose(
            _collections(alone)[0].get_facecolors()[0],
            _collections(shifted)[0].get_facecolors()[0],
        )

    def test_no_two_values_share_a_color(self):
        """Past what the wheel holds a palette must widen, not repeat.

        A legend whose swatch means two things is worse than none.
        """
        names = [f"value {x:02d}" for x in range(len(WHEEL_ORDER) + 8)]
        frame = pd.DataFrame(
            {
                "start": np.arange(float(len(names))),
                "end": np.arange(float(len(names))) + 1.0,
                "v": names,
            }
        )
        ax = plot_lanes(frame, value="v")
        colors = {tuple(x) for x in _collections(ax)[0].get_facecolors()}
        assert len(colors) == len(names)

    def test_a_wide_palette_is_still_stable(self):
        """A value keeps its color whether or not the others are drawn."""
        names = [f"value {x:02d}" for x in range(len(WHEEL_ORDER) + 8)]
        frame = pd.DataFrame({"start": [0.0], "end": [1.0], "v": [names[0]]})
        alone = plot_lanes(frame, value="v", vocabulary=names)
        first = _collections(alone)[0].get_facecolors()[0]
        plt.close("all")
        whole = pd.DataFrame(
            {
                "start": np.arange(float(len(names))),
                "end": np.arange(float(len(names))) + 1.0,
                "v": names,
            }
        )
        together = plot_lanes(whole, value="v")
        assert np.allclose(first, _collections(together)[0].get_facecolors()[0])

    @pytest.mark.parametrize("length", range(4, 34, 3))
    def test_labels_decided_the_same_at_any_dpi(self, length):
        """Whether a label fits is a question about the figure, not its dpi.

        A width is swept because only a label near the edge of its box
        can be decided two ways, and every width is near some box's edge.
        """
        frame = pd.DataFrame({"start": [0.0], "end": [1.0], "v": ["x" * length]})
        drawn = []
        for dpi in (50, 100, 300):
            _, ax = plt.subplots(figsize=(2, 1), dpi=dpi)
            plot_lanes(frame, ax=ax, value="v")
            drawn.append((_texts(ax), [x.get_rotation() for x in ax.texts]))
            plt.close("all")
        # A renderer rounds each glyph to whole pixels, so measuring what
        # it drew is how the same figure saved at two resolutions keeps
        # different labels.
        assert len(set(map(str, drawn))) == 1

    @pytest.mark.parametrize("engine", [None, "constrained", "tight"])
    def test_a_tall_legend_stays_on_the_page(self, engine):
        """A column naming more than the axes is tall runs off the figure.

        Only a constrained layout keeps room for a legend outside the
        axes, so the other figures have to be given it explicitly.
        """
        names, frame = _many_values()
        figure, ax = plt.subplots(figsize=(8, 3), layout=engine)
        plot_lanes(frame, ax=ax, value="v")
        legend = _legend_of(figure, ax)
        figure.draw_without_rendering()
        box = legend.get_window_extent(figure.canvas.get_renderer())
        assert box.x0 >= 0 and box.x1 <= figure.bbox.width
        assert box.y0 >= 0 and box.y1 <= figure.bbox.height
        # Every value is still named; none was dropped to make it fit.
        assert len(legend.get_texts()) == len(names)

    def test_a_legend_below_stays_inside_the_axes_it_was_given(self):
        """A figure nobody laid out may hold other axes under this one.

        The legend is the axes' own, so it takes the axes' room rather
        than the space a neighbor below is sitting in.
        """
        _, frame = _many_values()
        figure, (ax, below) = plt.subplots(2, 1, figsize=(8, 6))
        before = ax.get_window_extent().frozen()
        plot_lanes(frame, ax=ax, value="v")
        figure.draw_without_rendering()
        box = _legend_of(figure, ax).get_window_extent(figure.canvas.get_renderer())
        # Under the lanes, clear of the neighbor, and neither the lanes
        # nor the legend pushed off the page to make room.
        assert box.y1 <= ax.get_window_extent().y0 + 1
        assert box.y0 >= below.get_window_extent().y1
        assert ax.get_position().y0 >= 0
        assert ax.get_window_extent().height >= before.height / 2 - 1

    def test_a_short_legend_stays_beside_them(self, string_frame):
        """Few enough values still read best in one column at the side."""
        ax = plot_lanes(string_frame, lane="group", value="value")
        assert ax.get_figure().legends == []
        box = ax.get_legend().get_window_extent()
        assert box.x0 >= ax.get_window_extent().x1

    def test_a_backend_which_renders_no_pixels(self, string_frame):
        """Not every canvas hands out a renderer when asked for one.

        A vector backend has none until it draws, so a figure bound for
        a pdf must be laid out without asking the canvas for one.
        """
        figure = Figure(figsize=(4, 3), layout="constrained")
        FigureCanvasPdf(figure)
        ax = figure.subplots()
        names, frame = _many_values()
        plot_lanes(frame, ax=ax, value="v")
        assert len(_legend_of(figure, ax).get_texts()) == len(names)

    def test_legend_off(self, string_frame):
        """legend=False draws none."""
        ax = plot_lanes(string_frame, lane="group", value="value", legend=False)
        assert ax.get_legend() is None


class TestSeparator:
    """The stroke which parts one box from the next."""

    @staticmethod
    def _edges(frame):
        """The edge color of each box, as the renderer settles it."""
        ax = plot_lanes(frame)
        ax.get_figure().canvas.draw()
        return _collections(ax)[0].get_edgecolor()

    def test_a_box_with_room_carries_the_separator(self):
        """Two boxes wide enough to be parted are parted by it."""
        frame = pd.DataFrame({"start": [0.0, 10.0], "end": [5.0, 15.0]})
        edges = self._edges(frame)
        assert np.allclose(edges, to_rgba_array(SEPARATOR_COLOR))

    def test_a_box_without_room_keeps_its_own_color(self):
        """A box thinner than the separator would be painted out by it.

        Drawing it as the separator says the interval is not there,
        which for a short gap between two long runs is a lie.
        """
        frame = pd.DataFrame({"start": [0.0, 5.0], "end": [5.0, 5.0 + 1e-6]})
        edges = self._edges(frame)
        faces = _collections(plot_lanes(frame))[0].get_facecolor()
        assert np.allclose(edges[0], to_rgba_array(SEPARATOR_COLOR))
        assert np.allclose(edges[1], faces[1])

    def test_recoloring_a_box_recolors_the_stroke(self):
        """A box edged in its own color is edged in the one it states now."""
        frame = pd.DataFrame({"start": [0.0, 5.0], "end": [5.0, 5.0 + 1e-6]})
        ax = plot_lanes(frame)
        boxes = _collections(ax)[0]
        # One color for every box, which is what a caller reaching for
        # the collection is most likely to set.
        boxes.set_facecolor("red")
        ax.get_figure().canvas.draw()
        assert np.allclose(boxes.get_edgecolor()[1], to_rgba_array("red"))

    def test_zooming_in_gives_a_box_its_separator_back(self):
        """The room a box has is the room the axis gives it, at each draw."""
        frame = pd.DataFrame({"start": [0.0, 5.0], "end": [5.0, 5.0 + 1e-6]})
        ax = plot_lanes(frame)
        ax.set_xlim(5.0 - 1e-7, 5.0 + 2e-6)
        ax.get_figure().canvas.draw()
        edges = _collections(ax)[0].get_edgecolor()
        assert np.allclose(edges[1], to_rgba_array(SEPARATOR_COLOR))
