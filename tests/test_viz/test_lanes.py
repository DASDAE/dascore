"""Tests for the interval-lane renderer."""

from __future__ import annotations

import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PatchCollection

from dascore.exceptions import ParameterError
from dascore.viz._lanes import UNCOVERED_COLOR, _pack_rows, plot_lanes


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
        """Numbers state themselves; membership rows draw no text."""
        ax = plot_lanes(kinds_frame, lane="lane", value="value")
        assert sorted(_texts(ax)) == ["1", "2", "a", "b"]

    def test_x_label_and_show(self, monkeypatch):
        """x_label is applied and show calls plt.show."""
        called = []
        monkeypatch.setattr(plt, "show", lambda: called.append(True))
        ax = plot_lanes({"start": [0.0], "end": [1.0]}, x_label="Time", show=True)
        assert ax.get_xlabel() == "Time"
        assert called


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
        """Every row takes the lane's one color; the legend names the lane."""
        ax = plot_lanes(kinds_frame, lane="lane", value="value", lanes=("flag",))
        colors = _collections(ax)[0].get_facecolors()
        assert colors[0][3] == pytest.approx(1.0)
        assert colors[1][3] == pytest.approx(1.0)
        assert np.allclose(colors[0], colors[1])
        assert [x.get_text() for x in ax.get_legend().get_texts()] == ["flag"]

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

    def test_a_value_which_is_not_a_value(self):
        """A lane value of NaN is refused, the way the model refuses it."""
        n = 10
        values = [float(x) for x in range(n)]
        values[3] = float("nan")
        frame = pd.DataFrame(
            {"start": np.arange(n) * 1.0, "end": np.arange(n) + 1.0, "v": values}
        )
        with pytest.raises(ParameterError, match="must be finite"):
            plot_lanes(frame, value="v")

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

    def test_labels_decided_the_same_at_any_dpi(self):
        """Whether a label fits is a question about the figure, not its dpi."""
        frame = pd.DataFrame(
            {"start": [0.0], "end": [1.0], "v": ["a rather long label"]}
        )
        drawn = []
        for dpi in (50, 200):
            _, ax = plt.subplots(figsize=(2, 1), dpi=dpi)
            plot_lanes(frame, ax=ax, value="v")
            drawn.append(_texts(ax))
            plt.close("all")
        # Measuring text in points against a box in pixels answers this
        # differently at each dpi, which is how the same figure saved at
        # two resolutions loses its labels.
        assert drawn[0] == drawn[1]

    def test_legend_off(self, string_frame):
        """legend=False draws none."""
        ax = plot_lanes(string_frame, lane="group", value="value", legend=False)
        assert ax.get_legend() is None
