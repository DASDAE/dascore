"""Tests for drawing an annotation set."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Polygon, Rectangle

import dascore as dc
from dascore.exceptions import ParameterError

DIMS = ("distance", "time")
T0 = np.datetime64("2020-01-01T00:00:00", "ns")
SECOND = np.timedelta64(1, "s")


def _set(frame, dims=DIMS) -> dc.AnnotationSet:
    """A set from a dict of columns."""
    return dc.AnnotationSet(pd.DataFrame(frame), dims=dims)


def _legend(ax) -> list[str]:
    """The legend entries of an axis."""
    legend = ax.get_legend()
    return [] if legend is None else [x.get_text() for x in legend.get_texts()]


@pytest.fixture(scope="module")
def shapes():
    """A path, a polygon with a hole, and a curve-only path."""
    moveout = {
        "object_type": "Moveout",
        "apex_distance": 50.0,
        "apex_time": T0,
        "velocity": 100.0,
        "distance_min": 0.0,
        "distance_max": 100.0,
    }
    outer = {"time": T0 + SECOND * np.arange(3), "distance": [0.0, 0.0, 50.0]}
    hole = {"time": T0 + SECOND * np.array([0, 1, 1]), "distance": [1.0, 1.0, 9.0]}
    empty = dc.AnnotationSet(dims=DIMS, bases={"m": moveout})
    times, speeds = T0 + SECOND * np.arange(2), [2.0, 2.5]
    out = empty.add_path("car", time=times, distance=[0, 9.0], speed=speeds)
    out = out.add_path("wave", basis="m")
    return out.add_polygon("noise", rings=[outer, hole])


class TestDrawing:
    """Each spelling draws its artist."""

    def test_markers(self):
        """A value on both axes is a marker, a group member's too; style reaches it."""
        frame = {"time": [1.0, 2.0], "distance": [5.0, 6.0], "feature_id": [None, "e"]}
        lines = _set(frame).viz.plot(linewidth=7).lines
        assert {(x.get_marker(), x.get_linestyle()) for x in lines} == {("o", "None")}
        assert [x.get_linewidth() for x in lines] == [7, 7]

    def test_lines_and_spans(self):
        """With the other axis spanned, a value is a line and a range a span."""
        rows = [{"time": 1.0}, {"distance": 5.0}]
        rows += [{"time_min": 1.0, "time_max": 2.0}]
        rows += [{"distance_min": 3.0, "distance_max": 4.0}]
        ax = _set(rows).viz.plot()
        assert sorted(tuple(x.get_xdata()) for x in ax.lines) == [(0, 1), (1, 1)]
        assert len(ax.patches) == 2

    def test_box(self):
        """Two ranges are a rectangle."""
        frame = {"time_min": [1.0], "time_max": [2.0]}
        frame |= {"distance_min": [3.0], "distance_max": [5.0]}
        (box,) = _set(frame).viz.plot().patches
        assert isinstance(box, Rectangle)
        assert (box.get_x(), box.get_y(), box.get_width()) == (1.0, 3.0, 1.0)

    def test_segment(self):
        """A value and a range are a segment."""
        frame = {"time": [1.0], "distance_min": [3.0], "distance_max": [5.0]}
        (line,) = _set(frame).viz.plot().lines
        assert list(line.get_ydata()) == [3.0, 5.0]

    def test_paths(self, shapes):
        """A path is a polyline, and a curve-only path is its sampled curve."""
        lengths = sorted(len(x.get_xdata()) for x in shapes.viz.plot().lines)
        assert lengths == [2, 64]

    def test_polygon(self, shapes):
        """A polygon is filled; its hole is an outline."""
        polygons = [x for x in shapes.viz.plot().patches if isinstance(x, Polygon)]
        assert [x.get_fill() for x in polygons] == [True, False]


class TestAxes:
    """Which dimension goes where."""

    def test_default(self):
        """The last dimension is x, the first y, as waterfall has them."""
        ax = _set({"time": [1.0], "distance": [5.0]}).viz.plot()
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time", "Distance")
        assert list(ax.lines[0].get_xdata()) == [1.0]

    def test_explicit(self):
        """Named axes swap the dimensions."""
        ax = _set({"time": [1.0], "distance": [5.0]}).viz.plot(x="distance")
        assert list(ax.lines[0].get_xdata()) == [5.0]
        assert ax.get_ylabel() == "Time"

    def test_datetime(self):
        """A datetime axis is converted and formatted as dates; on y it runs down."""
        ann = _set({"time": [T0], "distance": [5.0]})
        ax = ann.viz.plot()
        formatter = ax.xaxis.get_major_formatter()
        assert isinstance(formatter, mdates.ConciseDateFormatter)
        assert ax.lines[0].get_xdata()[0] == mdates.date2num(T0)
        assert ann.viz.plot(y="time").yaxis_inverted()

    def test_one_dim(self):
        """A one-dimensional set draws along x only."""
        ax = _set({"time": [1.0, 2.0]}, dims=("time",)).viz.plot()
        assert [list(x.get_xdata()) for x in ax.lines] == [[1, 1], [2, 2]]

    @pytest.mark.parametrize(
        ("axes", "match"),
        [({"x": "depth"}, "not a dimension"), ({"y": "time"}, "both")],
    )
    def test_refused(self, axes, match):
        """An undeclared name, or one dimension on both axes, is refused."""
        with pytest.raises(ParameterError, match=match):
            _set({"time": [1.0]}).viz.plot(**{"x": "time", **axes})

    def test_many_dims(self):
        """A set of three dimensions names both; a row on neither is skipped."""
        frame = {"time": [1.0, None], "depth": [None, 2.0]}
        ann = _set(frame, dims=("distance", "time", "depth"))
        with pytest.raises(ParameterError, match="x and y named"):
            ann.viz.plot()
        assert len(ann.viz.plot(x="time", y="distance").lines) == 1


class TestColor:
    """Color and labels from columns."""

    def test_by_column(self):
        """One color per value, and a legend entry each."""
        frame = {"time": [1.0, 2.0, 3.0], "distance": [1.0, 2.0, 3.0]}
        ax = _set(frame | {"phase": ["P", "S", "P"]}).viz.plot(color="phase")
        colors = [x.get_color() for x in ax.lines]
        assert colors[0] == colors[2] != colors[1]
        assert _legend(ax) == ["P", "S"]

    def test_feature_column(self, shapes):
        """A features column colors its features."""
        ann = shapes.update(feature="car", vehicle="road")
        assert _legend(ann.viz.plot(color="vehicle")) == ["road"]

    def test_member_column(self, shapes):
        """An ordered feature takes its first member's value of a row column."""
        ann = shapes.add(pd.DataFrame({"time": [T0], "speed": [3.0]}))
        assert _legend(ann.viz.plot(color="speed")) == ["2.0", "3.0"]

    def test_plain_color(self):
        """A matplotlib color colors everything, with no legend."""
        ax = _set({"time": [1.0], "distance": [5.0]}).viz.plot(color="red")
        assert ax.lines[0].get_color() == "red"
        assert ax.get_legend() is None

    def test_label(self):
        """A label column gives one entry per value, blanks left out."""
        frame = {"time": [1.0, 2.0, 3.0], "distance": [1.0, 2.0, 3.0]}
        ann = _set(frame | {"note": ["a", "a", None]})
        assert _legend(ann.viz.plot(label="note")) == ["a"]
        with pytest.raises(ParameterError, match="neither"):
            ann.viz.plot(label="nope")
