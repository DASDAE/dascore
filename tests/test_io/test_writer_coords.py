"""
Every writer keeps a coordinate's step and segments, or refuses.

Rule 13 of the coordinate guidelines (`notes/coordinates.qmd`): a writer
never silently splits, stretches, fills or drops a step; a format which
cannot store segments or labels raises, saying to regularize or split first.
"""

from __future__ import annotations

import pickle
from contextlib import contextmanager, nullcontext

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import CoordError, MissingOptionalDependencyError
from dascore.io import FiberIO
from dascore.io.arrow import ArrowV1
from dascore.io.dasdae import DASDAEV1, DASDAEV2
from dascore.io.netcdf import NetCDFCFV18
from dascore.io.pickle import PickleIO
from dascore.io.prodml import ProdMLV2_1
from dascore.io.rsf import RSFV1
from dascore.io.segy.core import SegyV0_0, SegyV0_100, SegyV1_0, SegyV2_0, SegyV2_1
from dascore.io.wav import WavIO
from dascore.io.zarr import ZarrV2, ZarrV3
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning
from tests.conftest import segment_layout

# Every writer DASCore ships, checked against the plugin manager below.
WRITERS = [
    ArrowV1(),
    DASDAEV1(),
    DASDAEV2(),
    NetCDFCFV18(),
    PickleIO(),
    ProdMLV2_1(),
    RSFV1(),
    SegyV0_0(),
    SegyV0_100(),
    SegyV1_0(),
    SegyV2_0(),
    SegyV2_1(),
    WavIO(),
    ZarrV2(),
    ZarrV3(),
]
# Formats which store any coordinate's labels with its step.
EXACT = {"ARROW 1", "DASDAE 2", "PICKLE ", "NETCDF_CF 1.8", "ZARR 2", "ZARR 3"}
# File names, for formats which care.
NAMES = {"WAV": "out.wav", "RSF": "out.rsf", "ZARR": "out.zarr"}


def _label(fio) -> str:
    """The writer's name and version."""
    return f"{fio.name.upper()} {fio.version}"


@pytest.fixture(scope="module")
def cases():
    """Patches whose time or distance is even, gapped, jittered or stepless."""
    patch = dc.get_example_patch()
    time = patch.get_coord("time")
    step = time.step
    ticks = np.random.default_rng(0).uniform(-0.2, 0.2, len(time)) * 4e6
    jittered = dc.get_coord(data=time.values + ticks.astype("m8[ns]"), step=step)
    stepless = time.values.copy()
    stepless[len(stepless) // 2 :] += 3 * step // 4
    return {
        "even": patch,
        "time gaps": patch.unselect(time=(100, 120), samples=True),
        "jittered": patch.update_coords(time=jittered),
        "stepless": patch.update_coords(time=dc.get_coord(data=stepless, snap=False)),
        "distance gaps": patch.unselect(distance=(10, 20), samples=True),
        "one channel": patch.select(distance=(0, 1), samples=True),
    }


@contextmanager
def _skip_without_dependency(fio):
    """Skip a writer whose optional dependency is not installed."""
    try:
        yield
    except MissingOptionalDependencyError:
        pytest.skip(f"{fio.name} needs an optional dependency")


def _stored(fio, path):
    """What the file holds, as metadata, read without snapping."""
    if fio.name.upper() == "PICKLE":  # the stored object itself
        with path.open("rb") as handle:
            stored = pickle.load(handle)
        return [stored] if isinstance(stored, dc.Patch) else list(stored)
    return fio.get_metadata(path, snap=False)


class TestWriterCoordinates:
    """Each writer keeps a coordinate's step and segments, or refuses."""

    def test_every_writer_listed(self):
        """The list covers every writer DASCore registers."""
        FiberIO.manager.load_plugins()
        registered = {
            _label(fio)
            for versions in FiberIO.manager._format_version.values()
            for fio in versions.values()
            if fio.implements_write and type(fio).__module__.startswith("dascore.io")
        }
        assert registered == {_label(x) for x in WRITERS}

    @pytest.mark.parametrize("fio", WRITERS, ids=_label)
    @pytest.mark.parametrize(
        "case",
        ["even", "one channel", "time gaps", "jittered", "stepless", "distance gaps"],
    )
    def test_keeps_or_refuses(self, fio, case, cases, tmp_path):
        """A writer stores the coordinate as it is, or says how to fix it."""
        patch = cases[case]
        path = tmp_path / NAMES.get(fio.name.upper(), "out.h5")
        # SEGY writes channel numbers, and WAV no distance grid at all.
        time_only = fio.name.upper() in {"SEGY", "WAV"}
        if case == "one channel" and fio.name.upper() == "SEGY":
            pytest.skip("SEGY states a channel spacing, which one channel lacks")
        keeps = (
            case in {"even", "one channel"}
            or _label(fio) in EXACT
            # jittered labels state no step, as stepless ones do
            or (_label(fio) == "DASDAE 1" and case in {"stepless", "jittered"})
            or (time_only and case == "distance gaps")
        )
        args = (patch, path, fio.name, fio.version)
        if not keeps:
            with _skip_without_dependency(fio):
                with pytest.raises(CoordError, match="snap_coords"):
                    dc.write(*args)
            return
        with _skip_without_dependency(fio), suppress_warnings(UserWarning):
            dc.write(*args)  # SEGY warns its distance is lost
        if fio.name.upper() == "WAV":  # no reader
            return
        (meta,) = _stored(fio, path)
        for dim in ["time"] if time_only else patch.dims:
            expected, got = patch.get_coord(dim), meta.get_coord(dim)
            assert got.step == expected.step
            assert segment_layout(got) == segment_layout(expected)
            assert np.array_equal(got.values, expected.values)

    @pytest.mark.parametrize("fio", WRITERS, ids=_label)
    def test_refusal_names_split_gaps(self, fio, cases, tmp_path):
        """A format refusing holes also offers splitting at them."""
        if _label(fio) in EXACT:
            return
        path = tmp_path / NAMES.get(fio.name.upper(), "out.h5")
        with _skip_without_dependency(fio):
            with pytest.raises(CoordError, match="split_gaps"):
                dc.write(cases["time gaps"], path, fio.name, fio.version)

    @pytest.mark.parametrize("size", [2_000, 2_100_000])
    @pytest.mark.parametrize("fio", [NetCDFCFV18(), ZarrV3()], ids=_label)
    def test_fractional_step_with_holes_refused(self, fio, size, tmp_path):
        """A 1024 Hz grid with a hole has no whole-tick step to store."""
        start = np.datetime64("2020-01-01T00:00:00", "ns")
        time = dc.get_coord(start=start, step=(1, 1024), shape=(size,))
        data = np.zeros(size, dtype=np.int8)
        patch = dc.Patch(data=data, coords={"time": time}, dims=("time",))
        holed = patch.unselect(time=(500, 501), samples=True)
        path = tmp_path / NAMES.get(fio.name.upper(), "out.h5")
        with _skip_without_dependency(fio):
            with pytest.raises(CoordError, match="NetCDF and Zarr"):
                dc.write(holed, path, fio.name, fio.version)

    @pytest.mark.parametrize("fio", [NetCDFCFV18(), ZarrV3()], ids=_label)
    def test_segments_of_two_steps_refused(self, fio, tmp_path):
        """Segments sharing no step have none to store beside their labels."""
        first = dc.get_coord(start=0.0, step=1.0, shape=(4,))
        second = dc.get_coord(start=10.0, step=2.0, shape=(3,))
        coord = dc.core.coords.concat_coords(first, second)
        data = np.zeros(len(coord))
        patch = dc.Patch(data=data, coords={"distance": coord}, dims=("distance",))
        path = tmp_path / NAMES.get(fio.name.upper(), "out.h5")
        with _skip_without_dependency(fio):
            with pytest.raises(CoordError, match=r"2 segments.*split_gaps"):
                dc.write(patch, path, fio.name, fio.version)

    @pytest.mark.parametrize("second_start", [6.5, 5.3])
    @pytest.mark.parametrize("fio", [NetCDFCFV18(), ZarrV3()], ids=_label)
    def test_seam_off_the_grid_refused(self, fio, second_start, tmp_path):
        """Segments on one step whose seam misses its grid have none to store."""
        first = dc.get_coord(start=0.0, step=1.0, shape=(5,))
        second = dc.get_coord(start=second_start, step=1.0, shape=(5,))
        coord = dc.core.coords.concat_coords(first, second)
        data = np.zeros(len(coord))
        patch = dc.Patch(data=data, coords={"distance": coord}, dims=("distance",))
        path = tmp_path / NAMES.get(fio.name.upper(), "out.h5")
        with _skip_without_dependency(fio):
            with pytest.raises(CoordError, match=r"2 segments.*split_gaps"):
                dc.write(patch, path, fio.name, fio.version)


class TestDASDAESegments:
    """DASDAE keeps a coordinate's segments and their steps."""

    @pytest.mark.parametrize("start", [0.0, 12.345, 1.7e9, -3.3])
    @pytest.mark.parametrize("step", [1.0209523, 0.1, 0.001, 1 / 3])
    def test_float_segments_keep_steps(self, start, step, tmp_path):
        """Float segments read back as the same segments on the same step."""
        full = dc.get_coord(start=start, step=step, shape=(50,))
        keep = np.ones(50, dtype=bool)
        keep[14:18] = False
        coords = {"distance": full, "time": np.arange(2.0)}
        patch = dc.Patch(data=np.zeros((50, 2)), coords=coords, dims=tuple(coords))
        patch = patch.select(distance=keep)
        patch.io.write(tmp_path / "out.h5", "dasdae")
        back = dc.read(tmp_path / "out.h5")[0].get_coord("distance")
        segments = patch.get_coord("distance").segments
        assert [(len(x), x.step) for x in back.segments] == [
            (len(x), x.step) for x in segments
        ]
        assert back == patch.get_coord("distance")

    def test_single_sample_segments_keep_steps(self, tmp_path):
        """Lone samples between holes read back on their segments' step."""
        start = np.datetime64("2020-01-01", "ns")
        full = dc.get_coord(start=start, step=np.timedelta64(1, "ms"), shape=(12,))
        keep = np.isin(np.arange(12), [0, 1, 4, 8, 9])
        patch = dc.Patch(data=np.zeros(12), coords={"time": full}, dims=("time",))
        patch = patch.select(time=keep)
        patch.io.write(tmp_path / "out.h5", "dasdae")
        back = dc.read(tmp_path / "out.h5")[0].get_coord("time")
        assert [len(x) for x in back.segments] == [2, 1, 2]
        assert [x.step for x in back.segments] == [np.timedelta64(1, "ms")] * 3


# each label-list format's file name and the module it writes with
LIST_FORMATS = {
    "DASDAE": ("out.h5", "h5py"),
    "NETCDF_CF": ("out.nc", "h5netcdf"),
    "ZARR": ("out.zarr", "zarr"),
}


class TestManySegments:
    """Heavily masked and long concatenated coordinates write, or refuse."""

    @staticmethod
    def _patch(stretches, keep=2, period=4):
        """A patch masked to `stretches` stretches of `keep` samples."""
        size = period * stretches
        start = np.datetime64("2020-01-01", "ns")
        time = dc.get_coord(start=start, step=np.timedelta64(1, "ms"), shape=(size,))
        patch = dc.Patch(data=np.zeros(size), coords={"time": time}, dims=("time",))
        return patch.select(time=np.arange(size) % period < keep)

    @pytest.mark.parametrize("name", list(LIST_FORMATS))
    def test_masked_round_trip(self, name, tmp_path):
        """Past the segment limit a mask leaves one list, which every one writes."""
        file_name, module = LIST_FORMATS[name]
        pytest.importorskip(module)
        patch = self._patch(300)
        patch.io.write(tmp_path / file_name, name)
        back = dc.read(tmp_path / file_name)[0].get_coord("time")
        assert back.runs_count == 1 and back.step is None
        assert back == patch.get_coord("time")

    @pytest.mark.parametrize("name", list(LIST_FORMATS))
    def test_concatenated_segments_round_trip(self, name, tmp_path):
        """
        1,500 segments on one step write; they read back as the same
        segments up to max_index_runs, else as one list of the same labels.
        """
        file_name, module = LIST_FORMATS[name]
        pytest.importorskip(module)
        with dc.config_context(max_index_runs=2_000):
            patch = self._patch(1_500, keep=10, period=13)
        patch.io.write(tmp_path / file_name, name)
        coord = patch.get_coord("time")
        # DASDAE stores the segments; the others state one step beside labels
        dasdae = name == "DASDAE"
        warns = pytest.warns(DASCoreWarning, match="1500 segments")
        with nullcontext() if dasdae else warns:
            back = dc.read(tmp_path / file_name)[0].get_coord("time")
        assert back.runs_count == (1_500 if dasdae else 1) and back == coord
        with dc.config_context(max_index_runs=2_000):
            back = dc.read(tmp_path / file_name)[0].get_coord("time")
        assert back.runs_count == 1_500 and back == coord
        assert segment_layout(back) == segment_layout(coord)
        assert {x.step for x in back.segments} == {coord.segments[0].step}

    def test_dasdae_records_up_to_its_segment_limit(self, tmp_path):
        """8,000 segments write and read back; one more is refused, saying why."""
        with dc.config_context(max_index_runs=10_000):
            most, patch = (self._patch(x, keep=2, period=3) for x in (8_000, 8_001))
        assert most.get_coord("time").runs_count == 8_000
        most.io.write(tmp_path / "most.h5", "DASDAE")
        assert dc.read(tmp_path / "most.h5")[0].get_coord("time") == most.get_coord(
            "time"
        )
        with pytest.raises(
            CoordError, match=r"8,000 segments: it has 8001 segments.*split_gaps"
        ):
            patch.io.write(tmp_path / "out.h5", "DASDAE")
