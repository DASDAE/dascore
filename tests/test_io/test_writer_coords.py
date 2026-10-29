"""
Every writer keeps a coordinate's step, flags and holes, or refuses.

Rule 13 of the coordinate guidelines (`notes/coordinates.qmd`): a writer
never silently splits, stretches, fills or drops a step; a format which
cannot store holes or jitter raises, saying to regularize or split first.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import CoordError, MissingOptionalDependencyError
from dascore.io import FiberIO
from dascore.io.dasdae import DASDAEV1, DASDAEV2
from dascore.io.netcdf import NetCDFCFV18
from dascore.io.pickle import PickleIO
from dascore.io.prodml import ProdMLV2_1
from dascore.io.rsf import RSFV1
from dascore.io.segy.core import SegyV0_0, SegyV0_100, SegyV1_0, SegyV2_0, SegyV2_1
from dascore.io.wav import WavIO
from dascore.io.zarr import ZarrV2, ZarrV3
from dascore.utils.misc import suppress_warnings

# Every writer DASCore ships, checked against the plugin manager below.
WRITERS = [
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
EXACT = {"DASDAE 2", "PICKLE ", "NETCDF_CF 1.8", "ZARR 2", "ZARR 3"}
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


def _stored(fio, path):
    """What the file holds, as metadata, read without snapping."""
    if fio.name.upper() == "PICKLE":  # the stored object itself
        with path.open("rb") as handle:
            stored = pickle.load(handle)
        return [stored] if isinstance(stored, dc.Patch) else list(stored)
    return fio.get_metadata(path, snap=False)


class TestWriterCoordinates:
    """Each writer keeps a coordinate's step, flags and holes, or refuses."""

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
            or (_label(fio) == "DASDAE 1" and case == "stepless")
            or (time_only and case == "distance gaps")
        )
        args = (patch, path, fio.name, fio.version)
        if not keeps:
            with pytest.raises(CoordError, match="snap_coords"):
                dc.write(*args)
            return
        try:
            with suppress_warnings(UserWarning):  # SEGY's lost distance
                dc.write(*args)
        except MissingOptionalDependencyError:
            pytest.skip(f"{fio.name} needs an optional dependency")
        if fio.name.upper() in {"WAV", "RSF"}:  # no reader
            return
        (meta,) = _stored(fio, path)
        for dim in ["time"] if time_only else patch.dims:
            expected, got = patch.get_coord(dim), meta.get_coord(dim)
            assert got.step == expected.step
            assert (got.jittered, got.gapped) == (expected.jittered, expected.gapped)
            assert np.array_equal(got.values, expected.values)

    @pytest.mark.parametrize("fio", WRITERS, ids=_label)
    def test_refusal_names_split_gaps(self, fio, cases, tmp_path):
        """A format refusing holes also offers splitting at them."""
        if _label(fio) in EXACT:
            return
        path = tmp_path / NAMES.get(fio.name.upper(), "out.h5")
        with pytest.raises(CoordError, match="split_gaps"):
            dc.write(cases["time gaps"], path, fio.name, fio.version)

    @pytest.mark.parametrize("fio", [NetCDFCFV18(), ZarrV3()], ids=_label)
    def test_fractional_step_with_holes_refused(self, fio, tmp_path):
        """A 1024 Hz grid with a hole has no whole-tick step to store."""
        start = np.datetime64("2020-01-01T00:00:00", "ns")
        time = dc.get_coord(start=start, step=(1, 1024), shape=(2000,))
        patch = dc.Patch(data=np.zeros(2000), coords={"time": time}, dims=("time",))
        holed = patch.unselect(time=(500, 501), samples=True)
        path = tmp_path / NAMES.get(fio.name.upper(), "out.h5")
        with pytest.raises(CoordError, match="NetCDF and Zarr"):
            dc.write(holed, path, fio.name, fio.version)
