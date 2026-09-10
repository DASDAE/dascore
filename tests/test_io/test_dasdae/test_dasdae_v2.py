"""Tests for DASDAE format version 2: coordinates stored as descriptions."""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import (
    CoordMonotonicArray,
    CoordRange,
    CoordSegmented,
    CoordString,
    concat_coords,
    get_coord,
)
from dascore.exceptions import ParameterError
from dascore.io.dasdae.utils import _read_coord, _save_coord

T0 = np.datetime64("2020-01-01T00:00:00")


@pytest.fixture(scope="module")
def hz_1024_patch():
    """The example patch resampled onto an exact 1024 Hz grid."""
    patch = dc.get_example_patch()
    time = get_coord(start=T0, step=(1, 1024), shape=(patch.shape[1],))
    return patch.update_coords(time=time)


@pytest.fixture(scope="module")
def gapped_patch():
    """A patch whose time coordinate holds two exact runs around a gap."""
    patch = dc.get_example_patch()
    t0 = patch.get_coord("time").min()
    first = patch.select(time=(None, t0 + np.timedelta64(1, "s")))
    second = patch.select(time=(t0 + np.timedelta64(1012, "ms"), None))
    spool = dc.spool([first, second]).chunk(time=None, tolerance=5, snap_coords=False)
    (out,) = spool
    assert isinstance(out.get_coord("time"), CoordSegmented)
    return out


CASES = {
    "fraction": get_coord(start=T0, step=(1, 1024), shape=(4096,)),
    "whole_ms": get_coord(start=T0, step=np.timedelta64(4, "ms"), shape=(100,)),
    "days": get_coord(start=T0, stop=T0 + np.timedelta64(30, "D"), step=(1, 1)),
    "int": get_coord(start=0, stop=100, step=1),
    "int_reversed": get_coord(start=100, stop=0, step=-1),
    "float32": get_coord(start=np.float32(0), stop=np.float32(10), step=0.1),
    "float_units": get_coord(start=0.0, stop=10.0, step=0.1, units="m"),
    "duration": get_coord(start=np.timedelta64(0, "s"), step=(3, 2), shape=(9,)),
    "segmented": concat_coords(
        get_coord(start=0.0, stop=5.0, step=1.0),
        get_coord(start=8.0, stop=13.0, step=1.0),
    ),
    "array": get_coord(data=np.array([1.0, 2.0, 4.5, 9.0]), units="m"),
    "strings": get_coord(data=np.array(["a", "b"])),
}


class TestNodeCodec:
    """Each coordinate kind survives its own node."""

    @pytest.fixture(scope="class")
    def h5(self):
        """An in-memory HDF5 file."""
        return h5py.File(io.BytesIO(), "w")

    @pytest.mark.parametrize("name", list(CASES))
    def test_round_trip(self, h5, name):
        """A coordinate reads back equal to what was written."""
        coord = CASES[name]
        _save_coord(coord, name, h5, compact=True)
        back = _read_coord(h5[name], name, {}, snap=True)
        assert back == coord
        assert back.dtype == coord.dtype

    def test_ranges_store_no_values(self, h5):
        """A range costs a description, however long it is."""
        coord = get_coord(start=T0, step=(1, 1024), shape=(10**9,))
        _save_coord(coord, "long", h5, compact=True)
        assert h5["long"].shape == (0,)
        back = _read_coord(h5["long"], "long", {}, snap=True)
        assert back == coord

    def test_segments_are_a_group(self, h5):
        """A segmented coordinate is a group holding one node per segment."""
        _save_coord(CASES["segmented"], "seg", h5, compact=True)
        assert isinstance(h5["seg"], h5py.Group)
        assert set(h5["seg"]) == {"0", "1"}

    def test_arrays_keep_their_values(self, h5):
        """An irregular coordinate still writes its values."""
        _save_coord(CASES["array"], "arr", h5, compact=True)
        assert h5["arr"].shape == (4,)

    def test_float_step_keeps_its_precision(self, h5):
        """A float64 step on a float32 start counts the same samples back."""
        coord = get_coord(
            start=np.float32(92.97747), step=0.49223355932786406, shape=(5_482_063,)
        )
        _save_coord(coord, "mixed", h5, compact=True)
        back = _read_coord(h5["mixed"], "mixed", {}, snap=True)
        assert len(back) == len(coord)
        assert back == coord


class TestVersion2Files:
    """Whole-file behavior of the default (version 2) writer."""

    def test_default_version_is_2(self, random_patch, tmp_path):
        """A plain DASDAE write produces a version 2 file."""
        path = dc.write(random_patch, tmp_path / "v2.h5", "dasdae")
        assert dc.get_format(path) == ("DASDAE", "2")

    def test_version_1_still_writes(self, random_patch, tmp_path):
        """Version 1 remains available by request."""
        path = dc.write(random_patch, tmp_path / "v1.h5", "dasdae", file_version="1")
        assert dc.get_format(path) == ("DASDAE", "1")
        assert dc.read(path)[0] == random_patch

    def test_fraction_round_trip(self, hz_1024_patch, tmp_path):
        """A fractional step reads and scans back exactly."""
        path = dc.write(hz_1024_patch, tmp_path / "hz.h5", "dasdae")
        back = dc.read(path)[0]
        assert back.get_coord("time") == hz_1024_patch.get_coord("time")
        assert back == hz_1024_patch
        summary = dc.scan(path)[0].coords["time"]
        assert summary.to_coord() == hz_1024_patch.get_coord("time")

    def test_gapped_patch_written_whole(self, gapped_patch, tmp_path):
        """A gapped patch is stored as one patch with its segments."""
        path = dc.write(gapped_patch, tmp_path / "gap.h5", "dasdae")
        (back,) = dc.read(path)
        assert isinstance(back.get_coord("time"), CoordSegmented)
        assert back.get_coord("time") == gapped_patch.get_coord("time")
        assert np.array_equal(back.data, gapped_patch.data)
        (scanned,) = dc.scan(path)
        summary = scanned.coords["time"]
        assert summary.step is None
        assert summary.fingerprint == gapped_patch.get_coord("time").fingerprint()

    def test_split_still_honored(self, gapped_patch, tmp_path):
        """An explicit split writes each run as its own patch."""
        path = dc.write(gapped_patch, tmp_path / "split.h5", "dasdae", split=True)
        spool = dc.spool(path)
        assert len(spool) == 2
        for patch in spool:
            assert isinstance(patch.get_coord("time"), CoordRange)

    def test_gapped_file_still_guarded(self, gapped_patch, tmp_path):
        """A gapped patch read from a file is guarded like one in memory."""
        spool = dc.spool(dc.write(gapped_patch, tmp_path / "gap.h5", "dasdae"))
        with pytest.raises(ParameterError, match="split=True"):
            dc.write(spool, tmp_path / "v1.h5", "dasdae", file_version="1")
        path = dc.write(spool, tmp_path / "split.h5", "dasdae", split=True)
        assert len(dc.spool(path)) == 2

    def test_append_keeps_the_higher_version(self, random_patch, tmp_path):
        """Appending version 1 patches to a version 2 file leaves it version 2."""
        path = dc.write(random_patch, tmp_path / "both.h5", "dasdae")
        second = random_patch.update_coords(
            time_min=random_patch.get_coord("time").max()
        )
        dc.write(second, path, "dasdae", file_version="1")
        assert dc.get_format(path) == ("DASDAE", "2")
        assert len(dc.read(path)) == 2

    def test_array_coordinates(self, random_patch, tmp_path):
        """Irregular and string coordinates round trip as arrays."""
        n = len(random_patch.get_coord("distance"))
        uneven = np.sort(np.random.default_rng(0).random(n)) * 100
        patch = random_patch.update_coords(
            distance=uneven, tag=("distance", np.array(["a"] * n))
        )
        path = dc.write(patch, tmp_path / "arr.h5", "dasdae")
        back = dc.read(path)[0]
        assert isinstance(back.get_coord("distance"), CoordMonotonicArray)
        assert isinstance(back.get_coord("tag"), CoordString)
        assert back == patch

    def test_lazy_array_sizes_by_grid(self, tmp_path):
        """The lazy xarray view of a long fractional grid counts its samples."""
        pytest.importorskip("xarray")
        time = get_coord(start=T0, step=(1, 1024), shape=(2_500_000,))
        data = np.zeros((1, len(time)), dtype=np.float32)
        coords = {"distance": [0.0], "time": time}
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "time"))
        spool = dc.spool(dc.write(patch, tmp_path / "long.h5", "dasdae"))
        tree = spool.io.to_xarray()
        leaf = next(node for node in tree.subtree if "data" in node.dataset)
        assert leaf["data"].shape == data.shape
        assert leaf["data"].data.compute().shape == data.shape
