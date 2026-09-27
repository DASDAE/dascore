"""Tests for DASDAE format version 2: coordinates stored as descriptions."""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import (
    CoordString,
    concat_coords,
    get_coord,
)
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
    assert out.get_coord("time").runs_count > 1
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

    def test_every_node_states_its_class(self, h5):
        """Each node names its coordinate class in plain typed attributes."""
        for name in ("fraction", "array"):
            _save_coord(CASES[name], f"typed_{name}", h5, compact=True)
        assert h5["typed_fraction"].attrs["object_type"] == "CoordRange"
        assert h5["typed_array"].attrs["object_type"] == "NumericCoord"
        attrs = dict(h5["typed_fraction"].attrs)
        assert attrs["step_denominator"] == 2 and attrs["length"] == 4096

    def test_extended_float_range_keeps_its_values(self, h5):
        """A long-double range exceeds a JSON double, so its values are stored."""
        start = np.nextafter(np.longdouble(1), np.longdouble(2))
        coord = get_coord(start=start, step=np.longdouble("0.1"), shape=(10,))
        _save_coord(coord, "wide", h5, compact=True)
        assert h5["wide"].shape == (10,)
        back = _read_coord(h5["wide"], "wide", {}, snap=True)
        assert back.dtype == coord.dtype
        assert np.array_equal(back.values, coord.values)

    def test_arrays_keep_their_values(self, h5):
        """An irregular coordinate still writes its values."""
        _save_coord(CASES["array"], "arr", h5, compact=True)
        assert h5["arr"].shape == (4,)

    @pytest.mark.parametrize(
        "indexer",
        [slice(3, 73), slice(2, None, 3), slice(None, None, -1), slice(-1, None)],
    )
    def test_float_slice_round_trip(self, h5, indexer):
        """Compact storage retains the parent's rounding and the integer window."""
        parent = get_coord(start=0.1, step=0.1, shape=(100,))
        coord = parent[indexer]
        name = f"slice_{indexer.start}_{indexer.stop}_{indexer.step}"
        _save_coord(coord, name, h5, compact=True)
        assert h5[name].shape == (0,)
        back = _read_coord(h5[name], name, {}, snap=True)
        assert back.values.tobytes() == parent.values[indexer].tobytes()
        assert back.data_id == coord.data_id

    @pytest.mark.parametrize("step", [0.1, np.float32(0.1), np.float64(0.1)])
    def test_float_window_step_precision(self, h5, step):
        """The scalar step's precision is part of the original expression."""
        parent = get_coord(start=np.float64(0.1), step=step, shape=(100,))
        coord = parent[89:2:-2]
        name = f"window_{type(step).__name__}"
        _save_coord(coord, name, h5, compact=True)
        back = _read_coord(h5[name], name, {}, snap=True)
        assert back.values.tobytes() == parent.values[89:2:-2].tobytes()
        assert back.data_id == coord.data_id

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

    def test_gapped_patch_written_as_pieces(self, gapped_patch, tmp_path):
        """A gapped patch is written as its contiguous pieces, which read back."""
        path = dc.write(gapped_patch, tmp_path / "gap.h5", "dasdae")
        pieces = list(dc.spool([gapped_patch]))
        back = sorted(dc.spool(path), key=lambda x: x.get_coord("time").min())
        assert len(back) == len(pieces) > 1
        for read, piece in zip(back, pieces, strict=True):
            assert read.get_coord("time").evenly_sampled
            assert read.get_coord("time") == piece.get_coord("time")
            assert np.array_equal(read.data, piece.data)

    def test_gapped_directory_to_xarray(self, gapped_patch, tmp_path):
        """A directory holding a written gapped patch converts (#1218)."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")
        dc.write(gapped_patch, tmp_path / "gap.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        assert len(spool.get_gaps()) == 1
        tree = spool.io.to_xarray()
        leaves = [x for x in tree.subtree if "data" in x.data_vars]
        pieces = list(dc.spool([gapped_patch]))
        assert len(leaves) == len(pieces)
        for leaf, piece in zip(leaves, pieces, strict=True):
            assert np.array_equal(leaf["data"].values, piece.data)

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
        distance = back.get_coord("distance")
        assert distance.sorted and not distance.evenly_sampled
        assert isinstance(back.get_coord("tag"), CoordString)
        assert back == patch

    def test_exact_scan_of_array_values(self, random_patch, tmp_path):
        """An exact scan keeps a version 2 array's values without fitting."""
        n = len(random_patch.get_coord("distance"))
        uneven = np.sort(np.random.default_rng(0).random(n)) * 100
        patch = random_patch.update_coords(distance=uneven)
        path = dc.write(patch, tmp_path / "uneven.h5", "dasdae")
        (payload,) = dc.scan_payloads(path, snap=False)
        distance = payload.coords.coord_map["distance"]
        assert distance.sorted and not distance.evenly_sampled
        np.testing.assert_array_equal(distance.values, uneven)

    def test_lazy_array_sizes_by_grid(self, tmp_path):
        """The lazy xarray view of a long fractional grid counts its samples."""
        pytest.importorskip("xarray")
        pytest.importorskip("dask")
        time = get_coord(start=T0, step=(1, 1024), shape=(2_500_000,))
        data = np.zeros((1, len(time)), dtype=np.float32)
        coords = {"distance": [0.0], "time": time}
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "time"))
        spool = dc.spool(dc.write(patch, tmp_path / "long.h5", "dasdae"))
        tree = spool.io.to_xarray()
        leaf = next(node for node in tree.subtree if "data" in node.dataset)
        assert leaf["data"].shape == data.shape
        assert leaf["data"].data.compute().shape == data.shape


class TestMultiRunRoundTrip:
    """A coordinate of several exact runs is written as its pieces."""

    @pytest.mark.parametrize("rate", [1000, 1024, 3000])
    @pytest.mark.parametrize("layout", ["hole", "strided", "off_lattice"])
    def test_round_trip(self, rate, layout, tmp_path):
        """Each piece reads back with the coordinate and data it was written with."""
        full = get_coord(start=T0, step=(1, rate), shape=(60,))
        if layout == "hole":
            coord = concat_coords(full[:10], full[11:])
        elif layout == "strided":
            coord = concat_coords(full[:12], full[16:])[::2]
        else:
            later = full.min() + np.timedelta64(1, "s") + np.timedelta64(5, "ns")
            other = get_coord(start=later, step=(1, rate), shape=(20,))
            coord = concat_coords(full, other)
        data = np.arange(float(len(coord)))
        patch = dc.Patch(data=data, coords={"time": coord}, dims=("time",))
        pieces = list(dc.spool([patch]))
        assert len(pieces) > 1
        back = dc.read(dc.write(patch, tmp_path / "runs.h5", "dasdae"))
        back = sorted(back, key=lambda x: x.get_coord("time").min())
        assert len(back) == len(pieces)
        for read, piece in zip(back, pieces, strict=True):
            out, expected = read.get_coord("time"), piece.get_coord("time")
            assert out == expected
            assert out.step_exact == expected.step_exact
            np.testing.assert_array_equal(read.data, piece.data)
