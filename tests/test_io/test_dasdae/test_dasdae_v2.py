"""Tests for DASDAE format version 2: coordinates stored as descriptions."""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.core._run_kernels import float_rows
from dascore.core.coords import (
    CoordString,
    NumericND,
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
        if isinstance(coord, NumericND):
            # the shape of the table too, not only the labels it makes
            assert back.runs_count == coord.runs_count
            assert (back.sources is None) == (coord.sources is None)
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

    def test_every_node_states_its_class(self, h5):
        """Each node names its coordinate class in plain typed attributes."""
        for name in ("fraction", "segmented", "array", "strings"):
            _save_coord(CASES[name], f"typed_{name}", h5, compact=True)
        # the names the layout has always used, so older readers still read it
        expected = {
            "fraction": "CoordRange",
            "segmented": "CoordSegmented",
            "array": "CoordMonotonicArray",
        }
        for name, tag in expected.items():
            assert h5[f"typed_{name}"].attrs["object_type"] == tag
        assert h5["typed_strings"].attrs["object_type"] == "CoordString"
        attrs = dict(h5["typed_fraction"].attrs)
        assert attrs["step_denominator"] == 2 and attrs["length"] == 4096

    @pytest.mark.parametrize(
        "name,tag",
        [
            ("segmented", "CoordSegmented"),
            ("fraction", "CoordRange"),
            ("array", "CoordMonotonicArray"),
            ("array", "CoordArray"),
        ],
    )
    def test_legacy_tags_still_read(self, h5, name, tag):
        """A node tagged before the numeric coord classes merged still reads."""
        coord = CASES[name]
        node = f"legacy_{name}_{tag}"
        _save_coord(coord, node, h5, compact=True)
        h5[node].attrs["object_type"] = tag
        assert _read_coord(h5[node], node, {}, snap=True) == coord

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

    def test_array_segment_stays_exact(self, h5):
        """A near-uniform array segment is not snapped to a range on read."""
        jitter = NumericND.from_array(
            np.array([0.0, 1.0, 2.0005, 3.0, 4.0]), detect=False
        )
        coord = concat_coords(jitter, get_coord(start=10.0, stop=15.0, step=1.0))
        _save_coord(coord, "jitter", h5, compact=True)
        back = _read_coord(h5["jitter"], "jitter", {}, snap=True)
        assert back == coord
        assert not back.segments[0].evenly_sampled and (
            back.segments[0].sorted or back.segments[0].reverse_sorted
        )

    def test_float_step_keeps_its_precision(self, h5):
        """A float64 step on a float32 start counts the same samples back."""
        coord = get_coord(
            start=np.float32(92.97747), step=0.49223355932786406, shape=(5_482_063,)
        )
        _save_coord(coord, "mixed", h5, compact=True)
        back = _read_coord(h5["mixed"], "mixed", {}, snap=True)
        assert len(back) == len(coord)
        assert back == coord


class TestUnorderedRuns:
    """Runs the segmented layout was never able to hold go out as values."""

    @pytest.fixture(scope="class")
    def h5(self):
        """An in-memory HDF5 file."""
        return h5py.File(io.BytesIO(), "w")

    @pytest.fixture(scope="class")
    def cases(self):
        """Multi-run time coordinates the segmented group cannot describe."""
        ms = np.timedelta64(1, "ms")
        overlapping = T0 + np.concatenate([np.arange(20), np.arange(15, 35)]) * ms
        unsorted = (
            T0
            + np.concatenate([np.arange(5), np.arange(40, 44), np.arange(5, 16)]) * ms
        )
        return {
            "overlapping": dc.get_coord(data=overlapping),
            "unsorted": dc.get_coord(data=unsorted),
        }

    @pytest.mark.parametrize("name", ["overlapping", "unsorted"])
    def test_runs_are_not_segments(self, cases, name):
        """These are several runs, but neither sorted nor reverse sorted."""
        coord = cases[name]
        assert coord.runs_count > 1
        assert not coord.sorted and not coord.reverse_sorted

    @pytest.mark.parametrize("name", ["overlapping", "unsorted"])
    def test_node_holds_its_values(self, h5, cases, name):
        """Runs the segments group cannot hold go out as one values dataset."""
        coord = cases[name]
        _save_coord(coord, name, h5, compact=True)
        assert isinstance(h5[name], h5py.Dataset)
        assert h5[name].attrs["object_type"] == "CoordArray"

    @pytest.mark.parametrize("name", ["overlapping", "unsorted"])
    def test_values_round_trip(self, cases, name, tmp_path):
        """The labels come back as they went in, in their own order."""
        coord = cases[name]
        patch = dc.Patch(
            data=np.zeros((3, len(coord))),
            coords={"distance": np.arange(3.0), "time": coord},
            dims=("distance", "time"),
        )
        path = dc.write(patch, tmp_path / f"{name}.h5", "dasdae")
        (back,) = dc.read(path)
        np.testing.assert_array_equal(back.get_coord("time").values, coord.values)


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
        assert back.get_coord("time").runs_count > 1
        assert back.get_coord("time") == gapped_patch.get_coord("time")
        assert np.array_equal(back.data, gapped_patch.data)
        (scanned,) = dc.scan(path)
        summary = scanned.coords["time"]
        assert summary.step is None
        assert summary.data_id == gapped_patch.get_coord("time").data_id

    def test_split_still_honored(self, gapped_patch, tmp_path):
        """An explicit split writes each run as its own patch."""
        path = dc.write(gapped_patch, tmp_path / "split.h5", "dasdae", split=True)
        spool = dc.spool(path)
        assert len(spool) == 2
        for patch in spool:
            assert patch.get_coord("time").evenly_sampled

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
        assert not back.get_coord("distance").evenly_sampled and (
            back.get_coord("distance").sorted
            or back.get_coord("distance").reverse_sorted
        )
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
        assert not distance.evenly_sampled and (
            distance.sorted or distance.reverse_sorted
        )
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


class TestFloatGridTerms:
    """A float range which is not counted from its first label states its grid."""

    divided = get_coord(
        runs=float_rows("float64", [0.0], [400], [250.0], [-1], [0]),
        dtype="float64",
        units="m",
    )
    descending = get_coord(
        runs=float_rows("float64", [0.0], [400], [-1000.0], [-1], [0]),
        dtype="float64",
        units="m",
    )
    # A multiplied axis and explicit divided grids exercise persisted terms.
    COORDS = (
        get_coord(data=np.arange(400) * 0.1, units="m")[37::3],
        divided,
        divided[11:],
        descending[::-1],
    )

    @pytest.mark.parametrize("coord", COORDS)
    def test_round_trip_is_bit_for_bit(self, tmp_path, coord):
        """Writing and reading gives back the same doubles and the same run."""
        assert coord.evenly_sampled
        patch = dc.Patch(
            data=np.zeros(len(coord)), coords={"distance": coord}, dims=("distance",)
        )
        path = tmp_path / "grid.h5"
        patch.io.write(path, "dasdae")
        back = dc.read(path)[0].get_coord("distance")
        np.testing.assert_array_equal(back.values, coord.values)
        assert back == coord
        np.testing.assert_array_equal(back.runs, coord.runs)

    def test_a_narrow_float_keeps_its_wider_origin(self, tmp_path):
        """A float32 run counted from a float64 origin reads back label for label."""
        coord = NumericND.from_run(0.1, 0.001, 6, dtype="float32")
        patch = dc.Patch(
            data=np.zeros(6), coords={"distance": coord}, dims=("distance",)
        )
        path = tmp_path / "narrow.h5"
        patch.io.write(path, "dasdae")
        back = dc.read(path, snap=False)[0].get_coord("distance")
        np.testing.assert_array_equal(back.values, coord.values)

    def test_a_plain_range_writes_no_extra_terms(self, tmp_path):
        """Counted from its own first label, a range is its start and step."""
        patch = dc.Patch(
            data=np.zeros(10),
            coords={"distance": get_coord(start=3.0, step=0.5, shape=(10,))},
            dims=("distance",),
        )
        path = tmp_path / "plain.h5"
        patch.io.write(path, "dasdae")
        with h5py.File(path) as h5:
            group = next(iter(h5["waveforms"].values()))
            assert "grid_terms" not in group["_coord_distance"].attrs


class TestRangeNodesFromEarlierWriters:
    """A range an earlier DASCore described by its step alone still reads."""

    def test_a_time_range_stated_by_a_whole_step(self):
        """Start, stop and a step in the node's own unit rebuild the range."""
        with h5py.File(io.BytesIO(), "w") as h5:
            node = h5.create_dataset("_coord_time", shape=(0,), dtype="int64")
            node.attrs["dtype"] = "datetime64[ms]"
            node.attrs["start"] = int(T0.astype("datetime64[ms]").astype("int64"))
            node.attrs["stop"] = node.attrs["start"] + 40
            node.attrs["step"] = 4
            node.attrs["length"] = 10
            node.attrs["object_type"] = "CoordRange"
            coord = _read_coord(node, "time", {}, snap=True)
        expected = get_coord(start=T0, step=np.timedelta64(4, "ms"), shape=(10,))
        assert coord == expected
