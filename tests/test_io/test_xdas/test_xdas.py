"""Tests for reading the XDAS NetCDF layout from generated files."""

from __future__ import annotations

import sys

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError, PatchAttributeError
from dascore.io.h5simple import H5Simple
from dascore.io.netcdf import NetCDFCFV18
from dascore.io.xdas import XdasV1
from dascore.utils.downloader import fetch

xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")

START = np.datetime64("2025-02-03T01:02:03.123456789", "ns")
TIMES = START + np.arange(31) * np.timedelta64(4, "ms")
DISTANCES = -4.0 + np.arange(9) * 2.0
DATA = np.arange(31 * 9, dtype="float32").reshape(31, 9)


def xdas_dataset(legacy=False, name="strain rate", times=TIMES, indices=None):
    """Return a dataset whose coordinates are stored as tie points."""
    data = np.arange(len(times) * 9, dtype="float32").reshape(len(times), 9)
    ds = xr.Dataset(
        {name: (("time", "distance"), data)},
        attrs={"Conventions": "CF-1.9" if legacy else "CF-1.13"},
    )
    mapping = []
    for dim, values, ties in (
        ("time", times, indices),
        ("distance", DISTANCES, None),
    ):
        ties = [0, len(values) - 1] if ties is None else ties
        ds[f"{dim}_indices"] = (f"{dim}_points", ties)
        ds[f"{dim}_values"] = (f"{dim}_points", values[ties])
        tie_mapping = f"{dim}: {dim}_indices {dim}_points"
        if legacy:
            mapping.append(f"{dim}: {dim}_indices {dim}_values")
        else:
            mapping.append(f"{dim}_values: {dim}_interpolation")
            attrs = {"interpolation_name": "linear", "tie_point_mapping": tie_mapping}
            if dim == "time":  # XDAS states the step of a regular time axis
                attrs |= {"sampling_interval": 4, "sampling_interval_units": "ms"}
                attrs["sampling_interval_dtype"] = "timedelta64[ns]"
            ds[f"{dim}_interpolation"] = xr.DataArray(np.nan, attrs=attrs)
    ds[name].attrs = {"coordinate_interpolation": " ".join(mapping), "tag": "a"}
    ds["distance_values"].attrs["units"] = "m"
    return ds


def sampled_dataset(legacy=False, rate=(3, 1)):
    """Return a dataset whose time is two segments sampled at one rate."""
    numerator, denominator = rate
    if legacy:
        mapping = "time: time_values time_lengths"
        attrs = {"tie_point_mapping": mapping, "units": "milliseconds"}
        attrs["dtype"] = "timedelta64[ns]"
        descriptor, label = numerator, "time"
    else:
        mapping = "time: time_lengths time_points"
        key = "sampling_numerator" if denominator > 1 else "sampling_interval"
        attrs = {"tie_point_mapping": mapping, key: numerator}
        attrs[f"{key}_units"] = "milliseconds"
        attrs[f"{key}_dtype"] = "timedelta64[ns]"
        if denominator > 1:
            attrs |= {"sampling_interval": 0, "sampling_denominator": denominator}
        descriptor, label = np.nan, "time_values"
    ds = xr.Dataset(
        {
            "signal": (("time",), np.arange(7)),
            "time_values": ("time_points", [START, START + np.timedelta64(1, "s")]),
            "time_lengths": ("time_points", [3, 4]),
            "time_sampling": ((), descriptor, attrs),
        },
        attrs={"Conventions": "CF-1.13"},
    )
    ds["signal"].attrs["coordinate_sampling"] = f"{label}: time_sampling"
    return ds


def write(ds, path, **kwargs):
    """Write a dataset with the HDF5 netCDF engine and return the path."""
    ds.to_netcdf(path, engine="h5netcdf", **kwargs)
    return path


@pytest.fixture(params=[False, True], ids=["current", "legacy"])
def xdas_path(request, tmp_path):
    """A file in each tie-point spelling describing the same signal."""
    return write(xdas_dataset(legacy=request.param), tmp_path / "xdas.nc")


@pytest.fixture
def gapped_path(tmp_path):
    """A file whose time tie points leave a 2 s hole after sample 9."""
    times = TIMES.copy()
    times[10:] += np.timedelta64(2, "s")
    ds = xdas_dataset(times=times, indices=[0, 9, 10, 30])
    return write(ds, tmp_path / "gapped.nc")


def _replace_with_virtual(path, source, name="strain rate"):
    """Swap a file's signal for a virtual dataset of `source` "values"."""
    with h5py.File(path, "r+") as handle:
        attrs = dict(handle[name].attrs)
        del handle[name]
        layout = h5py.VirtualLayout(shape=DATA.shape, dtype=DATA.dtype)
        layout[:] = h5py.VirtualSource(source, "values", shape=DATA.shape)
        node = handle.create_virtual_dataset(name, layout)
        node.attrs.update(attrs)


class TestDetection:
    """The layout is claimed by XDAS alone, and only when it is identifiable."""

    def test_claimed(self, xdas_path):
        """Generic readers defer files carrying tie-point signals."""
        assert dc.get_format(xdas_path) == ("XDAS", "1")
        assert NetCDFCFV18().get_format(xdas_path) is False

    def test_h5simple_defers(self, tmp_path):
        """A tie-point signal named "data" beside a "time" dim is not H5Simple."""
        path = write(xdas_dataset(name="data"), tmp_path / "data.nc")
        assert H5Simple().get_format(path) is False
        assert dc.get_format(path) == ("XDAS", "1")

    def test_plain_cf_not_claimed(self, tmp_path):
        """A CF file with stored coordinates stays generic NetCDF."""
        ds = xr.Dataset(
            {"data": (("time",), np.arange(5))},
            coords={"time": np.arange(5)},
            attrs={"Conventions": "CF-1.8"},
        )
        path = write(ds, tmp_path / "plain.nc")
        assert XdasV1().get_format(path) is False
        assert dc.get_format(path) == ("NETCDF_CF", "1.8")

    def test_other_cf_interpolation_not_claimed(self, tmp_path):
        """CF interpolation XDAS does not write stays generic NetCDF."""
        quadratic = xdas_dataset()
        quadratic["time_interpolation"].attrs["interpolation_name"] = "quadratic"
        shared = xdas_dataset()
        shared["strain rate"].attrs["coordinate_interpolation"] = (
            "time_values: distance_values: time_interpolation"
        )
        broken = xdas_dataset()
        broken["strain rate"].attrs["coordinate_interpolation"] = "time_values"
        dangling = xdas_dataset(legacy=True)
        dangling["strain rate"].attrs["coordinate_interpolation"] = (
            "time: time_a time_b"
        )
        for number, ds in enumerate([quadratic, shared, broken, dangling]):
            path = write(ds, tmp_path / f"{number}.nc")
            assert XdasV1().get_format(path) is False
            assert dc.get_format(path)[0] == "NETCDF_CF"
        dc.spool(tmp_path).update()


class TestRead:
    """Signals become patches with exact coordinates and their own attrs."""

    def test_read(self, xdas_path):
        """Data, tie-point coordinates, units and attrs come back exactly."""
        patch = dc.read(xdas_path)[0]
        assert patch.dims == ("time", "distance")
        np.testing.assert_array_equal(patch.data, DATA)
        np.testing.assert_array_equal(patch.get_array("time"), TIMES)
        np.testing.assert_array_equal(patch.get_array("distance"), DISTANCES)
        assert patch.get_coord("time").step == np.timedelta64(4, "ms")
        assert str(patch.get_coord("distance").units) == "1 m"
        assert patch.attrs.tag == "a"
        names = set(patch.attrs.model_dump())
        assert not {x for x in names if x.startswith("_") and x != "_source_patch_key"}
        assert not names & {"coordinate_interpolation", "coordinates"}
        assert patch._source.key == "strain rate"

    def test_shipped_file(self):
        """The shipped file, in the original spelling, is one 50 Hz patch."""
        patch = dc.read(fetch("xdas_netcdf.nc"))[0]
        assert patch.shape == (300, 401)
        assert patch.get_coord("time").step == np.timedelta64(20, "ms")

    def test_packed(self, tmp_path):
        """CF scale, offset and fill values apply, and do not become attrs."""
        ds = xdas_dataset()
        ds["strain rate"][0, 0] = np.nan
        packing = {"scale_factor": 0.5, "add_offset": 10.0, "_FillValue": -999}
        encoding = {"strain rate": {"dtype": "int16", **packing}}
        path = write(ds, tmp_path / "packed.nc", encoding=encoding)
        patch = dc.read(path)[0]
        expected = xr.open_dataset(path, engine="h5netcdf")["strain rate"].values
        np.testing.assert_array_equal(patch.data, expected)
        assert np.isnan(patch.data[0, 0])
        assert not set(packing) & set(patch.attrs.model_dump())

    def test_several_missing_values(self, tmp_path):
        """Every value a missing_value list names becomes NaN."""
        path = write(xdas_dataset(), tmp_path / "missing.nc")
        with h5py.File(path, "r+") as handle:
            handle["strain rate"].attrs["missing_value"] = np.array([0.0, 1.0], "f4")
        data = dc.read(path)[0].data
        assert np.isnan(data.flat[:2]).all() and not np.isnan(data.flat[2:]).any()

    def test_spool_and_selection(self, xdas_path):
        """Selecting while reading, or lazily, matches selecting afterwards."""
        select = {"time": (TIMES[3], TIMES[12]), "distance": (-2.0, 4.0)}
        expected = dc.read(xdas_path)[0].select(**select)
        assert dc.read(xdas_path, **select)[0].equals(expected)
        assert dc.spool(xdas_path).select(**select)[0].equals(expected)

    def test_read_array(self, xdas_path):
        """Windows are positional and half-open."""
        out = XdasV1().read_array(xdas_path, ((3, 12), (2, 5)))
        np.testing.assert_array_equal(out, DATA[3:12, 2:5])
        with pytest.raises(PatchAttributeError, match="No patch named"):
            XdasV1().read_array(xdas_path, (), key="missing")

    def test_unnamed(self, tmp_path):
        """An unnamed signal is keyed by the variable it is stored as."""
        path = write(xdas_dataset(name="__values__"), tmp_path / "unnamed.nc")
        spool = dc.spool(path)
        assert spool[0]._source.key == "__values__"
        np.testing.assert_array_equal(spool[0].data, DATA)

    def test_collection(self, tmp_path):
        """Signals in nested groups keep distinct keys, even with one name."""
        path = tmp_path / "collection.nc"
        groups = ["net/fiber_a/event", "net/fiber_b/event"]
        for number, group in enumerate(groups):
            ds = xdas_dataset()
            ds["strain rate"] += number
            write(ds, path, group=group, mode="a" if number else "w")
        spool = dc.spool(path)
        keys = [patch._source.key for patch in spool]
        assert sorted(keys) == [f"{x}/strain rate" for x in groups]
        for patch in spool:
            offset = groups.index(patch._source.key.rsplit("/", 1)[0])
            np.testing.assert_array_equal(patch.data, DATA + offset)

    def test_tile_manifest(self, tmp_path):
        """A tile manifest placeholder is refused rather than read as data."""
        ds = xdas_dataset()
        ds["strain rate"].attrs["__tiling__"] = "__tiles__"
        path = write(ds, tmp_path / "tiles.nc")
        with pytest.raises(NotImplementedError, match="tile manifests"):
            dc.read(path)


class TestGaps:
    """A hole in the tie points splits the signal at its evenly sampled runs."""

    def test_one_patch_per_run(self, gapped_path):
        """Read, scan and spool agree on two keyed pieces of one signal."""
        spool = dc.spool(gapped_path)
        keys = ["strain rate#0", "strain rate#1"]
        assert [x.source_patch_key for x in dc.scan(gapped_path)] == keys
        assert [x._source.key for x in dc.read(gapped_path)] == keys
        first, second = spool
        np.testing.assert_array_equal(first.data, DATA[:10])
        np.testing.assert_array_equal(second.data, DATA[10:])
        assert first.get_coord("time").step == np.timedelta64(4, "ms")
        assert second.get_coord("time").min() == TIMES[10] + np.timedelta64(2, "s")

    def test_select_second_run(self, gapped_path):
        """Selecting after the hole reads only from the second piece."""
        start = TIMES[12] + np.timedelta64(2, "s")
        spool = dc.spool(gapped_path).select(time=(start, None))
        assert len(spool) == 1
        np.testing.assert_array_equal(spool[0].data, DATA[12:])
        out = XdasV1().read_array(gapped_path, ((1, 3),), key="strain rate#1")
        np.testing.assert_array_equal(out, DATA[11:13])

    def test_overlap(self, tmp_path):
        """Tie points which jump back keep each stretch with its own samples."""
        times = TIMES.copy()
        times[10:] -= np.timedelta64(20, "ms")
        path = write(
            xdas_dataset(times=times, indices=[0, 9, 10, 30]), tmp_path / "o.nc"
        )
        _, second = dc.spool(path)
        np.testing.assert_array_equal(second.data, DATA[10:])
        np.testing.assert_array_equal(second.get_array("time"), times[10:])
        assert len(dc.scan(path)) == 2

    def test_backward_segments(self, tmp_path):
        """Sampled segments keep file order when the second starts earlier."""
        ds = sampled_dataset()
        ds["time_values"] = ("time_points", [START + np.timedelta64(1, "s"), START])
        first, _ = dc.read(write(ds, tmp_path / "back.nc"))
        assert first.get_coord("time").min() == START + np.timedelta64(1, "s")
        np.testing.assert_array_equal(first.data, np.arange(3))

    def test_contiguous_ties_one_patch(self, tmp_path):
        """Tie points on one grid, floored to whole ns, are one patch."""
        offsets = (np.arange(3072) * 10**9 // 1024).astype("timedelta64[ns]")
        ties = [0, 1023, 1024, 2047, 2048, 3071]
        ds = xdas_dataset(times=START + offsets, indices=ties)
        ds["time_interpolation"].attrs |= {
            "sampling_numerator": 1_000_000_000,
            "sampling_numerator_units": "nanoseconds",
            "sampling_numerator_dtype": "timedelta64[ns]",
            "sampling_denominator": 1024,
        }
        ds = ds.drop_vars(["distance_indices", "distance_values"])
        ds["distance_indices"] = ("distance_points", [0, 3, 8])
        ds["distance_values"] = ("distance_points", np.arange(9)[[0, 3, 8]] * 0.1)
        path = write(ds, tmp_path / "1024.nc")
        (patch,) = dc.read(path)
        assert patch.get_coord("time").step_exact == 1 / 1024
        np.testing.assert_allclose(patch.get_array("distance"), np.arange(9) * 0.1)


class TestCoordinates:
    """Each way of storing a coordinate decodes to exact labels."""

    def test_integer(self, tmp_path):
        """Integer tie points give an integer grid."""
        ds = xdas_dataset()
        ds["distance_values"] = ("distance_points", [11, 27])
        path = write(ds, tmp_path / "int.nc")
        coord = dc.read(path)[0].get_coord("distance")
        np.testing.assert_array_equal(coord.values, np.arange(11, 28, 2))

    def test_integer_rounded(self, tmp_path):
        """Integer ties a fractional step apart give the labels they round to."""
        ds = xdas_dataset()
        ds["distance_values"] = ("distance_points", [0, 10])
        coord = dc.read(write(ds, tmp_path / "round.nc"))[0].get_coord("distance")
        np.testing.assert_array_equal(coord.values, [0, 1, 2, 4, 5, 6, 8, 9, 10])

    def test_unsigned_descending(self, tmp_path):
        """Unsigned ties which descend do not wrap around."""
        ds = xdas_dataset()
        ds["distance_values"] = ("distance_points", np.array([18, 2], "uint16"))
        coord = dc.read(write(ds, tmp_path / "uint.nc"))[0].get_coord("distance")
        np.testing.assert_array_equal(coord.values, np.arange(18, 1, -2))

    def test_float_gap_far_from_zero(self, tmp_path):
        """A small gap in floats far from zero is still a gap."""
        ds = xdas_dataset().drop_vars(["distance_indices", "distance_values"])
        values = 1e9 + np.arange(9.0)
        values[4:] += 0.5
        ds["distance_indices"] = ("distance_points", [0, 3, 4, 8])
        ds["distance_values"] = ("distance_points", values[[0, 3, 4, 8]])
        assert len(dc.read(write(ds, tmp_path / "far.nc"))) == 2

    def test_sampled_distance(self, tmp_path):
        """A sampled interval in metres stays in metres."""
        mapping = "distance: distance_lengths distance_points"
        attrs = {"tie_point_mapping": mapping, "sampling_interval": 2.0}
        attrs["sampling_interval_units"] = "m"
        ds = xr.Dataset(
            {
                "signal": (("distance",), np.arange(7)),
                "distance_values": ("distance_points", [0.0, 100.0]),
                "distance_lengths": ("distance_points", [3, 4]),
                "distance_sampling": ((), np.nan, attrs),
            },
            attrs={"Conventions": "CF-1.13"},
        )
        ds["signal"].attrs["coordinate_sampling"] = "distance_values: distance_sampling"
        _, second = dc.read(write(ds, tmp_path / "metres.nc"))
        np.testing.assert_array_equal(
            second.get_array("distance"), [100, 102, 104, 106]
        )

    def test_time_zone_offset(self, tmp_path):
        """A reference time with a UTC offset is converted to UTC."""
        ds = xdas_dataset()
        ds["time_values"] = ("time_points", [0, 120])
        units = "milliseconds since 2020-01-01T00:00:00+02:00"
        ds["time_values"].attrs["units"] = units
        time = dc.read(write(ds, tmp_path / "tz.nc"))[0].get_coord("time")
        assert time.min() == np.datetime64("2019-12-31T22:00:00")

    @pytest.mark.parametrize("legacy", [False, True])
    def test_sampled(self, tmp_path, legacy):
        """Segments at a stated rate are two runs, so two patches."""
        path = write(sampled_dataset(legacy=legacy), tmp_path / "sampled.nc")
        first, second = dc.read(path)
        step = np.timedelta64(3, "ms")
        expected = START + np.arange(3) * step
        np.testing.assert_array_equal(first.get_array("time"), expected)
        expected = START + np.timedelta64(1, "s") + np.arange(4) * step
        np.testing.assert_array_equal(second.get_array("time"), expected)

    def test_rational_rate(self, tmp_path):
        """A rate given as a ratio is an exact fractional step."""
        ds = sampled_dataset(rate=(1000, 1024))
        path = write(ds, tmp_path / "rational.nc")
        assert dc.read(path)[0].get_coord("time").step_exact == 1 / 1024

    def test_lengths_must_cover(self, tmp_path):
        """Segment lengths which miss samples are refused."""
        ds = sampled_dataset()
        ds["time_lengths"] = ("time_points", [3, 3])
        path = write(ds, tmp_path / "short.nc")
        with pytest.raises(InvalidFiberFileError, match="do not sum"):
            dc.read(path, file_format="XDAS")

    @pytest.mark.parametrize("ties", [[1, 30], [0, 29], [0, 10, 10, 30]])
    def test_bad_tie_points(self, tmp_path, ties):
        """Tie points which do not span the dimension are refused."""
        ds = xdas_dataset()
        ds = ds.drop_vars(["time_indices", "time_values"])
        ds["time_indices"] = ("time_points", ties)
        ds["time_values"] = ("time_points", TIMES[: len(ties)])
        path = write(ds, tmp_path / "bad.nc")
        with pytest.raises(InvalidFiberFileError, match="do not span"):
            dc.read(path)

    def test_single_sample(self, tmp_path):
        """One tie point describes a one-sample dimension."""
        ds = xdas_dataset(times=TIMES[:1], indices=[0])
        path = write(ds, tmp_path / "one.nc")
        assert dc.read(path)[0].get_array("time").tolist() == TIMES[:1].tolist()

    def test_stored(self, tmp_path):
        """Stored labels, and those of an auxiliary coordinate, are kept."""
        jittered = TIMES + np.arange(31) ** 2 * np.timedelta64(1, "ns")
        ds = xdas_dataset().drop_vars(["time_indices", "time_values"])
        ds = ds.drop_vars("time_interpolation").assign_coords(time=jittered)
        ds = ds.assign_coords(lat=("distance", np.linspace(40, 41, 9)))
        ds["lon_indices"] = ("lon_points", [0, 8])
        ds["lon_values"] = ("lon_points", [10.0, 14.0])
        lon_mapping = "distance: lon_indices lon_points"
        attrs = {"interpolation_name": "linear", "tie_point_mapping": lon_mapping}
        ds["lon_interpolation"] = xr.DataArray(np.nan, attrs=attrs)
        ds["strain rate"].attrs["coordinate_interpolation"] = (
            "distance_values: distance_interpolation lon_values: lon_interpolation"
        )
        path = write(ds, tmp_path / "stored.nc")
        patch = dc.read(path, snap=False)[0]
        np.testing.assert_array_equal(patch.get_array("time"), jittered)
        assert patch.coords.dim_map["lat"] == ("distance",)
        np.testing.assert_array_equal(patch.get_array("lon"), np.arange(10, 14.5, 0.5))
        assert len(dc.spool(path)) == 1

    def test_positional(self, tmp_path):
        """A dimension with no coordinate is numbered by sample."""
        ds = xdas_dataset(legacy=True).drop_vars(["distance_indices"])
        ds = ds.drop_vars("distance_values")
        ds["strain rate"].attrs["coordinate_interpolation"] = (
            "time: time_indices time_values"
        )
        path = write(ds, tmp_path / "positional.nc")
        distance = dc.read(path)[0].get_array("distance")
        np.testing.assert_array_equal(distance, np.arange(9))


class TestVirtual:
    """A signal stored as an HDF5 virtual dataset reads through its sources."""

    @pytest.fixture
    def virtual_path(self, tmp_path):
        """A file whose signal is a virtual dataset of a sibling file."""
        with h5py.File(tmp_path / "source.h5", "w") as handle:
            handle.create_dataset("values", data=DATA)
        path = write(xdas_dataset(), tmp_path / "virtual.nc")
        _replace_with_virtual(path, "source.h5")
        return path

    def test_relative_source(self, virtual_path):
        """A source named relative to the file is found beside it."""
        np.testing.assert_array_equal(dc.read(virtual_path)[0].data, DATA)

    def test_missing_source(self, virtual_path):
        """A missing source raises, rather than reading fill values."""
        (virtual_path.parent / "source.h5").unlink()
        assert dc.scan(virtual_path)[0].shape == DATA.shape
        with pytest.raises(FileNotFoundError, match=r"source\.h5"):
            dc.read(virtual_path)

    def test_missing_source_dataset(self, virtual_path):
        """A source file without the mapped dataset raises too."""
        with h5py.File(virtual_path.parent / "source.h5", "a") as handle:
            handle.move("values", "other")
        with pytest.raises(FileNotFoundError, match="values"):
            dc.read(virtual_path)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="HDF5's C runtime does not see os.environ changes on Windows.",
    )
    def test_prefix_source(self, virtual_path, tmp_path_factory, monkeypatch):
        """A source HDF5 finds through HDF5_VDS_PREFIX is read."""
        elsewhere = tmp_path_factory.mktemp("sources")
        (virtual_path.parent / "source.h5").rename(elsewhere / "source.h5")
        monkeypatch.setenv("HDF5_VDS_PREFIX", str(elsewhere))
        np.testing.assert_array_equal(dc.read(virtual_path)[0].data, DATA)

    def test_same_file_source(self, tmp_path):
        """A virtual dataset may read from another dataset in its own file."""
        path = write(xdas_dataset(), tmp_path / "same.nc")
        with h5py.File(path, "r+") as handle:
            handle.create_dataset("values", data=DATA)
        _replace_with_virtual(path, ".")
        np.testing.assert_array_equal(dc.read(path)[0].data, DATA)
