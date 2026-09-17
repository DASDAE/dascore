"""XDAS input contracts exercised with generated files."""

from __future__ import annotations

import subprocess
import sys
import tracemalloc
import types

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchAttributeError
from dascore.io.h5simple.core import H5Simple
from dascore.io.netcdf.core import NetCDFCFV18
from dascore.io.xdas.core import XdasV1
from dascore.io.xdas.utils import interpolate_coord
from tests.test_io.test_common_io import _CountingHandle
from tests.test_io.test_xdas._fixtures import write_xdas, xdas_dataset

xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")


@pytest.fixture(params=[False, True], ids=["current", "legacy"])
def example(request, tmp_path):
    """The two on-disk grammars describe identical signals and coordinates."""
    path = tmp_path / "generated.nc"
    return path, *write_xdas(path, legacy=request.param)


class TestXdas:
    """Public format detection, metadata, reads and lazy access."""

    def test_detect(self, example):
        """Generic CF detection must defer independently of registration order."""
        path, *_ = example
        assert XdasV1().get_format(path) == ("xdas", "1")
        assert NetCDFCFV18().get_format(path) is False
        assert H5Simple().get_format(path) is False
        assert dc.get_format(path) == ("xdas", "1")

    def test_read(self, example):
        """Named signals retain exact timestamps, units, data and user metadata."""
        path, data, times, distances = example
        spool = dc.read(path)
        assert len(spool) == 1
        patch = spool[0]
        assert patch.dims == ("time", "distance")
        np.testing.assert_array_equal(patch.data, data)
        np.testing.assert_array_equal(patch.get_array("time"), times)
        np.testing.assert_array_equal(patch.get_array("distance"), distances)
        assert str(patch.get_coord("distance").units) == "1 m"
        assert patch.attrs.tag == "synthetic"
        assert patch.attrs["_source_patch_key"] == "/strain rate"
        assert "coordinate_interpolation" not in patch.attrs

    def test_scan_and_spool(self, example):
        """Scanning and lazy loading agree on identity and exact coordinates."""
        path, data, times, distances = example
        payload = dc.scan_payloads(path)[0]
        assert payload["source_patch_key"] == "/strain rate"
        np.testing.assert_array_equal(payload["coords"].get_array("time"), times)
        assert payload["shape"] == data.shape
        lazy = dc.spool(path)
        assert len(lazy) == 1
        np.testing.assert_array_equal(lazy[0].data, data)
        np.testing.assert_array_equal(lazy[0].get_array("distance"), distances)

    def test_selection(self, example):
        """Read-time selection and lazy selection match selecting a loaded patch."""
        path, _, times, distances = example
        selection = {
            "time": (times[3], times[12]),
            "distance": (distances[2], distances[5]),
        }
        expected = dc.read(path)[0].select(**selection)
        assert dc.read(path, **selection)[0].equals(expected)
        assert dc.spool(path).select(**selection)[0].equals(expected)
        assert len(dc.read(path, time=(times[-1] + np.timedelta64(1, "s"), None))) == 0

    def test_array_window(self, example):
        """The array fast path uses absolute half-open sample windows."""
        path, data, *_ = example
        actual = XdasV1().read_array(path, {"time": (3, 12), "distance": (2, 5)})
        np.testing.assert_array_equal(actual, data[3:12, 2:5])
        with pytest.raises(PatchAttributeError, match="No patch"):
            XdasV1().read_array(path, {}, source_patch_key="missing")

    def test_multiple_signals(self, tmp_path):
        """Two measurements in one dataset become two independently keyed patches."""
        ds, data, *_ = xdas_dataset()
        ds["temperature"] = ds["strain rate"].copy(data=data + 50)
        path = tmp_path / "multiple.nc"
        ds.to_netcdf(path, engine="h5netcdf")
        assert len(dc.read(path)) == len(dc.scan(path)) == 2
        np.testing.assert_array_equal(
            dc.read(path, source_patch_key="/temperature")[0].data, data + 50
        )
        with pytest.raises(PatchAttributeError, match="several patches"):
            XdasV1().read_array(path, {})
        np.testing.assert_array_equal(
            XdasV1().read_array(path, {}, source_patch_key="/strain rate"), data
        )

    @pytest.mark.parametrize(
        "name", ["__values__", "data", "strain _rate [n_strain|s]"]
    )
    def test_signal_names(self, tmp_path, name):
        """Signal names never determine which coordinate helpers become patches."""
        path = tmp_path / "names.nc"
        data, *_ = write_xdas(path, name=name)
        spool = dc.read(path)
        assert len(spool) == 1
        np.testing.assert_array_equal(spool[0].data, data)

    def test_collection(self, tmp_path):
        """Nested groups with duplicate signal names retain stable distinct keys."""
        path = tmp_path / "collection.nc"
        ds, data, *_ = xdas_dataset()
        groups = ["network/fiber_a/event_1", "network/fiber_b/event_1"]
        for i, group in enumerate(groups):
            part = ds.copy(deep=True)
            part["strain rate"].data += i
            part.to_netcdf(
                path, group=group, mode="w" if i == 0 else "a", engine="h5netcdf"
            )
        keys = {f"/{group}/strain rate" for group in groups}
        assert {item.source_patch_key for item in dc.scan(path)} == keys
        spool = dc.spool(path)
        assert len(spool) == 2
        for patch in spool:
            index = groups.index(patch.attrs["_source_patch_key"].rsplit("/", 1)[0][1:])
            np.testing.assert_array_equal(patch.data, data + index)

    def test_plain_cf(self, tmp_path):
        """Explicit XDAS reads also accept arrays with ordinary dense coordinates."""
        path = tmp_path / "ordinary.nc"
        ds = xr.Dataset(
            {"data": (("time",), np.arange(8))},
            coords={"time": np.arange(8)},
            attrs={"Conventions": "CF-1.8"},
        )
        ds.to_netcdf(path, engine="h5netcdf")
        assert dc.get_format(path) == ("NETCDF_CF", "1.8")
        assert H5Simple().get_format(path) is False
        np.testing.assert_array_equal(
            dc.read(path, file_format="xdas", file_version="1")[0].data, ds.data.values
        )

    def test_scan_does_not_load_samples(self, tmp_path):
        """Metadata scanning consumes far fewer bytes than reading a full signal."""
        path = tmp_path / "large.nc"
        write_xdas(path, shape=(1000, 400))
        counts = []
        for method in (XdasV1().scan, XdasV1().read):
            with path.open("rb") as stream:
                handle = _CountingHandle(stream)
                method(handle)
                counts.append(handle.bytes_read)
        assert counts[0] < counts[1] / 4

    def test_zfp(self, tmp_path):
        """A fresh process registers ZFP without importing XDAS or hdf5plugin first."""
        plugin = pytest.importorskip("hdf5plugin")
        path = tmp_path / "compressed.nc"
        ds, data, *_ = xdas_dataset()
        ds.to_netcdf(
            path,
            engine="h5netcdf",
            encoding={"strain rate": dict(plugin.Zfp(reversible=True))},
        )
        script = (
            "import dascore as dc; import sys; p = dc.read(sys.argv[1])[0]; "
            "assert p.shape == (31, 9); assert p.data[30, 8] == 278"
        )
        subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        np.testing.assert_array_equal(dc.read(path)[0].data, data)

    def test_unsupported_interpolation(self, tmp_path):
        """Unsupported metadata must never silently produce plausible coordinates."""
        path = tmp_path / "unsupported.nc"
        ds, *_ = xdas_dataset()
        ds.time_interpolation.attrs["interpolation_name"] = "quadratic"
        ds.to_netcdf(path, engine="h5netcdf")
        with pytest.raises(NotImplementedError, match="linear"):
            dc.read(path)

    @pytest.mark.parametrize("indices", [[1, 30], [0, 29], [0, 0]])
    def test_invalid_ties(self, tmp_path, indices):
        """Out-of-bounds or duplicate knots cannot define the stored dimension."""
        path = tmp_path / "invalid.nc"
        ds, *_ = xdas_dataset()
        ds.time_indices.data = indices
        ds.to_netcdf(path, engine="h5netcdf")
        with pytest.raises(ValueError, match="tie points"):
            dc.read(path)

    def test_tile_manifest(self, tmp_path):
        """A virtual placeholder cannot be mistaken for real signal samples."""
        path = tmp_path / "tiles.nc"
        ds, *_ = xdas_dataset()
        ds["strain rate"].attrs["__tiling__"] = "tiles"
        ds.to_netcdf(path, engine="h5netcdf")
        with pytest.raises(NotImplementedError, match="tile manifests"):
            dc.read(path)

    def test_legacy_netcdf_reader(self, tmp_path):
        """Existing callers forcing NETCDF_CF retain exact XDAS tie-point labels."""
        path = tmp_path / "legacy.nc"
        data, times, distances = write_xdas(path, legacy=True, name="data")
        patch = NetCDFCFV18().read(path)[0]
        np.testing.assert_array_equal(patch.data, data)
        np.testing.assert_array_equal(patch.get_array("time"), times)
        np.testing.assert_array_equal(patch.get_array("distance"), distances)
        assert patch.attrs["_source_patch_key"] == "data"


class TestXdasCoordinateEdges:
    """Small on-disk fixtures cover validation and less common coordinate forms."""

    @pytest.mark.parametrize(
        "mapping",
        ["bad tokens", "", "time:", "time: a b c", "time: missing time_values"],
    )
    def test_bad_mapping(self, tmp_path, mapping):
        """Invalid or dangling coordinate references fail during reading."""
        path = tmp_path / "mapping.nc"
        ds, *_ = xdas_dataset()
        ds["strain rate"].attrs["coordinate_interpolation"] = mapping
        ds.to_netcdf(path, engine="h5netcdf")
        with pytest.raises(ValueError, match=r"mapping|references missing"):
            XdasV1().read(path)

    def test_bad_dimension_mapping(self, tmp_path):
        """A descriptor must identify exactly one dimension and its tie points."""
        path = tmp_path / "mapping.nc"
        ds, *_ = xdas_dataset()
        ds.time_interpolation.attrs["tie_point_mapping"] = "time: time_indices"
        ds.to_netcdf(path, engine="h5netcdf")
        with pytest.raises(ValueError, match="one-dimensional"):
            dc.read(path)

    def test_nonfinite_ties(self, tmp_path):
        """Non-finite labels must not silently become an apparently regular grid."""
        path = tmp_path / "nan.nc"
        ds, *_ = xdas_dataset()
        ds.distance_values.data[0] = np.nan
        ds.to_netcdf(path, engine="h5netcdf")
        with pytest.raises(ValueError, match="non-finite"):
            dc.read(path)

    def test_original_grammar_without_descriptors(self, tmp_path):
        """The oldest XDAS files carry only indices, values and the signal mapping."""
        path = tmp_path / "oldest.nc"
        ds, data, times, _ = xdas_dataset(legacy=True)
        ds = ds.drop_vars(["time_interpolation", "distance_interpolation"])
        ds.to_netcdf(path, engine="h5netcdf")
        patch = dc.read(path)[0]
        np.testing.assert_array_equal(patch.data, data)
        np.testing.assert_array_equal(patch.get_array("time"), times)

    @pytest.mark.parametrize("legacy", [False, True])
    def test_sampled(self, tmp_path, legacy):
        """Both sampling grammars reconstruct unequal segments and their gap."""
        path = tmp_path / "sampled.nc"
        start = np.datetime64("2025-05-06T12:34:56.123456789", "ns")
        attrs = (
            {
                "tie_point_mapping": "time: time_values time_lengths",
                "units": "milliseconds",
                "dtype": "timedelta64[ns]",
            }
            if legacy
            else {
                "tie_point_mapping": "time: time_lengths time_points",
                "sampling_interval": 3,
                "sampling_interval_units": "milliseconds",
                "sampling_interval_dtype": "timedelta64[ns]",
            }
        )
        ds = xr.Dataset(
            {
                "signal": (("time",), np.arange(7)),
                "time_values": (
                    ("time_points",),
                    [start, start + np.timedelta64(1, "s")],
                ),
                "time_lengths": (("time_points",), [3, 4]),
                "time_sampling": ((), 3 if legacy else np.nan, attrs),
            },
            attrs={"Conventions": "CF-1.13"},
        )
        ds.signal.attrs["coordinate_sampling"] = (
            "time: time_sampling" if legacy else "time_values: time_sampling"
        )
        ds.to_netcdf(path, engine="h5netcdf")
        expected = start + np.array(
            [0, 3, 6, 1000, 1003, 1006, 1009], dtype="timedelta64[ms]"
        )
        np.testing.assert_array_equal(dc.read(path)[0].get_array("time"), expected)
        with h5py.File(path, "r+") as handle:
            handle["time_lengths"][0] = 2
        with pytest.raises(ValueError, match="lengths"):
            dc.read(path)

    def test_integer_coordinates(self, tmp_path):
        """Integer tie points use nearest-even offsets, even with an odd origin."""
        path = tmp_path / "integer.nc"
        ds, *_ = xdas_dataset()
        ds["distance_values"] = (("distance_points",), [11, 15])
        ds.to_netcdf(path, engine="h5netcdf")
        expected = [11, 11, 12, 13, 13, 13, 14, 15, 15]
        np.testing.assert_array_equal(dc.read(path)[0].get_array("distance"), expected)

    def test_dimension_without_coordinate(self, tmp_path):
        """A stated dimension without labels uses its positional sample indices."""
        path = tmp_path / "positional.nc"
        ds, data, *_ = xdas_dataset(legacy=True)
        ds = ds.drop_vars(
            ["distance_indices", "distance_values", "distance_interpolation"]
        )
        ds["strain rate"].attrs["coordinate_interpolation"] = (
            "time: time_indices time_values"
        )
        ds.to_netcdf(path, engine="h5netcdf")
        patch = dc.read(path)[0]
        np.testing.assert_array_equal(
            patch.get_array("distance"), np.arange(data.shape[1])
        )

    @pytest.mark.parametrize("sampled", [False, True])
    def test_empty_signal(self, tmp_path, sampled):
        """An empty coordinate and signal scan consistently and read as no patches."""
        path = tmp_path / "empty.nc"
        kind = "sampling" if sampled else "interpolation"
        indices = "lengths" if sampled else "indices"
        ds = xr.Dataset(
            {
                "signal": (("time",), np.empty(0)),
                "time_values": (("time_points",), np.empty(0)),
                f"time_{indices}": (("time_points",), np.empty(0, dtype="int64")),
                f"time_{kind}": (
                    (),
                    np.nan,
                    {
                        "tie_point_mapping": f"time: time_{indices} time_points",
                        "sampling_interval": 1,
                    },
                ),
            },
            attrs={"Conventions": "CF-1.13"},
        )
        ds.signal.attrs[f"coordinate_{kind}"] = f"time_values: time_{kind}"
        ds.to_netcdf(path, engine="h5netcdf")
        assert len(dc.read(path)) == 0
        assert XdasV1().scan(path)[0]["shape"] == (0,)

    @pytest.mark.parametrize("missing", [True, False])
    def test_missing_filter(self, tmp_path, missing, monkeypatch):
        """A missing codec has a clear error both with and without hdf5plugin."""
        plugin = pytest.importorskip("hdf5plugin")
        path = tmp_path / "zfp.nc"
        ds, *_ = xdas_dataset()
        ds.to_netcdf(
            path,
            engine="h5netcdf",
            encoding={"strain rate": dict(plugin.Zfp(reversible=True))},
        )
        monkeypatch.setattr(h5py.h5z, "filter_avail", lambda code: code != 32013)
        monkeypatch.setitem(
            sys.modules,
            "hdf5plugin",
            None if missing else types.ModuleType("hdf5plugin"),
        )
        with pytest.raises(
            dc.exceptions.MissingOptionalDependencyError, match=r"hdf5plugin|32013"
        ):
            dc.read(path)


class TestXdasVirtual:
    """A local VDS must resolve real source data, including compressed sources."""

    @pytest.fixture
    def virtual_path(self, tmp_path):
        """Build a small virtual signal with a relative source path."""
        source = tmp_path / "source.h5"
        target = tmp_path / "virtual.nc"
        plugin = pytest.importorskip("hdf5plugin")
        data, *_ = write_xdas(target)
        with h5py.File(source, "w") as handle:
            handle.create_dataset("values", data=data, **plugin.Zfp(reversible=True))
        with h5py.File(target, "r+") as handle:
            old = handle["strain rate"]
            attrs = dict(old.attrs)
            del handle["strain rate"]
            layout = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
            layout[:] = h5py.VirtualSource("source.h5", "values", shape=data.shape)
            node = handle.create_virtual_dataset("strain rate", layout)
            for key, value in attrs.items():
                node.attrs[key] = value
        return target, source

    def test_compressed_source(self, virtual_path):
        """Codecs on the source are registered even though the VDS has no filter."""
        path, _ = virtual_path
        script = (
            "import dascore as dc; import sys; "
            "assert dc.read(sys.argv[1])[0].data[30, 8] == 278"
        )
        subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_missing_source(self, virtual_path):
        """A deleted source raises instead of returning plausible all-zero data."""
        path, source = virtual_path
        source.unlink()
        assert XdasV1().scan(path)[0]["shape"] == (31, 9)
        with pytest.raises(FileNotFoundError):
            dc.read(path)

    def test_same_file_source(self, tmp_path):
        """A VDS can refer to a signal stored elsewhere in its own file."""
        path = tmp_path / "same_file.nc"
        expected, *_ = write_xdas(path)
        with h5py.File(path, "r+") as handle:
            attrs = dict(handle["strain rate"].attrs)
            handle.move("strain rate", "storage/values")
            layout = h5py.VirtualLayout(shape=expected.shape, dtype=expected.dtype)
            layout[:] = h5py.VirtualSource(".", "/storage/values", shape=expected.shape)
            node = handle.create_virtual_dataset("strain rate", layout)
            for key, value in attrs.items():
                node.attrs[key] = value
        np.testing.assert_array_equal(dc.read(path)[0].data, expected)


class TestReviewRegressions:
    """Public regressions for detection and compact metadata decoding."""

    @pytest.mark.parametrize("declared", [False, True])
    def test_h5simple_without_cf(self, tmp_path, declared):
        """h5netcdf's storage marker alone must not strand H5Simple data."""
        path = tmp_path / "simple.nc"
        attrs = {"file_format": "h5simple"} if declared else {}
        dataset = xr.Dataset(
            {"data": (("time", "distance"), np.ones((4, 3)))},
            coords={"time": 1_700_000_000.0 + np.arange(4), "distance": np.arange(3)},
            attrs=attrs,
        )
        dataset.to_netcdf(path, engine="h5netcdf")
        assert dc.get_format(path) == ("H5Simple", "1")
        np.testing.assert_array_equal(dc.read(path)[0].data, dataset.data.values)

    def test_generic_cf_with_quality_group(self, tmp_path):
        """An incidental CF subgroup does not turn a root signal into a collection."""
        path = tmp_path / "generic.nc"
        dataset = xr.Dataset(
            {"data": ("time", np.arange(5))},
            coords={"time": np.arange(5)},
            attrs={"Conventions": "CF-1.8"},
        )
        dataset.to_netcdf(path, engine="h5netcdf")
        dataset.rename({"data": "quality"}).to_netcdf(
            path, group="quality_control", mode="a", engine="h5netcdf"
        )
        assert XdasV1().get_format(path) is False
        assert dc.get_format(path) == ("NETCDF_CF", "1.8")
        assert len(dc.read(path)) == len(dc.spool(path)) == 1

    @pytest.mark.parametrize("sampled", [False, True])
    def test_scan_large_compact_grid(self, tmp_path, sampled):
        """A million-sample scan keeps its coordinate memory use below 20 MB."""
        h5netcdf = pytest.importorskip("h5netcdf")
        path = tmp_path / "compact.nc"
        size = 1_000_000
        with h5netcdf.File(path, "w") as handle:
            handle.attrs["Conventions"] = "CF-1.13"
            handle.dimensions = {"time": size, "points": 2}
            variable = handle.create_variable(
                "signal", ("time",), dtype="float32", chunks=(1000,)
            )
            times = handle.create_variable("time_values", ("points",), dtype="int64")
            times.attrs["units"] = "nanoseconds since 2025-01-01 00:00:00"
            indices = handle.create_variable("time_indices", ("points",), dtype="int64")
            if sampled:
                times[:] = [0, size * 1_000_000]
                indices[:] = [size // 2, size // 2]
                descriptor = handle.create_variable("time_sampling", (), dtype="int64")
                descriptor.attrs.update(
                    {
                        "tie_point_mapping": "time: time_indices points",
                        "sampling_interval": 1_000_000,
                        "sampling_interval_units": "nanoseconds",
                    }
                )
                variable.attrs["coordinate_sampling"] = "time_values: time_sampling"
            else:
                times[:] = [0, (size - 1) * 1_000_000]
                indices[:] = [0, size - 1]
                variable.attrs["coordinate_interpolation"] = (
                    "time: time_indices time_values"
                )
        # Warm backend imports before measuring metadata allocations.
        XdasV1().get_format(path)
        tracemalloc.start()
        try:
            payload = XdasV1().scan(path)[0]
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 20_000_000
        coord = payload["coords"].get_coord("time")
        assert len(coord) == size
        start = np.datetime64("2025-01-01", "ns")
        patch = dc.read(path, time=(start, start + np.timedelta64(2, "ms")))[0]
        assert patch.shape == (3,)

    def test_nonmonotonic_interpolation(self):
        """Turning points retain labels when segments cannot form an ordered chain."""
        actual = interpolate_coord(np.array([0, 4, 0]), np.array([0, 2, 4]), 5)
        np.testing.assert_array_equal(actual, [0, 2, 4, 2, 0])

    def test_sampled_order_is_preserved(self, tmp_path):
        """Separated sampled runs retain file order, including backward jumps."""
        path = tmp_path / "backward.nc"
        ds = xr.Dataset(
            {
                "signal": ("time", np.arange(4)),
                "time_values": ("points", [10, 0]),
                "time_lengths": ("points", [2, 2]),
                "time_sampling": (
                    (),
                    0,
                    {
                        "tie_point_mapping": "time: time_lengths points",
                        "sampling_interval": 1,
                    },
                ),
            },
            attrs={"Conventions": "CF-1.13"},
        )
        ds.signal.attrs["coordinate_sampling"] = "time_values: time_sampling"
        ds.to_netcdf(path, engine="h5netcdf")
        np.testing.assert_array_equal(
            dc.read(path)[0].get_array("time"), [10, 11, 0, 1]
        )

    @pytest.mark.parametrize("convention", ["CF-1.5", "CF-1.8", "CF-invalid"])
    def test_h5simple_with_cf_label(self, tmp_path, convention):
        """A CF label without NetCDF dimensions still permits H5Simple inference."""
        path = tmp_path / "labelled.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["Conventions"] = convention
            handle.create_dataset("data", data=np.ones((5, 3)))
            time = handle.create_dataset("time", data=1_700_000_000.0 + np.arange(5))
            if convention == "CF-invalid":
                time.make_scale("time")
                handle["data"].dims[0].attach_scale(time)
        assert H5Simple().get_format(path) == ("H5Simple", "1")
        patch = H5Simple().read(path)[0]
        assert patch.dims == ("time", "channel")

    def test_generic_fractional_grid_consistency(self, tmp_path):
        """A CF grid rounded to nanoseconds has identical exact scan/read labels."""
        path = tmp_path / "3000hz.nc"
        times = np.datetime64("2025-01-01", "ns") + np.rint(
            np.arange(20) * 1e9 / 3000
        ).astype("timedelta64[ns]")
        xr.Dataset(
            {"data": ("time", np.arange(20))},
            coords={"time": times},
            attrs={"Conventions": "CF-1.8"},
        ).to_netcdf(path, engine="h5netcdf")
        for snap in (False, True):
            payload = NetCDFCFV18().scan(path, snap=snap)[0]
            read = dc.read(path, snap=snap)[0]
            np.testing.assert_array_equal(payload["coords"].get_array("time"), times)
            np.testing.assert_array_equal(read.get_array("time"), times)
            assert (
                payload["coords"].get_coord("time").step == read.get_coord("time").step
            )

    def test_virtual_selection_ignores_unrelated_missing_source(self, tmp_path):
        """Selecting one virtual block does not open an unrelated archived block."""
        path = tmp_path / "two_sources.nc"
        data, times, _ = write_xdas(path)
        source = tmp_path / "present.h5"
        with h5py.File(source, "w") as handle:
            handle.create_dataset("values", data=data[:15])
        with h5py.File(path, "r+") as handle:
            attrs = dict(handle["strain rate"].attrs)
            del handle["strain rate"]
            layout = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
            layout[:15] = h5py.VirtualSource(str(source), "values", shape=(15, 9))
            layout[15:] = h5py.VirtualSource(
                str(tmp_path / "missing.h5"), "values", shape=(16, 9)
            )
            node = handle.create_virtual_dataset("strain rate", layout)
            for name, value in attrs.items():
                node.attrs[name] = value
        selected = dc.read(path, time=(times[1], times[4]))[0]
        np.testing.assert_array_equal(selected.data, data[1:5])
        np.testing.assert_array_equal(
            dc.spool(path).select(time=(times[1], times[4]))[0].data, data[1:5]
        )
        np.testing.assert_array_equal(
            XdasV1().read_array(path, {"time": (1, 5)}), data[1:5]
        )
        with pytest.raises(FileNotFoundError):
            dc.read(path, time=(times[20], times[25]))
        assert len(dc.read(path, time=(times[-1] + np.timedelta64(1, "s"), None))) == 0

    def test_noncontiguous_and_scalar_selection(self, tmp_path):
        """Exact sample lists and scalar coordinates do not widen the loaded data."""
        path = tmp_path / "selection.nc"
        ds, data, times, distances = xdas_dataset()
        ds = ds.assign_coords(experiment=7)
        ds.to_netcdf(path, engine="h5netcdf")
        patch = dc.read(
            path, time=times[[1, 4]], distance=distances[[1, 3, 5]], experiment=(7, 7)
        )[0]
        np.testing.assert_array_equal(patch.data, data[np.ix_([1, 4], [1, 3, 5])])
