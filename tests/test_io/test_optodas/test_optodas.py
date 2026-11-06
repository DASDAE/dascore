"""
Tests for optoDAS files.
"""

import json
import shutil
from fractions import Fraction

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io._index import planned
from dascore.io.optodas import OptoDASV8
from dascore.utils.downloader import fetch


class TestOptoDASIssues:
    """Test case related to issues in OptoDAS parser."""

    def test_scan_distance_units_preserved(self):
        """Snapped and exact scan coordinates should retain header units."""
        path = fetch("decimated_optodas.hdf5")
        fiber_io = OptoDASV8()

        for snap in (True, False):
            payload = fiber_io.scan(path, snap=snap)[0]
            distance = payload.coords.get_coord("distance")
            assert distance.units == dc.get_quantity("m")


class TestTimeStep:
    """The time step comes from the header's unitScale."""

    @pytest.mark.parametrize("first", [0, 3])
    def test_exact_step(self, tmp_path, first):
        """A unitScale of 1/1024 s gives an exact step, and a shifted start."""
        path = tmp_path / "hz_1024.hdf5"
        shutil.copyfile(fetch("opto_das_1.hdf5"), path)
        with h5py.File(path, "r+") as fi:
            crange = fi["header/dimensionRanges/dimension0"]
            crange["unitScale"][()] = 1 / 1024
            crange["min"][()] = first
            crange["max"][()] = first + fi["data"].shape[0] - 1
            t1 = dc.to_datetime64(fi["header/time"][()])
        for coord in (
            OptoDASV8().read(path)[0].get_coord("time"),
            OptoDASV8().scan(path)[0].get_coord("time"),
        ):
            assert coord.step_exact == Fraction(1, 1024)
            offset = round(first * 10**9 / 1024)
            assert coord.min() == t1 + np.timedelta64(offset, "ns")


class TestReadArray:
    """Tests for slicing the data dataset directly."""

    @pytest.fixture(scope="class")
    def transposed_path(self, tmp_path_factory):
        """An OptoDAS file whose header states the other dimension order."""
        source = fetch("opto_das_1.hdf5")
        path = tmp_path_factory.mktemp("optodas_transposed") / "transposed.hdf5"
        with h5py.File(source, "r") as src, h5py.File(path, "w") as dest:
            for name in src:
                src.copy(name, dest)
            names = [x.decode() for x in dest["header"]["dimensionNames"][:]]
            del dest["header"]["dimensionNames"]
            dest["header"]["dimensionNames"] = np.array(
                [x.encode() for x in names[::-1]]
            )
            data = dest["data"][:]
            del dest["data"]
            dest["data"] = data.T
        return path

    def test_dimension_order_follows_the_header(self, transposed_path):
        """The header names the stored order; a hardcoded one would transpose."""
        io = OptoDASV8()
        with h5py.File(transposed_path, "r") as h5:
            stored = h5["data"][:] * h5["header/dataScale"][()]
        # the header now calls the first axis distance, so the metadata's
        # order is (distance, time) and the time window is the second
        assert io.get_metadata(transposed_path)[0].dims == ("distance", "time")
        out = io.read_array(transposed_path, (None, (1, 4)))
        np.testing.assert_allclose(out, stored[:, 1:4], rtol=1e-6)


class TestDataScale:
    """Storage scaling must produce values matching the declared units."""

    @pytest.fixture
    def scaled_file(self, tmp_path):
        """Make a writable copy of the existing OptoDAS fixture."""
        path = tmp_path / "scaled.hdf5"
        shutil.copyfile(fetch("decimated_optodas.hdf5"), path)
        return path

    @pytest.mark.parametrize("version", [8, 9, 10, 11])
    @pytest.mark.parametrize("dtype", ["int16", "float32", "float64"])
    @pytest.mark.parametrize("scale", [0.004693713039159775, 1.0, None])
    def test_scaled_values(self, scaled_file, dtype, scale, version):
        """Scale all storage dtypes, preserving legacy files without a scale."""
        with h5py.File(scaled_file, "r+") as fi:
            shape = fi["data"].shape
            values = np.resize(np.array([-12, 0, 7, 103], dtype=dtype), shape)
            del fi["data"]
            fi["data"] = values
            del fi["fileVersion"]
            fi["fileVersion"] = version
            if "dataScale" in fi["header"]:
                del fi["header/dataScale"]
            if scale is not None:
                fi["header/dataScale"] = scale
        result = dc.read(scaled_file)[0]
        expected = values.astype("float64") * (1.0 if scale is None else scale)
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)
        expected_dtype = dtype if scale is None else np.result_type(dtype, np.float32)
        assert result.data.dtype == expected_dtype
        summary = dc.scan(scaled_file)[0]
        assert summary.dtype == str(expected_dtype)
        assert result.attrs.data_units == summary.attrs.data_units

    def test_subset(self, scaled_file):
        """Reading a subset scales exactly once and keeps coordinates aligned."""
        with h5py.File(scaled_file, "r+") as fi:
            fi["header/dataScale"][...] = 0.25
            stored = fi["data"][:]
        full = dc.read(scaled_file)[0]
        np.testing.assert_allclose(full.data, stored.astype("float64") * 0.25)
        bounds = {
            dim: (full.get_array(dim)[1], full.get_array(dim)[-2]) for dim in full.dims
        }
        subset = dc.read(scaled_file, **bounds)[0]
        expected = full.select(**bounds)
        np.testing.assert_array_equal(subset.data, expected.data)
        assert subset.coords == expected.coords

    def test_read_array_scaled(self, scaled_file):
        """read_array applies the same scale as read, and scan reports its dtype."""
        with h5py.File(scaled_file, "r+") as fi:
            fi["header/dataScale"][...] = 0.25
        out = OptoDASV8().read_array(scaled_file, ((1, 4),))
        expected = dc.read(scaled_file)[0].select(samples=True, time=(1, 4))
        np.testing.assert_array_equal(out, expected.data)
        assert str(out.dtype) == dc.scan(scaled_file)[0].dtype


class TestIndexedMerge:
    """An index row restates the OptoDAS attrs, so merges plan from it."""

    @pytest.fixture(scope="class")
    def directory(self, tmp_path_factory):
        """Two adjacent OptoDAS files, the second shifted by 10 s."""
        path = tmp_path_factory.mktemp("optodas_adjacent")
        for num in range(2):
            out = path / f"file_{num}.hdf5"
            shutil.copyfile(fetch("opto_das_1.hdf5"), out)
            with h5py.File(out, "r+") as h5:
                h5["header/time"][()] = 1695736872.0 + 10 * num
        dc.spool(path).update()
        return path

    def test_merge_equals_reader_merge(self, directory, monkeypatch):
        """The indexed merge loads no member through the reader."""
        loads = []
        original = planned.PlanResolver._load_member

        def counted(resolver, kwargs):
            loads.append(kwargs)
            return original(resolver, kwargs)

        monkeypatch.setattr(planned.PlanResolver, "_load_member", counted)
        merged = dc.spool(directory).chunk(time=None)[0]
        assert not loads
        files = sorted(directory.glob("*.hdf5"))
        patches = [dc.read(x)[0] for x in files]
        expected = dc.spool(patches).chunk(time=None)[0]
        assert merged.shape == (40000, 550)
        np.testing.assert_array_equal(merged.data, expected.data)
        assert merged.coords == expected.coords
        assert merged.attrs.model_dump() == expected.attrs.model_dump()
        assert merged.attrs.schema_version == ""
        assert merged.attrs.gauge_length == patches[0].attrs.gauge_length

    def test_reopened_index_keeps_defaults(self, directory):
        """A persisted index states the defaults it was written with."""
        row = dc.spool(directory)._df.iloc[0]
        assert row["_attrs_complete"]
        defaults = {"gauge_length": None, "schema_version": ""}
        assert json.loads(row["_attr_defaults"]) == defaults
