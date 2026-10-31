"""
Tests for optoDAS files.
"""

import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.lazy_array import LazyArray
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


class TestOffsetReads:
    """OptoDAS loads from offsets, scaling the stored samples as its reader does."""

    @pytest.fixture(params=["opto_das_1.hdf5", "decimated_optodas.hdf5"])
    def source(self, request):
        """The source of an example file, which records offsets and a scale."""
        return dc.read(fetch(request.param))[0]._source

    def test_equals_reader(self, source, stored_reads):
        """Offset loads equal the reader's, dtype too, without opening HDF5."""
        width = source.shape[1]
        windows = [(), (slice(2, 9),), (slice(3, 11), slice(width // 3, width))]
        loaded = [source[x].load() for x in windows]
        assert source.scale is not None and source.stored_dtype
        assert not stored_reads
        for window, out in zip(windows, loaded):
            expected = source._without_layout()[window].load()
            assert out.dtype == expected.dtype and out.flags.c_contiguous
            np.testing.assert_array_equal(out, expected)

    def test_indexed(self, source, tmp_path, stored_reads):
        """
        A reopened index keeps the scale, so a member it names loads from
        offsets. (An OptoDAS distance is no grid the index states, so merges
        read the patches instead.)
        """
        expected = source._without_layout().load()
        shutil.copy2(source.path, tmp_path / "a.hdf5")
        dc.spool(tmp_path).update()
        resolver = dc.spool(tmp_path).chunk(time=None)._catalog.resolver
        row = resolver.member_rows.iloc[0].to_dict()
        stored_reads.clear()
        out = resolver._member_array_source(row, expected.shape).load()
        assert out.dtype == expected.dtype and not stored_reads
        np.testing.assert_array_equal(out, expected)

    def test_frame_parquet(self, source, tmp_path):
        """A frame keeps the scale and stored dtype through parquet."""
        pytest.importorskip("pyarrow")
        array = LazyArray.from_sources([source[0:10], source[20:30]])
        array.to_frame().to_parquet(tmp_path / "frame.parquet")
        stored = pd.read_parquet(tmp_path / "frame.parquet")
        back = LazyArray.from_frame(stored, array.shape, array.dtype)
        assert {(x.scale, x.stored_dtype) for x in back.sources} == {
            (source.scale, source.stored_dtype)
        }
        np.testing.assert_array_equal(back.load(), array.load())

    def test_no_scale(self, tmp_path):
        """A file stating no scale records none, and loads its stored samples."""
        path = shutil.copy2(fetch("decimated_optodas.hdf5"), tmp_path / "a.hdf5")
        with h5py.File(path, "r+") as fi:
            del fi["header/dataScale"]
            stored = fi["data"][:]
        source = dc.read(path)[0]._source
        assert source.offsets and source.scale is None and not source.stored_dtype
        np.testing.assert_array_equal(source.load(), stored)
