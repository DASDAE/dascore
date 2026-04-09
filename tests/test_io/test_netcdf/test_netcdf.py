"""Tests for NetCDF IO with CF conventions."""

from __future__ import annotations

import importlib.util
from typing import ClassVar

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.netcdf import core as netcdf_core
from dascore.io.netcdf import utils as netcdf_utils
from dascore.io.netcdf.utils import (
    get_cf_version,
    is_netcdf4_file,
)

pytest.importorskip("xarray")


def _get_xarray_netcdf_engine() -> str | None:
    """Return an xarray engine that can open NetCDF-4 files, if available."""
    if importlib.util.find_spec("netCDF4"):
        return "netcdf4"
    if importlib.util.find_spec("h5netcdf"):
        return "h5netcdf"
    return None


def _require_xarray_netcdf_engine() -> str:
    """Return an xarray NetCDF backend or skip the test."""
    if not importlib.util.find_spec("xarray"):
        pytest.skip("xarray not installed")
    engine = _get_xarray_netcdf_engine()
    if engine is None:
        pytest.skip("xarray NetCDF-4 backend not installed")
    return engine


def _assert_patch_round_trip_equal(expected: dc.Patch, observed: dc.Patch) -> None:
    """Assert two patches are equivalent for round-trip testing."""
    assert expected.equals(observed)
    assert expected.dims == observed.dims
    assert set(expected.coords.coord_map) == set(observed.coords.coord_map)
    expected_attrs = expected.attrs.model_dump()
    observed_attrs = observed.attrs.model_dump()
    expected_attrs.pop("_source_patch_id", None)
    observed_attrs.pop("_source_patch_id", None)
    assert expected_attrs == observed_attrs


def _assert_patch_compatible_with_xarray_output(
    expected: dc.Patch, observed: dc.Patch
) -> None:
    """Assert xarray-read DASCore output preserves patch content plus CF attrs."""
    assert expected.equals(observed)
    assert expected.dims == observed.dims
    assert set(expected.coords.coord_map) == set(observed.coords.coord_map)
    observed_attrs = observed.attrs.model_dump()
    for key, value in expected.attrs.model_dump().items():
        assert observed_attrs.get(key) == value


@pytest.fixture
def example_patch():
    """Create an example patch for NetCDF round-trip tests."""
    return dc.get_example_patch("random_das")


@pytest.fixture
def patch_with_attrs(example_patch):
    """Create a patch with representative string attrs preserved in NetCDF."""
    return example_patch.update(
        attrs={
            "station": "TEST_STATION",
            "network": "TEST_NET",
            "instrument_id": "TEST_INST",
            "acquisition_id": "ACQ_01",
            "tag": "netcdf_roundtrip",
            "data_type": "strain_rate",
            "data_category": "DAS",
        }
    )


@pytest.fixture
def patch_with_non_dim_coords(example_patch):
    """Create a patch with 1D and 2D non-dimensional coordinates."""
    shape = example_patch.shape
    latitude = np.linspace(40.0, 41.0, shape[0])
    quality = np.broadcast_to(np.linspace(0.0, 1.0, shape[1]), shape)
    coords = example_patch.coords.update(
        latitude=("distance", latitude),
        quality=(("distance", "time"), quality),
    )
    return example_patch.new(coords=coords)


@pytest.fixture
def minimal_cf_netcdf_path(tmp_path):
    """Create a minimal CF-compliant NetCDF file using only h5py."""
    path = tmp_path / "minimal_cf.nc"
    with h5py.File(path, "w") as h5file:
        h5file.attrs["Conventions"] = "CF-1.8"
        h5file.attrs["_NCProperties"] = "version=2,netcdf=4.9.0"
        time_ds = h5file.create_dataset("time", data=np.arange(10))
        time_ds.make_scale("time")
        distance_ds = h5file.create_dataset("distance", data=np.arange(5))
        distance_ds.make_scale("distance")
        data_ds = h5file.create_dataset("data", data=np.zeros((5, 10)))
        data_ds.dims[0].attach_scale(distance_ds)
        data_ds.dims[1].attach_scale(time_ds)
    return path


@pytest.fixture
def rng():
    """Return a seeded generator for synthetic NetCDF test data."""
    return np.random.default_rng(0)


class TestNetCDFUtils:
    """Tests for NetCDF utility functions."""

    def test_get_cf_version(self, tmp_path):
        """Test extracting CF version from NetCDF file."""
        path = tmp_path / "test_cf_version.nc"

        # Test CF-1.8 format
        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = np.bytes_("CF-1.8")

        with h5py.File(path, "r") as h5file:
            version = get_cf_version(h5file)
            assert version == "1.8"

    def test_get_cf_version_space_format(self, tmp_path):
        """Test extracting CF version with space format."""
        path = tmp_path / "test_cf_space.nc"

        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = b"CF 1.7"

        with h5py.File(path, "r") as h5file:
            version = get_cf_version(h5file)
            assert version == "1.7"

    def test_get_cf_version_with_comma_separated_conventions(self, tmp_path):
        """Version parsing should tolerate additional comma-separated conventions."""
        path = tmp_path / "test_cf_comma.nc"

        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = "CF-1.8, ACDD-1.3"

        with h5py.File(path, "r") as h5file:
            version = get_cf_version(h5file)
            assert version == "1.8"

    def test_get_cf_version_none(self, tmp_path):
        """Test extracting CF version when none present."""
        path = tmp_path / "test_no_cf.nc"

        with h5py.File(path, "w") as h5file:
            h5file.attrs["other_attr"] = b"value"

        with h5py.File(path, "r") as h5file:
            version = get_cf_version(h5file)
            assert version is None

    def test_is_netcdf4_file_from_conventions_bytes(self, tmp_path):
        """CF conventions alone should identify a NetCDF-like file."""
        path = tmp_path / "test_conventions.nc"
        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = b"CF-1.8"

        with h5py.File(path, "r") as h5file:
            assert is_netcdf4_file(h5file)

    def test_is_netcdf4_file_from_dimension_scales(self, tmp_path):
        """Dimension scales alone should not identify a NetCDF-like file."""
        path = tmp_path / "test_dimension_scale.nc"
        with h5py.File(path, "w") as h5file:
            time_ds = h5file.create_dataset("time", data=np.arange(5))
            time_ds.make_scale("time")

        with h5py.File(path, "r") as h5file:
            assert not is_netcdf4_file(h5file)

    def test_is_netcdf4_file_handles_attribute_error(self):
        """Attribute errors during detection should return False."""

        class BrokenFile:
            @property
            def attrs(self):
                raise AttributeError

        assert not is_netcdf4_file(BrokenFile())

    def test_is_netcdf4_file_decodes_byte_conventions_from_mapping(self):
        """Conventions bytes from a mapping-like attrs object should decode."""

        class FakeFile:
            def __init__(self):
                self.attrs = {"Conventions": b"CF-1.8"}

            def values(self):
                return []

        assert is_netcdf4_file(FakeFile())

    def test_is_netcdf4_file_from_dimension_list_attr(self, tmp_path):
        """DIMENSION_LIST attrs alone should not mark a file as NetCDF-like."""
        path = tmp_path / "test_dimension_list.nc"
        with h5py.File(path, "w") as h5file:
            data_ds = h5file.create_dataset("data", data=np.ones((2, 2)))
            data_ds.attrs["DIMENSION_LIST"] = "present"

        with h5py.File(path, "r") as h5file:
            assert not is_netcdf4_file(h5file)

    def test_get_cf_version_decodes_bytes(self, tmp_path):
        """CF version extraction should decode byte attrs."""
        path = tmp_path / "cf_version_bytes.nc"
        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = np.bytes_("CF-1.9")

        with h5py.File(path, "r") as h5file:
            assert get_cf_version(h5file) == "1.9"


class TestNetCDFCoreHelpers:
    """Direct tests for lightweight NetCDF core helpers."""

    def test_get_xarray_data_var_name(self):
        """Dataset helper should find expected data variables or raise."""
        xr = pytest.importorskip("xarray")
        ds_with_data = xr.Dataset({"data": (("x",), [1, 2])})
        ds_single = xr.Dataset({"signal": (("x",), [1, 2])})
        ds_multi = xr.Dataset({"signal": (("x",), [1, 2]), "other": (("x",), [3, 4])})

        class _DatasetWithNone:
            data_vars: ClassVar = {None: object(), "distance_indices": object()}

        assert netcdf_utils.get_xarray_data_var_name(ds_with_data) == "data"
        assert netcdf_utils.get_xarray_data_var_name(_DatasetWithNone()) is None
        assert netcdf_utils.get_xarray_data_var_name(ds_single) == "signal"
        with pytest.raises(ValueError, match="No suitable data variable found"):
            netcdf_utils.get_xarray_data_var_name(ds_multi)

    def test_get_format_false_cases(self, minimal_cf_netcdf_path, tmp_path):
        """Format detection should reject missing and invalid CF versions."""
        formatter = netcdf_core.NetCDFCFV18()

        no_cf_path = tmp_path / "no_cf_version.nc"
        with h5py.File(no_cf_path, "w") as h5file:
            h5file.attrs["_NCProperties"] = "version=2"

        invalid_cf_path = tmp_path / "invalid_cf_version.nc"
        with h5py.File(invalid_cf_path, "w") as h5file:
            h5file.attrs["Conventions"] = "CF-not-a-number"

        assert dc.get_format(minimal_cf_netcdf_path) == ("NETCDF_CF", "1.8")
        with h5py.File(no_cf_path, "r") as h5file:
            assert formatter.get_format(h5file) is False
        with h5py.File(invalid_cf_path, "r") as h5file:
            assert formatter.get_format(h5file) is False

    def test_get_format_accepts_comma_separated_conventions(self, tmp_path):
        """Format detection should accept CF versions followed by extra conventions."""
        formatter = netcdf_core.NetCDFCFV18()
        path = tmp_path / "comma_conventions.nc"

        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = "CF-1.8, ACDD-1.3"
            h5file.attrs["_NCProperties"] = "version=2"
            time_ds = h5file.create_dataset("time", data=np.arange(3))
            time_ds.make_scale("time")
            dist_ds = h5file.create_dataset("distance", data=np.arange(2))
            dist_ds.make_scale("distance")
            data_ds = h5file.create_dataset("data", data=np.ones((2, 3)))
            data_ds.dims[0].attach_scale(dist_ds)
            data_ds.dims[1].attach_scale(time_ds)

        with h5py.File(path, "r") as h5file:
            assert formatter.get_format(h5file) == ("NETCDF_CF", "1.8")

    def test_get_write_encoding_invalid_compression_raises(self):
        """Write encoding should reject unsupported compression values."""
        formatter = netcdf_core.NetCDFCFV18()

        with pytest.raises(ValueError, match="only gzip compression"):
            formatter._get_write_encoding(compression="szip")

    def test_get_write_encoding_explicit_chunks(self):
        """Write encoding should pass explicit chunk sizes through."""
        formatter = netcdf_core.NetCDFCFV18()

        out = formatter._get_write_encoding(chunks=(10, 20))

        assert out["chunksizes"] == (10, 20)

    def test_read_returns_empty_spool_for_empty_filtered_patch(
        self, minimal_cf_netcdf_path, monkeypatch
    ):
        """Read should return an empty spool after filtering removes all data."""

        class FakeDataArray:
            def __init__(self):
                self.data = np.ones((1, 1))

                def _make_coord(dims, vals):
                    return type("Coord", (), {"dims": dims, "values": vals})()

                self.coords = {
                    "distance": _make_coord(("distance",), np.array([0])),
                    "time": _make_coord(("time",), np.array([0])),
                }
                self.attrs = {}
                self.dims = ("distance", "time")
                self.shape = self.data.shape

            def load(self):
                return self

        class FakeDataset:
            def __init__(self):
                self.data_vars = {"data": FakeDataArray()}
                self.attrs = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, name):
                return self.data_vars["data"]

            def __getitem__(self, item):
                return self.data_vars[item]

        formatter = netcdf_core.NetCDFCFV18()
        fake_xarray = type(
            "FakeXarray",
            (),
            {"open_dataset": staticmethod(lambda *args, **kwargs: FakeDataset())},
        )
        empty_patch = dc.Patch(
            data=np.empty((0, 0)),
            coords=dc.get_coord_manager(
                coords={"distance": np.array([]), "time": np.array([])},
                dims=("distance", "time"),
            ),
            dims=("distance", "time"),
            attrs={"tag": "empty"},
        )

        def _optional_import(name, on_missing="raise"):
            if name == "xarray":
                return fake_xarray
            if name == "netCDF4":
                return object()
            return None

        monkeypatch.setattr(netcdf_core, "optional_import", _optional_import)
        monkeypatch.setattr(
            netcdf_core,
            "xarray_to_patch",
            lambda data_array: dc.Patch(
                data=data_array.data,
                coords=dc.get_coord_manager(
                    coords={
                        name: (coord.dims, coord.values)
                        for name, coord in data_array.coords.items()
                    },
                    dims=data_array.dims,
                ),
                dims=data_array.dims,
                attrs=dict(data_array.attrs),
            ),
        )
        monkeypatch.setattr(dc.Patch, "select", lambda self, **kwargs: empty_patch)

        spool = formatter.read(minimal_cf_netcdf_path, time=(0, 1))
        assert len(spool) == 0

    def test_write_uses_xarray_dataset_path(self, example_patch, tmp_path, monkeypatch):
        """Write should pass the xarray dataset and encoding through unchanged."""

        class FakeCoord:
            def __init__(self):
                self.attrs = {}

        class FakeDataVar:
            def __init__(self):
                self.attrs = {}

        class FakeDataset:
            def __init__(self):
                self.attrs = {}
                self.data = FakeDataVar()
                self.to_netcdf_calls = []

            def __getitem__(self, item):
                assert item == "data"
                return self.data

            def to_netcdf(self, *args, **kwargs):
                self.to_netcdf_calls.append((args, kwargs))

        class FakeDataArray:
            def __init__(self):
                self.coords = {
                    "distance": FakeCoord(),
                    "time": FakeCoord(),
                    "latitude": FakeCoord(),
                }
                self.dataset = FakeDataset()

            def rename(self, name):
                assert name == "data"
                return self

            def to_dataset(self):
                return self.dataset

        fake_data_array = FakeDataArray()
        formatter = netcdf_core.NetCDFCFV18()

        def _optional_import(name, on_missing="raise"):
            if name == "xarray":
                return object()
            if name == "netCDF4":
                return object()
            return None

        monkeypatch.setattr(
            netcdf_core,
            "optional_import",
            _optional_import,
        )
        monkeypatch.setattr(
            netcdf_core,
            "patch_to_xarray",
            lambda patch: fake_data_array,
        )

        out_path = tmp_path / "write_stub.nc"
        formatter.write(example_patch, out_path)

        dataset = fake_data_array.dataset
        assert dataset.attrs == {"Conventions": "CF-1.8"}
        assert dataset.data.attrs == {}
        assert dataset.to_netcdf_calls
        _args, kwargs = dataset.to_netcdf_calls[0]
        assert kwargs["encoding"] is None


class TestNetCDFIO:
    """Tests for NetCDF IO functionality."""

    @pytest.fixture
    def example_patch(self):
        """Create an example patch for testing."""
        return dc.get_example_patch("random_das")

    @pytest.fixture
    def netcdf_path(self, example_patch, tmp_path):
        """Create a test NetCDF file."""
        _require_xarray_netcdf_engine()
        path = tmp_path / "test.nc"
        # Write patch to NetCDF format
        dc.write(example_patch, path, file_format="netcdf_cf")
        return path

    def test_write_netcdf(self, example_patch, tmp_path):
        """Test writing a patch to NetCDF format."""
        _require_xarray_netcdf_engine()
        path = tmp_path / "test_write.nc"

        # Write patch
        dc.write(example_patch, path, file_format="netcdf_cf")

        # Check file exists and is valid HDF5/NetCDF
        assert path.exists()

        # Open with h5py to verify structure
        with h5py.File(path, "r") as h5file:
            # Check it's detected as NetCDF
            assert is_netcdf4_file(h5file)

            # Check coordinates exist
            assert "time" in h5file
            assert "distance" in h5file

            # Check data variable exists
            assert "data" in h5file

            # Check dimension scales
            assert h5file["time"].is_scale
            assert h5file["distance"].is_scale

    def test_read_netcdf(self, netcdf_path):
        """Test reading a NetCDF file."""
        spool = dc.read(netcdf_path, file_format="netcdf_cf")
        patch = spool[0]
        assert isinstance(patch, dc.Patch)
        assert "time" in patch.coords
        assert "distance" in patch.coords
        assert patch.data.ndim == 2
        assert patch.attrs["_source_patch_id"] == "data"

    def test_scan_netcdf(self, netcdf_path):
        """Test scanning a NetCDF file for metadata."""
        summary_list = dc.scan(netcdf_path, file_format="netcdf_cf")
        assert len(summary_list) == 1
        summary = summary_list[0]
        assert summary.source_format == "NETCDF_CF"
        assert "time" in summary.coords
        assert "distance" in summary.coords
        assert summary.source_patch_id == "data"

    def test_get_format(self, minimal_cf_netcdf_path):
        """Test format detection."""
        assert dc.get_format(minimal_cf_netcdf_path) == ("NETCDF_CF", "1.8")

    def test_get_format_without_xarray_import(
        self, minimal_cf_netcdf_path, monkeypatch
    ):
        """Format detection should not depend on xarray being importable."""
        import importlib

        original_import_module = importlib.import_module

        def _import_module(name, package=None):
            if name == "xarray":
                raise ImportError("xarray disabled for test")
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _import_module)

        assert dc.get_format(minimal_cf_netcdf_path) == ("NETCDF_CF", "1.8")

    def test_round_trip(self, example_patch, tmp_path):
        """Test round-trip: patch -> NetCDF -> patch."""
        _require_xarray_netcdf_engine()
        path = tmp_path / "roundtrip.nc"

        # Write and read back
        dc.write(example_patch, path, file_format="netcdf_cf")
        spool = dc.read(path, file_format="netcdf_cf")
        recovered_patch = spool[0]

        # Check data preservation
        np.testing.assert_array_almost_equal(
            example_patch.data, recovered_patch.data, decimal=6
        )

        # Check coordinate preservation
        for coord_name in example_patch.coords.coord_map:
            orig_coord = example_patch.coords.get_array(coord_name)
            recovered_coord = recovered_patch.coords.get_array(coord_name)

            if coord_name == "time":
                # Time coordinates might have slight precision differences
                # due to CF time conversion (float64 seconds -> datetime64[ns])
                time_diff = np.abs(orig_coord - recovered_coord)
                assert np.all(
                    time_diff < np.timedelta64(200, "us")
                )  # 200 microsecond tolerance
            else:
                np.testing.assert_array_almost_equal(
                    orig_coord, recovered_coord, decimal=6
                )


class TestNetCDFXarrayCompatibility:
    """Tests for compatibility between DASCore NetCDF output and xarray."""

    @pytest.fixture(
        params=("example_patch", "patch_with_attrs", "patch_with_non_dim_coords")
    )
    def patch_variant(self, request):
        """Parametrize representative patch variants for compatibility checks."""
        return request.getfixturevalue(request.param)

    def test_patch_to_xarray_can_be_serialized_with_xarray(
        self, patch_variant, tmp_path
    ):
        """A patch converted to xarray should round-trip through xarray IO."""
        xr = pytest.importorskip("xarray")
        path = tmp_path / "xarray_roundtrip.nc"

        data_array = dc.io.patch_to_xarray(patch_variant)
        data_array.to_netcdf(path, engine="scipy")

        reopened = xr.open_dataarray(path, engine="scipy")
        round_tripped = dc.io.xarray_to_patch(reopened)
        reopened.close()

        _assert_patch_round_trip_equal(patch_variant, round_tripped)

    def test_current_netcdf_output_is_readable_by_xarray(self, patch_variant, tmp_path):
        """DASCore NetCDF output should be consumable by xarray backends."""
        xr = pytest.importorskip("xarray")
        engine = _require_xarray_netcdf_engine()

        path = tmp_path / "dascore_netcdf.nc"
        dc.write(patch_variant, path, file_format="netcdf_cf")

        data_array = xr.open_dataarray(path, engine=engine)
        round_tripped = dc.io.xarray_to_patch(data_array)
        data_array.close()

        _assert_patch_compatible_with_xarray_output(patch_variant, round_tripped)

    def test_written_file_exposes_expected_metadata_to_xarray(
        self, patch_with_attrs, tmp_path
    ):
        """Xarray should see the basic dataset structure on DASCore output."""
        xr = pytest.importorskip("xarray")
        engine = _require_xarray_netcdf_engine()

        path = tmp_path / "metadata_visible.nc"
        dc.write(patch_with_attrs, path, file_format="netcdf_cf")

        with xr.open_dataset(path, engine=engine) as dataset:
            assert list(dataset.data_vars) == ["data"]
            assert dataset.attrs["Conventions"] == "CF-1.8"
            assert set(dataset.coords) == {"distance", "time"}
            assert dataset["data"].attrs["station"] == "TEST_STATION"
            assert dataset["data"].attrs["network"] == "TEST_NET"
            assert dataset["data"].attrs["data_type"] == "strain_rate"

    def test_dascore_and_xarray_roundtrip_agree_for_non_dim_coords(
        self, patch_with_non_dim_coords, tmp_path
    ):
        """The same patch should round-trip through both IO paths identically."""
        xr = pytest.importorskip("xarray")
        _require_xarray_netcdf_engine()
        path = tmp_path / "xarray_non_dim.nc"

        dascore_path = tmp_path / "dascore_non_dim.nc"
        dc.write(patch_with_non_dim_coords, dascore_path, file_format="netcdf_cf")
        dascore_patch = dc.read(dascore_path, file_format="netcdf_cf")[0]

        data_array = dc.io.patch_to_xarray(patch_with_non_dim_coords)
        data_array.to_netcdf(path, engine="scipy")
        reopened = xr.open_dataarray(path, engine="scipy")
        xarray_patch = dc.io.xarray_to_patch(reopened)
        reopened.close()

        _assert_patch_round_trip_equal(patch_with_non_dim_coords, dascore_patch)
        _assert_patch_round_trip_equal(patch_with_non_dim_coords, xarray_patch)
        _assert_patch_round_trip_equal(dascore_patch, xarray_patch)


class TestNetCDFEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def multi_patch_spool(self):
        """Create a spool with multiple patches for testing."""
        patch1 = dc.get_example_patch("random_das")
        patch2 = dc.get_example_patch("random_das")
        return dc.spool([patch1, patch2])

    @pytest.fixture
    def invalid_hdf5_file(self, tmp_path):
        """Create an invalid HDF5 file (not NetCDF)."""
        path = tmp_path / "invalid.nc"
        with h5py.File(path, "w") as h5file:
            rng = np.random.default_rng()
            h5file.create_dataset("random_data", data=rng.standard_normal((100, 50)))
        return path

    @pytest.fixture
    def compressed_netcdf_file(self, tmp_path):
        """Create a compressed NetCDF file for testing."""
        _require_xarray_netcdf_engine()
        patch = dc.get_example_patch("random_das")
        path = tmp_path / "compressed.nc"
        dc.write(
            patch,
            path,
            file_format="netcdf_cf",
            compression="gzip",
            compression_opts=9,
        )
        return path, patch

    def test_empty_spool_write_error(self, tmp_path):
        """Test that writing empty spool raises error."""
        path = tmp_path / "empty.nc"
        empty_spool = dc.spool([])

        with pytest.raises(ValueError, match="Cannot write empty spool"):
            dc.write(empty_spool, path, file_format="netcdf_cf")

    def test_multi_patch_write_error(self, multi_patch_spool, tmp_path):
        """Test that multi-patch spool raises NotImplementedError."""
        path = tmp_path / "multi.nc"

        with pytest.raises(
            NotImplementedError, match="Multi-patch spools not yet supported"
        ):
            dc.write(multi_patch_spool, path, file_format="netcdf_cf")

    def test_invalid_netcdf_file(self, invalid_hdf5_file):
        """Test behavior with invalid NetCDF file."""
        with h5py.File(invalid_hdf5_file, "r") as h5file:
            assert not is_netcdf4_file(h5file)

    def test_compression_options(self, compressed_netcdf_file):
        """Test NetCDF file creation with compression options."""
        path, original_patch = compressed_netcdf_file

        spool = dc.read(path, file_format="netcdf_cf")
        recovered_patch = spool[0]

        np.testing.assert_array_almost_equal(
            original_patch.data, recovered_patch.data, decimal=6
        )


class TestNetCDFUtilsAdvanced:
    """Additional tests for NetCDF utility functions."""

    def test_coordless_coord_manager_falls_back_without_tie_points(self):
        """Coord fallback should use direct vars or arange without tie points."""
        xr = pytest.importorskip("xarray")
        dataset = xr.Dataset(
            data_vars={"distance": (("distance",), np.array([0.0, 2.0, 4.0]))}
        )

        coords = netcdf_utils.get_coord_manager_for_coordless_data_var(
            dataset,
            dims=("distance", "time"),
            shape=(3, 4),
        )

        np.testing.assert_array_equal(
            coords.coord_map["distance"].values, [0.0, 2.0, 4.0]
        )
        np.testing.assert_array_equal(coords.coord_map["time"].values, np.arange(4))

    @pytest.fixture
    def cf_compliant_file(self, tmp_path):
        """Create a CF-compliant NetCDF file for testing."""
        _require_xarray_netcdf_engine()
        path = tmp_path / "cf_compliant.nc"
        patch = dc.get_example_patch("random_das")
        patch = patch.update(
            attrs={
                "station": "TEST_STATION",
                "network": "TEST_NET",
                "data_type": "strain_rate",
            }
        )
        dc.write(patch, path, file_format="netcdf_cf")
        return path

    def test_coordinate_filtering_during_read(self, cf_compliant_file):
        """Test coordinate filtering during NetCDF read."""
        # Read with time filtering
        spool = dc.read(cf_compliant_file, file_format="netcdf_cf")
        original_patch = spool[0]

        # Get time bounds for filtering
        time_coord = original_patch.coords.get_array("time")
        time_start = time_coord[10]
        time_end = time_coord[50]

        # Read with filtering
        filtered_spool = dc.read(
            cf_compliant_file, file_format="netcdf_cf", time=(time_start, time_end)
        )
        filtered_patch = filtered_spool[0]

        # Should have fewer time samples
        assert filtered_patch.data.shape[1] < original_patch.data.shape[1]

    def test_netcdf_format_detection_edge_cases(self, tmp_path, rng):
        """Test NetCDF format detection edge cases."""
        _require_xarray_netcdf_engine()
        # Test with newer CF version via file read
        path1 = tmp_path / "newer_cf.nc"
        with h5py.File(path1, "w") as h5file:
            h5file.attrs["Conventions"] = b"CF-2.0"
            h5file.attrs["_NCProperties"] = b"version=2,netcdf=test"
            # Add required data for valid NetCDF
            time_ds = h5file.create_dataset("time", data=np.arange(10))
            time_ds.make_scale("time")
            dist_ds = h5file.create_dataset("distance", data=np.arange(50))
            dist_ds.make_scale("distance")
            h5file.create_dataset("data", data=rng.random((50, 10)))

        # Test that it can be detected and read
        spool = dc.read(path1, file_format="netcdf_cf")
        assert len(spool) == 1

        # Test with older supported CF version
        path2 = tmp_path / "older_cf.nc"
        with h5py.File(path2, "w") as h5file:
            h5file.attrs["Conventions"] = b"CF-1.6"
            h5file.attrs["_NCProperties"] = b"version=2,netcdf=test"
            time_ds = h5file.create_dataset("time", data=np.arange(10))
            time_ds.make_scale("time")
            dist_ds = h5file.create_dataset("distance", data=np.arange(50))
            dist_ds.make_scale("distance")
            h5file.create_dataset("data", data=rng.random((50, 10)))

        # Should be able to read older CF versions
        spool = dc.read(path2, file_format="netcdf_cf")
        assert len(spool) == 1
        patch = spool[0]
        assert set(patch.coords.coord_map) == {"time", "distance"}
        assert patch.attrs.tag == ""
        assert patch.attrs["_source_patch_id"] == "data"

    def test_error_conditions(self, tmp_path):
        """Test various error conditions."""
        _require_xarray_netcdf_engine()
        # Test reading file with no data variables
        path = tmp_path / "no_data.nc"
        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = b"CF-1.8"
            time_ds = h5file.create_dataset("time", data=np.arange(10))
            time_ds.make_scale("time")

        # Use DASCore's standard reading interface to test error conditions
        with pytest.raises(ValueError, match="No suitable data variable found"):
            dc.read(path, file_format="netcdf_cf")
