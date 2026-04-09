"""Tests for NetCDF IO with CF conventions."""

from __future__ import annotations

import importlib.util

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io.netcdf import core as netcdf_core
from dascore.io.netcdf import utils as netcdf_utils
from dascore.io.netcdf.utils import (
    cf_time_to_datetime64,
    datetime64_to_cf_time,
    extract_patch_attrs_from_netcdf,
    find_main_data_variable,
    get_cf_data_attrs,
    get_cf_global_attrs,
    get_cf_version,
    is_netcdf4_file,
    read_netcdf_coordinates,
    validate_cf_compliance,
)
from dascore.utils.downloader import fetch


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

    @pytest.fixture
    def test_datetime_array(self):
        """Create test datetime array for CF time conversion tests."""
        return np.array(
            [
                "2023-01-01T00:00:00",
                "2023-01-01T01:00:00",
                "2023-01-01T02:00:00",
            ],
            dtype="datetime64[ns]",
        )

    @pytest.fixture
    def expected_cf_times(self):
        """Expected CF time values for test datetime array."""
        expected_base = 1672531200.0  # 2023-01-01T00:00:00 in epoch seconds
        return np.array([expected_base, expected_base + 3600, expected_base + 7200])

    @pytest.fixture
    def test_patch_attrs(self):
        """Create test patch attributes for global attrs tests."""
        return dc.PatchAttrs(
            station="TEST_STATION",
            network="TEST_NET",
            instrument_id="TEST_INST",
        )

    def test_datetime64_to_cf_time(self, test_datetime_array, expected_cf_times):
        """Test conversion of datetime64 to CF time format."""
        cf_times = datetime64_to_cf_time(test_datetime_array)
        np.testing.assert_array_almost_equal(cf_times, expected_cf_times)

    def test_get_cf_data_attrs(self):
        """Test CF data attributes generation."""
        attrs = get_cf_data_attrs("strain_rate")
        assert attrs["standard_name"] == "strain_rate"
        assert attrs["units"] == "1/s"
        assert "_FillValue" in attrs

    def test_get_cf_global_attrs(self, test_patch_attrs):
        """Test CF global attributes generation."""
        attrs = get_cf_global_attrs(test_patch_attrs, "1.8")
        assert attrs["Conventions"] == "CF-1.8"
        assert attrs["station"] == "TEST_STATION"
        assert attrs["network"] == "TEST_NET"
        assert attrs["instrument"] == "TEST_INST"
        assert "institution" not in attrs

    def test_cf_time_to_datetime64(self):
        """Test CF time to datetime64 conversion."""
        # Test with seconds since epoch
        cf_times = np.array([0, 3600, 7200])  # 0, 1, 2 hours since epoch
        units = "seconds since 1970-01-01 00:00:00"

        result = cf_time_to_datetime64(cf_times, units)

        expected = np.array(
            ["1970-01-01T00:00:00", "1970-01-01T01:00:00", "1970-01-01T02:00:00"],
            dtype="datetime64[ns]",
        )

        np.testing.assert_array_equal(result, expected)

    def test_cf_time_to_datetime64_different_units(self):
        """Test CF time conversion with different time units."""
        # Test with days since epoch
        cf_times = np.array([0, 1, 2])
        units = "days since 2023-01-01 00:00:00"

        result = cf_time_to_datetime64(cf_times, units)

        expected = np.array(
            ["2023-01-01T00:00:00", "2023-01-02T00:00:00", "2023-01-03T00:00:00"],
            dtype="datetime64[ns]",
        )

        np.testing.assert_array_equal(result, expected)

    def test_cf_time_to_datetime64_invalid_units(self):
        """Test CF time conversion with invalid units."""
        cf_times = np.array([0, 1, 2])
        invalid_units = "invalid format"

        with pytest.raises(ValueError, match="Invalid CF time units format"):
            cf_time_to_datetime64(cf_times, invalid_units)

    def test_find_main_data_variable(self, tmp_path, rng):
        """Test finding main data variable in NetCDF file."""
        path = tmp_path / "test_data_var.nc"

        with h5py.File(path, "w") as h5file:
            # Create dimension scales
            time_ds = h5file.create_dataset("time", data=np.arange(100))
            time_ds.make_scale("time")

            # Create various datasets.
            h5file.create_dataset("metadata", data=np.array([1, 2, 3]))
            h5file.create_dataset("other_data", data=rng.random((50, 50)))

            # Create priority data variable
            strain_data = h5file.create_dataset(
                "strain_data", data=rng.random((100, 50))
            )
            strain_data.attrs["standard_name"] = b"strain"

        with h5py.File(path, "r") as h5file:
            main_var = find_main_data_variable(h5file)
            assert main_var == "strain_data"

    def test_find_main_data_variable_no_priority(self, tmp_path, rng):
        """Test finding main data variable when no priority match."""
        path = tmp_path / "test_no_priority.nc"

        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("first_candidate", data=rng.random((50, 50)))
            h5file.create_dataset("second_candidate", data=rng.random((60, 60)))

        with h5py.File(path, "r") as h5file:
            main_var = find_main_data_variable(h5file)
            assert main_var == "first_candidate"  # Returns first candidate

    def test_find_main_data_variable_none_found(self, tmp_path):
        """Test finding main data variable when none found."""
        path = tmp_path / "test_none_found.nc"

        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("only_1d", data=np.array([1, 2, 3]))  # Only 1D data

        with h5py.File(path, "r") as h5file:
            main_var = find_main_data_variable(h5file)
            assert main_var is None

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

    def test_get_cf_data_attrs_partial_match(self):
        """Partial data-type matches should map to known CF names."""
        attrs = get_cf_data_attrs("my_acceleration_trace")
        assert attrs["standard_name"] == "acceleration"

    def test_get_cf_global_attrs_optional_fields(self):
        """Optional patch attrs should be propagated to global metadata."""
        attrs = dc.PatchAttrs(
            station="TEST_STATION",
            network="TEST_NET",
            instrument_id="TEST_INST",
            acquisition_id="ACQ_01",
            tag="taggy",
            data_category="DAS",
            data_type="strain_rate",
            history=("step-1", "step-2"),
            category="processed",
        )

        out = get_cf_global_attrs(attrs, "1.8")

        assert out["acquisition"] == "ACQ_01"
        assert out["tag"] == "taggy"
        assert out["data_category"] == "DAS"
        assert out["data_type"] == "strain_rate"
        assert out["category"] == "processed"
        assert out["processing_history"] == "step-1 | step-2"

    def test_extract_patch_attrs_source_data_type_fallback(self, tmp_path):
        """Source data type should populate data_type when primary attr is absent."""
        path = tmp_path / "source_data_type.nc"
        with h5py.File(path, "w") as h5file:
            h5file.attrs["network"] = np.bytes_("TEST_NET")
            h5file.attrs["source_data_type"] = np.bytes_("strain_rate")

        with h5py.File(path, "r") as h5file:
            attrs = extract_patch_attrs_from_netcdf(h5file)

        assert attrs["network"] == "TEST_NET"
        assert attrs["data_type"] == "strain_rate"

    def test_handle_time_interpolation_without_indices(self, tmp_path):
        """Time interpolation should fall back to raw time_values when needed."""
        path = tmp_path / "time_interp.nc"
        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("time", data=np.arange(3))
            time_values = h5file.create_dataset("time_values", data=np.arange(3))
            time_values.attrs["units"] = "seconds since 1970-01-01 00:00:00"

        with h5py.File(path, "r") as h5file:
            out = netcdf_utils._handle_time_interpolation(h5file, "time", np.arange(3))

        expected = np.array(
            ["1970-01-01T00:00:00", "1970-01-01T00:00:01", "1970-01-01T00:00:02"],
            dtype="datetime64[ns]",
        )
        np.testing.assert_array_equal(out, expected)

    def test_handle_time_interpolation_invalid_units_returns_none(self, tmp_path):
        """Interpolation should fail quietly when auxiliary units are invalid."""
        path = tmp_path / "time_interp_invalid.nc"
        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("time", data=np.arange(3))
            time_values = h5file.create_dataset("time_values", data=np.arange(3))
            time_values.attrs["units"] = "invalid"

        with h5py.File(path, "r") as h5file:
            out = netcdf_utils._handle_time_interpolation(h5file, "time", np.arange(3))

        assert out is None

    def test_read_netcdf_coordinates_with_non_dim_coord_attrs(self, tmp_path):
        """Non-dimensional coordinates should honor _DASCORE_DIMS metadata."""
        path = tmp_path / "non_dim_coords.nc"
        with h5py.File(path, "w") as h5file:
            time_ds = h5file.create_dataset("time", data=np.arange(4))
            time_ds.make_scale("time")
            dist_ds = h5file.create_dataset("distance", data=np.arange(3))
            dist_ds.make_scale("distance")
            data_ds = h5file.create_dataset("data", data=np.ones((3, 4)))
            data_ds.dims[0].attach_scale(dist_ds)
            data_ds.dims[1].attach_scale(time_ds)
            data_ds.attrs["coordinates"] = np.bytes_("latitude")
            lat_ds = h5file.create_dataset("latitude", data=np.linspace(1.0, 2.0, 3))
            lat_ds.attrs["_DASCORE_DIMS"] = np.bytes_("distance")

        with h5py.File(path, "r") as h5file:
            coords = read_netcdf_coordinates(h5file)

        assert coords.dims == ("distance", "time")
        assert "latitude" in coords.coord_map
        assert coords.dim_map["latitude"] == ("distance",)

    def test_validate_cf_compliance_invalid_time_units(self, tmp_path):
        """Invalid time units should be reported as a CF issue."""
        path = tmp_path / "invalid_time_units.nc"
        with h5py.File(path, "w") as h5file:
            h5file.attrs["Conventions"] = "CF-1.8"
            time_ds = h5file.create_dataset("time", data=np.arange(4))
            time_ds.make_scale("time")
            time_ds.attrs["units"] = "seconds"

        with h5py.File(path, "r") as h5file:
            issues = validate_cf_compliance(h5file)

        assert any("invalid units" in issue for issue in issues)

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

    def test_handle_time_interpolation_decodes_units_and_handles_bad_units(
        self, tmp_path
    ):
        """Interpolation should decode bytes and handle unsupported units."""
        decoded_path = tmp_path / "time_interp_bytes.nc"
        with h5py.File(decoded_path, "w") as h5file:
            h5file.create_dataset("time", data=np.arange(3))
            time_values = h5file.create_dataset("time_values", data=np.arange(3))
            time_values.attrs["units"] = np.bytes_("seconds since 1970-01-01 00:00:00")
            time_indices = h5file.create_dataset("time_indices", data=np.array([0]))
            time_indices[...] = np.array([0])

        with h5py.File(decoded_path, "r") as h5file:
            out = netcdf_utils._handle_time_interpolation(h5file, "time", np.arange(3))

        assert out.dtype == "datetime64[ns]"

        invalid_path = tmp_path / "time_interp_unsupported.nc"
        with h5py.File(invalid_path, "w") as h5file:
            h5file.create_dataset("time", data=np.arange(3))
            time_values = h5file.create_dataset("time_values", data=np.arange(3))
            time_values.attrs["units"] = "fortnights since 2023-01-01"

        with h5py.File(invalid_path, "r") as h5file:
            out = netcdf_utils._handle_time_interpolation(h5file, "time", np.arange(3))

        assert out is None

    def test_read_netcdf_coordinates_handles_bad_refs_and_auxiliaries(self, tmp_path):
        """Coordinate reader should tolerate bad refs and skip auxiliaries."""
        path = tmp_path / "coords_bad_refs.nc"
        with h5py.File(path, "w") as h5file:
            h5file.create_group("aux_group")
            h5file.create_dataset("time_values", data=np.arange(3))
            data_ds = h5file.create_dataset("data", data=np.ones((2, 3)))
            data_ds.attrs["DIMENSION_LIST"] = np.array([[b"/missing"]], dtype="S8")
            data_ds.attrs["coordinates"] = np.bytes_("missing aux_group latitude")
            lat_ds = h5file.create_dataset("latitude", data=np.arange(2))
            lat_ds.attrs["_DASCORE_DIMS"] = np.bytes_("distance")
            dist_ds = h5file.create_dataset("distance", data=np.arange(2))
            dist_ds.make_scale("distance")
            time_ds = h5file.create_dataset("time", data=np.arange(3))
            time_ds.make_scale("time")
            time_ds.attrs["units"] = np.bytes_("fortnights since 2023-01-01")

        with h5py.File(path, "r") as h5file:
            coords = read_netcdf_coordinates(h5file)

        assert coords.dims == ("distance", "time")
        assert "latitude" in coords.coord_map

    def test_read_netcdf_coordinates_without_dimension_list(self, tmp_path):
        """Coordinate reader should fall back to discovered order when needed."""
        path = tmp_path / "coords_no_dimension_list.nc"
        with h5py.File(path, "w") as h5file:
            dist_ds = h5file.create_dataset("distance", data=np.arange(2))
            dist_ds.make_scale("distance")
            time_ds = h5file.create_dataset("time", data=np.arange(3))
            time_ds.make_scale("time")
            time_ds.attrs["units"] = "seconds since 1970-01-01 00:00:00"
            h5file.create_dataset("data", data=np.ones((2, 3)))

        with h5py.File(path, "r") as h5file:
            coords = read_netcdf_coordinates(h5file)

        assert coords.dims == ("distance", "time")
        np.testing.assert_array_equal(coords.coord_map["distance"].values, np.arange(2))

    def test_read_netcdf_coordinates_adds_unordered_dim_coords(self, tmp_path):
        """Coords discovered outside DIMENSION_LIST should still be retained."""
        path = tmp_path / "coords_extra_scale.nc"
        with h5py.File(path, "w") as h5file:
            dist_ds = h5file.create_dataset("distance", data=np.arange(2))
            dist_ds.make_scale("distance")
            time_ds = h5file.create_dataset("time", data=np.arange(3))
            time_ds.make_scale("time")
            time_ds.attrs["units"] = np.bytes_("seconds since 1970-01-01 00:00:00")
            extra_ds = h5file.create_dataset("channel", data=np.arange(2))
            extra_ds.make_scale("channel")
            data_ds = h5file.create_dataset("data", data=np.ones((2, 3)))
            data_ds.dims[0].attach_scale(dist_ds)
            data_ds.dims[1].attach_scale(time_ds)

        with h5py.File(path, "r") as h5file:
            coords = read_netcdf_coordinates(h5file)

        assert "channel" in coords.coord_map

    def test_read_netcdf_coordinates_skips_auxiliary_named_scales(self, tmp_path):
        """Auxiliary scale names should be skipped before coord collection."""
        path = tmp_path / "coords_aux_skip.nc"
        with h5py.File(path, "w") as h5file:
            dist_ds = h5file.create_dataset("distance", data=np.arange(2))
            dist_ds.make_scale("distance")
            time_ds = h5file.create_dataset("time", data=np.arange(3))
            time_ds.make_scale("time")
            time_ds.attrs["units"] = "seconds since 1970-01-01 00:00:00"
            aux_ds = h5file.create_dataset("time_values", data=np.arange(3))
            aux_ds.make_scale("time_values")
            h5file.create_dataset("data", data=np.ones((2, 3)))

        with h5py.File(path, "r") as h5file:
            coords = read_netcdf_coordinates(h5file)

        assert "time_values" not in coords.coord_map


class TestNetCDFCoreHelpers:
    """Direct tests for lightweight NetCDF core helpers."""

    def test_coord_attrs_cover_distance_and_depth(self, example_patch):
        """Coordinate helper should populate distance and depth CF attrs."""
        distance_attrs = netcdf_utils.coord_attrs(
            "distance", example_patch.coords.coord_map["distance"]
        )
        depth_coord = dc.get_coord(data=np.arange(3), units="ft")
        depth_attrs = netcdf_utils.coord_attrs("sensor_depth", depth_coord)

        assert distance_attrs["standard_name"] == "distance"
        assert distance_attrs["units"] == "m"
        assert depth_attrs["units"] == "ft"

    def test_get_xarray_data_var_name(self):
        """Dataset helper should find expected data variables or raise."""
        xr = pytest.importorskip("xarray")
        ds_with_data = xr.Dataset({"data": (("x",), [1, 2])})
        ds_single = xr.Dataset({"signal": (("x",), [1, 2])})
        ds_multi = xr.Dataset({"signal": (("x",), [1, 2]), "other": (("x",), [3, 4])})

        class _DatasetWithNone:
            data_vars = {None: object(), "distance_indices": object()}

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

    def test_get_data_variable_name_raises_for_missing_data(self, tmp_path):
        """Formatter should raise when no main data variable exists."""
        formatter = netcdf_core.NetCDFCFV18()
        path = tmp_path / "missing_data.nc"
        with h5py.File(path, "w") as h5file:
            h5file.create_dataset("time", data=np.arange(3))

        with h5py.File(path, "r") as h5file:
            with pytest.raises(ValueError, match="No suitable data variable found"):
                formatter._get_data_variable_name(h5file)

    def test_apply_coordinate_filtering_handles_selected_and_unrelated_kwargs(
        self, example_patch
    ):
        """Filtering should only use known coordinate kwargs."""
        formatter = netcdf_core.NetCDFCFV18()

        unfiltered = formatter._apply_coordinate_filtering(example_patch, {"tag": "x"})
        filtered = formatter._apply_coordinate_filtering(
            example_patch, {"distance": (10, 20)}
        )

        assert unfiltered.equals(example_patch)
        assert filtered.data.shape[0] < example_patch.data.shape[0]

    def test_read_uses_xarray_dataset_and_merges_missing_coords(
        self, minimal_cf_netcdf_path, monkeypatch
    ):
        """Read should accept xarray-backed data and merge extra coords."""

        class FakeCoord:
            def __init__(self, dims, values):
                self.dims = dims
                self.values = values

        class FakeDataArray:
            def __init__(self, data):
                self.data = data
                self.coords = {
                    "distance": FakeCoord(("distance",), np.arange(data.shape[0])),
                    "time": FakeCoord(("time",), np.arange(data.shape[1])),
                    "latitude": FakeCoord(
                        ("distance",), np.linspace(1.0, 2.0, data.shape[0])
                    ),
                }

            def load(self):
                return self

        class FakeDataset:
            def __init__(self, data_array):
                self.data_vars = {None: data_array}
                self._data_array = data_array

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, name):
                return None

            def __getitem__(self, item):
                assert item is None
                return self._data_array

        fake_coords = dc.get_coord_manager(
            coords={"distance": np.arange(2), "time": np.arange(3)},
            dims=("distance", "time"),
        )
        fake_data_array = FakeDataArray(np.arange(6).reshape(2, 3))
        fake_dataset = FakeDataset(fake_data_array)
        fake_xarray = type(
            "FakeXarray",
            (),
            {"open_dataset": staticmethod(lambda *args, **kwargs: fake_dataset)},
        )
        formatter = netcdf_core.NetCDFCFV18()

        def _optional_import(name, on_missing="raise"):
            if name == "xarray":
                return fake_xarray
            if name == "netCDF4":
                return object()
            return None

        monkeypatch.setattr(netcdf_core, "optional_import", _optional_import)
        monkeypatch.setattr(
            netcdf_core,
            "read_netcdf_coordinates",
            lambda *args: fake_coords,
        )
        monkeypatch.setattr(
            formatter, "_get_data_variable_name", lambda resource: "__values__"
        )
        monkeypatch.setattr(
            formatter, "_get_patch_attrs", lambda resource: {"tag": "fake"}
        )

        spool = formatter.read(minimal_cf_netcdf_path)
        patch = spool[0]

        np.testing.assert_array_equal(patch.data, fake_data_array.data)
        assert patch.attrs.tag == "fake"
        assert "latitude" in patch.coords.coord_map

    def test_read_falls_back_to_single_xarray_data_var(
        self, minimal_cf_netcdf_path, monkeypatch
    ):
        """Read should use the only xarray data var when HDF-derived name misses."""

        class FakeDataArray:
            def __init__(self):
                self.data = np.ones((2, 2))
                self.coords = {}

            def load(self):
                return self

        class FakeDataset:
            def __init__(self):
                self.data_vars = {"signal": FakeDataArray()}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, name):
                return None

            def __getitem__(self, item):
                return self.data_vars[item]

        fake_coords = dc.get_coord_manager(
            coords={"distance": np.arange(2), "time": np.arange(2)},
            dims=("distance", "time"),
        )
        fake_xarray = type(
            "FakeXarray",
            (),
            {"open_dataset": staticmethod(lambda *args, **kwargs: FakeDataset())},
        )
        formatter = netcdf_core.NetCDFCFV18()

        def _optional_import(name, on_missing="raise"):
            if name == "xarray":
                return fake_xarray
            if name == "netCDF4":
                return object()
            return None

        monkeypatch.setattr(netcdf_core, "optional_import", _optional_import)
        monkeypatch.setattr(
            netcdf_core, "read_netcdf_coordinates", lambda *args: fake_coords
        )
        monkeypatch.setattr(
            formatter, "_get_data_variable_name", lambda resource: "missing"
        )
        monkeypatch.setattr(
            formatter, "_get_patch_attrs", lambda resource: {"tag": "fallback"}
        )

        spool = formatter.read(minimal_cf_netcdf_path)
        assert spool[0].attrs.tag == "fallback"

    def test_read_returns_empty_spool_for_empty_filtered_patch(
        self, minimal_cf_netcdf_path, monkeypatch
    ):
        """Read should return an empty spool after filtering removes all data."""

        class FakeDataArray:
            def __init__(self):
                self.data = np.ones((1, 1))
                self.coords = {}

            def load(self):
                return self

        class FakeDataset:
            def __init__(self):
                self.data_vars = {"data": FakeDataArray()}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, name):
                return self.data_vars["data"]

        formatter = netcdf_core.NetCDFCFV18()
        fake_xarray = type(
            "FakeXarray",
            (),
            {"open_dataset": staticmethod(lambda *args, **kwargs: FakeDataset())},
        )
        fake_coords = dc.get_coord_manager(
            coords={"distance": np.arange(1), "time": np.arange(1)},
            dims=("distance", "time"),
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
            netcdf_core, "read_netcdf_coordinates", lambda *args: fake_coords
        )
        monkeypatch.setattr(
            formatter, "_get_data_variable_name", lambda resource: "data"
        )
        monkeypatch.setattr(
            formatter, "_get_patch_attrs", lambda resource: {"tag": "empty"}
        )
        monkeypatch.setattr(
            formatter, "_apply_coordinate_filtering", lambda patch, kwargs: empty_patch
        )

        spool = formatter.read(minimal_cf_netcdf_path)
        assert len(spool) == 0

    def test_write_uses_xarray_dataset_path(self, example_patch, tmp_path, monkeypatch):
        """Write should pass CF attrs and encoding through the xarray path."""

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

        patch_with_partial = example_patch.new(
            coords=example_patch.coords.update(
                latitude=("distance", np.linspace(0.0, 1.0, example_patch.shape[0])),
                quality=("distance", dc.get_coord(shape=(example_patch.shape[0],))),
            )
        )
        out_path = tmp_path / "write_stub.nc"
        formatter.write(
            patch_with_partial.update(attrs={"data_type": "strain_rate"}), out_path
        )

        dataset = fake_data_array.dataset
        assert dataset.attrs["Conventions"] == "CF-1.8"
        assert dataset.attrs["source_data_type"] == "strain_rate"
        assert dataset.data.attrs["standard_name"] == "strain_rate"
        assert dataset.data.attrs["coordinates"] == "latitude"
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

            # Check global attributes
            assert "Conventions" in h5file.attrs
            conventions = h5file.attrs["Conventions"]
            if isinstance(conventions, bytes):
                conventions = conventions.decode()
            assert "CF-" in conventions

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
        """Xarray should see the expected CF metadata on DASCore output."""
        xr = pytest.importorskip("xarray")
        engine = _require_xarray_netcdf_engine()

        path = tmp_path / "metadata_visible.nc"
        dc.write(patch_with_attrs, path, file_format="netcdf_cf")

        with xr.open_dataset(path, engine=engine) as dataset:
            assert dataset.attrs["Conventions"] == "CF-1.8"
            assert dataset.attrs["source_data_type"] == "strain_rate"
            assert dataset.attrs["station"] == "TEST_STATION"
            assert dataset.attrs["network"] == "TEST_NET"

            assert list(dataset.data_vars) == ["data"]
            assert list(dataset.coords) == ["distance", "time"]
            assert dataset["data"].attrs["standard_name"] == "strain_rate"
            assert dataset["data"].attrs["units"] == "1/s"
            assert dataset["time"].attrs["axis"] == "T"
            assert dataset["time"].attrs["standard_name"] == "time"
            assert dataset["distance"].attrs["standard_name"] == "distance"
            assert dataset["distance"].attrs["units"] == "m"
            assert "featureType" not in dataset.attrs
            assert "coordinates" not in dataset["data"].attrs

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

    def test_extract_patch_attrs_from_netcdf(self, cf_compliant_file):
        """Test extracting patch attributes from NetCDF file."""
        with h5py.File(cf_compliant_file, "r") as h5file:
            attrs = extract_patch_attrs_from_netcdf(h5file)
            assert isinstance(attrs, dict)

    def test_read_netcdf_coordinates(self, cf_compliant_file):
        """Test reading coordinates from NetCDF file."""
        with h5py.File(cf_compliant_file, "r") as h5file:
            coords = read_netcdf_coordinates(h5file)

            assert "time" in coords.coord_map
            assert "distance" in coords.coord_map

            # Check that time coordinate is properly converted from CF format
            time_coord = coords.coord_map["time"]
            assert len(time_coord) > 0

    def test_validate_cf_compliance(self, cf_compliant_file):
        """Test CF compliance validation."""
        with h5py.File(cf_compliant_file, "r") as h5file:
            issues = validate_cf_compliance(h5file)

            # Our implementation should produce CF-compliant files
            assert len(issues) == 0, f"CF compliance issues found: {issues}"

    def test_validate_cf_compliance_with_issues(self, tmp_path, rng):
        """Test CF compliance validation with non-compliant file."""
        path = tmp_path / "non_compliant.nc"

        # Create a file with CF issues
        with h5py.File(path, "w") as h5file:
            # Missing Conventions attribute
            time_ds = h5file.create_dataset("time", data=np.arange(100))
            time_ds.make_scale("time")
            # Missing units attribute

            h5file.create_dataset("data", data=rng.random((100, 50)))
            # Missing long_name and units attributes

        with h5py.File(path, "r") as h5file:
            issues = validate_cf_compliance(h5file)

            assert len(issues) > 0
            assert any("Conventions" in issue for issue in issues)
            assert any("units" in issue for issue in issues)

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

    def test_different_data_types_cf_attrs(self):
        """Test CF attributes for different data types."""
        # Test various data types
        test_cases = [
            ("strain", "1", "Strain"),
            ("velocity", "m/s", "Velocity"),
            ("temperature", "K", "Temperature"),
            ("pressure", "Pa", "Pressure"),
            ("unknown_type", "1", "Distributed Acoustic Sensing data"),
        ]

        for data_type, expected_units, expected_long_name in test_cases:
            attrs = get_cf_data_attrs(data_type)
            assert attrs["units"] == expected_units
            assert attrs["long_name"] == expected_long_name
            assert "_FillValue" in attrs

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

    def test_read_external_xdas_netcdf_file(self):
        """External xdas NetCDF should keep its current readable structure."""
        _require_xarray_netcdf_engine()
        path = fetch("xdas_netcdf.nc")

        patch = dc.read(path, file_format="netcdf_cf")[0]

        assert patch.dims == ("time", "distance")
        assert patch.shape == (300, 401)
        assert set(patch.coords.coord_map) == {"time", "distance"}
        assert patch.attrs.tag == ""
        assert patch.attrs["_source_patch_id"] == netcdf_utils.XDAS_PAYLOAD_VARIABLE

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

    def test_cf_time_edge_cases(self):
        """Test CF time conversion edge cases."""
        # Test unsupported time unit
        cf_times = np.array([0, 1, 2])
        invalid_units = "fortnights since 2023-01-01"

        with pytest.raises(ValueError, match="Unsupported time unit"):
            cf_time_to_datetime64(cf_times, invalid_units)

        # Test various supported units
        supported_units = [
            ("hours since 2023-01-01", "h"),
            ("minutes since 2023-01-01", "min"),
            ("milliseconds since 2023-01-01", "ms"),
            ("microseconds since 2023-01-01", "us"),
        ]

        for units, _ in supported_units:
            cf_times = np.array([0, 1, 2])
            result = cf_time_to_datetime64(cf_times, units)
            assert result.dtype == "datetime64[ns]"
            assert len(result) == 3
