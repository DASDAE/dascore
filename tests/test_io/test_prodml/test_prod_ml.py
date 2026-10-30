"""Generic tests for prodml support."""

from __future__ import annotations

import itertools
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import get_coord
from dascore.io.core import read
from dascore.io.prodml.utils import (
    _as_datetime,
    _get_prodml_version_str,
    _get_root_attrs,
)
from dascore.utils.downloader import fetch
from dascore.utils.misc import suppress_warnings
from tests.conftest import needs_chunk_iter


@pytest.fixture(scope="session")
def quantx_v2_example_path():
    """Return the path to the example QuantXV2 file."""
    out = fetch("opta_sense_quantx_v2.h5")
    assert out.exists()
    return out


@pytest.fixture(scope="session")
def quantx_v2_das_patch(quantx_v2_example_path):
    """Read the QuantXV2 data, return contained DataArray."""
    out = read(quantx_v2_example_path, "prodml")[0]
    attr_time = out.summary.get_coord_summary("time").max
    coord_time = out.coords.max("time")
    assert attr_time == coord_time
    return out


class TestProdMLFile:
    """
    Ensure we can read the ProdML provided by Silixa.

    We do this since the other tests read the prodML files, even though
    the Silixa file is technical just ProdML v2.1.
    """

    @pytest.mark.parametrize("name", ["prodml_2.0.h5", "prodml_fbe_1.h5"])
    def test_dtype_is_not_an_attr(self, name):
        """The data type is the patch's, not an attr, so a spool does not warn."""
        with suppress_warnings(UserWarning, action="error"):
            spool = dc.spool(fetch(name))
        patch = spool[0]
        assert patch.dtype == patch.data.dtype

    @pytest.fixture(scope="class")
    def issue_221_patch_path(self, tmp_path_factory):
        """Ensure dims are correctly ascertained."""
        tmp_path = tmp_path_factory.mktemp("issue_221")
        path = fetch("prodml_2.0.h5")
        new_path = shutil.copy2(path, tmp_path / "prod_2_monkey_patched.h5")
        with h5py.File(new_path, "a") as fi:
            # monkey patch dimensions to simulate issue.
            fi["Acquisition/Raw[0]/RawData"].attrs["Dimensions"] = "time, locus"
        return new_path

    @pytest.fixture(scope="class")
    def issue_514_patch_path(self, tmp_path_factory):
        """Make a patch with bad endtime metadata. See #412."""
        tmp_path = tmp_path_factory.mktemp("issue_514")
        path = fetch("prodml_2.0.h5")
        new_path = shutil.copy2(path, tmp_path / "prod_2_issue_514.h5")
        with h5py.File(new_path, "a") as fi:
            # monkey patch dimensions to simulate issue.
            parent_node = fi["Acquisition"]["Raw[0]"]
            time_node = parent_node["RawDataTime"]
            time = time_node[:].astype("datetime64[us]")
            time_coord = get_coord(data=time)
            new_time = str(time_coord.max() + time_coord.step * 2)
            time_node.attrs["PartEndTime"] = new_time
        return new_path

    @pytest.fixture(scope="class")
    def unitless_patch_path(self, tmp_path_factory):
        """Make a patch whose raw data units are "unitless"."""
        tmp_path = tmp_path_factory.mktemp("unitless")
        path = dc.utils.downloader.fetch("prodml_2.0.h5")
        new_path = shutil.copy2(path, tmp_path / "prod_2_unitless.h5")
        with h5py.File(new_path, "a") as fi:
            fi["Acquisition"]["Raw[0]"].attrs["RawDataUnit"] = "unitless"
        return new_path

    @pytest.fixture(scope="class")
    def silixa_h5_patch(self, idas_h5_example_path):
        """Get the silixa file, return Patch."""
        return dc.spool(idas_h5_example_path)[0]

    def test_has_gauge_length(self, silixa_h5_patch):
        """Ensure gauge-length is found in patch attrs."""
        patch = silixa_h5_patch
        assert hasattr(patch.attrs, "gauge_length")
        # The value, for this dataset, should be non-null.
        gauge_length = getattr(patch.attrs, "gauge_length", None)
        assert not pd.isnull(gauge_length)

    def test_issue_221(self, issue_221_patch_path):
        """Ensure dims are correctly ascertained."""
        patch = dc.read(issue_221_patch_path)[0]
        assert isinstance(patch, dc.Patch)

    def test_issue_514(self, issue_514_patch_path):
        """Ensure the patch can be read despite bad attribute info."""
        patch = dc.read(issue_514_patch_path)[0]
        assert isinstance(patch, dc.Patch)

    def test_unitless_data_units(self, unitless_patch_path):
        """Ensure "unitless" data units don't prevent reading the file."""
        patch = dc.read(unitless_patch_path)[0]
        assert patch.attrs.data_units is None


class TestReadQuantXV2:
    """Tests for reading the QuantXV2 format."""

    def test_precision_of_time_array(self, quantx_v2_das_patch):
        """
        Ensure the time array is in ns, not native ms, in order to
        be consistent with other patches.
        """
        time = quantx_v2_das_patch.coords.get_array("time")
        dtype = time.dtype
        assert "[ns]" in str(dtype)


class TestVersionDetection:
    """Tests for the ProdML version fingerprint helper."""

    def test_acquisition_without_expected_attrs(self, tmp_path):
        """An Acquisition group lacking the fingerprint attrs is not ProdML."""
        path = tmp_path / "not_prodml.h5"
        with h5py.File(path, "w") as file:
            file.create_group("Acquisition").attrs["unrelated"] = "x"
        with h5py.File(path, "r") as file:
            assert _get_prodml_version_str(file) == ""


class TestFacilityId:
    """An acquisition names its facilities in one attr."""

    def test_one_facility(self):
        """The usual spelling is a length-1 array holding the name."""
        out = _get_root_attrs({"FacilityId": np.array([b"well_a"])})
        assert out["facility_id"] == "well_a"

    def test_several_facilities(self):
        """Several names are one comma separated name, not a tuple."""
        out = _get_root_attrs({"FacilityId": np.array([b"well_a", b"well_b"])})
        assert out["facility_id"] == "well_a,well_b"


class TestOffsetReads:
    """A PRODML array's chunks are read from their offsets, gaps and all."""

    @needs_chunk_iter
    def test_windows_equal_hdf5(self, stored_reads):
        """Windows across separated chunks equal h5py's, with no HDF5 open."""
        path = fetch("prodml_2.1.h5")
        source = dc.read(path)[0]._source
        offsets = source.offsets
        assert any(b - a != 1024 * 512 * 2 for a, b in itertools.pairwise(offsets))
        with h5py.File(path, "r") as h5:
            node = h5["Acquisition/Raw[0]/RawData"]
            for window in [(slice(None),), (slice(1000, 3100),), (slice(11999, None),)]:
                np.testing.assert_array_equal(source[window].load(), node[window])
        assert not stored_reads


def _write_variant(path, patch, rows=None, ns_times=False, rate=None):
    """
    Write a patch as ProdML, then bend it like some producers do: keep only
    the first `rows` samples (ending the part where it starts, so for
    rows=1), store times in ns, or change OutputDataRate (None keeps it,
    False deletes it).
    """
    patch.io.write(path, "prodml")
    with h5py.File(path, "r+") as h5:
        raw = h5["Acquisition/Raw[0]"]
        times, data = raw["RawDataTime"][:], raw["RawData"][:]
        time_attrs = dict(raw["RawDataTime"].attrs)
        data_attrs = dict(raw["RawData"].attrs)
        if rows is not None:
            time_attrs["PartEndTime"] = time_attrs["PartStartTime"]
            times, data = times[:rows], data[:rows]
        for name, values, attrs in (
            ("RawDataTime", times * (1000 if ns_times else 1), time_attrs),
            ("RawData", data, data_attrs),
        ):
            del raw[name]
            raw.create_dataset(name, data=values).attrs.update(attrs)
        if rate is False:
            del raw.attrs["OutputDataRate"]
        elif rate is not None:
            raw.attrs["OutputDataRate"] = rate
    return path


class TestProducerVariants:
    """ProdML files that bend the schema the way real producers do."""

    @pytest.fixture(scope="class")
    def patch(self):
        """A small patch with whole microsecond times."""
        patch = dc.get_example_patch().select(distance=(0, 20), samples=True)
        return patch.transpose("time", "distance")

    def test_capitalized_version_and_ns_times(self, patch, tmp_path):
        """
        A SchemaVersion attr and nanosecond times read like the standard
        spelling and microseconds; several samples ignore a wrong
        OutputDataRate.
        """
        path = _write_variant(tmp_path / "v.h5", patch, ns_times=True, rate=25.0)
        with h5py.File(path, "r+") as h5:
            attrs = h5["Acquisition"].attrs
            attrs["SchemaVersion"] = attrs.pop("schemaVersion")
        assert dc.get_format(path) == ("PRODML", "2.1")
        expected = patch.get_coord("time")
        for snap in (True, False):
            out = dc.read(path, snap=snap)[0]
            assert out.attrs.schema_version == "2.1"
            assert np.array_equal(out.get_coord("time").values, expected.values)
            assert np.array_equal(out.data, patch.transpose(*out.dims).data)

    @pytest.mark.parametrize("rate, step", [(0.5, 2), (False, None), (0.0, None)])
    def test_single_time_uses_output_rate(self, patch, tmp_path, rate, step):
        """One sample takes its step from a usable OutputDataRate, or has none."""
        path = _write_variant(tmp_path / "one.h5", patch, rows=1, rate=rate)
        time = dc.read(path)[0].get_coord("time")
        assert time.min() == patch.get_coord("time").min() and len(time) == 1
        assert time.step == (step and np.timedelta64(step, "s"))

    def test_wrong_end_time_uses_ns_labels(self, patch, tmp_path):
        """A PartEndTime a few steps off yields the recorded ns labels (#414)."""
        path = _write_variant(tmp_path / "late.h5", patch, ns_times=True)
        with h5py.File(path, "r+") as h5:
            times = h5["Acquisition/Raw[0]/RawDataTime"]
            late = times[-1] + 2 * 4_000_000
            end = str(np.datetime64(int(late), "ns")) + "Z"
            times.attrs["PartEndTime"] = end.encode()
        time = dc.read(path)[0].get_coord("time")
        assert np.array_equal(time.values, patch.get_coord("time").values)

    @pytest.mark.parametrize(
        "ns_times, last_shift, end_shift", [(True, 1, 1), (False, 0, 500)]
    )
    def test_jittered_times_keep_a_step(
        self, patch, tmp_path, ns_times, last_shift, end_shift
    ):
        """
        Jittered labels between ends that agree to the labels' precision
        snap to an even grid: ns labels with a step of no whole ns, and µs
        labels beside an end stated to the ns.
        """
        path = _write_variant(tmp_path / "jitter.h5", patch, ns_times=ns_times)
        with h5py.File(path, "r+") as h5:
            times = h5["Acquisition/Raw[0]/RawDataTime"]
            values = times[:]
            jitter = 20_000 if ns_times else 20
            values[1:-1] += np.tile([jitter, -jitter], len(values) // 2)[:-2]
            values[-1] += last_shift
            times[:] = values
            end = _as_datetime(values[-1:])[0] + np.timedelta64(end_shift, "ns")
            times.attrs["PartEndTime"] = (str(end) + "Z").encode()
        time = dc.read(path)[0].get_coord("time")
        assert time.step is not None
        assert time.min() == patch.get_coord("time").min()

    def test_drifting_clock_keeps_sample_count(self, tmp_path):
        """
        A long file whose spacing is not whole nanoseconds snaps to exactly
        as many samples as it holds, ending within a step of its stated end.
        """
        count, start = 140_001, np.datetime64("2024-01-01", "ns")
        even = start + np.arange(count) * np.timedelta64(100, "us")
        patch = dc.Patch(
            data=np.zeros((count, 2), dtype=np.float32),
            coords={"time": even, "distance": [0.0, 1.0]},
            dims=("time", "distance"),
        )
        path = _write_variant(tmp_path / "drift.h5", patch, ns_times=True)
        # Spacing 100,000.45 ns: rounding the step loses 0.6 of a sample.
        drift = np.round(np.arange(count) * 100_000.45).astype("timedelta64[ns]")
        time = start + drift
        with h5py.File(path, "r+") as h5:
            times = h5["Acquisition/Raw[0]/RawDataTime"]
            times[:] = time.astype(np.int64)
            times.attrs["PartEndTime"] = (str(time[-1]) + "Z").encode()
        out = dc.read(path)[0]
        assert out.shape == (count, 2)
        assert abs(out.get_coord("time").max() - time[-1]) < np.timedelta64(100, "us")
