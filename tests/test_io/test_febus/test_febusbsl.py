"""
FEBUS G1 BSL HDF5 specific tests.
"""

import shutil

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import dascore as dc
from dascore.io.febus import FebusBSLH5V1
from dascore.io.febus.g1utils import _get_g1_h5_base_coords
from dascore.utils.downloader import fetch

BSL_NAME = "febusg1_C2_2026-06-03T17.18.13+0200.bsl.h5"


class TestFebusBSL:
    """Tests for the FEBUS G1 BSL HDF5 reader."""

    parser = FebusBSLH5V1()

    @pytest.fixture(scope="class")
    def bsl_path(self):
        """Path to a G1 BSL HDF5 test file."""
        return fetch(BSL_NAME)

    @pytest.fixture(scope="class")
    def bsl_patch(self, bsl_path):
        """Return the parsed G1 BSL patch."""
        return self.parser.read(bsl_path)[0]

    def test_future_format_version_not_claimed(self, bsl_path, tmp_path):
        """Future BSL format versions should not be claimed by the v1 reader."""
        new_path = tmp_path / bsl_path.name
        shutil.copy2(bsl_path, new_path)
        with h5py.File(new_path, "a") as fi:
            fi.attrs["formatVersion"] = np.array([2], dtype=np.uint64)
        assert not self.parser.get_format(new_path)

    def test_scan(self, bsl_path):
        """Scan returns one patch attrs object with expected metadata."""
        payloads = self.parser.scan(bsl_path)
        assert len(payloads) == 1
        payload = payloads[0]
        attr = payload["attrs"]
        assert isinstance(attr, dc.PatchAttrs)
        assert "path" not in attr.model_dump()
        assert "file_format" not in attr.model_dump()
        assert "file_version" not in attr.model_dump()
        assert payload["dims"] == ("time", "distance")
        assert attr.data_category == "DSS"
        assert attr.data_type == "strain"
        assert attr.data_units == dc.get_quantity("microstrain")

    def test_read(self, bsl_patch):
        """Ensure the BSL file is read into a patch with expected shape."""
        assert isinstance(bsl_patch, dc.Patch)
        assert bsl_patch.shape == (120, 100)
        assert bsl_patch.attrs.data_units == dc.get_quantity("microstrain")
        assert "temperature" in bsl_patch.coords.coord_map
        assert bsl_patch.coords.dim_map["temperature"] == ("time",)

    def test_distance_range(self, bsl_patch):
        """Distance should span 50-149 m."""
        dist = bsl_patch.get_coord("distance")
        assert_allclose(dist.min(), 50.0)
        assert_allclose(dist.max(), 149.0)
        assert_allclose(dist.step, 1.0)

    def test_time_coord(self, bsl_patch):
        """Time should be monotonic but irregularly sampled."""
        time = bsl_patch.get_coord("time")
        assert "datetime64" in str(np.dtype(time.dtype))
        assert time.min() == np.datetime64("2026-06-03T15:18:13.422442752")
        assert time.max() == np.datetime64("2026-06-03T15:28:08.829897728")
        assert time.step is None
        assert bsl_patch.summary.get_coord_summary("time").step is None

    def test_sample_span_coord(self, bsl_path, bsl_patch):
        """Each sample covers a window; its exact length should be kept."""
        with h5py.File(bsl_path) as fi:
            expected = fi["end_times"][...] - fi["start_times"][...]
        span = bsl_patch.get_coord("sample_span")
        assert bsl_patch.coords.dim_map["sample_span"] == ("time",)
        assert np.array_equal(span.values, dc.to_timedelta64(expected))

    def test_sample_span_differenced_before_snapping(self):
        """Spans come off the raw arrays, not from subtracting built coords.

        Starts and ends are each near-regular and snap to slightly different
        steps, so differencing the two coords would show a linear drift where
        the file has jitter.
        """
        starts = np.array([0.0, 1.0, 2.0, 3.0])
        ends = starts + np.array([0.9, 1.2, 0.8, 1.1])
        coords = _get_g1_h5_base_coords(
            {
                "start_times": starts,
                "end_times": ends,
                "distances": np.arange(2, dtype=np.float64),
                "temperatures": np.zeros(4, dtype=np.float64),
            },
            dims=("time", "distance"),
        )
        span = coords.coord_map["sample_span"]
        assert np.array_equal(span.values, dc.to_timedelta64(ends - starts))
        # the snapped time coord would have given a constant span
        assert len(np.unique(span.values)) > 1

    def test_sample_span_keeps_a_monotonic_drift(self):
        """Spans that drift steadily are regular enough to be snapped away.

        They survive because the coord manager no longer collapses a coord
        when doing so would move its values (#896).
        """
        starts = np.arange(4, dtype=np.float64)
        ends = starts + np.array([1.0, 1.001, 1.0020005, 1.0030015])
        coords = _get_g1_h5_base_coords(
            {
                "start_times": starts,
                "end_times": ends,
                "distances": np.arange(2, dtype=np.float64),
                "temperatures": np.zeros(4, dtype=np.float64),
            },
            dims=("time", "distance"),
        )
        span = coords.coord_map["sample_span"]
        assert np.array_equal(span.values, dc.to_timedelta64(ends - starts))

    def test_mismatched_time_dataset_lengths_raise(self, bsl_path, tmp_path):
        """A half-written file should fail loudly, not broadcast to garbage."""
        new_path = tmp_path / bsl_path.name
        shutil.copy2(bsl_path, new_path)
        with h5py.File(new_path, "a") as fi:
            ends = fi["end_times"][...]
            del fi["end_times"]
            fi.create_dataset("end_times", data=ends[:1])
        with pytest.raises(ValueError, match="truncated or still being written"):
            self.parser.read(new_path)

    def test_redundant_time_attrs_absent(self, bsl_patch):
        """Neither the epoch-float time attrs nor format_version reach attrs."""
        names = set(dict(bsl_patch.attrs))
        assert not names & {"start_time", "end_time", "format_version"}

    def test_dropped_time_attrs_lose_nothing(self, bsl_path, bsl_patch):
        """What the dropped attrs said is still recoverable from the coords.

        This is what justifies removing them rather than retyping them.
        """
        with h5py.File(bsl_path) as fi:
            start_attr = float(fi.attrs["start_time"][0])
            end_attr = float(fi.attrs["end_time"][0])
        time = bsl_patch.get_coord("time")
        span = bsl_patch.get_coord("sample_span")
        assert time.min() == dc.to_datetime64(start_attr)
        # The end rebuilds through two independent float-seconds-to-ns
        # roundings, so it lands within a sample of the stored attr rather
        # than on it; the file's own epoch floats are only good to ~238 ns
        # here anyway.
        rebuilt = time.max() + span.values[-1]
        assert abs(rebuilt - dc.to_datetime64(end_attr)) < np.timedelta64(1, "us")

    def test_select_slices_sample_span(self, bsl_path, bsl_patch):
        """Selecting on time should carry the associated span along."""
        time = bsl_patch.get_coord("time")
        out = self.parser.read(bsl_path, time=(time.values[3], time.values[8]))[0]
        expected = bsl_patch.get_coord("sample_span").values[3:9]
        assert np.array_equal(out.get_coord("sample_span").values, expected)

    def test_select(self, bsl_path, bsl_patch):
        """Partial reads should reduce coords and data consistently."""
        time = bsl_patch.get_coord("time")
        dist = bsl_patch.get_coord("distance")
        out = self.parser.read(
            bsl_path,
            time=(time.values[10], time.values[20]),
            distance=(dist.min() + 5, dist.min() + 10),
        )[0]
        assert out.shape == (11, 6)
        assert out.get_coord("time").min() == time.values[10]
        assert out.get_coord("time").max() == time.values[20]
        assert_allclose(out.get_coord("distance").min(), 55.0)
        assert_allclose(out.get_coord("distance").max(), 60.0)
