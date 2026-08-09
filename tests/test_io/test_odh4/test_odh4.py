"""Tests for the OptaSense ODH4 format."""

import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.odh4 import ODH4V1
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def odh4_path():
    """Return the path to the ODH4 test file."""
    return fetch("optasense_odh4_1.h5")


@pytest.fixture(scope="module")
def odh4_patch(odh4_path):
    """Read the ODH4 test file into a patch."""
    return ODH4V1().read(odh4_path)[0]


class TestODH4:
    """Format-specific tests not covered by the common IO tests."""

    def test_coords(self, odh4_patch):
        """Coordinates come from the root attrs."""
        time = odh4_patch.get_coord("time")
        distance = odh4_patch.get_coord("distance")
        assert time.step == dc.to_timedelta64(1 / 50)
        assert time.min() == dc.to_datetime64("2023-02-13T23:59:59.995000")
        # distance = channel index * channel spacing (4 m), channels 500-532.
        assert distance.step == 4.0
        assert distance.min() == 500 * 4.0
        assert len(distance) == 32

    def test_attrs(self, odh4_patch):
        """Acquisition metadata lands on the patch attrs."""
        attrs = odh4_patch.attrs
        assert attrs.gauge_length == 16.0
        assert np.isclose(attrs.scale_factor_to_strain, 2.6926631095510917e-08)
        assert dc.get_quantity(attrs.data_units) == dc.get_quantity("radians")

    def test_channel_coord(self, odh4_patch):
        """Interrogator channel numbers ride along the distance dim."""
        channel = odh4_patch.get_coord("channel")
        assert channel.min() == 500
        assert channel.max() == 531
        trimmed = odh4_patch.select(distance=(2004.0, 2020.0))
        trimmed_channel = trimmed.get_coord("channel")
        assert trimmed_channel.min() == 501

    def test_inconsistent_attrs_raise(self, odh4_path, tmp_path):
        """A file whose attrs disagree with the data shape refuses to read."""
        path = tmp_path / "bad_rate.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            h5.attrs["sampling rate Hz"] = 100
        with pytest.raises(InvalidFiberFileError, match="inconsistent"):
            ODH4V1().read(path)

    def test_transposed_data_raises(self, odh4_path, tmp_path):
        """A transposed data layout must not be read with swapped coords."""
        path = tmp_path / "transposed.h5"
        with h5py.File(odh4_path, "r") as src, h5py.File(path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.create_dataset("raw_data", data=src["raw_data"][()].T)
        with pytest.raises(InvalidFiberFileError, match="inconsistent"):
            ODH4V1().read(path)

    def test_unparsable_unit_dropped(self, odh4_path, tmp_path):
        """An unknown unit description yields unset units, not a crash."""
        path = tmp_path / "bad_unit.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            h5.attrs["raw_data_units"] = "definitely not a unit"
        patch = ODH4V1().read(path)[0]
        assert not patch.attrs.data_units

    def test_non_positive_rate_raises(self, odh4_path, tmp_path):
        """A zero sampling rate raises a format error, not ZeroDivisionError."""
        path = tmp_path / "zero_rate.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            h5.attrs["sampling rate Hz"] = 0
        with pytest.raises(InvalidFiberFileError, match="sampling rate"):
            ODH4V1().read(path)

    @pytest.mark.parametrize("bad_rate", [np.nan, np.inf, "not a rate"])
    def test_unusable_rate_raises(self, odh4_path, tmp_path, bad_rate):
        """Non-finite or non-numeric rates raise a format error."""
        path = tmp_path / "unusable_rate.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            h5.attrs["sampling rate Hz"] = bad_rate
        with pytest.raises(InvalidFiberFileError, match="sampling rate"):
            ODH4V1().read(path)

    def test_oversized_channel_span_raises(self, odh4_path, tmp_path):
        """A channel span larger than the data must not silently read."""
        path = tmp_path / "wide_span.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            h5.attrs["channel_end"] = int(h5.attrs["channel_end"]) + 1
        with pytest.raises(InvalidFiberFileError, match="inconsistent"):
            ODH4V1().read(path)

    def test_inclusive_endtime_keeps_step(self, odh4_path, tmp_path):
        """An inclusive-bound endtime yields the exact sampling step."""
        path = tmp_path / "inclusive.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            start = pd.Timestamp(h5.attrs["starttime"])
            rate = int(h5.attrs["sampling rate Hz"])
            n_time = h5["raw_data"].shape[1]
            inclusive_end = start + pd.Timedelta(seconds=(n_time - 1) / rate)
            h5.attrs["endtime"] = inclusive_end.isoformat()
        patch = ODH4V1().read(path)[0]
        assert patch.get_coord("time").step == dc.to_timedelta64(1 / rate)

    def test_near_miss_not_claimed(self, tmp_path):
        """A file with raw_data but not the full attr set isn't claimed."""
        path = tmp_path / "near_miss.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("raw_data", data=np.zeros((3, 4)))
            h5.attrs["starttime"] = "2023-01-01T00:00:00"
            h5.attrs["sampling rate Hz"] = 50
        assert not ODH4V1().get_format(path)
