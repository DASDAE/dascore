"""
Febus specific tests.
"""

import shutil

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchAttributeError
from dascore.io.core import FiberIO
from dascore.io.febus import Febus2
from dascore.io.febus.a1utils import (
    _flatten_febus_info,
    _get_source_patch_key,
    _get_zone_time,
)
from dascore.utils.downloader import fetch
from dascore.utils.misc import unbyte
from dascore.utils.time import to_float


class TestFebus:
    """Special test cases for febus."""

    @pytest.fixture(scope="class")
    def febus_path(self):
        """Return the path to a test febus file."""
        return fetch("febus_1.h5")

    def test_time_coords_consistent_with_metadata(self, febus_path):
        """
        Ensure the time coords returned have the same length as
        metadata indicates.
        """
        patch = Febus2().read(febus_path)[0]
        coords = patch.coords
        time = coords.get_coord("time")
        time_span = to_float((time.max() - time.min()) + time.step)

        with h5py.File(febus_path, "r") as f:
            feb = _flatten_febus_info(f)[0]
            # First check total time extent
            n_blocks = feb.zone[feb.data_name].shape[0]
            block_time = 1 / (feb.zone.attrs["BlockRate"] / 1_000)
            expected_time_span = block_time * n_blocks
            assert np.isclose(expected_time_span, time_span)
            # Then check absolute time
            time_offset = feb.zone.attrs["Origin"][1] / 1_000
            time_start = feb.source["time"][0] + time_offset
            assert np.isclose(to_float(time.min()), time_start)

    def test_valencia_data_sampling_rate(self):
        """
        The valencia file has two indicated sampling rates. One from
        the spacing array (500 Hz) and one from the sampling_rate attributes (250).
        Looking at the entire archive on pubDAS, it is clear the correct value
        is 250, otherwise there 50% gaps every file. This test just ensure that
        The sampling rate doesnt change.
        """
        patch = dc.get_example_patch("valencia_febus_example.h5")
        time = patch.get_coord("time")
        sampling_rate = np.timedelta64(1, "s") / time.step
        assert np.isclose(sampling_rate, 250)

    def test_scan_summaries_include_source_patch_key(self, febus_path):
        """Scanned Febus summaries should carry a reloadable patch id."""
        summaries = dc.scan(febus_path)
        assert summaries
        assert all(summary.source_patch_key for summary in summaries)

    def test_read_source_patch_key_selects_single_patch(self, febus_path):
        """Reading by source_patch_key should return the matching Febus patch."""
        summaries = dc.scan(febus_path)
        target = summaries[0]
        out = dc.read(febus_path, source_patch_key=target.source_patch_key)
        assert len(out) == 1
        assert out[0].attrs["_source_patch_key"] == target.source_patch_key
        assert (
            out[0].summary.get_coord_summary("time").min
            == target.get_coord_summary("time").min
        )

    def test_read_source_patch_key_sequence_input(self, febus_path):
        """Reading Febus with a sequence source_patch_key input should work."""
        summaries = dc.scan(febus_path)
        targets = [summaries[0].source_patch_key]
        out = dc.read(febus_path, source_patch_key=targets)
        assert len(out) == 1
        assert out[0].attrs["_source_patch_key"] == targets[0]
        assert (
            out[0].summary.get_coord_summary("time").min
            == summaries[0].get_coord_summary("time").min
        )

    def test_read_source_patch_key_non_matching_returns_empty(self, febus_path):
        """A non-matching source_patch_key should filter out Febus patches."""
        out = dc.read(febus_path, source_patch_key="not-a-real-febus-patch")
        assert len(out) == 0

    @staticmethod
    def _rows_per_block(path, key):
        """How many time rows of each block survive its padding."""
        with h5py.File(path, "r") as h5:
            zones = {_get_source_patch_key(z): z for z in _flatten_febus_info(h5)}
            info = _get_zone_time(zones[key])
        return info.idx_stop - info.idx_start + 1

    def test_read_array_spans_blocks(self, febus_path):
        """A window crossing a block boundary matches the default."""
        io = Febus2()
        payload = dc.scan(febus_path)[0]
        key = payload.source_patch_key
        rows = self._rows_per_block(febus_path, key)
        # one window straddling a block boundary, one wholly inside a
        # later block, one at the start: the offset arithmetic both ways
        cases = [(rows - 2, rows + 3), (2 * rows + 1, 2 * rows + 4), (0, 3)]
        for start, stop in cases:
            if stop > payload.shape[0]:
                continue
            windows = {"time": (start, stop)}
            out = io.read_array(febus_path, windows, source_patch_key=key)
            expected = FiberIO.read_array(io, febus_path, windows, source_patch_key=key)
            assert out.shape == expected.shape == (stop - start, payload.shape[1])
            assert np.array_equal(out, expected, equal_nan=True), (start, stop)

    def test_read_array_empty_window(self, febus_path):
        """A reversed or zero-width window is empty, with the right width."""
        io = Febus2()
        payload = dc.scan(febus_path)[0]
        key = payload.source_patch_key
        out = io.read_array(febus_path, {"time": (5, 5)}, source_patch_key=key)
        assert out.shape == (0, payload.shape[1])
        out = io.read_array(febus_path, {"distance": (4, 2)}, source_patch_key=key)
        assert out.shape == (payload.shape[0], 0)

    def test_read_array_without_key_raises(self, febus_path):
        """A file holding several zones cannot resolve a keyless read."""
        if len(dc.scan(febus_path)) < 2:
            pytest.skip("file holds one zone")
        with pytest.raises(PatchAttributeError):
            Febus2().read_array(febus_path, {})


class TestFebusA1Interrogator:
    """A1 Source attrs name the interrogator which wrote the file."""

    @pytest.fixture(scope="class", params=["febus_1.h5", "febus_2.h5"])
    def a1_path(self, request):
        """Paths to A1 files of both schema versions."""
        return fetch(request.param)

    def test_interrogator_name_is_hostname(self, a1_path):
        """The name is the Hostname the Source states."""
        with h5py.File(a1_path, "r") as f:
            hostname = _flatten_febus_info(f)[0].source.attrs["Hostname"]
        attrs = dict(dc.scan(a1_path)[0].attrs)
        assert attrs["interrogator.name"] == unbyte(hostname)

    def test_hostname_wins_over_group_name(self, a1_path, tmp_path):
        """A renamed container group does not rename the interrogator."""
        path = tmp_path / "renamed_group.h5"
        shutil.copy2(a1_path, path)
        with h5py.File(path, "r+") as f:
            group_name = next(iter(f))
            hostname = unbyte(f[f"{group_name}/Source1"].attrs["Hostname"])
            f.move(group_name, "container-alias")
        attrs = dict(dc.scan(path)[0].attrs)
        assert attrs["group"] == "container-alias"
        assert attrs["interrogator.name"] == hostname

    def test_blank_hostname_dropped(self, a1_path, tmp_path):
        """An empty Hostname is not passed off as a name."""
        path = tmp_path / "blank_hostname.h5"
        shutil.copy2(a1_path, path)
        with h5py.File(path, "r+") as f:
            group_name = next(iter(f))
            for node in (f[group_name]["Source1"], f[group_name]["Source1"]["Zone1"]):
                node.attrs["Hostname"] = b"  "
        assert "interrogator.name" not in dict(dc.scan(path)[0].attrs)

    def test_scan_and_read_agree(self, a1_path):
        """A read states the same interrogator a scan does."""
        scanned = dict(dc.scan(a1_path)[0].attrs)
        read = dict(Febus2().read(a1_path)[0].attrs)
        assert scanned["interrogator.name"] == read["interrogator.name"]
