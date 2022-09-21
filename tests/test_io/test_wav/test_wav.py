"""
Tests module for wave format.
"""
from pathlib import Path

import pytest

import dascore as dc
from dascore.constants import ONE_SECOND


class TestWriteWav:
    """Tests for writing wav format to disk."""

    @pytest.fixture(scope="class")
    def trimmed_down_patch(self, random_patch):
        """Trim down the random patch for a faster test."""
        dists = random_patch.coords["distance"]
        time = random_patch.coords["time"]
        out = random_patch.select(
            distance=(dists[0], dists[2]), time=(time[0], time[0] + ONE_SECOND)
        )
        return out

    @pytest.fixture(scope="class")
    def wave_dir(self, trimmed_down_patch, tmp_path_factory):
        """Create a wave directory, return path."""
        new = Path(tmp_path_factory.mktemp("wavs"))
        dc.write(trimmed_down_patch, new, "wav")
        return new

    def test_directory(self, wave_dir, trimmed_down_patch):
        """Sanity checks on wav directory"""
        assert wave_dir.exists()
        wavs = list(wave_dir.rglob("*.wav"))
        assert len(wavs) == len(trimmed_down_patch.coords["distance"])

    def test_write_single_file(self, trimmed_down_patch, tmp_path_factory):
        """Ensure a single file can be written"""
        path = tmp_path_factory.mktemp("wave_temp") / "temp.wav"
        dc.write(trimmed_down_patch, path, "wav")
        assert path.exists()
