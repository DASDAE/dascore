"""
Tests module for wave format.
"""
from pathlib import Path

import pytest

import dascore


class TestWriteWav:
    """Tests for writing wav format to disk."""

    @pytest.fixture(scope="class")
    def trimmed_down_patch(self, random_patch):
        """Trim down the random patch for a faster test."""
        dists = random_patch.coords["distance"]
        out = random_patch.select(distance=(dists[0], dists[2]))
        return out

    @pytest.fixture(scope="class")
    def wave_dir(self, trimmed_down_patch, tmp_path_factory):
        """Create a wave directory, return path."""
        new = Path(tmp_path_factory.mktemp("wavs"))
        dascore.write(trimmed_down_patch, new, "wav")
        return new

    def test_directory(self, wave_dir, trimmed_down_patch):
        """Sanity checks on wav directory"""
        assert wave_dir.exists()
        wavs = list(wave_dir.rglob("*.wav"))
        assert len(wavs) == len(trimmed_down_patch.coords["distance"])
