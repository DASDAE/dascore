"""Tests module for wave format."""

from __future__ import annotations

from pathlib import Path

import pytest
from scipy.io.wavfile import read as read_wav
from upath import UPath

import dascore as dc
from dascore.constants import ONE_SECOND


class TestWriteWav:
    """Tests for writing wav format to disk."""

    @pytest.fixture(scope="class")
    def audio_patch(self):
        """Return the example sin wave patch."""
        return dc.get_example_patch("sin_wav", sample_rate=500)

    @pytest.fixture(scope="class")
    def wave_dir(self, audio_patch, tmp_path_factory):
        """Create a wave directory, return path."""
        new = Path(tmp_path_factory.mktemp("wavs"))
        dc.write(audio_patch, new, "wav")
        return new

    @pytest.fixture(scope="class")
    def audio_patch_non_distance_dim(self, audio_patch):
        """Create a patch that has a non-distance dimension in addition to time."""
        patch = audio_patch.rename_coords(distance="microphone")
        return patch

    def test_directory(self, wave_dir, audio_patch):
        """Sanity checks on wav directory."""
        assert wave_dir.exists()
        wavs = list(wave_dir.rglob("*.wav"))
        assert len(wavs) == len(audio_patch.coords.get_array("distance"))

    def test_write_single_file(self, audio_patch, tmp_path_factory):
        """Ensure a single file can be written."""
        path = tmp_path_factory.mktemp("wave_temp") / "temp.wav"
        dc.write(audio_patch, path, "wav")
        assert path.exists()

    def test_write_single_local_upath(self, audio_patch, tmp_path_factory):
        """Local UPath file destinations should work for wav writes."""
        path = UPath(tmp_path_factory.mktemp("wave_temp_upath") / "temp.wav")
        dc.write(audio_patch, path, "wav")
        assert path.exists()

    def test_write_single_remote_upath(self, audio_patch):
        """Remote UPath file destinations should work for wav writes."""
        path = UPath("memory://dascore/temp.wav")
        dc.write(audio_patch, path, "wav")
        assert path.exists()

    def test_resample(self, audio_patch, tmp_path_factory):
        """Ensure resampling changes sampling rate in file."""
        path = tmp_path_factory.mktemp("wav_resample") / "resampled.wav"
        dc.write(audio_patch, path, "wav", resample_frequency=1000)
        (sr, ar) = read_wav(str(path))
        assert sr == 1000

    def test_write_non_distance_dims(
        self, audio_patch_non_distance_dim, tmp_path_factory
    ):
        """Ensure any non-time dimension still works."""
        path = tmp_path_factory.mktemp("wav_resample")
        patch = audio_patch_non_distance_dim
        patch.io.write(path, "wav")
        assert path.exists()
        # Verify number of WAV files
        wavs = list(path.rglob("*.wav"))
        assert len(wavs) == len(patch.coords.get_array("microphone"))
        # Verify file naming
        for mic_val in patch.coords.get_array("microphone"):
            assert path / f"microphone_{mic_val}.wav" in wavs
            # Verify content of first file
            sr, data = read_wav(str(wavs[0]))
        assert sr == int(ONE_SECOND / patch.get_coord("time").step)

    def test_write_remote_directory(self, audio_patch_non_distance_dim):
        """Remote directory destinations should work for wav writes."""
        path = UPath("memory://dascore/wav_dir")
        patch = audio_patch_non_distance_dim
        patch.io.write(path, "wav")
        wavs = list(path.glob("*.wav"))
        assert len(wavs) == len(patch.coords.get_array("microphone"))
