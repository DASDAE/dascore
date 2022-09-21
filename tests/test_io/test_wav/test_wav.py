"""
Tests module for wave format.
"""
from pathlib import Path

import pytest
from scipy.io.wavfile import read as read_wav

import dascore as dc


class TestWriteWav:
    """Tests for writing wav format to disk."""

    @pytest.fixture(scope="class")
    def audio_patch(self):
        """Return the example sin wave patch"""
        return dc.get_example_patch("sin_wav", sample_rate=500)

    @pytest.fixture(scope="class")
    def wave_dir(self, audio_patch, tmp_path_factory):
        """Create a wave directory, return path."""
        new = Path(tmp_path_factory.mktemp("wavs"))
        dc.write(audio_patch, new, "wav")
        return new

    def test_directory(self, wave_dir, audio_patch):
        """Sanity checks on wav directory"""
        assert wave_dir.exists()
        wavs = list(wave_dir.rglob("*.wav"))
        assert len(wavs) == len(audio_patch.coords["distance"])

    def test_write_single_file(self, audio_patch, tmp_path_factory):
        """Ensure a single file can be written"""
        path = tmp_path_factory.mktemp("wave_temp") / "temp.wav"
        dc.write(audio_patch, path, "wav")
        assert path.exists()

    def test_resample(self, audio_patch, tmp_path_factory):
        """Ensure resampling changes sampling rate in file"""
        path = tmp_path_factory.mktemp("wav_resample") / "resampled.wav"
        dc.write(audio_patch, path, "wav", resample_frequency=1000)
        (sr, ar) = read_wav(str(path))
        assert sr == 1000
