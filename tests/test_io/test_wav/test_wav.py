"""Tests module for wave format."""

from __future__ import annotations

from pathlib import Path

import pytest
from scipy.io.wavfile import read as read_wav

import dascore as dc


class TestWriteWav:
    """Tests for writing wav format to disk."""

    @pytest.fixture(scope="class")
    def audio_patch(self):
        """Return the example sin wave patch."""
        return dc.get_example_patch("sin_wav", sample_rate=500)

    @pytest.fixture(scope="class")
    def wave_dir(self, audio_patch, tmp_path_factory):
        """
        Create a temporary directory for WAV files and write the provided audio patch into it.
        
        This function uses the temporary path factory to generate a new directory named "wavs" and writes
        the audio patch into this directory in WAV format using the dc.write function.
        
        Parameters:
            audio_patch: The audio patch object containing waveform data to be written.
            tmp_path_factory (pytest.TempPathFactory): A factory fixture for creating temporary directories.
        
        Returns:
            pathlib.Path: The path to the temporary directory where the WAV file has been written.
        
        Raises:
            Exception: Propagates any exception raised by the dc.write function during the write operation.
        """
        new = Path(tmp_path_factory.mktemp("wavs"))
        dc.write(audio_patch, new, "wav")
        return new

    @pytest.fixture(scope="class")
    def audio_patch_non_distance_dim(self, audio_patch):
        """
        Create a modified audio patch by renaming its 'distance' coordinate to 'microphone'.
        
        Parameters:
            audio_patch (Patch): An audio patch object that contains a 'distance' coordinate along with a time dimension.
        
        Returns:
            Patch: A new audio patch with the 'distance' coordinate renamed to 'microphone'.
        
        Example:
            modified_patch = self.audio_patch_non_distance_dim(original_patch)
            # The modified_patch now uses 'microphone' as the coordinate name instead of 'distance'.
        """
        patch = audio_patch.rename_coords(distance="microphone")
        return patch

    def test_directory(self, wave_dir, audio_patch):
        """
        Perform sanity checks on the WAV directory.
        
        This test verifies that the WAV directory exists and that the number of WAV files in the directory matches the number of 'distance' coordinates in the provided audio patch.
        
        Parameters:
            wave_dir (pathlib.Path): Temporary directory containing the generated WAV files.
            audio_patch: Audio patch data containing a 'distance' coordinate array used to determine the expected number of WAV files.
        
        Raises:
            AssertionError: If the directory does not exist or if the number of WAV files does not match the expected count.
        """
        assert wave_dir.exists()
        wavs = list(wave_dir.rglob("*.wav"))
        assert len(wavs) == len(audio_patch.coords.get_array("distance"))

    def test_write_single_file(self, audio_patch, tmp_path_factory):
        """Ensure a single file can be written."""
        path = tmp_path_factory.mktemp("wave_temp") / "temp.wav"
        dc.write(audio_patch, path, "wav")
        assert path.exists()

    def test_resample(self, audio_patch, tmp_path_factory):
        """
        Test that resampling updates the WAV file's sampling rate correctly.
        
        This test writes an audio patch to a WAV file in a temporary directory with a specified
        resample frequency of 1000 Hz using the dc.write function. After writing, the file is read
        back using read_wav to verify that its sampling rate has been correctly set to 1000 Hz.
        
        Parameters:
            audio_patch (AudioPatch): The audio data patch to write as a WAV file.
            tmp_path_factory (Fixture): Pytest fixture that provides a temporary directory for file creation.
        
        Raises:
            AssertionError: If the WAV file's sampling rate does not equal 1000 Hz.
        """
        path = tmp_path_factory.mktemp("wav_resample") / "resampled.wav"
        dc.write(audio_patch, path, "wav", resample_frequency=1000)
        (sr, ar) = read_wav(str(path))
        assert sr == 1000

    def test_write_non_distance_dims(
        self, audio_patch_non_distance_dim, tmp_path_factory
    ):
        """
        Test writing a WAV file with an audio patch that includes a non-time dimension.
        
        This test verifies that the WAV file writing functionality correctly handles audio patches that contain
        dimensions other than time (for example, a "microphone" dimension instead of the usual "distance" dimension).
        It uses a temporary directory provided by tmp_path_factory to write the file using the audio_patch_non_distance_dim
        and then asserts that the expected output path exists after writing.
        
        Parameters:
            audio_patch_non_distance_dim: An audio patch object that includes a non-time dimension.
            tmp_path_factory: A factory fixture to create temporary directories for file writing.
        
        Raises:
            AssertionError: If the output path does not exist after writing the audio patch.
        """
        path = tmp_path_factory.mktemp("wav_resample")
        patch = audio_patch_non_distance_dim
        patch.io.write(path, "wav")
        assert path.exists()
