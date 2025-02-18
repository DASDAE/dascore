"""Core module for wave format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

from dascore.constants import ONE_SECOND, SpoolType
from dascore.io.core import FiberIO
from dascore.utils.patch import check_patch_coords


class WavIO(FiberIO):
    """IO support for wav (audio) format."""

    name = "WAV"

    def write(
        self, spool: SpoolType, resource: str | Path, resample_frequency=None, **kwargs
    ):
        """
        Write the contents of the patch to one or more wav files.

        Parameters
        ----------
        resource
            If a path that ends with .wav, write all non-time channels
            to a single file. If not, assume the path is a directory and
            write each non-time channel to its own wav file.
        resample_frequency
            A resample frequency in Hz. If None, do not perform resampling.
            Often DAS has non-int sampling rates, so the default resampling
            rate is usually safe.

        Notes
        -----
            - The sampling rate in the wav format must be an int, so the
            patch's sampling rate is cast to an integer before writing wav file.
            This may cause some distortion (but it isn't likely to be noticeable).

            - The array data type is converted to np.float32 before writing. This
            requires values to be between (-1, 1) so the data are detrended
            and normalized before writing.

            - If a single wavefile is specified with the path argument, and
            the output the patch has more than one len along the non-time
            dimension, a multi-channel wavefile is created. There may be some
            players that do not support multi-channel wavefiles.

            - If using VLC, often it won't play the file unless the sampling
            rate is 44100, in this case just set resample_frequency=44100 to
            see if this fixes the issue.
        """
        resource = Path(resource)
        assert len(spool) == 1, "Only single patch spools can be written to wav"
        patch = spool[0]
        # write a single wav file, maybe multi-channeled.
        data, sr = self._get_wav_data(patch, resample_frequency)
        if resource.name.endswith(".wav"):
            write(filename=str(resource), rate=int(sr), data=data)
        else:  # write data to directory, one file for each non-time
            resource.mkdir(exist_ok=True, parents=True)
            non_time_set = set(patch.dims) - {"time"}
            non_time_name = next(iter(non_time_set))
            non_time = patch.coords.get_array(non_time_name)
            for ind, val in enumerate(non_time):
                sub_data = np.take(data, ind, axis=1)
                sub_path = resource / f"{non_time_name}_{val}.wav"
                write(filename=str(sub_path), rate=int(sr), data=sub_data)

    @staticmethod
    def _get_wav_data(patch, resample):
        """Pre-condition patch data for writing. Return array and sample rate."""
        # Ensure we have a 2D patch which has a time dimension.
        check_patch_coords(patch, ("time",))
        assert len(patch.dims) == 2, "only 2D patches supported for this function."
        time = patch.get_coord("time").step

        # handle resampling and normalization
        pat = patch.transpose("time", ...)
        if resample is not None:
            pat = pat.resample(time=1 / resample)
        # normalize and detrend
        pat = pat.detrend("time", "linear").normalize("time", norm="max")
        data = pat.data
        sample_rate = resample or np.round(ONE_SECOND / time)
        return data.astype(np.float32), int(sample_rate)
