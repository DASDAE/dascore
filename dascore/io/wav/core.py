"""
Core module for wave format.
"""
from pathlib import Path
from typing import Union

import numpy as np
from scipy.io.wavfile import write

from dascore.constants import ONE_SECOND, SpoolType
from dascore.io.core import FiberIO
from dascore.utils.patch import check_patch_dims


class WavIO(FiberIO):
    """
    IO support for wav (audio) format.
    """

    name = "WAV"

    def write(self, spool: SpoolType, path: Union[str, Path], resample_frequency=None):
        """
        Write the contents of the patch to one or more wav files.

        Parameters
        ----------
        path
            If a path that ends with .wav, write all the distance channels
            to a single file. If not, assume the path is a directory and write
            each distance channel to its own wav file.
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
            the output the patch has more than one len along the distance
            dimension, a multi-channel wavefile is created. There may be some
            players that do not support multi-channel wavefiles.

            - If using VLC, often it won't play the file unless the sampling
            rate is 44100, in this case just set resample_frequency=44100 to
            see if this fixes the issue.
        """
        path = Path(path)
        assert len(spool) == 1, "Only single patch spools can be written to wav"
        patch = spool[0]
        # write a single wav file, maybe multi-channeled.
        data, sr = self._get_wav_data(patch, resample_frequency)
        if path.name.endswith(".wav"):
            write(filename=str(path), rate=int(sr), data=data)
        else:  # write data to directory, one file for each distance
            path.mkdir(exist_ok=True, parents=True)
            distances = patch.coords["distance"]
            for ind, dist in enumerate(distances):
                sub_data = np.take(data, ind, axis=1)
                sub_path = path / f"{dist}.wav"
                write(filename=str(sub_path), rate=int(sr), data=sub_data)

    @staticmethod
    def _get_wav_data(patch, resample):
        """Pre-condition patch data for writing. Return array and sample rate."""
        check_patch_dims(patch, ("time", "distance"))
        assert len(patch.dims) == 2, "only 2D patches supported for this function."
        # handle resampling and normalization
        pat = patch.transpose("time", "distance")
        if resample is not None:
            pat = pat.resample(time=1 / resample)
        # normalize and detrend
        pat = pat.detrend("time", "linear").normalize("time", norm="max")
        data = pat.data
        sample_rate = resample or np.round(ONE_SECOND / pat.attrs["d_time"])
        return data.astype(np.float32), int(sample_rate)
