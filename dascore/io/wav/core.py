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
        Write the contents of a single patch from the spool to WAV file(s).
        
        Parameters:
            spool (SpoolType): A collection containing a single data patch to write. An AssertionError is raised if more than one patch is provided.
            resource (str or Path): Destination path. If the path ends with ".wav", all non-time channels from the patch are written as a single multi-channel WAV file.
                Otherwise, the path is treated as a directory and a separate WAV file is created for each channel corresponding to the non-time dimension.
            resample_frequency (Optional[float]): Sampling frequency in Hz to which the patch data should be resampled. If None, the original sampling rate is used.
            **kwargs: Additional keyword arguments (currently unused).
        
        Raises:
            AssertionError: If the spool contains more than one patch.
        
        Notes:
            - The WAV format requires an integer sampling rate; hence, the sample rate is cast to an int before writing,
              which may introduce minor distortion if the original value is non-integral.
            - The patch data is transposed to ensure time is the leading dimension, then detrended, normalized to the range (-1, 1),
              and converted to np.float32 prior to writing.
            - When writing multiple WAV files (i.e., when resource is a directory), the non-time dimension is determined dynamically,
              allowing for flexible handling of patches with varying coordinate names.
            - Some audio players may not support multi-channel WAV files. If playback issues occur (for example, with VLC),
              try setting resample_frequency to 44100.
        
        Returns:
            None
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
            non_time_name = next(
                iter(
                    set(patch.dims)
                    - {
                        "time",
                    }
                )
            )
            non_time = patch.coords.get_array(non_time_name)
            for ind, val in enumerate(non_time):
                sub_data = np.take(data, ind, axis=1)
                sub_path = resource / f"{non_time_name}_{val}.wav"
                write(filename=str(sub_path), rate=int(sr), data=sub_data)

    @staticmethod
    def _get_wav_data(patch, resample):
        """
        Pre-processes a 2D patch for WAV file writing by ensuring proper dimensions, optional resampling, detrending, and normalization.
        
        This function validates that the input patch has exactly two dimensions and includes a 'time' coordinate. It transposes the patch so that the time dimension comes first, optionally resamples the data if a resample frequency is provided, and then applies linear detrending and max normalization along the time axis. The function returns the processed data as a NumPy float32 array along with the calculated sample rate.
        
        Parameters:
            patch: An object representing the data patch. It must:
                   - Contain a 'time' coordinate (validated via check_patch_coords).
                   - Have exactly 2 dimensions (asserted).
                   - Provide methods such as get_coord, transpose, resample, detrend, and normalize.
            resample (optional): A float specifying the desired sample rate in Hz. If provided, the patch data
                   will be resampled using a time interval of 1/resample. If None, the sample rate is derived from the
                   patch's time coordinate step (using ONE_SECOND / time).
        
        Returns:
            tuple: A tuple containing:
                   - data (np.ndarray): The processed audio data as a NumPy array of type np.float32.
                   - sample_rate (int): The sample rate in Hz, determined by the resample parameter if provided, or
                     computed from the time coordinate's step.
        
        Raises:
            AssertionError: If the patch does not have exactly 2 dimensions.
            Exception: Any exceptions raised by check_patch_coords if the 'time' coordinate is missing.
        """
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
