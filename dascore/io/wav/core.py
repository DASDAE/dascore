"""
Core module for wave format.
"""

from pathlib import Path
from typing import Union

import numpy as np
from scipy.io.wavfile import write

from dascore.constants import PatchType
from dascore.utils.patch import patch_function


@patch_function(required_dims=("time", "distance"))
def _write_wavfolder(patch: PatchType, path: Union[str, Path]):
    """
    Write the contents of the patch to wavefiles in a folder.

    Each distance channel is writen as its own file.

    Parameters
    ----------
    path
        The *directory* path to which the wave files are written.

    Notes
    -----
    The sampling rate in the wav format must be an int, so the patch's sampling
    rate is cast to an integer before writing wav file.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    assert len(patch.dims) == 2, "only 2D patches supported for this function."
    axis = patch.dims.index("distance")
    data = patch.data

    distance = patch.coords["distance"]
    sr = 1 / (patch.attrs["d_time"] / np.timedelta64(1, "s"))
    inds = np.arange(len(distance))

    for ind, dist in enumerate(patch.coords["distance"]):
        data = np.take(data, inds, axis=axis)
        sub_path = path / f"{dist}.wav"
        write(filename=str(sub_path), rate=int(sr), data=data)
