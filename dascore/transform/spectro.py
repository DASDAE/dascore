"""
Module to transform a Patch into spectrograms.
"""

from scipy.signal import spectrogram as scipy_spectrogram

from dascore.constants import PatchType
from dascore.utils.misc import _get_sampling_rate
from dascore.utils.patch import patch_function
from dascore.utils.time import to_timedelta64


def _get_new_dims(dims, dim):
    """Get the new dimensions for the spectrogram output."""
    out = []
    for val in dims:
        if val == dim:
            out.append("frequency")
        else:
            out.append(val)
    out.append(dim)
    return out


@patch_function()
def spectrogram(patch: PatchType, dim: str = "time", **kwargs) -> PatchType:
    """
    Convert a patch to spectrogram arrays along a desired dimension.

    Parameters
    ----------
    patch
        If not None, an axis on which to plot.
    dim
        The dimension along which the spectrograms are calculated.
    **kwargs
        Passed directly to `scipy.signal.spectrogram` to control spectrogram
        construction.

    """
    assert dim == "time", "only supporting time for now."
    axis = patch.dims.index(dim)
    fs = _get_sampling_rate(patch.attrs[f"d_{dim}"])
    # returns frequency, new values for original dimension (eg time) and spectrogram
    freqs, original, spec = scipy_spectrogram(patch.data, fs=fs, axis=axis, **kwargs)
    if dim == "time":
        original = to_timedelta64(original)

    new_coord = original + patch.attrs[f"{dim}_min"]
    new_dims = _get_new_dims(patch.dims, dim)
    coords = {x: patch.coords[x] for x in patch.dims}
    coords.update({dim: new_coord, "frequency": freqs})
    return patch.__class__(spec, coords=coords, dims=new_dims, attrs=patch.attrs)
