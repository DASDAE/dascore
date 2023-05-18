"""
Module to transform a Patch into spectrograms.
"""

from scipy.signal import spectrogram as scipy_spectrogram

from dascore.constants import PatchType
from dascore.utils.misc import _get_sampling_rate
from dascore.utils.patch import patch_function
from dascore.utils.time import to_timedelta64
from dascore.utils.transformatter import FourierTransformatter


def _get_new_attrs_coords(patch, new_dims, axis, new_array, old_array):
    """Get new attributes and coords for spectrogram transform."""
    new_name = new_dims[axis]
    old_name = new_dims[-1]
    coords = dict(patch.coords)
    coords[new_name] = new_array
    coords[old_name] = old_array
    return dict(patch.attrs), coords


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
    ft = FourierTransformatter()
    assert dim == "time", "only supporting time for now."
    axis = patch.dims.index(dim)
    fs = _get_sampling_rate(patch.attrs[f"d_{dim}"])
    # returns frequency, new values for original dimension (eg time) and spectrogram
    freqs, original, spec = scipy_spectrogram(patch.data, fs=fs, axis=axis, **kwargs)
    if dim == "time":
        original = to_timedelta64(original)
    new_coord = original + patch.attrs[f"{dim}_min"]
    new_dims = list(ft.rename_dims(patch.dims, index=axis)) + [dim]
    attrs, coords = _get_new_attrs_coords(patch, new_dims, axis, freqs, new_coord)
    return patch.__class__(spec, coords=coords, dims=new_dims, attrs=patch.attrs)
