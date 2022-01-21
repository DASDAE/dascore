"""
Core module for reading and writing pickle format.
"""
import numpy as np
import tables

import dascore as dc
from dascore.io.core import FiberIO


def write_meta(hfile):
    """Write metadata to hdf5 file."""
    attrs = hfile.root._v_attrs
    attrs["__format__"] = "DASDAE"
    attrs["__DASDAE_version__"] = dc.io.dasdae.__version__


def generate_patch_node_name(patch):
    """Generates the name of the node."""

    def _format_datetime64(dt):
        """Format the datetime string in a sensible way."""
        out = str(np.datetime64(dt).astype("datetime64[ns]"))
        return out.replace(":", "_").replace("-", "_").replace(".", "_")

    attrs = patch.attrs
    start = _format_datetime64(attrs.get("time_min", ""))
    end = _format_datetime64(attrs.get("time_max", ""))
    net = attrs.get("network", "")
    sta = attrs.get("station", "")
    tag = attrs.get("tag", "")
    return f"DAS__{net}__{sta}__{tag}__{start}__{end}"


def save_patch(patch, wave_group, h5):
    """Save the patch to disk."""
    name = generate_patch_node_name(patch)
    patch_group = h5.create_group(wave_group, name)
    for i, v in patch.attrs.items():
        patch_group._v_attrs[i] = v


class DASDAEIO(FiberIO):
    """
    Provides IO support for the DASDAE format.
    """

    name = "DASDAE"
    preferred_extensions = ("h5", "hdf5")

    def write(self, patch, path, **kwargs):
        """Read a Patch/Stream from disk."""
        with tables.open_file(path, mode="a") as h5:
            write_meta(h5)
            # get an iterable of patches and save them
            patches = [patch] if isinstance(patch, dc.Patch) else patch
            waveforms = h5.create_group(h5.root, "waveforms")
            for patch in patches:
                save_patch(patch, waveforms, h5)
