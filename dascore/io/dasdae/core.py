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


def _save_attrs_and_dim(patch, patch_group):
    """Save the attributes."""
    # copy attrs to group attrs
    # TODO will need to test if objects are serializable
    for i, v in patch.attrs.items():
        patch_group._v_attrs[i] = v
    patch_group._v_attrs["dims"] = ",".join(patch.dims)


def _save_array(data, name, group, h5):
    """Save an array to a group, handle datetime flubbery."""
    # handle datetime conversions
    is_dt = np.issubdtype(data, np.datetime64)
    is_td = np.issubdtype(data, np.timedelta64)
    if is_dt or is_td:
        data = data

    # out = h5.create_array(
    #     group,
    #     f"coord_{name}",
    #     data,
    # )


def _save_coords(patch, patch_group, h5):
    """Save coordinates"""
    for name, coord in patch._data_array.coords.items():
        _save_array(coord, name, patch_group, h5)

        # array = patch.coords[name]
        # out = h5.create_array(
        #     patch_group,
        #     f"coord_{name}",
        #     array,
        # )
        # # breakpoint()


def _save_patch(patch, wave_group, h5):
    """Save the patch to disk."""
    name = generate_patch_node_name(patch)
    patch_group = h5.create_group(wave_group, name)
    _save_attrs_and_dim(patch, patch_group)
    _save_coords(patch, patch_group, h5)

    # add data
    h5.create_array(patch_group, "data", patch.data)


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
                _save_patch(patch, waveforms, h5)
