"""
A simple script to create and downsize a hdf5 files.

This script can be used as a template for creating small test files.
"""

from __future__ import annotations

import h5py
from h5py import Dataset, Group


def _copy_attrs(
    old_obj: Group | Dataset,
    new_obj: Group | Dataset,
    address: str,
    replace=None,
):
    """Copy attrs from obj1 to obj2"""
    replace = {} if not replace else replace
    # Iterate each of the attributes
    for key, value in old_obj.attrs.items():
        addr = f"{address}.{key}"
        new_obj.attrs[key] = replace.get(addr, value)
    return new_obj


def _recurse_n_replace(f_in, f_out, address, replaces, funcs):
    """Recurse the object applying replacements when needed."""
    iterator = f_in[address].items() if address else f_in.items()

    for name, value in iterator:
        new_address = f"{address}/{name}" if address else name
        if func := funcs.get(new_address):
            func(f_in, f_out, new_address, replaces)
        elif isinstance(value, Dataset):
            group = f_out[address]
            old_dataset = f_in[new_address]
            new_dataset = group.create_dataset_like(name, other=value)
            # copy the actual data.
            new_dataset[:] = value[:]
            _copy_attrs(old_dataset, new_dataset, new_address, replaces)
        if isinstance(value, Group):
            new_group = f_out.create_group(new_address)
            _copy_attrs(value, new_group, new_address, replaces)
            _recurse_n_replace(f_in, f_out, new_address, replaces, funcs)


def copy_h5(path_old, path_new, attrs_to_replace=None, funcs=None):
    """
    Copy an old h5 to a new one while apply specific modifications.

    Parameters
    ----------
    path_old
        The old path
    path_new
        The new path
    attrs_to_replace
        A dict of attributes which will be replaced with new values.
    funcs
        A dict of functions whose keys are addresses and values are
        custome functions to apply.
    """
    replace = attrs_to_replace if attrs_to_replace else {}
    funcs = funcs if funcs else {}
    with h5py.File(path_old, "r") as fi_in:
        with h5py.File(path_new, "w") as fi_out:
            _recurse_n_replace(fi_in, fi_out, "", replace, funcs)


def downsize_raw_data(f_in, f_out, address, replaces):
    """A custom function to replace the raw data."""
    group_name = "/".join(address.split("/")[:-1])
    dataset_name = address.split("/")[-1]

    group = f_out[group_name]
    old_dataset = f_in[address]
    new_data = old_dataset[:, 0:10]

    new_dataset = group.create_dataset(name=dataset_name, data=new_data)
    _copy_attrs(old_dataset, new_dataset, address, replaces)


if __name__ == "__main__":
    path_old = "Fibre_Pos_DD00_160308174030.h5"
    path_new = "gdr_example_1.h5"

    new_overview = "This is a modified file from GDR created for DASCore's test suite"
    acq_path = "DasMetadata/Interrogator/Acquisition"
    attrs_to_replace = {
        f"{acq_path}.NumberOfChannels": 10,
        f"{acq_path}.Acquisition/ChannelGroup.LastUsableChannelID": 10,
        "DasMetadata.Overview": new_overview,
    }
    replace_funcs = {"DasRawData/RawData": downsize_raw_data}

    copy_h5(path_old, path_new, funcs=replace_funcs, attrs_to_replace=attrs_to_replace)
