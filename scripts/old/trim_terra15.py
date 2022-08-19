"""
Script to trim down the terra15 test file to be a reasonable size.
"""
from pathlib import Path

import h5py
import numpy as np


def _get_velocity_ind_start_stop(old, frame_ind):
    """Get the indexes for which the velocity arrays should start and stop."""
    # get expected start/stop times for desired frame
    frame_times = old["posix_frame_time"]["data"][:]
    start, stop = frame_times[frame_ind], frame_times[frame_ind + 1]
    v_times = old["velocity"]["posix_time"][:]
    assert np.all(v_times[:-1] <= v_times[1:]), "why arent the times sorted?"
    ind_start = np.searchsorted(v_times, start, side="left")
    ind_stop = np.searchsorted(v_times, stop, side="left")
    return ind_start, ind_stop


def _set_attrs(ar, old_attrs):
    """Set attributes on ar from old_attrs."""

    attr_name = [x for x in dir(old_attrs) if not x.startswith("_")]
    for x in attr_name:
        setattr(ar.attrs, x, getattr(old_attrs, x))


def copy_attrs(old_node, new_node):
    """Copy all attributes form one node to the other"""
    for i, v in dict(old_node.attrs).items():
        new_node.attrs[i] = v


if __name__ == "__main__":
    here = Path(__file__).parent.absolute().parent
    path = here / "tests" / "test_large.hdf5"

    assert path.exists(), f"{path} does not exist!"

    new_path = path.parent / "terra15_das_1_trimmed.hdf5"
    frame_ind = 0  # the frame index to extract
    frame_groups = ["frame_id", "posix_frame_time", "gps_frame_time"]

    with h5py.File(path) as old, h5py.File(new_path, "w") as new:
        copy_attrs(old["/"], new["/"])
        for frame_group in frame_groups:
            old_frame = old[frame_group]
            old_data = old_frame["data"]

            old_array = old_data[:]

            new_array = np.array([old_array[frame_ind]])

            gr = new.create_group(f"/{frame_group}")
            ar = gr.create_dataset(
                "data",
                dtype=old_data.dtype,
                data=new_array,
                chunks=old_data.chunks,
            )
            copy_attrs(ar, old_data)
        # now get velocity info
        velgroup = old["velocity"]
        gr = new.create_group("velocity")
        copy_attrs(velgroup, gr)
        start, stop = _get_velocity_ind_start_stop(old, frame_ind)
        for dataset_name in velgroup:
            old_dataset = velgroup[dataset_name]
            old_data = old_dataset[:]
            new_data = old_data[start:stop]
            new_array = gr.create_dataset(
                data=new_data,
                name=dataset_name,
                dtype=old_dataset.dtype,
                chunks=old_dataset.chunks,
            )
            copy_attrs(old_dataset, new_array)
            # _set_attrs(new_array, vel_array.attrs)
