"""
Script to trim down the terra15 test file to be a reasonable size.
"""
from pathlib import Path
import tables as tb

import numpy as np

here = Path(__file__).parent.absolute().parent
path = here / "tests" / "test_data" / "terra15_v2_das_1.hdf5"


def _get_velocity_ind_start_stop(old_root, frame_ind):
    """Get the indexes for which the velocity arrays should start and stop."""
    # get expected start/stop times for desired frame
    frame_times = old_root["posix_frame_time"]["data"].read()
    start, stop = frame_times[frame_ind], frame_times[frame_ind + 1]
    v_times = old.root["velocity"]["posix_time"].read()
    assert np.all(v_times[:-1] <= v_times[1:]), "why arent the times sorted?"
    ind_start = np.searchsorted(v_times, start, side="left")
    ind_stop = np.searchsorted(v_times, stop, side="left")
    return ind_start, ind_stop


def _set_attrs(ar, old_attrs):
    """Set attributes on ar from old_attrs."""
    attr_name = [x for x in dir(old_attrs) if not x.startswith("_")]
    for x in attr_name:
        setattr(ar.attrs, x, getattr(old_attrs, x))


if __name__ == "__main__":
    assert path.exists(), f"{path} does not exist!"

    new_path = path.parent / "terra15_v2_das_1_trimmed.hdf5"
    frame_ind = 10  # the frame index to extract
    frame_groups = ["frame_id", "posix_frame_time", "gps_frame_time"]

    with tb.open_file(path) as old, tb.open_file(new_path, "w") as new:
        for frame_group in frame_groups:
            old_frame = old.root[frame_group]
            old_data = old_frame["data"]
            old_array = old_data.read()

            new_array = np.array([old_array[frame_ind]])

            gr = new.create_group(new.root, frame_group)
            ar = new.create_carray(
                gr,
                "data",
                atom=old_data.atom,
                obj=new_array,
                chunkshape=old_data.chunkshape,
            )
            _set_attrs(ar, old_data.attrs)

        # now get velocity info
        velgroup = old.root["velocity"]
        gr = new.create_group(new.root, "velocity")
        start, stop = _get_velocity_ind_start_stop(old.root, frame_ind)
        for vel_array in velgroup:
            old_data = vel_array.read()
            new_data = old_data[start:stop]
            new_array = new.create_carray(
                gr,
                obj=new_data,
                name=vel_array.name,
                chunkshape=vel_array.chunkshape,
                atom=vel_array.atom,
            )
            _set_attrs(new_array, vel_array.attrs)
