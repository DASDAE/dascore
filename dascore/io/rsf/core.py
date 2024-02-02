"""IO module for reading and writing RSF file format support of MADAGASCAR."""
# from __future__ import annotations

# import segyio
import datetime as dt
from pathlib import Path

import numpy as np

import dascore as dc
from dascore.io.core import FiberIO
from dascore.utils.time import to_float

RSFKEYS_WRITE = ("in", "esize", "data_format")


class RSFV1(FiberIO):
    """An IO class supporting the RSF format of MADAGASCAR."""

    name = "rsf"
    preferred_extensions = "rsf"
    # also specify a version so when version 2 is released you can
    # just make another class in the same module named rsfV2.
    version = "1"

    def write(self, spool, path, data_path=None, **kwargs):
        """
        Write a patch to RSF format.

        if no data_path is NOT specified, the header and binary will be
        packed together
        data_path needs to be bindata_file.rsf or /location/of/bindata_file.rsf
        (NO '@')

        path needs to be hdr_file.rsf or /location/of/hdr_file.rsf

        spool needs to have a single patch in it

        time origin is forced to 0.0 in the new RSF file, but a dummy
        variable named 'starttime' still holds the real start time

        Parameters
        ----------
        spool
            The input spool to convert to rsf, must have exactly one patch.
        path
            Path to create the rsf file
        data_path
            If data and rsf header information are to be separate, the
            location of the data file. Needs to be bindata_file.rsf or
            /location/of/bindata_file.rsf (no '@')

        Notes
        -----
        - Patch datatype is converted to float32 for compatibility with
        Madagascar (may be able to keep dytpe in the future)
        """
        assert len(spool) == 1
        patch = spool[0]
        axis_lengs = patch.shape
        axis_origs = [to_float(patch.get_coord(x).start) for x in patch.dims]
        axis_steps = [to_float(patch.get_coord(x).step) for x in patch.dims]
        axis_names = patch.dims
        axis_units = [patch.get_coord(x).units.units for x in patch.dims]

        data = patch.data
        dtype = np.dtype(data.dtype)

        if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
            raise ValueError("Data format is not integer or floating.")
        ## we are casting to float32 to ensure that esize=4 for m8r
        data = patch.data.astype(np.float32)
        dtype = np.dtype(data.dtype)
        file_esize = dtype.itemsize
        file_formt = 'data_format="native_float"'

        data_bytes = data.astype(np.float32).tobytes("F")

        hdr_str = f"DASCORE {dc.__version__}   {dt.datetime.now()} \n"

        length = len(axis_lengs)
        hdr_info = [hdr_str, file_formt, f"esize={file_esize}"]
        for i in range(length):
            hdr_info.append(f"n{i+1}={axis_lengs[i]}")
            if axis_names[i] == "time":
                hdr_info.append(f"o{i+1}=0.0")
                hdr_info.append(f"starttime={axis_origs[i]}")
            else:
                hdr_info.append(f"o{i+1}={axis_origs[i]}")
            hdr_info.append(f"d{i+1}={axis_steps[i]}")
            hdr_info.append(f'label{i+1}="{axis_names[i]}"')
            hdr_info.append(f'unit{i+1}="{axis_units[i]}"')

        if data_path is not None:
            # outputs header and binary separately (.rsf and .rsf@)
            outdatapath = Path(str(data_path) + "@")
            outdatapath.parent.mkdir(exist_ok=True, parents=True)
            hdr_info.append(f'in="{data_path}@"')
            with outdatapath.open("wb") as fi:
                fi.write(data_bytes)
            out = "\n".join(hdr_info)
            outpath = Path(str(path))
            outpath.parent.mkdir(exist_ok=True, parents=True)
            with outpath.open("w") as fi:
                fi.write(out)
        else:
            # outputs header and binary combined (.rsf with both hdr and bin)
            hdr_info.append('in="stdin"\n\n')
            # hdr_info.append(data)
            out = "\n".join(hdr_info)
            outpath = Path(str(path))
            outpath.parent.mkdir(exist_ok=True, parents=True)
            with outpath.open("w") as fi:
                fi.write(out)
            with outpath.open("ab") as fi:
                fi.write(data_bytes)
