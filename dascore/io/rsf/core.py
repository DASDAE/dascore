"""IO module for reading and writing RSF file format support of MADAGASCAR."""
# from __future__ import annotations

# import segyio
import datetime as dt
import warnings

import numpy as np

import dascore as dc
from dascore.exceptions import CoordError
from dascore.io.core import FiberIO
from dascore.io.utils import _DASCORE_DIR, require_grid
from dascore.units import get_quantity_str
from dascore.utils.paths import coerce_to_local_path, coerce_to_upath, is_local_path
from dascore.utils.time import to_float
from dascore.warnings import DASCoreWarning

RSFKEYS_WRITE = ("in", "esize", "data_format")


def _coerce_output_path(path):
    """Return a local Path or remote UPath-like path for writing."""
    return coerce_to_local_path(path) if is_local_path(path) else coerce_to_upath(path)


def _get_step(coord):
    """Return the coordinate's step as a float, exactly when it has an exact form."""
    step = coord.step_exact
    return float(step) if step is not None else to_float(coord.step)


def _get_epoch(coord):
    """Return a datetime coordinate's start as an ISO 8601 UTC string, or None."""
    if not np.issubdtype(coord.dtype, np.datetime64):
        return None
    start = np.datetime64(coord[0], "ns")
    return f"{np.datetime_as_string(start, unit='ns')}Z"


class RSFV1(FiberIO):
    """An IO class supporting the RSF format of MADAGASCAR."""

    name = "rsf"
    preferred_extensions = ("rsf",)
    # also specify a version so when version 2 is released you can
    # just make another class in the same module named rsfV2.
    version = "1"

    def write(self, spool, resource, data_path=None, **kwargs):
        """
        Write a patch to RSF format.

        if no data_path is NOT specified, the header and binary will be
        packed together
        data_path needs to be bindata_file.rsf or /location/of/bindata_file.rsf
        (NO '@')

        resource needs to be hdr_file.rsf or /location/of/hdr_file.rsf

        spool needs to have a single patch in it

        for a datetime time axis i, o<i> is 0.0, epoch<i> holds the start
        time as an ISO 8601 UTC string with nanoseconds, and the legacy
        key 'starttime' holds it as float seconds since 1970; any other
        axis, including a time axis of offsets, holds its start in o<i>

        Parameters
        ----------
        spool
            The input spool to convert to rsf, must have exactly one patch.
        resource
            Path to create the rsf file
        data_path
            If data and rsf header information are to be separate, the
            location of the data file. Needs to be bindata_file.rsf or
            /location/of/bindata_file.rsf (no '@')

        Notes
        -----
        - Patch datatype is converted to float32 for compatibility with
        Madagascar (may be able to keep dtype in the future), with a
        DASCoreWarning when the data are not already float32
        - With data_path, the header's in= holds the binary's absolute path
        - Without data_path, the header ends with a blank line and the
        separator bytes 0x0c 0x0c 0x04 before the binary
        """
        patches = [spool] if isinstance(spool, dc.Patch) else spool
        assert len(patches) == 1
        patch = patches[0]
        require_grid(patch, patch.dims, "RSF")
        axis_lengths = patch.shape
        axis_origs = [to_float(patch.get_coord(x)[0]) for x in patch.dims]
        if not np.all(np.isfinite(axis_origs)):
            msg = "The RSF writer needs a finite start on every dimension."
            raise CoordError(msg)
        axis_steps = [_get_step(patch.get_coord(x)) for x in patch.dims]
        axis_names = patch.dims
        axis_units = [get_quantity_str(patch.get_coord(x).units) for x in patch.dims]
        axis_epochs = [_get_epoch(patch.get_coord(x)) for x in patch.dims]

        data = patch.data
        dtype = np.dtype(data.dtype)

        if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
            raise ValueError("Data format is not integer or floating.")
        if dtype != np.float32:
            msg = f"RSF stores float32; casting {dtype} data to float32."
            # Point at the caller's write, past dascore's dispatch.
            warnings.warn(msg, DASCoreWarning, skip_file_prefixes=(_DASCORE_DIR,))
        ## we are casting to float32 to ensure that esize=4 for m8r
        data = data.astype(np.float32, copy=False)
        dtype = np.dtype(data.dtype)
        file_esize = dtype.itemsize
        file_format = 'data_format="native_float"'

        data_bytes = data.tobytes("F")

        hdr_str = f"DASCORE {dc.__version__}   {dt.datetime.now()} \n"

        length = len(axis_lengths)
        hdr_info = [hdr_str, file_format, f"esize={file_esize}"]
        for i in range(length):
            hdr_info.append(f"n{i + 1}={axis_lengths[i]}")
            if axis_names[i] == "time" and axis_epochs[i] is not None:
                hdr_info.append(f"o{i + 1}=0.0")
                hdr_info.append(f'epoch{i + 1}="{axis_epochs[i]}"')
                # legacy key, kept for readers of earlier output
                hdr_info.append(f"starttime={axis_origs[i]}")
            else:
                hdr_info.append(f"o{i + 1}={axis_origs[i]}")
            hdr_info.append(f"d{i + 1}={axis_steps[i]}")
            hdr_info.append(f'label{i + 1}="{axis_names[i]}"')
            if axis_units[i]:
                hdr_info.append(f'unit{i + 1}="{axis_units[i]}"')

        if data_path is not None:
            # outputs header and binary separately (.rsf and .rsf@)
            outdatapath = _coerce_output_path(str(data_path) + "@")
            outdatapath.parent.mkdir(exist_ok=True, parents=True)
            in_path = outdatapath.resolve() if is_local_path(data_path) else outdatapath
            hdr_info.append(f'in="{in_path}"')
            with outdatapath.open("wb") as fi:
                fi.write(data_bytes)
            out = "\n".join(hdr_info) + "\n"
            outpath = _coerce_output_path(resource)
            outpath.parent.mkdir(exist_ok=True, parents=True)
            with outpath.open("w") as fi:
                fi.write(out)
        else:
            # outputs header and binary combined (.rsf with both hdr and bin)
            hdr_info.append('in="stdin"\n\n')
            out = "\n".join(hdr_info).encode()
            outpath = _coerce_output_path(resource)
            outpath.parent.mkdir(exist_ok=True, parents=True)
            with outpath.open("wb") as fi:
                fi.write(out)
                fi.write(b"\x0c\x0c\x04")
                fi.write(data_bytes)
