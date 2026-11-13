"""IO module for reading and writing RSF file format support of MADAGASCAR."""

from __future__ import annotations

import json
import warnings

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.exceptions import CoordError
from dascore.io.core import FiberIO
from dascore.io.utils import _DASCORE_DIR, require_grid, slice_dataset
from dascore.units import get_quantity_str
from dascore.utils.io import BinaryReader, LocalPath
from dascore.utils.misc import iterate
from dascore.utils.paths import coerce_to_local_path, coerce_to_upath, is_local_path
from dascore.utils.time import to_float
from dascore.warnings import DASCoreWarning

from .utils import (
    HEADER_MAX,
    HISTORY,
    PREFIX,
    RESERVED,
    SEPARATOR,
    get_attrs,
    get_coords,
    get_dtype,
    get_layout,
    read_header,
)

# A history entry is prose: no token in it may bind, and no control
# character (a newline, NUL, or the separator bytes) may end its line.
_HISTORY_TEXT = str.maketrans({"=": ":"} | {chr(x): " " for x in range(32)})


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


def _get_history_lines(patch) -> list[str]:
    """Return a history line per entry of the patch's history."""
    version = dc.__version__
    entries = iterate(patch.attrs.history)
    return [f"{HISTORY}{version} {x.translate(_HISTORY_TEXT)}" for x in entries]


def _get_attr_text(value) -> str:
    """Return an attr value as header text, quoted unless a number, bool or null."""
    if isinstance(value, np.number | np.bool_):
        value = value.item()
    if value is None or isinstance(value, bool | int | float):
        return json.dumps(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _is_one_binding(line: str) -> bool:
    """Return True when Madagascar reads a header line as a single binding."""
    # simtab.c opens or closes a string at every '"', with no escape, and
    # ends a word at whitespace outside one. A control character could end
    # the header early, and a name that is not an identifier cannot bind.
    name, _, _ = line.partition("=")
    parts = line.split('"')
    outside = "".join(parts[::2])
    return (
        name.isascii()
        and name.isidentifier()
        and len(parts) % 2 == 1
        and line.isprintable()
        and not any(x.isspace() for x in outside)
    )


def _get_attr_lines(patch) -> list[str]:
    """Return a key=value line per attr that is set and not stored elsewhere."""
    # the history has its own lines, a read derives the ids from the file,
    # and an attr named for a coordinate's field repeats the coordinate.
    skip = {"history", "origin_id", "data_id"}
    fields = dc.core.CoordSummary.model_fields
    skip |= {f"{x}_{y}" for x in patch.coords.coord_map for y in fields}
    out = []
    for name, value in patch.attrs.model_dump(exclude_defaults=True).items():
        if name in skip:
            continue
        if RESERVED.fullmatch(name) or name.startswith(PREFIX):
            name = PREFIX + name
        line = f"{name}={_get_attr_text(value)}"
        if not _is_one_binding(line):
            msg = (
                f"The {name} attr is not written to RSF: its name is not an "
                "identifier, or Madagascar would misread its quotes, control "
                "characters, or spacing."
            )
            warnings.warn(msg, DASCoreWarning, skip_file_prefixes=(_DASCORE_DIR,))
            continue
        out.append(line)
    return out


class RSFV1(FiberIO):
    """An IO class supporting the RSF format of MADAGASCAR."""

    name = "rsf"
    preferred_extensions = ("rsf",)
    # also specify a version so when version 2 is released you can
    # just make another class in the same module named rsfV2.
    version = "1"

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the version when the resource starts with an RSF header."""
        return self.version if read_header(resource) is not None else None

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Describe the patch from the header alone; see `get_attrs`."""
        hdr, _, text = read_header(resource, raise_on_fail=True)
        coords = get_coords(hdr)
        dtype = get_dtype(hdr).newbyteorder("=")
        cm = dc.core.get_coord_manager(coords=coords, dims=tuple(coords))
        return [dc.PatchMeta(coords=cm, dtype=str(dtype), attrs=get_attrs(text))]

    def read_array(
        self, resource: LocalPath, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Memory-map the samples, n1 fastest, and select the windows."""
        binary, offset, dtype, shape = get_layout(resource)
        data = np.memmap(binary, dtype, "r", offset, shape, order="F")
        out = slice_dataset(data, windows)
        # copy the window, so a later write to the file cannot change it
        return np.array(out, dtype=dtype.newbyteorder("="))

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
        - Each attr that is set, except the ids and coordinate fields such
        as time_min, is one key=value line, prefixed
        'dascore_' when its name is an RSF key such as n1. An attr Madagascar
        would misread (an odd quote, a control character, a non-identifier
        name) is skipped with a DASCoreWarning.
        - Each history entry is a line '# dascore <version> <entry>', with
        '=' written as ':' and control characters as spaces, so it never
        binds; it reads back with those substitutions.
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

        length = len(axis_lengths)
        hdr_info = [*_get_history_lines(patch), file_format, f"esize={file_esize}"]
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
        hdr_info.extend(_get_attr_lines(patch))

        if data_path is not None:
            # outputs header and binary separately (.rsf and .rsf@)
            outdatapath = _coerce_output_path(str(data_path) + "@")
            in_path = outdatapath.resolve() if is_local_path(data_path) else outdatapath
            # RSF has no escape for these inside a quoted header value.
            if '"' in str(in_path) or "\n" in str(in_path):
                msg = (
                    f"RSF cannot record a data path with a quote or newline: {in_path}"
                )
                raise ValueError(msg)
            hdr_info.append(f'in="{in_path}"')
            tail = ()
        else:
            # outputs header and binary combined (.rsf with both hdr and bin)
            hdr_info.append('in="stdin"')
            # chunks, so the samples are not copied into one bytes object
            tail = (b"\n" + SEPARATOR, data_bytes)
        header = ("\n".join(hdr_info) + "\n").encode()
        # A reader looks for in= and the separator in the first HEADER_MAX bytes.
        if len(header) + len(SEPARATOR) + 1 > HEADER_MAX:
            msg = (
                f"The RSF header is {len(header)} bytes, over the {HEADER_MAX} a "
                "reader reads; shorten the patch's history or attrs."
            )
            raise ValueError(msg)
        if data_path is not None:
            outdatapath.parent.mkdir(exist_ok=True, parents=True)
            with outdatapath.open("wb") as fi:
                fi.write(data_bytes)
        outpath = _coerce_output_path(resource)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        with outpath.open("wb") as fi:
            fi.writelines([header, *tail])
