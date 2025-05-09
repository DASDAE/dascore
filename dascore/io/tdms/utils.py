"""Utilities for TDMS format."""

from __future__ import annotations

import datetime
import mmap
import os
import struct

import numpy as np

from dascore.core.attrs import PatchAttrs
from dascore.core.coords import get_coord
from dascore.utils.time import to_datetime64, to_timedelta64

DEFAULT_ATTRS = set(PatchAttrs.model_fields)


def type_not_supported(vargin):
    """Function raises a NotImplementedException."""
    raise NotImplementedError("Reading of this tdsDataType is not implemented")


# Enum mapping TDM data types to description string, numpy type where exists
# See Ref[2] for enum values
TDS_DATA_TYPE = dict(
    {
        0x00: "void",  # tdsTypeVoid
        0x01: "int8",  # tdsTypeI8
        0x02: "int16",  # tdsTypeI16
        0x03: "int32",  # tdsTypeI32
        0x04: "int64",  # tdsTypeI64
        0x05: "uint8",  # tdsTypeU8
        0x06: "uint16",  # tdsTypeU16
        0x07: "uint32",  # tdsTypeU32
        0x08: "uint64",  # tdsTypeU64
        0x09: "float32",  # tdsTypeSingleFloat
        0x0A: "float64",  # tdsTypeDoubleFloat
        0x0B: "float128",  # tdsTypeExtendedFloat
        0x19: "singleFloatWithUnit",  # tdsTypeSingleFloatWithUnit
        0x1A: "doubleFloatWithUnit",  # tdsTypeDoubleFloatWithUnit
        0x1B: "extendedFloatWithUnit",  # tdsTypeExtendedFloatWithUnit
        0x20: "str",  # tdsTypeString
        0x21: "bool",  # tdsTypeBoolean
        0x44: "datetime",  # tdsTypeTimeStamp
        0xFFFFFFFF: "raw",  # tdsTypeDAQmxRawData
    }
)

# Function mapping for reading TDMS data types
TDS_READ_VAL = dict(
    {
        "void": lambda f: None,  # tdsTypeVoid
        "int8": lambda f: struct.unpack("<b", f.read(1))[0],
        "int16": lambda f: struct.unpack("<h", f.read(2))[0],
        "int32": lambda f: struct.unpack("<i", f.read(4))[0],
        "int64": lambda f: struct.unpack("<q", f.read(8))[0],
        "uint8": lambda f: struct.unpack("<B", f.read(1))[0],
        "uint16": lambda f: struct.unpack("<H", f.read(2))[0],
        "uint32": lambda f: struct.unpack("<I", f.read(4))[0],
        "uint64": lambda f: struct.unpack("<Q", f.read(8))[0],
        "float32": lambda f: struct.unpack("<f", f.read(4))[0],
        "float64": lambda f: struct.unpack("<d", f.read(8))[0],
        "float128": type_not_supported,
        "singleFloatWithUnit": type_not_supported,
        "doubleFloatWithUnit": type_not_supported,
        "extendedFloatWithUnit": type_not_supported,
        "str": lambda f: f.read(struct.unpack("<i", f.read(4))[0]),
        "bool": lambda f: struct.unpack("<?", f.read(1))[0],
        "datetime": lambda f: parse_time_stamp(
            struct.unpack("<Q", f.read(8))[0], struct.unpack("<q", f.read(8))[0]
        ),
        "raw": type_not_supported,
    }
)

DECIMATE_MASK = 0b00100000
FILEINFO_NAMES = (
    "file_tag",
    "toc",
    "version",
    "next_segment_offset",
    "raw_data_offset",
)


def parse_time_stamp(fractions, seconds):
    """
    Convert time TDMS time representation to datetime
    fractions   -- fractional seconds (2^-64)
    seconds     -- The number of seconds since 1/1/1904
    @rtype : datetime.datetime.
    """
    if fractions is not None and seconds is not None and fractions + seconds > 0:
        return datetime.timedelta(0, fractions * 2**-64 + seconds) + datetime.datetime(
            1904, 1, 1
        )
    else:
        return None


def _get_version_str(tdms_file, lead_in_length=28) -> str:
    """Return True if we have the right file type."""
    lead_in = tdms_file.read(lead_in_length)
    # lead_in is 28 bytes:
    # [string of length 4][int32][int32][int64][int64]
    fields = struct.unpack("<4siiQQ", lead_in)
    # TODO: validate file
    if fields[0].decode() in "TDSm":
        version_str = str(fields[2])
        return version_str
    else:
        return False


def _get_time_coord(attrs, num_samps):
    """Get the time array for the file."""
    dt = to_timedelta64(1 / attrs["SamplingFrequency[Hz]"])
    t_min = to_datetime64(str(attrs["GPSTimeStamp"]))
    # Note: Previously this was:
    # out["time_min"] + np.timedelta64(
    t_max = t_min + dt * (num_samps - 1)
    coord = get_coord(start=t_min, stop=t_max + dt, step=dt, units="s")
    return coord


def _get_default_attrs(tdms_file, attrs=None):
    """Return the required/default attributes which can be fetched from attributes."""
    all_attrs = attrs if attrs is not None else _get_all_attrs(tdms_file)[0]
    # cull attributes to only include defaults (TODO: think about why?)
    out = {
        default_attr: all_attrs[default_attr]
        for default_attr in DEFAULT_ATTRS
        if default_attr in all_attrs
    }
    return out


def _read_attr(tdms_file):
    """
    Read a single property from the TDMS file.
    Return the name, type and value of the property as a list.
    """
    # Read length of object path:
    var = struct.unpack("<i", tdms_file.read(4))[0]
    # Read property name and type:
    name, data_type = struct.unpack(f"<{var}si", tdms_file.read(var + 4))
    # Lookup function to read and parse property value based on type:
    value = TDS_READ_VAL[TDS_DATA_TYPE[data_type]](tdms_file)
    name = name.decode()
    if data_type == 32:
        value = value.decode()

    return name, value  # data_type, value


def _get_distance_coord(attr):
    """Get distance coordinate from attribute."""
    # Note: some TDMS files actually have "Start Distance (m)" and
    # "Stop Distance (m)" fields, but not all. These also don't really
    # match the distance calculated below. We need to figure out why and what
    # is the correct way to do this, but this seems safe for now.
    multiplier = attr["Fibre Length Multiplier"]
    total_length = attr["MeasureLength[m]"] * multiplier
    start = attr["StartPosition[m]"]
    step = attr["SpatialResolution[m]"] * multiplier
    stop = start + total_length
    d_coord = get_coord(start=start, stop=stop, step=step, units="m")
    return d_coord


def _get_all_attrs(tdms_file, lead_in_length=28):
    """Return all the attributes which can be fetched from attributes."""
    # read leadin infomation into fileinfo
    lead_in = tdms_file.read(lead_in_length)
    # lead_in is 28 bytes:
    fields = struct.unpack("<4siiQQ", lead_in)
    # Keep track of information about file in fileinfo
    fileinfo = dict(zip(FILEINFO_NAMES, fields))
    fileinfo["decimated"] = not bool(fileinfo["toc"] & DECIMATE_MASK)
    # Make offsets relative to beginning of file:
    fileinfo["next_segment_offset"] += lead_in_length
    fileinfo["raw_data_offset"] += lead_in_length
    fileinfo["file_size"] = os.path.getsize(tdms_file.name)
    # Make sure next segment does not go beyond file capacity
    if fileinfo["next_segment_offset"] > fileinfo["file_size"]:
        fileinfo["next_segment_offset"] = fileinfo["file_size"]
    # navigate pointer to immediately after lead in data
    tdms_file.seek(lead_in_length, 0)
    # Read number of channels
    n_channels = struct.unpack("i", tdms_file.read(4))[0] - 2
    fileinfo["n_channels"] = n_channels
    # Read length of object path:
    var = struct.unpack("<i", tdms_file.read(4))[0]
    # skip over object path and raw data index:
    tdms_file.seek(var + 4, 1)
    # Read number of properties in this group:
    var = struct.unpack("<i", tdms_file.read(4))[0]
    # loop through and read each property
    out = [_read_attr(tdms_file) for _ in range(var)]
    # Returns a pandas dataframe that we convert to dictionary
    out = dict(out)
    # Add other attributes not yet included
    out["n_channels"] = n_channels
    out["data_type"] = "strain_rate"
    out["data_units"] = ""
    out["dims"] = "time,distance"
    out["instrument_id"] = out.get("SystemInfomation.Devices0.SerialNum")
    # Rename some attributes to preferred names
    d_coord = _get_distance_coord(out)
    fileinfo["end_of_properties_offset"] = tdms_file.tell()
    tdms_file.seek(fileinfo["end_of_properties_offset"], 0)
    # skip over Group Information:
    var = struct.unpack("<i", tdms_file.read(4))[0]
    tdms_file.seek(var + 8, 1)
    # skip over first channel path and length of index information:
    var = struct.unpack("<i", tdms_file.read(4))[0]
    tdms_file.seek(var + 4, 1)
    fileinfo["data_type"] = TDS_DATA_TYPE.get(struct.unpack("<i", tdms_file.read(4))[0])
    if fileinfo["data_type"] not in ("int16", "float32"):
        raise Exception("Unsupported TDMS data type: " + fileinfo["data_type"])
    # get number of samples by dividing amount of unread data by the
    # size of data per channel
    numofsamples = (
        (fileinfo["file_size"] - fileinfo["raw_data_offset"])
        / n_channels
        / np.dtype(fileinfo["data_type"]).itemsize
    )
    t_coord = _get_time_coord(out, numofsamples)
    out["coords"] = {"time": t_coord, "distance": d_coord}
    out.update(t_coord.get_attrs_dict("time"))
    out.update(d_coord.get_attrs_dict("distance"))
    return out, fileinfo


def _get_fileinfo(tdms_file, lead_in_length=28):
    """Get info about file not included in the attributes."""
    attrs, fileinfo = _get_all_attrs(tdms_file)
    # Read Dimension of the raw data array (has to be 1):
    _ = struct.unpack("<i", tdms_file.read(4))[0]
    fileinfo["chunk_size"] = struct.unpack("<i", tdms_file.read(4))[0]
    return fileinfo, attrs


def _get_data(tdms_file, lead_in_length=28):
    """Get all the data saved in the current file."""

    def get_segment_data(fileinfo, nch, dmap, nso, rdo):
        # seg1_length: length of recording indicated as raw_data in metadata for
        # each channel in bytes
        seg_length = int((nso - rdo) / nch / np.dtype(fileinfo["data_type"]).itemsize)
        channel_length = seg_length

        if fileinfo["decimated"]:
            # number of completely full chunks
            n_complete_blk = int(seg_length / fileinfo["chunk_size"])
            ax_ord = "C"
        else:
            n_complete_blk = 0
            ax_ord = "F"
        # use data from mapped file to fill variable raw_data
        raw_data = np.ndarray(
            (n_complete_blk, nch, fileinfo["chunk_size"]),
            dtype=fileinfo["data_type"],
            buffer=dmap,
            offset=rdo,
        )
        # Rotate the axes to [chunk_size, nblk, nch]
        raw_data = np.rollaxis(raw_data, 2)
        data_node = np.reshape(raw_data, (n_complete_blk * fileinfo["chunk_size"], nch))
        if n_complete_blk != seg_length / fileinfo["chunk_size"]:
            # If the last chunk isn't full there is some data left
            additional_samples = int(
                seg_length - n_complete_blk * fileinfo["chunk_size"]
            )
            additional_samples_offset = (
                rdo
                + n_complete_blk
                * nch
                * fileinfo["chunk_size"]
                * np.dtype(fileinfo["data_type"]).itemsize
            )
            raw_last_chunk = np.ndarray(
                (nch, additional_samples),
                dtype=fileinfo["data_type"],
                buffer=dmap,
                offset=additional_samples_offset,
                order=ax_ord,
            )
            # Rotate the axes to [samples, nch]
            raw_last_chunk = np.rollaxis(raw_last_chunk, 1)
            data_node = np.append(data_node, raw_last_chunk, axis=0)
            return data_node, channel_length

    fileinfo, attrs = _get_fileinfo(tdms_file)

    # map file contents to a variable dmap
    dmap = mmap.mmap(tdms_file.fileno(), 0, access=mmap.ACCESS_READ)
    # rdo: the start of the data in the file
    rdo = int(fileinfo["raw_data_offset"])
    # nch: number of channels
    nch = int(fileinfo["n_channels"])

    # nso: the beginning of the segment that comes next after the data
    nso = fileinfo["next_segment_offset"]
    # seg1_length: length of recording indicated as raw_data in metadata for
    # each channel in bytes

    flag = 0
    while flag == 0:
        cdata_node, cchannel_length = get_segment_data(fileinfo, nch, dmap, nso, rdo)
        if fileinfo["file_size"] == nso:
            flag = 1
        else:
            tdms_file.seek(nso + 12, 0)
            (next_seg_nso, next_seg_rdo) = struct.unpack("<qq", tdms_file.read(2 * 8))
            rdo = min(fileinfo["file_size"], nso + lead_in_length + next_seg_rdo)
            nso = min(fileinfo["file_size"], nso + lead_in_length + next_seg_nso)
        try:
            data_node = np.append(data_node, cdata_node, axis=1)
            channel_length += cchannel_length
        except NameError:
            data_node = cdata_node
            channel_length = cchannel_length

    return data_node, channel_length, attrs
