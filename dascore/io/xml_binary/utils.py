"""Utilities for Binary."""
from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from functools import lru_cache


@lru_cache
def read_xml_metadata(path):
    """A function to read metadata from the xml file."""
    tree = ElementTree.parse(path)
    root = tree.getroot()
    # Key metadata
    total_n_chans = int(root.find("NumberOfChannels").text)
    gauge_length = float(root.find("GaugeLengthM").text)
    distance_step = float(root.find("OriginalSpatialSamplingInterval").text)
    n_time = int(root.find("NumberOfFrames").text)
    time_step = 1 / float(root.find("OutputTemporalSamplingRate").text)
    data_type = root.find("DataType").text
    file_format = root.find("FileFormat").text
    # Secondary metadata
    iu_id = root.find("DasInterrogatorSerial/Interrogator_1").text
    time_start = root.find("DateTime").text
    itu_channels_laser_1 = root.find("ITUChannels/Laser_1").text
    itu_channels_laser_2 = root.find("ITUChannels/Laser_2").text
    n_lasers = root.find("NumberOfLasers").text
    original_time_step = 1 / float(root.find("OriginalTemporalSamplingRate").text)
    pulse_width_ns = float(root.find("PulseWidthNs").text)
    transposed_data = root.find("TransposedData").text
    units = root.find("Units").text
    use_relative_strain = root.find("UseRelativeStrain").text
    start_channel = int(root.find("Zones/Zone/StartChannel").text)
    end_channel = int(root.find("Zones/Zone/EndChannel").text)
    channel_stride = int(root.find("Zones/Zone/Stride").text)
    zone_n_chans = int(root.find("Zones/Zone/NumberOfChannels").text)

    metadata = dict(
        total_n_chans=total_n_chans,
        n_time=n_time,
        time_start=time_start,
        time_step=time_step,
        distance_step=distance_step,
        pulse_width_ns=pulse_width_ns,
        units=units,
        data_type=data_type,
        gauge_length=gauge_length,
        iu_id=iu_id,
        n_lasers=n_lasers,
        itu_channels_laser_1=itu_channels_laser_1,
        itu_channels_laser_2=itu_channels_laser_2,
        file_format=file_format,
        transposed_data=transposed_data,
        use_relative_strain=use_relative_strain,
        original_time_step=original_time_step,
        start_channel=start_channel,
        end_channel=end_channel,
        channel_stride=channel_stride,
        zone_n_chans=zone_n_chans,
    )
    return metadata


# def read_xml_binary(path, distance=None, time=None):
#     """Read the binary values into a patch."""

#     # Define metadata
#     metadata_name = 'metadata.xml'
#     data_dir = "/same/as/spool/"

#     metadata = read_xml_metadata(path_to_xml)

#     if transposed_data:
#         data = np.float64(np.fromfile(data_path, dtype=np.dtype(data_type)
#                          .reshape((n_chans, n_time)))
#         dims = ('distance', 'time')
#         axes = [1,0]
#     else:
#        data = np.float64(np.fromfile(data_path, dtype=np.dtype(data_type)
#                         .reshape((n_time, n_chans)))
#        dims = ('time', 'distance')
#        axes = [0,1]
#     # Create coordinates, labels for each axis in the array
#     time_start = dc.to_datetime64(time_start)
#     patch_index = to_timedelta64(patch_index) # need to figure out
#     time_step = to_timedelta64(time_step)
#     time = time_start + np.arange(data.shape[axes[0]]) * time_step

#     distance_start = 0
#     distance = distance_start + np.arange(data.shape[axes[1]]) * distance_step

#     # Define coordinates
#     coords = dict(time=time, distance=distance)

#     patch = dc.Patch(data=data, attrs=attrs, coords=coords)
#     return patch
