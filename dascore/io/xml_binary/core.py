"""IO module for reading binary raw format DAS data."""
from __future__ import annotations

import dascore as dc
from dascore.constants import timeable_types
from dascore.io import FiberIO

# from .utils import read_xml_binary


# class BinaryPatchAttrs(dc.PatchAttrs):
#     """Patch attrs for Binary."""

#     pulse_width_ns: float = np.NAN
#     units: UnitQuantity | None = None
#     data_type: UTF8Str = ""
#     gauge_length: float = np.NaN
#     iu_id: UTF8Str = ""
#     n_lasers: int = np.NaN
#     itu_channels_laser_1: UTF8Str = ""
#     itu_channels_laser_2: UTF8Str = ""
#     file_format: UTF8Str = ""
#     transposed_data: bool = False
#     use_relative_strain: bool = False
#     original_time_step: float = np.NaN


class XMLBinary(FiberIO):
    """Support for binary data format with xml metadata."""

    name = "XMLBinary"
    preferred_extensions = "raw"

    def read(
        self,
        path,
        time: tuple[timeable_types, timeable_types] | None = None,
        distance: tuple[float, float] | None = None,
        snap_dims: bool = True,
        **kwargs,
    ) -> dc.BaseSpool:
        """
        Read a binary file.

        Parameters
        ----------
        path
            The path to the file.
        time
            A tuple for filtering time.
        distance
            A tuple for filtering distance.
        snap_dims
            If True, ensure the coordinates are evenly sampled monotonic.
            This will cause some loss in precision but it is usually
            negligible.
        """
        # patch = read_xml_binary(
        #     path,
        #     time=time,
        #     distance=distance,
        # )
        # return dc.spool(patch)
