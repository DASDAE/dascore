"""Tests for the AP Sensing format."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dascore.io.ap_sensing.utils import _get_version_string


class TestDetection:
    """The format needs both its root attrs and its groups."""

    @pytest.mark.parametrize("with_attrs", [True, False])
    def test_partial_layout_is_not_ap_sensing(self, tmp_path, with_attrs):
        """
        Half a layout is not the format; detection used to claim a file
        unless both halves were missing.
        """
        path = tmp_path / "partial.h5"
        with h5py.File(path, "w") as h5:
            if with_attrs:
                h5.attrs["AppVersion"] = "1.0"
                h5.attrs["FileVersion"] = "10"
                h5.attrs["OpticalChannelNumber"] = 1
            else:
                for name in ("DAQ", "Interrogator", "Metadata", "ProcessingServer"):
                    h5.create_group(name)
            h5["DAS"] = np.zeros((2, 2))
        with h5py.File(path, "r") as h5:
            assert _get_version_string(h5) is False
