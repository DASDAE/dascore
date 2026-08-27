"""The interrogator a Silixa file names, and the components it does not."""

import shutil

import h5py
import pytest

import dascore as dc
from dascore.utils.downloader import fetch

# Serials the file states for parts inside the unit rather than for the
# unit: an NI timing card, a GPIB card, an ADLINK PXI crate. None is the
# interrogator's, so none may be read as one.
_COMPONENT_SERIAL_ATTRS = (
    "SystemInfomation.Chassis.SerialNum",
    "SystemInfomation.Devices0.SerialNum",
    "SystemInfomation.Devices1.SerialNum",
    "SystemInfomation.ProcessingUnit.FPGA1.SerialNum",
)


@pytest.fixture(scope="module", params=["silixa_h5_1.hdf5", "silixa_h5_ingv_1.h5"])
def silixa_path(request):
    """Paths to the Acoustic and Carina variants."""
    return fetch(request.param)


def _host_name(path):
    """Read the host name from wherever the variant keeps its attrs."""
    with h5py.File(path, "r") as f:
        attrs = f["Acoustic"].attrs if "Acoustic" in f else f.attrs
        value = attrs["SystemInfomation.OS.HostName"]
    return value.decode() if isinstance(value, bytes) else str(value)


class TestSilixaInterrogator:
    """Both Silixa variants name the interrogator by its host name."""

    def test_name_is_host_name(self, silixa_path):
        """HostName is the only field naming the unit itself."""
        attrs = dict(dc.scan(silixa_path)[0].attrs)
        assert attrs["interrogator.name"] == _host_name(silixa_path)

    def test_no_component_serial_as_interrogator(self, silixa_path):
        """No card or crate serial is passed off as the interrogator's."""
        with h5py.File(silixa_path, "r") as f:
            raw = f["Acoustic"].attrs if "Acoustic" in f else f.attrs
            serials = {
                v.decode() if isinstance(v, bytes) else str(v)
                for k in _COMPONENT_SERIAL_ATTRS
                if (v := raw.get(k)) is not None
            }
        attrs = dict(dc.scan(silixa_path)[0].attrs)
        stated = {v for k, v in attrs.items() if k.startswith("interrogator.")}
        assert not (stated & (serials - {""}))

    def test_no_serial_claimed(self, silixa_path):
        """These files state no interrogator serial, so the reader sets none."""
        attrs = dict(dc.scan(silixa_path)[0].attrs)
        assert "interrogator.serial_number" not in attrs

    def test_scan_and_read_agree(self, silixa_path):
        """A read states the same interrogator a scan does."""
        scanned = dict(dc.scan(silixa_path)[0].attrs)
        patch = dc.spool(silixa_path)[0]
        assert dict(patch.attrs)["interrogator.name"] == scanned["interrogator.name"]

    def test_detection_survives_missing_host_name(self, silixa_path, tmp_path):
        """A file which omits HostName is still claimed, just unnamed."""
        path = tmp_path / "no_host.h5"
        shutil.copy2(silixa_path, path)
        with h5py.File(path, "r+") as f:
            node = f["Acoustic"] if "Acoustic" in f else f
            del node.attrs["SystemInfomation.OS.HostName"]
        assert dc.get_format(path)[0] == "Silixa_H5"
        assert "interrogator.name" not in dict(dc.scan(path)[0].attrs)

    def test_blank_host_name_dropped(self, silixa_path, tmp_path):
        """An empty HostName is not passed off as a name."""
        path = tmp_path / "blank_host.h5"
        shutil.copy2(silixa_path, path)
        with h5py.File(path, "r+") as f:
            is_acoustic = "Acoustic" in f
            node = f["Acoustic"] if is_acoustic else f
            node.attrs["SystemInfomation.OS.HostName"] = b""
        assert "interrogator.name" not in dict(dc.scan(path)[0].attrs)
