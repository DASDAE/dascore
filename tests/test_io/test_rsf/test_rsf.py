"""Tests for RSF format."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

import dascore as dc
from dascore.io.rsf import RSFV1
from dascore.warnings import DASCoreWarning

SEPARATOR = b"\x0c\x0c\x04"


def _parse_header(text):
    """Return the last binding of each key=value line, quotes removed."""
    out = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip('"')
    return out


def _split_combined(raw):
    """Split a combined RSF file into its header dict, header text, and binary."""
    head, sep, binary = raw.partition(SEPARATOR)
    assert sep, "combined RSF output has no header/binary separator"
    return _parse_header(head.decode()), head, binary


@pytest.fixture(scope="module")
def fractional_patch():
    """A patch with a unitless distance and a 1024 Hz time axis."""
    start = np.datetime64("2024-01-01T00:00:00.123456789")
    time = dc.get_coord(start=start, step=(1, 1024), shape=(8,))
    data = np.arange(24, dtype=np.float32).reshape(3, 8)
    coords = {"distance": np.arange(3), "time": time}
    return dc.Patch(data=data, coords=coords, dims=("distance", "time"))


class TestRsfWrite:
    """Tests for the header and binary the RSF writer produces."""

    def test_combined(self, random_patch, tmp_path):
        """Combined output is header, blank line, separator, then the samples."""
        path = tmp_path / "test_hdrdata.rsf"
        with pytest.warns(DASCoreWarning, match="float32"):
            RSFV1().write(dc.spool(random_patch), path)
        hdr, head, binary = _split_combined(path.read_bytes())
        assert head.endswith(b'in="stdin"\n\n')
        assert binary == random_patch.data.astype(np.float32).tobytes("F")
        assert hdr["data_format"] == "native_float"
        assert hdr["esize"] == "4"
        assert hdr["in"] == "stdin"
        for i, dim in enumerate(random_patch.dims, start=1):
            assert hdr[f"n{i}"] == str(random_patch.shape[i - 1])
            assert hdr[f"label{i}"] == dim
        assert (hdr["d1"], hdr["unit1"]) == ("1.0", "m")
        assert (hdr["o2"], hdr["d2"], hdr["unit2"]) == ("0.0", "0.004", "s")

    def test_split(self, random_patch, tmp_path):
        """Split output names its binary by absolute path and ends in a newline."""
        path = tmp_path / "test_hdr.rsf"
        datapath = tmp_path / "binary/test_data.rsf"
        with pytest.warns(DASCoreWarning, match="float32"):
            RSFV1().write(dc.spool(random_patch), path, data_path=datapath)
        text = path.read_text()
        assert text.endswith("\n")
        assert SEPARATOR.decode() not in text
        binary_path = Path(_parse_header(text)["in"])
        assert binary_path.is_absolute()
        assert binary_path == Path(f"{datapath}@").resolve()
        expected = random_patch.data.astype(np.float32).tobytes("F")
        assert binary_path.read_bytes() == expected

    def test_split_relative(self, random_patch, tmp_path, monkeypatch):
        """A relative data_path is written as an absolute in= path."""
        monkeypatch.chdir(tmp_path)
        patch = random_patch.new(data=random_patch.data.astype(np.float32))
        RSFV1().write(patch, "out/hdr.rsf", data_path="bin/data.rsf")
        hdr = _parse_header((tmp_path / "out/hdr.rsf").read_text())
        assert hdr["in"] == str((tmp_path / "bin/data.rsf@").resolve())

    def test_unitless_dim(self, fractional_patch, tmp_path):
        """A dimension without units is written without a unit key."""
        path = tmp_path / "unitless.rsf"
        RSFV1().write(fractional_patch, path)
        hdr, _, _ = _split_combined(path.read_bytes())
        assert hdr["label1"] == "distance"
        assert "unit1" not in hdr
        assert hdr["unit2"] == "s"

    def test_exact_step(self, fractional_patch, tmp_path):
        """A fractional step is written exactly, not rounded to whole ns."""
        path = tmp_path / "exact.rsf"
        RSFV1().write(fractional_patch, path)
        hdr, _, binary = _split_combined(path.read_bytes())
        assert hdr["d2"] == "0.0009765625"
        assert float(hdr["d2"]) == 1 / 1024
        assert binary == fractional_patch.data.tobytes("F")

    def test_float32_no_warning(self, fractional_patch, tmp_path):
        """Float32 data are written as they are, without a warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RSFV1().write(fractional_patch, tmp_path / "f4.rsf")

    def test_int_warns(self, random_patch, tmp_path):
        """Integer data are cast to float32 with a warning."""
        data = np.full(random_patch.shape, 2**24 + 1, dtype=np.int32)
        path = tmp_path / "int.rsf"
        with pytest.warns(DASCoreWarning, match="int32"):
            RSFV1().write(random_patch.new(data=data), path)
        _, _, binary = _split_combined(path.read_bytes())
        assert binary == data.astype(np.float32).tobytes("F")

    def test_write_cmplx(self, random_patch, tmp_path):
        """Complex data are refused."""
        complex_patch = random_patch.dft("time")
        path = tmp_path / "test_hdrcmplx.rsf"
        with pytest.raises(ValueError, match="not integer or floating"):
            RSFV1().write(dc.spool(complex_patch), path)

    def test_preferred_extensions(self):
        """The writer registers the rsf extension, not its letters."""
        assert tuple(RSFV1.preferred_extensions) == ("rsf",)

    def test_write_remote_memory_combined(self, random_patch):
        """Combined RSF output should support remote UPath destinations."""
        path = UPath("memory://dascore/test_hdrdata_remote.rsf")
        with pytest.warns(DASCoreWarning, match="float32"):
            RSFV1().write(dc.spool(random_patch), path)
        _, _, binary = _split_combined(path.read_bytes())
        assert binary == random_patch.data.astype(np.float32).tobytes("F")

    def test_write_remote_memory_split(self, random_patch):
        """Split RSF output should support remote UPath destinations."""
        path = UPath("memory://dascore/test_hdr_remote.rsf")
        datapath = UPath("memory://dascore/binary/test_data_remote.rsf")
        with pytest.warns(DASCoreWarning, match="float32"):
            RSFV1().write(dc.spool(random_patch), path, data_path=datapath)
        assert _parse_header(path.read_text())["in"] == f"{datapath}@"
        newdatapath = UPath(f"{datapath}@")
        expected = random_patch.data.astype(np.float32).tobytes("F")
        assert newdatapath.read_bytes() == expected


class TestRsfEpoch:
    """Tests for the absolute time the writer stores on a datetime time axis."""

    def test_header_text(self, random_patch, tmp_path):
        """The time axis carries epoch, relative o and d, unit s, and starttime."""
        path = tmp_path / "epoch.rsf"
        patch = random_patch.new(data=random_patch.data.astype(np.float32))
        RSFV1().write(patch, path)
        _, head, _ = _split_combined(path.read_bytes())
        lines = head.decode().splitlines()
        start = lines.index("n2=2000")
        assert lines[start : start + 7] == [
            "n2=2000",
            "o2=0.0",
            'epoch2="2017-09-18T00:00:00.000000000Z"',
            "starttime=1505692800.0",
            "d2=0.004",
            'label2="time"',
            'unit2="s"',
        ]
        assert not any(x.startswith("epoch1") for x in lines)

    def test_nanoseconds(self, fractional_patch, tmp_path):
        """The epoch keeps the start time to the nanosecond."""
        path = tmp_path / "epoch_ns.rsf"
        RSFV1().write(fractional_patch, path)
        hdr, _, _ = _split_combined(path.read_bytes())
        assert hdr["epoch2"] == "2024-01-01T00:00:00.123456789Z"
        start = fractional_patch.get_coord("time").min()
        assert np.datetime64(hdr["epoch2"][:-1], "ns") == start
