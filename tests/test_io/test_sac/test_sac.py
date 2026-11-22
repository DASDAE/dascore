"""Tests for SAC binary files, written with numpy."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError, UnknownFiberFormatError
from dascore.io.sac import SacV6, SacV7
from dascore.utils.downloader import fetch, get_registry_df
from tests.test_io._common_io_test_utils import skip_timeout

_START = np.datetime64("2024-02-03T04:05:06.789", "ns")
# Positions of the fields `write_sac` sets by name.
_FLOATS = {"delta": 0, "b": 5}
_INTS = {
    "nzyear": 0,
    "nzjday": 1,
    "nzhour": 2,
    "nzmin": 3,
    "nzsec": 4,
    "nzmsec": 5,
    "nvhdr": 6,
    "npts": 9,
    "iftype": 15,
    "leven": 35,
}
_STRINGS = {"kstnm": 0, "khole": 3, "kcmpnm": 20, "knetwk": 21}
_DEFAULTS = {
    "delta": 0.01,
    "b": 0.0,
    "nzyear": 2024,
    "nzjday": 34,
    "nzhour": 4,
    "nzmin": 5,
    "nzsec": 6,
    "nzmsec": 789,
    "iftype": 1,
    "leven": 1,
    "knetwk": "XX",
    "kstnm": "STA",
    "khole": "00",
    "kcmpnm": "HHZ",
}


def write_sac(path, data=10, *, byte_order="<", version=6, footer=None, **fields):
    """
    Write a SAC binary file, every field undefined unless set; return its path.

    `data` is the samples or their count, and `fields` override header values
    by name, None leaving one undefined. Version 7 writes a footer of DELTA
    and B, overridden by `footer`, then undefined doubles.
    """
    data = np.arange(data) if np.isscalar(data) else np.asarray(data)
    values = {**_DEFAULTS, "nvhdr": version, "npts": len(data), **fields}
    floats = np.full(70, -12345.0, f"{byte_order}f4")
    ints = np.full(40, -12345, f"{byte_order}i4")
    strings = [b"-12345  "] * 24
    for name, value in values.items():
        if value is None:
            continue
        if name in _FLOATS:
            floats[_FLOATS[name]] = value
        elif name in _INTS:
            ints[_INTS[name]] = value
        else:
            strings[_STRINGS[name]] = value.encode().ljust(8)
    with open(path, "wb") as fi:
        fi.write(floats.tobytes() + ints.tobytes() + b"".join(strings))
        fi.write(data.astype(f"{byte_order}f4").tobytes())
        if version == 7:
            doubles = np.full(22, -12345.0, f"{byte_order}f8")
            doubles[:2] = footer or (values["delta"], values["b"])
            fi.write(doubles.tobytes())
    return path


@pytest.fixture
def sac_path(tmp_path):
    """A little-endian version 6 file of ten samples."""
    return write_sac(tmp_path / "trace.sac")


class TestGetFormat:
    """SAC files are found by a header check that refuses other files."""

    @pytest.mark.parametrize("byte_order", ["<", ">"])
    @pytest.mark.parametrize("version", [6, 7])
    def test_byte_orders_and_versions(self, tmp_path, byte_order, version):
        """Either byte order of either version is detected as its NVHDR."""
        path = write_sac(tmp_path / "x.sac", byte_order=byte_order, version=version)
        assert dc.get_format(path) == ("SAC", str(version))

    @pytest.mark.parametrize(
        "fields",
        [{"nvhdr": 5}, {"delta": 0.0}, {"delta": None}, {"delta": np.nan}],
    )
    def test_implausible_header(self, tmp_path, fields):
        """An unknown NVHDR, or a DELTA that is not positive, is not SAC."""
        path = write_sac(tmp_path / "x.sac", **fields)
        assert SacV6().get_format(path) is False

    @pytest.mark.parametrize("version", [6, 7])
    @pytest.mark.parametrize("npts", [9, 11, 5, -1])
    def test_size_disagrees_with_npts(self, tmp_path, npts, version):
        """A time series must be the size its NPTS samples make, no more or less."""
        path = write_sac(tmp_path / "x.sac", npts=npts, version=version)
        assert SacV6().get_format(path) is False

    def test_short_file(self, tmp_path):
        """A file shorter than the header is not SAC."""
        path = tmp_path / "short.sac"
        path.write_bytes(write_sac(tmp_path / "x.sac").read_bytes()[:631])
        assert SacV6().get_format(path) is False

    def test_alphanumeric(self, tmp_path):
        """An alphanumeric SAC file is refused cleanly."""
        obspy = pytest.importorskip("obspy")
        path = tmp_path / "alpha.sac"
        obspy.Trace(np.arange(10.0)).write(str(path), format="SACXY")
        assert SacV6().get_format(path) is False
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(path)

    @pytest.mark.parametrize("name", get_registry_df()["name"])
    def test_registry_files(self, name):
        """No other format's file in the data registry is SAC."""
        with skip_timeout():
            path = fetch(name)
        assert SacV6().get_format(path) is False


class TestRead:
    """A SAC trace reads as a one-channel patch."""

    def test_mapping(self, sac_path):
        """The trace's codes are per-channel coordinates and the tag."""
        (patch,) = dc.read(sac_path)
        assert patch.dims == ("channel", "time")
        assert patch.dtype == np.float32
        np.testing.assert_array_equal(patch.data, np.arange(10.0)[None])
        assert patch.get_coord("channel").values.tolist() == [0]
        coords = {
            name: patch.coords.get_array(name).tolist()
            for name in ("source_id", "network", "station", "location", "seed_channel")
        }
        assert coords == {
            "source_id": ["FDSN:XX_STA_00_H_H_Z"],
            "network": ["XX"],
            "station": ["STA"],
            "location": ["00"],
            "seed_channel": ["HHZ"],
        }
        assert patch.attrs.tag == "XX.STA.00.HHZ"
        time = patch.get_coord("time")
        assert time.min() == _START
        assert time.step == np.timedelta64(10, "ms")

    @pytest.mark.parametrize("byte_order", ["<", ">"])
    @pytest.mark.parametrize("version", [6, 7])
    def test_byte_orders_and_versions(self, tmp_path, byte_order, version):
        """Every layout reads the same native float32 samples and times."""
        data = np.linspace(-1, 1, 7)
        path = write_sac(
            tmp_path / "x.sac", data, byte_order=byte_order, version=version
        )
        expected = dc.read(write_sac(tmp_path / "le.sac", data))[0]
        (patch,) = dc.read(path)
        assert patch.data.dtype == np.dtype("=f4")
        assert patch == expected

    def test_decimal_steps_are_exact(self, tmp_path):
        """A float32 DELTA of 0.1 is 1/10 s, so labels do not drift."""
        path = write_sac(tmp_path / "x.sac", 1001, delta=0.1)
        time = dc.read(path)[0].get_coord("time")
        assert time.max() - time.min() == np.timedelta64(100, "s")

    def test_footer_times(self, tmp_path):
        """Version 7 reads DELTA and B from its footer's doubles."""
        footer = (1 / 3, 3600.123456789)
        path = write_sac(tmp_path / "x.sac", 4, version=7, footer=footer)
        time = dc.read(path)[0].get_coord("time")
        assert time.min() == _START + np.timedelta64(3600_123_456_789, "ns")
        assert time.max() - time.min() == np.timedelta64(1, "s")

    def test_undefined_codes(self, tmp_path):
        """Undefined strings are empty codes, B undefined is zero."""
        names = ("knetwk", "kstnm", "khole", "kcmpnm", "b")
        path = write_sac(tmp_path / "x.sac", **dict.fromkeys(names))
        (patch,) = dc.read(path)
        assert patch.coords.get_array("source_id").tolist() == ["FDSN:___"]
        assert patch.coords.get_array("station").tolist() == [""]
        assert patch.attrs.tag == "..."
        assert patch.get_coord("time").min() == _START

    def test_undefined_reference(self, tmp_path):
        """Without a reference time, time is an offset of B."""
        path = write_sac(tmp_path / "x.sac", b=-1.5, nzmsec=None)
        time = dc.read(path)[0].get_coord("time")
        assert time.min() == np.timedelta64(-1500, "ms")

    def test_year_out_of_range(self, tmp_path):
        """A reference year nanosecond times cannot hold raises."""
        path = write_sac(tmp_path / "x.sac", nzyear=3000)
        with pytest.raises(InvalidFiberFileError, match="3000"):
            dc.read(path)

    @pytest.mark.parametrize(
        "fields", [{"iftype": 2}, {"iftype": 4, "leven": 0}, {"leven": 0}]
    )
    def test_not_an_even_time_series(self, tmp_path, fields):
        """Spectral, x-y and unevenly sampled files are SAC, refused on read."""
        path = write_sac(tmp_path / "x.sac", 20, npts=10, **fields)
        assert dc.get_format(path) == ("SAC", "6")
        with pytest.raises(InvalidFiberFileError, match="IFTYPE"):
            dc.read(path)

    def test_bad_footer_delta(self, tmp_path):
        """A version 7 footer's DELTA must be positive too."""
        path = write_sac(tmp_path / "x.sac", version=7, footer=(0, 0))
        with pytest.raises(InvalidFiberFileError, match="DELTA"):
            dc.read(path)

    def test_no_samples(self, tmp_path):
        """A trace of no samples is no patch."""
        path = write_sac(tmp_path / "x.sac", 0)
        assert not len(dc.read(path))
        assert not len(dc.scan(path))

    def test_not_sac(self, tmp_path):
        """Reading another file as SAC raises."""
        path = tmp_path / "x.sac"
        path.write_bytes(bytes(700))
        with pytest.raises(InvalidFiberFileError, match="not a SAC"):
            SacV7().read(path)

    @pytest.mark.parametrize("channel", ["HHZ", "Z", "HHZ1", None])
    def test_source_id_matches_miniseed(self, tmp_path, channel):
        """The source ID is the one the MiniSEED library builds."""
        pymseed = pytest.importorskip("pymseed")
        path = write_sac(tmp_path / "x.sac", khole=None, kcmpnm=channel)
        (patch,) = dc.read(path)
        expected = pymseed.nslc2sourceid("XX", "STA", "", channel or "")
        assert patch.coords.get_array("source_id").tolist() == [expected]


class TestTimeSelection:
    """Reads select time inclusively, between samples or on them."""

    @pytest.fixture
    def times(self, sac_path):
        """The file's time labels."""
        return dc.read(sac_path)[0].get_coord("time").values

    def test_on_samples(self, sac_path, times):
        """Bounds on samples keep both."""
        (patch,) = dc.read(sac_path, time=(times[2], times[5]))
        np.testing.assert_array_equal(patch.get_coord("time").values, times[2:6])
        np.testing.assert_array_equal(patch.data[0], np.arange(2, 6))

    def test_between_samples(self, sac_path, times):
        """Bounds between samples keep the samples inside."""
        half = np.timedelta64(5, "ms")
        (patch,) = dc.read(sac_path, time=(times[2] + half, times[5] + half))
        np.testing.assert_array_equal(patch.data[0], np.arange(3, 6))

    def test_outside(self, sac_path, times):
        """Bounds past the trace select nothing."""
        later = times[-1] + np.timedelta64(1, "s")
        assert not len(dc.read(sac_path, time=(later, None)))


class TestScan:
    """Scans describe what reads return, from the header alone."""

    @pytest.mark.parametrize("version", [6, 7])
    def test_scan_matches_read(self, tmp_path, version):
        """The scan's coordinates and attrs are the read patch's."""
        path = write_sac(tmp_path / "x.sac", 100, byte_order=">", version=version)
        (summary,) = dc.scan(path)
        (patch,) = dc.read(path)
        assert summary.coords == patch.summary.coords
        assert summary.attrs.tag == patch.attrs.tag
        assert summary.source_version == str(version)


class TestDirectory:
    """A directory of one trace per file, as SAC archives are kept."""

    @pytest.fixture
    def spool(self, tmp_path):
        """Stations A and B for two seconds, one file a second, and D later."""
        files = [("A", 0), ("B", 0), ("A", 1), ("B", 1), ("D", 5)]
        for name, seconds in files:
            path = tmp_path / f"{name}_{seconds}.sac"
            write_sac(path, 100, kstnm=name, nzsec=6 + seconds)
        return dc.spool(tmp_path).update()

    def test_chunk_keeps_traces_apart(self, spool):
        """Chunking along time joins a trace's files and never mixes traces."""
        out = spool.chunk(time=None)
        stations = sorted((x.coords.get_array("station")[0], x.shape[1]) for x in out)
        assert stations == [("A", 200), ("B", 200), ("D", 100)]
        assert len(spool.chunk(channel=None)) == len(spool)

    def test_join_channels(self, spool):
        """The docstring's recipe joins traces along channel where times agree."""
        patches = list(spool)
        sources = sorted({x.coords.get_array("source_id")[0] for x in patches})

        def numbered(patch):
            channel = sources.index(patch.coords.get_array("source_id")[0])
            return patch.replace(coords=patch.coords.update(channel=[channel]))

        joined = dc.spool([numbered(x) for x in patches]).concatenate(
            channel=None, group=(), conflict="drop"
        )
        out = sorted(joined.chunk(time=None), key=lambda x: x.shape[0])
        assert [x.coords.get_array("station").tolist() for x in out] == [
            ["D"],
            ["A", "B"],
        ]
        assert [x.get_coord("channel").values.tolist() for x in out] == [[2], [0, 1]]
        assert out[1].shape == (2, 200)


class TestObspy:
    """Files ObsPy writes read as ObsPy reads them."""

    @pytest.mark.parametrize("byte_order", ["<", ">"])
    def test_obspy_written(self, tmp_path, byte_order):
        """Start, step, codes and samples agree with ObsPy's reading."""
        obspy = pytest.importorskip("obspy")
        stats = {
            "network": "XX",
            "station": "STA",
            "channel": "HHZ",
            "sampling_rate": 40.0,
            "starttime": obspy.UTCDateTime("2024-01-02T03:04:05.678901"),
        }
        path = tmp_path / "obspy.sac"
        trace = obspy.Trace(np.arange(50, dtype=np.float32), header=stats)
        trace.write(str(path), format="SAC", byteorder=byte_order)
        (expected,) = obspy.read(str(path))
        (patch,) = dc.read(path)
        time = patch.get_coord("time")
        assert time.min() == np.datetime64(expected.stats.starttime.ns, "ns")
        assert time.step == np.timedelta64(25, "ms")
        assert patch.coords.get_array("seed_channel").tolist() == ["HHZ"]
        np.testing.assert_array_equal(patch.data[0], expected.data)
