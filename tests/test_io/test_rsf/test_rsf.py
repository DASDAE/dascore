"""Tests for RSF format."""

import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

import dascore as dc
from dascore.exceptions import CoordError, InvalidFiberFileError
from dascore.io.rsf import RSFV1
from dascore.io.rsf.utils import HEADER_MAX, KINDS, SEPARATOR, read_header
from dascore.io.utils import step_from_interval
from dascore.warnings import DASCoreWarning


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


@pytest.fixture(scope="module")
def f4_patch(random_patch):
    """The random patch with float32 data."""
    return random_patch.new(data=random_patch.data.astype(np.float32))


class TestRsfWrite:
    """Tests for the header and binary the RSF writer produces."""

    @pytest.mark.parametrize(
        "get_path",
        [
            lambda tmp_path: tmp_path / "hdrdata.rsf",
            lambda _: UPath("memory://dascore/test_hdrdata_remote.rsf"),
        ],
        ids=["local", "memory"],
    )
    def test_combined(self, get_path, f4_patch, tmp_path):
        """Combined output is header, blank line, separator, then the samples."""
        path = get_path(tmp_path)
        RSFV1().write(dc.spool(f4_patch), path)
        hdr, head, binary = _split_combined(path.read_bytes())
        assert head.endswith(b'in="stdin"\n\n')
        assert binary == f4_patch.data.tobytes("F")
        assert hdr["data_format"] == "native_float"
        assert hdr["esize"] == "4"
        assert hdr["in"] == "stdin"
        for i, dim in enumerate(f4_patch.dims, start=1):
            assert hdr[f"n{i}"] == str(f4_patch.shape[i - 1])
            assert hdr[f"label{i}"] == dim
        assert (hdr["d1"], hdr["unit1"]) == ("1.0", "m")
        assert (hdr["o2"], hdr["d2"], hdr["unit2"]) == ("0.0", "0.004", "s")

    def test_split(self, f4_patch, tmp_path):
        """Split output names its binary by absolute path and ends in a newline."""
        path = tmp_path / "test_hdr.rsf"
        datapath = tmp_path / "binary/test_data.rsf"
        RSFV1().write(dc.spool(f4_patch), path, data_path=datapath)
        text = path.read_text()
        assert text.endswith("\n")
        assert SEPARATOR.decode() not in text
        binary_path = Path(_parse_header(text)["in"])
        assert binary_path.is_absolute()
        assert binary_path == Path(f"{datapath}@").resolve()
        assert binary_path.read_bytes() == f4_patch.data.tobytes("F")

    def test_split_relative(self, f4_patch, tmp_path, monkeypatch):
        """A relative data_path is written as an absolute in= path."""
        monkeypatch.chdir(tmp_path)
        RSFV1().write(f4_patch, "out/hdr.rsf", data_path="bin/data.rsf")
        hdr = _parse_header((tmp_path / "out/hdr.rsf").read_text())
        assert hdr["in"] == str((tmp_path / "bin/data.rsf@").resolve())

    def test_unitless_exact_step(self, fractional_patch, tmp_path):
        """A unitless dim has no unit key; a fractional step is written exactly."""
        path = tmp_path / "fractional.rsf"
        RSFV1().write(fractional_patch, path)
        hdr, _, binary = _split_combined(path.read_bytes())
        assert hdr["label1"] == "distance"
        assert "unit1" not in hdr
        assert hdr["unit2"] == "s"
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

    def test_warning_location(self, random_patch, tmp_path):
        """The cast warning points at the caller, not inside DASCore."""
        with pytest.warns(DASCoreWarning, match="float32") as record:
            dc.write(random_patch, tmp_path / "located.rsf", "rsf")
        assert record[0].filename == __file__

    def test_write_cmplx(self, random_patch, tmp_path):
        """Complex data are refused."""
        complex_patch = random_patch.dft("time")
        path = tmp_path / "test_hdrcmplx.rsf"
        with pytest.raises(ValueError, match="not integer or floating"):
            RSFV1().write(dc.spool(complex_patch), path)

    def test_preferred_extensions(self):
        """The writer registers the rsf extension, not its letters."""
        assert tuple(RSFV1.preferred_extensions) == ("rsf",)

    def test_write_remote_memory_split(self, f4_patch):
        """Split RSF output should support remote UPath destinations."""
        path = UPath("memory://dascore/test_hdr_remote.rsf")
        datapath = UPath("memory://dascore/binary/test_data_remote.rsf")
        RSFV1().write(dc.spool(f4_patch), path, data_path=datapath)
        assert _parse_header(path.read_text())["in"] == f"{datapath}@"
        newdatapath = UPath(f"{datapath}@")
        assert newdatapath.read_bytes() == f4_patch.data.tobytes("F")

    def test_split_quoted_path(self, fractional_patch):
        """A data_path the header cannot quote is refused before any write."""
        path = UPath("memory://dascore/test_hdr_quoted.rsf")
        datapath = UPath('memory://dascore/binary/we"ird.rsf')
        with pytest.raises(ValueError, match="quote or newline"):
            RSFV1().write(fractional_patch, path, data_path=datapath)
        assert not path.exists()
        assert not UPath(f"{datapath}@").exists()

    @pytest.mark.parametrize("split", [False, True])
    def test_header_too_long(self, split, fractional_patch, tmp_path):
        """A header the reader could not find in= in is refused before any write."""
        patch = fractional_patch.update_attrs(note="x" * HEADER_MAX)
        path, data_path = tmp_path / "long.rsf", tmp_path / "long_data.rsf"
        with pytest.raises(ValueError, match="RSF header"):
            RSFV1().write(patch, path, data_path=data_path if split else None)
        assert not path.exists()
        assert not Path(f"{data_path}@").exists()


class TestRsfEpoch:
    """Tests for the absolute time the writer stores on a datetime time axis."""

    def test_header_text(self, f4_patch, tmp_path):
        """The time axis carries epoch, relative o and d, unit s, and starttime."""
        path = tmp_path / "epoch.rsf"
        RSFV1().write(f4_patch, path)
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

    def test_relative_time(self, fractional_patch, tmp_path):
        """A time axis of offsets keeps its start in o, with no epoch, as offsets."""
        start, step = dc.to_timedelta64(5.0), dc.to_timedelta64(0.004)
        time = dc.get_coord(start=start, step=step, shape=(8,))
        path = tmp_path / "relative.rsf"
        RSFV1().write(fractional_patch.update_coords(time=time), path)
        hdr, _, _ = _split_combined(path.read_bytes())
        assert (hdr["o2"], hdr["d2"], hdr["unit2"]) == ("5.0", "0.004", "s")
        assert "epoch2" not in hdr
        assert "starttime" not in hdr
        assert dc.spool(path)[0].get_coord("time") == time

    def test_nat_start(self, fractional_patch, tmp_path):
        """A time axis whose start is NaT is refused, not written as an epoch."""
        start, step = np.datetime64("2024-01-01"), np.timedelta64(1, "s")
        time = dc.get_coord(start=start, step=step, shape=(8,))
        nat_time = time.update_limits(min=np.datetime64("NaT", "ns"))
        patch = fractional_patch.update_coords(time=nat_time)
        path = tmp_path / "nat.rsf"
        with pytest.raises(CoordError, match="finite start"):
            RSFV1().write(patch, path)
        assert not path.exists()


def _write_rsf(path, lines, data=None, data_path=None):
    """Write a hand-made RSF header, packing data unless data_path is given."""
    text = "\n".join(lines) + "\n"
    if data_path is not None:
        Path(data_path).write_bytes(data.tobytes("F"))
        path.write_text(text)
        return path
    raw = text.encode() + b"\n" + SEPARATOR
    path.write_bytes(raw if data is None else raw + data.tobytes("F"))
    return path


class TestRsfRead:
    """Tests for reading RSF files, including the writer's own output."""

    @pytest.mark.parametrize("split", [False, True])
    def test_roundtrip(self, split, f4_patch, tmp_path):
        """The writer's output reads back with the same samples and coords."""
        path, data_path = tmp_path / "a.rsf", tmp_path / "bin/data.rsf"
        RSFV1().write(f4_patch, path, data_path=data_path if split else None)
        assert dc.get_format(path) == ("rsf", "1")
        patch = dc.spool(path)[0]
        assert patch.dtype == np.float32
        assert np.array_equal(patch.data, f4_patch.data)
        assert patch.get_coord("time") == f4_patch.get_coord("time")
        distance = patch.get_coord("distance")
        assert np.array_equal(distance.values, f4_patch.get_coord("distance").values)
        assert distance.units == f4_patch.get_coord("distance").units

    @pytest.mark.parametrize("rate", [1024, 3000, -1024])
    def test_exact_time(self, rate, fractional_patch, tmp_path):
        """A time step that is no finite decimal, or is negative, reads back."""
        start = np.datetime64("2024-01-01T00:00:00.000001", "ns")
        step = step_from_interval(1 / abs(rate))
        time = dc.get_coord(start=start, step=step if rate > 0 else -step, shape=(8,))
        RSFV1().write(fractional_patch.update_coords(time=time), tmp_path / "e.rsf")
        assert dc.spool(tmp_path / "e.rsf")[0].get_coord("time") == time

    def test_window_and_overwrite(self, f4_patch, tmp_path):
        """A window reads alone and survives a later write to the same file."""
        path = tmp_path / "w.rsf"
        RSFV1().write(f4_patch, path)
        time = f4_patch.get_coord("time")
        span = (time.values[10], time.values[20])
        patch = dc.read(path, time=span, distance=(5, 9))[0]
        RSFV1().write(f4_patch.update_attrs(tag="x" * 100), path)
        expected = f4_patch.select(time=span, distance=(5, 9))
        assert np.array_equal(patch.data, expected.data)
        assert patch.coords == expected.coords

    def test_scan_header_only(self, f4_patch, tmp_path):
        """A scan reads the header, so it works with the binary missing."""
        binary = tmp_path / "data.rsf"
        RSFV1().write(f4_patch, tmp_path / "hdr.rsf", data_path=binary)
        Path(f"{binary}@").unlink()
        (meta,) = dc.scan(tmp_path / "hdr.rsf")
        assert (meta.dims, meta.shape) == (f4_patch.dims, f4_patch.shape)

    def test_relative_in(self, tmp_path, monkeypatch):
        """A relative in= resolves against the header's directory, then cwd."""
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        lines = ["n1=2", "n2=3", 'in="data.rsf@"']
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_rsf(sub / "a.rsf", lines, data, data_path=sub / "data.rsf@")
        assert np.array_equal(dc.spool(sub / "a.rsf")[0].data, data)
        monkeypatch.chdir(sub)
        _write_rsf(tmp_path / "b.rsf", lines)
        assert np.array_equal(dc.spool(tmp_path / "b.rsf")[0].data, data)

    def test_in_keeps_backslashes(self):
        """in= keeps its backslashes, as Madagascar does, for a UNC path."""
        hdr, _, _ = read_header(BytesIO(b'n1=1\nin="\\\\srv\\d.rsf@"\n'))
        assert hdr["in"] == r"\\srv\d.rsf@"

    @pytest.mark.parametrize("order", ["native", "xdr"])
    @pytest.mark.parametrize("kind", list(KINDS))
    def test_data_formats(self, order, kind, tmp_path):
        """Native and xdr samples of each supported kind read back as native."""
        dtype = np.dtype(KINDS[kind])
        data = (np.arange(12) * (1 + 1j if kind == "complex" else 1)).reshape(3, 4)
        data = data.astype(dtype)
        stored = data.astype(dtype.newbyteorder(">")) if order == "xdr" else data
        lines = [f'data_format="{order}_{kind}"', "n1=3", "n2=4", 'in="stdin"']
        patch = dc.spool(_write_rsf(tmp_path / "fmt.rsf", lines, stored))[0]
        assert patch.dims == ("axis1", "axis2")
        assert patch.dtype == dtype
        assert np.array_equal(patch.data, data)

    def test_starttime(self, tmp_path):
        """Without epoch<i>, the legacy starttime starts the axis labelled time."""
        lines = ["n1=4", "d1=0.5", 'label1="time"', "starttime=1.5", 'in="stdin"']
        path = _write_rsf(tmp_path / "t.rsf", lines, np.zeros(4, np.float32))
        time = dc.spool(path)[0].get_coord("time")
        assert time.min() == np.datetime64("1970-01-01T00:00:01.5", "ns")
        assert time.step == np.timedelta64(500, "ms")

    def test_free_text_units(self, tmp_path):
        """A unit pint cannot parse, such as hz, is dropped; the axis still reads."""
        lines = ["n1=2", "n2=3", 'unit1="hz"', 'unit2="Hz"', 'in="stdin"']
        path = _write_rsf(tmp_path / "u.rsf", lines, np.zeros(6, np.float32))
        patch = dc.spool(path)[0]
        assert patch.get_coord("axis1").units is None
        assert patch.get_coord("axis2").units == dc.get_quantity("Hz")

    def test_axis_names(self, tmp_path):
        """An unbound n<i> is one sample; a repeated label falls back to axis<i>."""
        lines = ["n1=2", "n3=3", 'label1="x"', 'label3="x"', 'in="stdin"']
        path = _write_rsf(tmp_path / "m.rsf", lines, np.zeros(6, np.float32))
        patch = dc.spool(path)[0]
        assert (patch.dims, patch.shape) == (("x", "axis2", "axis3"), (2, 1, 3))

    @pytest.mark.parametrize(
        "content",
        [b"", b"just text\n", b"n1=3\n", b'in="stdin"\n', b"\x89HDF\x00n1=3\nin=x\n"],
    )
    def test_not_rsf(self, content, tmp_path):
        """Text without in= and n1, and binary, are not RSF."""
        path = tmp_path / "other.rsf"
        path.write_bytes(content)
        assert RSFV1().get_version(path) is None
        with pytest.raises(InvalidFiberFileError, match="RSF header"):
            RSFV1().get_metadata(path)

    @pytest.mark.parametrize(
        "lines, data, match",
        [
            (["n1=4", 'in="stdin"'], None, "no binary separator"),
            (["n1=5", 'in="stdin"'], np.zeros(4, "f4"), "exactly 20 bytes"),
            (["n1=3", 'in="stdin"'], np.zeros(4, "f4"), "exactly 12 bytes"),
            (["n1=4", 'data_format="ascii_float"', 'in="stdin"'], None, "ascii"),
        ],
    )
    def test_bad_binary(self, lines, data, match, tmp_path):
        """A binary that is misplaced or mis-sized, or a text format, raises."""
        path = tmp_path / "bad.rsf"
        if data is None:
            path.write_text("\n".join(lines))
        else:
            _write_rsf(path, lines, data)
        with pytest.raises(InvalidFiberFileError, match=match):
            RSFV1().read(path)


HISTORY = ("pass_filter(time=(None, 10))", "a\nn1=99\x00b")


@pytest.fixture(scope="module")
def attrs_patch(fractional_patch):
    """A float32 patch with set, extra and colliding attrs and history."""
    attrs = {
        "data_type": "strain_rate",
        "data_units": "1/s",
        "tag": '{"a": 1}',
        "gauge_length": 10.5,
        "channel_count": np.int64(3),
        "loop": True,
        "empty": None,
        "code": "3",
        "station": 'a "quoted" \\ name',
        "in": "collides",
        "n1": 7,
        "dascore_note": "prefixed",
        "time_min": 0.0,
        "history": HISTORY,
    }
    return fractional_patch.update(attrs=attrs)


class TestRsfAttrs:
    """Tests for the attrs and history the writer stores and the reader restores."""

    def test_header(self, attrs_patch, tmp_path):
        """History lines open the header; attrs are one prefixed line each."""
        RSFV1().write(attrs_patch, tmp_path / "a.rsf")
        RSFV1().write(attrs_patch, tmp_path / "b.rsf")
        raw = (tmp_path / "a.rsf").read_bytes()
        assert raw == (tmp_path / "b.rsf").read_bytes()  # no clock in the header
        header = _split_combined(raw)[1].decode().splitlines()
        assert header[:2] == [
            f"# dascore {dc.__version__} pass_filter(time:(None, 10))",
            f"# dascore {dc.__version__} a n1:99 b",
        ]
        assert {
            "loop=true",
            "empty=null",
            r'station="a \"quoted\" \\ name"',
            'dascore_in="collides"',
            "dascore_n1=7",
            'dascore_dascore_note="prefixed"',
        } <= set(header)
        keys = {x.partition("=")[0] for x in header}
        assert not keys & {"data_category", "origin_id", "data_id", "time_min"}

    def test_roundtrip(self, attrs_patch, tmp_path):
        """Attrs read back equal; history reads back with its substitutions."""
        RSFV1().write(attrs_patch, tmp_path / "attrs.rsf")
        patch = dc.spool(tmp_path / "attrs.rsf")[0]
        skip = {"history", "origin_id", "data_id", "time_min"}
        expected = {i: v for i, v in dict(attrs_patch.attrs).items() if i not in skip}
        assert {i: v for i, v in dict(patch.attrs).items() if i not in skip} == expected
        assert patch.attrs.history == ("pass_filter(time:(None, 10))", "a n1:99 b")

    @pytest.mark.parametrize(
        "name, value",
        [("note", 'say "hi'), ("note", "a\nb"), ("my-note", 5)],
    )
    def test_misread_skipped(self, name, value, fractional_patch, tmp_path):
        """An attr Madagascar would misread, or DASCore not read, is skipped."""
        patch = fractional_patch.update_attrs(**{name: value, "tag": "kept"})
        with pytest.warns(DASCoreWarning, match=f"{name} attr") as record:
            dc.write(patch, tmp_path / "q.rsf", "rsf")
        assert record[0].filename == __file__
        _, head, _ = _split_combined((tmp_path / "q.rsf").read_bytes())
        assert f"{name}=".encode() not in head
        assert dc.spool(tmp_path / "q.rsf")[0].attrs["tag"] == "kept"

    def test_madagascar(self, tmp_path):
        """Madagascar's and gabbro's history lines are prose; its keys are extras."""
        lines = [
            "4.2-git\tsfspike\t/home/user:\tuser@host\tThu Nov  5 22:06:23 2026",
            "",
            "\to1=0",
            '\tlabel1="Time"',
            '\tdata_format="native_float"',
            "\tesize=4",
            '\tin="stdout"',
            '\tunit1="s"',
            "\td1=0.004",
            "\tn1=4",
            '\tin="stdin"',
            "",
            "4.2-git\tsfmath\t/home/user:\tuser@host\tThu Nov  5 22:06:23 2026",
            "# spike 0.4.0 n1:4",
            "made with o1=99 by hand",
            "",
            "\tinput=0",
            "\ttitle=Spike",
            '\tin="stdin"',
        ]
        data = np.arange(4, dtype=np.float32)
        patch = dc.spool(_write_rsf(tmp_path / "m8r.rsf", lines, data))[0]
        assert np.array_equal(patch.data, data)
        assert patch.attrs.history == ()
        assert (patch.attrs["input"], patch.attrs["title"]) == (0, "Spike")
        assert "sfspike" not in str(dict(patch.attrs))
        assert patch.get_coord("Time").min() == 0.0

    def test_unheld_value_skipped(self, tmp_path):
        """A bare value no attr holds, such as a JSON list, is skipped."""
        lines = ["n1=4", "d1=1", "o1=0", 'data_format="native_float"']
        lines += ["esize=4", 'in="stdin"', "pair=[1,2]", 'tag="{}"']
        path = _write_rsf(tmp_path / "list.rsf", lines, np.zeros(4, np.float32))
        with pytest.warns(UserWarning, match="pair"):
            patch = dc.spool(path)[0]
        assert "pair" not in dict(patch.attrs)
        assert patch.attrs["tag"] == "{}"
