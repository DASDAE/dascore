"""Tests for SEGY format."""

import tracemalloc

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import (
    InvalidSpoolError,
    PatchError,
)
from dascore.io.segy.core import SegyV1_0
from dascore.units import get_quantity
from dascore.utils.downloader import fetch
from dascore.utils.misc import suppress_warnings


class TestSegyGetFormat:
    """Tests for getting format codes of SEGY files."""

    @pytest.fixture(scope="class")
    def small_file(self, tmp_path_factory):
        """Creates a small file with only a few bytes."""
        parent = tmp_path_factory.mktemp("small_file")
        path = parent / "test_file.segy"
        with path.open("wb") as f:
            f.write(b"abd")
        return path

    def test_get_format_small_file(self, small_file):
        """
        Ensure a file that is too small to contain segy header doesn't throw
        an error.
        """
        segy = SegyV1_0()
        out = segy.get_format(small_file)
        assert out is False  # we actually want to make sure its False.


class TestSegyReadArray:
    """Tests for reading SEGY trace data."""

    def test_reads_without_a_second_copy(self):
        """The traces are read in bulk, so the peak is about one array."""
        pytest.importorskip("segyio")
        path = fetch("conoco_segy_1.sgy")
        tracemalloc.start()
        try:
            data = SegyV1_0().read_array(path)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 1.5 * data.nbytes
        assert data.flags.c_contiguous

    def test_narrow_window_holds_only_its_samples(self):
        """A short time window does not keep the traces' full length."""
        pytest.importorskip("segyio")
        data = SegyV1_0().read_array(fetch("conoco_segy_1.sgy"), ((0, 1), None))
        assert data.base is None and data.shape[0] == 1


class TestSegyWrite:
    """Tests for writing segy files."""

    @pytest.fixture(scope="class")
    def test_segy_directory(self, tmp_path_factory):
        """Make a tmp directory for saving files."""
        path = tmp_path_factory.mktemp("test_segy_write_directory")
        return path

    @pytest.fixture(scope="class")
    def channel_patch(self, random_patch):
        """Get a patch with channels rather than distance."""
        distance = random_patch.get_coord("distance")
        new = random_patch.rename_coords(distance="channel").update_coords(
            **{"channel": np.arange(len(distance))}
        )
        return new

    @pytest.fixture(scope="class")
    def channel_patch_path(self, channel_patch, test_segy_directory):
        """Write the channel patch to disk."""
        pytest.importorskip("segyio")
        path = test_segy_directory / "patch_with_channel_coord.segy"
        channel_patch.io.write(path, "segy")
        return path

    def test_empty_channel_window(self, channel_patch_path, channel_patch):
        """An empty channel window preserves time length and the decoded dtype."""
        out = SegyV1_0().read_array(channel_patch_path, (None, (0, 0)))
        assert out.shape == (len(channel_patch.get_coord("time")), 0)
        assert out.dtype == dc.scan(channel_patch_path)[0].dtype

    def test_can_get_format(self, channel_patch_path):
        """Ensure we can get the correct format/version."""
        segy = SegyV1_0()
        out = segy.get_format(channel_patch_path)
        assert out, "Failed to detect written segy file."
        assert out[0] == segy.name

    def test_channel_patch_round_trip(self, channel_patch_path, channel_patch):
        """The channel patch should round trip."""
        patch1 = channel_patch
        patch2 = dc.spool(channel_patch_path)[0].transpose(*patch1.dims)
        # We really don't have a way to transport attributes yet, so we
        # just check that data and coords are equal.
        assert np.allclose(patch1.data, patch2.data)
        assert patch1.coords == patch2.coords

    def test_write_non_channel_path(self, random_patch, tmp_path_factory):
        """Ensure a 'normal' patch can be written."""
        pytest.importorskip("segyio")
        path = tmp_path_factory.mktemp("test_write_segy") / "temppath.segy"
        match = "non-time dimension"
        with pytest.warns(match=match):
            random_patch.io.write(path, "segy")
        assert path.exists()
        patch2 = dc.spool(path)[0]
        assert set(random_patch.shape) == set(patch2.shape)

    def test_fractional_start_second_warns(self, channel_patch, tmp_path_factory):
        """A start time between whole seconds warns that it is floored."""
        pytest.importorskip("segyio")
        path = tmp_path_factory.mktemp("test_fractional_start") / "temppath.segy"
        start = np.datetime64("2020-01-01T00:00:00.846")
        patch = channel_patch.update_coords(time_min=start)
        with pytest.warns(UserWarning, match="written as 2020-01-01 00:00:00"):
            patch.io.write(path, "segy")
        # A start on a whole second is written as it is, without a warning.
        whole = channel_patch.update_coords(time_min=start.astype("datetime64[s]"))
        with suppress_warnings(UserWarning, message=".*whole seconds", action="error"):
            whole.io.write(path.with_name("whole.segy"), "segy")

    def test_loss_of_precision_raises(self, random_patch, tmp_path_factory):
        """Raise PatchError when writing would lose precision."""
        pytest.importorskip("segyio")
        path = tmp_path_factory.mktemp("test_loss_of_precision") / "temppath.segy"
        patch = random_patch.update_coords(time_step=np.timedelta64(10, "ns"))
        match = "will result in a loss of precision"
        with pytest.raises(PatchError, match=match):
            with suppress_warnings():
                patch.io.write(path, "segy")

    def test_bad_dims_raises(self, random_patch, tmp_path):
        """Ensure a bad dimension name raises."""
        pytest.importorskip("segyio")
        patch = random_patch.rename_coords(distance="bad_dim")
        with pytest.raises(PatchError, match="Can only save 2D patches"):
            patch.io.write(tmp_path, "segy")

    def test_multi_patch_spool_raises(self, random_spool, tmp_path):
        """Spools with more than one patch cant be written."""
        pytest.importorskip("segyio")
        segy = SegyV1_0()
        with pytest.raises(InvalidSpoolError):
            segy.write(random_spool, tmp_path)


def _write_segy_file(
    path,
    endian="big",
    revision=(1, 0),
    date=(2016, 22, 1, 23, 59),
    text=None,
    ascii_text=False,
):
    """
    Write a 3-trace SEG-Y file with the given byte order, revision bytes, date
    and textual header lines (a dict of line number to content), stored as
    EBCDIC or ASCII.
    """
    segyio = pytest.importorskip("segyio")
    data = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)
    spec = segyio.spec()
    spec.format, spec.samples, spec.endian = 5, range(20), endian
    spec.ilines, spec.xlines = range(3), [1]
    field = segyio.TraceField
    names = ("YearDataRecorded", "DayOfYear", "HourOfDay")
    names += ("MinuteOfHour", "SecondOfMinute")
    header = {getattr(field, name): value for name, value in zip(names, date)}
    header.update({field.TRACE_SAMPLE_INTERVAL: 1000, field.TRACE_SAMPLE_COUNT: 20})
    with segyio.create(str(path), spec) as fi:
        fi.bin.update({segyio.BinField.Interval: 1000})
        if text:
            fi.text[0] = segyio.tools.create_text_header(text)
        for num, trace in enumerate(data):
            fi.header[num] = header
            fi.trace[num] = trace
    with path.open("r+b") as fi:
        if ascii_text:
            ebcdic = fi.read(3200)
            fi.seek(0)
            fi.write(ebcdic.decode("cp037").encode("latin-1"))
        fi.seek(3500)
        fi.write(bytes(revision))
    return path, data.T


class TestSegyCompatibility:
    """Tests for SEG-Y files that bend the standard."""

    def test_hdf5_signature_is_not_segy(self, tmp_path):
        """An HDF5 file is not SEG-Y, whatever its bytes at 3200 hold."""
        path, _ = _write_segy_file(tmp_path / "h5.sgy")
        with open(path, "r+b") as fp:
            fp.write(b"\x89HDF\r\n\x1a\n")
        assert SegyV1_0().get_format(path) is False

    def test_nonstandard_minor_revision(self, tmp_path):
        """A minor revision such as 1.80 reads as revision 1.0."""
        path, data = _write_segy_file(tmp_path / "rev.sgy", revision=(1, 80))
        assert dc.get_format(path) == ("segy", "1.0")
        assert np.array_equal(dc.spool(path)[0].data, data)

    @pytest.mark.parametrize(
        "line, stamp",
        [
            # As quoted in #1311.
            (13, "UTC File Time stamp: 2019/04/24 22:24:24.846233 (UTC)"),
            # As in Booth et al. (2020), doi:10.6084/m9.figshare.12410390.
            (11, "UTC Time stamp: 2019/04/24 22:24:24.846233 (UTC)"),
            # As in conoco_segy_1.sgy.
            (13, "GPS Time stamp: 2019/04/24 22:24:24.846233 (UTC)"),
        ],
    )
    @pytest.mark.parametrize("ascii_text", [False, True])
    def test_text_header_sub_second_start(self, tmp_path, line, stamp, ascii_text):
        """
        Silixa's textual header time stamp, in EBCDIC or ASCII, gives the start
        below the second.
        """
        date, text = (2019, 114, 22, 24, 24), {line: stamp}
        path, _ = _write_segy_file(
            tmp_path / "s.sgy", date=date, text=text, ascii_text=ascii_text
        )
        time = dc.spool(path)[0].get_coord("time")
        assert time.min() == np.datetime64("2019-04-24T22:24:24.846233")
        assert time.step == np.timedelta64(1, "ms")

    def test_gps_time_stamp_file(self):
        """A Silixa file stating a GPS time stamp starts at that stamp."""
        pytest.importorskip("segyio")
        time = dc.spool(fetch("conoco_segy_1.sgy"))[0].get_coord("time")
        assert time.min() == np.datetime64("2017-07-28T18:05:57.734")

    @pytest.mark.parametrize(
        "stamp", ["2019/04/24 22:24:25.846233", "2019/13/24 22:24:24.846233"]
    )
    def test_text_header_other_second_ignored(self, tmp_path, stamp):
        """A stamp outside the trace header's second, or not a date, is ignored."""
        text = {13: f"UTC File Time stamp: {stamp} (UTC)"}
        date = (2019, 114, 22, 24, 24)
        path, _ = _write_segy_file(tmp_path / "silixa.sgy", date=date, text=text)
        time = dc.spool(path)[0].get_coord("time")
        assert time.min() == np.datetime64("2019-04-24T22:24:24")

    def test_text_header_without_trace_date(self, tmp_path):
        """Without a valid trace header date, time stays relative."""
        text = {13: "UTC File Time stamp: 2019/04/24 22:24:24.846233 (UTC)"}
        path, _ = _write_segy_file(tmp_path / "s.sgy", date=(0,) * 5, text=text)
        with pytest.warns(UserWarning, match="not valid"):
            time = dc.spool(path)[0].get_coord("time")
        assert time.min() == np.timedelta64(0, "ns")

    def test_nonstandard_minor_needs_whole_traces(self, tmp_path):
        """A non-standard revision is only claimed when whole traces fill the file."""
        path, _ = _write_segy_file(tmp_path / "rev.sgy", revision=(1, 80))
        with path.open("ab") as fi:
            fi.write(b"\x00" * 10)
        assert SegyV1_0().get_format(path) is False

    @pytest.mark.parametrize(
        "revision, version", [((1, 0), "1.0"), ((0, 1), "1.0"), ((0, 100), "0.100")]
    )
    def test_little_endian(self, tmp_path, revision, version):
        """
        Little-endian files are detected and read like big-endian ones, with
        the revision as two bytes or as one little-endian word (segyio).
        """
        big, data = _write_segy_file(tmp_path / "big.sgy")
        little, _ = _write_segy_file(
            tmp_path / "little.sgy", endian="little", revision=revision
        )
        assert dc.get_format(little) == ("segy", version)
        patch = dc.spool(little)[0]
        assert patch == dc.spool(big)[0]
        assert np.array_equal(patch.select(channel=(1, 2)).data, data[:, 1:])

    def test_second_60_rolls_over(self, tmp_path):
        """A header second of 60 is the start of the next minute."""
        path, _ = _write_segy_file(tmp_path / "s60.sgy", date=(2016, 22, 1, 23, 60))
        time = dc.spool(path)[0].get_coord("time")
        assert time.min() == np.datetime64("2016-01-22T01:24:00")

    @pytest.mark.parametrize(
        "date", [(0, 0, 0, 0, 0), (2016, 0, 1, 2, 3), (16, 22, 1, 2, 3)]
    )
    def test_invalid_date_is_relative(self, tmp_path, date):
        """
        Without a valid year and day (none, day 0, two-digit year), time
        starts at zero, keeps its step and takes the start the warning gives.
        """
        path, _ = _write_segy_file(tmp_path / "nodate.sgy", date=date)
        with pytest.warns(UserWarning, match="not valid"):
            patch = dc.spool(path)[0]
        time = patch.get_coord("time")
        assert time.min() == np.timedelta64(0, "ns")
        assert time.step == np.timedelta64(1, "ms")
        start = "2013-09-11T17:53:29"  # any form dc.to_datetime64 takes
        new = patch.update_coords(
            time=patch.get_coord("time") + dc.to_datetime64(start)
        )
        assert new.get_coord("time").min() == np.datetime64(start)


class TestSegyTextHeader:
    """Tests for reading the textual header into attrs."""

    @pytest.mark.parametrize("ascii_text", [False, True])
    def test_silixa_strain_rate_scale(self, tmp_path, ascii_text):
        """
        Silixa's strain rate line, in EBCDIC or ASCII, sets the data type and
        scaled units, and the header is kept line by line.
        """
        line = "AMPLITUDE VALUES PROPORTIONAL TO STRAINRATE; SC= X11600 TO NM/M/S"
        path, _ = _write_segy_file(
            tmp_path / "s.sgy", text={14: line}, ascii_text=ascii_text
        )
        attrs = dc.spool(path)[0].attrs
        assert attrs.data_type == "strain_rate"
        assert attrs.data_units == get_quantity("11600 nm/m/s")
        assert f"C14 {line}" in attrs.segy_text_header.split("\n")

    def test_strain_rate_without_scale(self):
        """A strain rate line without a scale sets the data type only."""
        pytest.importorskip("segyio")
        attrs = dc.spool(fetch("conoco_segy_1.sgy"))[0].attrs
        assert attrs.data_type == "strain_rate"
        assert attrs.data_units is None
        assert "C3 Manufacturer: Silixa" in attrs.segy_text_header.split("\n")

    def test_blank_header(self, tmp_path):
        """A blank textual header gives no attrs."""
        path, _ = _write_segy_file(tmp_path / "blank.sgy")
        with path.open("r+b") as fi:
            fi.write(b"\x40" * 3200)
        attrs = dc.spool(path)[0].attrs
        assert "segy_text_header" not in attrs.model_dump()
        assert attrs.data_type == ""
