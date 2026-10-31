"""Tests for seismic ASDF waveforms, written with h5py."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.asdf import ASDFV1

_START = 1_569_062_699_624_900_096
# (NET.STA.LOC.CHA, start offset ns, rate, count, tag); the first three join.
_TRACES = (
    ("MG.0000..ESN", 0, 100.0, 6, "raw_data"),
    ("MG.0001..ESN", 0, 100.0, 6, "raw_data"),
    ("MG.0002..ESN", 0, 100.0, 6, "raw_data"),
    ("MG.0003..ESN", 10_000_000, 100.0, 6, "raw_data"),
    ("MG.0004..ESN", 0, 50.0, 6, "raw_data"),
    ("MG.0005..ESN", 0, 100.0, 5, "raw_data"),
    ("MG.0006.01.ESN", 0, 100.0, 6, "raw_data"),
    ("MG.0000..ESN", 0, 100.0, 6, "processed"),
)


def _data(index, count):
    """Return distinct samples for one trace."""
    return np.arange(count, dtype=np.float64) + 100 * index


def write_asdf(path, traces=_TRACES, version="1.0.3"):
    """Write an ASDF file holding one dataset per trace, plus a StationXML."""
    with h5py.File(path, "w") as h5:
        h5.attrs["file_format"] = np.bytes_(b"ASDF")
        h5.attrs["file_format_version"] = np.bytes_(version.encode())
        waveforms = h5.create_group("Waveforms")
        for index, (codes, offset, rate, count, tag) in enumerate(traces):
            group = waveforms.require_group(".".join(codes.split(".")[:2]))
            group.require_dataset("StationXML", shape=(3,), dtype="u1")
            name = f"{codes}__start__end__{tag}"
            dataset = group.create_dataset(name, data=_data(index, count))
            dataset.attrs["starttime"] = np.int64(_START + offset)
            dataset.attrs["sampling_rate"] = np.float64(rate)
    return path


@pytest.fixture(scope="module")
def asdf_path(tmp_path_factory):
    """An ASDF file whose first three traces share one patch."""
    return write_asdf(tmp_path_factory.mktemp("asdf") / "example.h5")


@pytest.fixture(scope="module")
def spool(asdf_path):
    """Every patch of the example file."""
    return dc.read(asdf_path)


class TestRead:
    """Traces map to patches as in the MiniSEED reader."""

    def test_compatible_traces_join(self, spool):
        """Shared network, location, channel, timing and tag make one patch."""
        (patch,) = [x for x in spool if x.shape[0] == 3]
        assert patch.dims == ("channel", "time")
        expected = np.stack([_data(i, 6) for i in range(3)])
        assert np.array_equal(patch.data, expected)
        assert list(patch.get_coord("channel").values) == [0, 1, 2]
        assert list(patch.coords.get_array("source_id")) == [
            "MG.0000..ESN",
            "MG.0001..ESN",
            "MG.0002..ESN",
        ]
        assert list(patch.coords.get_array("station")) == ["0000", "0001", "0002"]
        assert set(patch.coords.get_array("seed_channel")) == {"ESN"}
        time = patch.get_coord("time")
        assert time.min() == np.datetime64(_START, "ns")
        assert time.step == np.timedelta64(10, "ms")
        assert patch.attrs.tag == "MG..ESN.raw_data"
        assert patch.attrs.sample_rate == 100.0

    def test_incompatible_traces_split(self, spool):
        """Start, rate, count, location and tag each split a patch."""
        assert len(spool) == 6
        singles = {
            x.coords.get_array("source_id")[0]: x for x in spool if x.shape[0] == 1
        }
        assert sorted(singles) == [
            "MG.0000..ESN",
            "MG.0003..ESN",
            "MG.0004..ESN",
            "MG.0005..ESN",
            "MG.0006.01.ESN",
        ]
        assert singles["MG.0000..ESN"].attrs.tag == "MG..ESN.processed"
        assert singles["MG.0006.01.ESN"].attrs.tag == "MG.01.ESN.raw_data"
        # The second tag of a source keeps its channel index.
        assert singles["MG.0000..ESN"].get_coord("channel").values[0] == 0
        assert np.array_equal(singles["MG.0005..ESN"].data[0], _data(5, 5))

    def test_selection(self, asdf_path):
        """A channel and time selection reads the right values."""
        start = np.datetime64(_START, "ns")
        times = (start + np.timedelta64(20, "ms"), start + np.timedelta64(40, "ms"))
        out = dc.read(asdf_path, channel=(1, 2), time=times, tag="MG..ESN.raw_data")
        (patch,) = [x for x in out if x.shape[0] == 2]
        assert np.array_equal(patch.data, np.stack([_data(i, 6)[2:5] for i in (1, 2)]))

    def test_keyed_read_reads_only_the_window(self, asdf_path, monkeypatch):
        """read_array slices only the selected channels' datasets."""
        (meta,) = [x for x in ASDFV1().get_metadata(asdf_path) if x.shape[0] == 3]
        reads = []
        original = h5py.Dataset.__getitem__

        def spy(self, item):
            reads.append((self.name.rsplit("/", 2)[1], item))
            return original(self, item)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
        array = ASDFV1().read_array(asdf_path, ((1, 3), (2, 5)), key=meta._source.key)
        assert np.array_equal(array, np.stack([_data(i, 6)[2:5] for i in (1, 2)]))
        assert reads == [("MG.0001", slice(2, 5)), ("MG.0002", slice(2, 5))]


class TestDetect:
    """Only seismic ASDF 1.0 files with waveforms are claimed."""

    def test_claims_asdf(self, asdf_path):
        """The example is ASDF 1.0."""
        assert dc.get_format(asdf_path) == ("ASDF", "1.0")

    def test_plain_hdf5(self, tmp_path):
        """An HDF5 file without the ASDF attrs is not claimed."""
        path = tmp_path / "plain.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("data", data=np.zeros(3))
        assert not ASDFV1().get_format(path)

    def test_other_format_name(self, tmp_path):
        """The ASDF layout under another format name is not claimed."""
        path = write_asdf(tmp_path / "other.h5")
        with h5py.File(path, "r+") as h5:
            h5.attrs["file_format"] = "OTHER"
        assert not ASDFV1().get_format(path)

    def test_other_version(self, tmp_path):
        """ASDF 2.0 is refused."""
        path = write_asdf(tmp_path / "v2.h5", version="2.0")
        assert not ASDFV1().get_format(path)

    def test_no_waveforms(self, tmp_path):
        """An ASDF file without a Waveforms group is not claimed."""
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["file_format"] = "ASDF"
            h5.attrs["file_format_version"] = "1.0.3"
        assert not ASDFV1().get_format(path)

    def test_astronomy_asdf(self, tmp_path):
        """The astronomy format's YAML file is not claimed."""
        path = tmp_path / "image.asdf"
        path.write_text(
            "#ASDF 1.0.0\n#ASDF_STANDARD 1.5.0\n%YAML 1.1\n--- !core/asdf-1.1.0\n...\n"
        )
        assert not ASDFV1().get_format(path)


class TestMalformed:
    """Waveforms the reader cannot parse raise, naming the file."""

    @pytest.mark.parametrize(
        "name", ["MG.0000.ESN__start__end__raw_data", "MG.0000..ESN__raw_data"]
    )
    def test_bad_name(self, tmp_path, name):
        """A name not shaped NET.STA.LOC.CHA__start__end__tag raises."""
        path = write_asdf(tmp_path / "bad.h5", traces=_TRACES[:1])
        with h5py.File(path, "r+") as h5:
            h5["Waveforms/MG.0000"].move("MG.0000..ESN__start__end__raw_data", name)
        with pytest.raises(InvalidFiberFileError, match=r"bad\.h5"):
            ASDFV1().get_metadata(path)

    @pytest.mark.parametrize("attr", ["starttime", "sampling_rate"])
    def test_missing_attr(self, tmp_path, attr):
        """A waveform lacking its timing attrs raises."""
        path = write_asdf(tmp_path / "bad.h5", traces=_TRACES[:1])
        with h5py.File(path, "r+") as h5:
            del h5["Waveforms/MG.0000/MG.0000..ESN__start__end__raw_data"].attrs[attr]
        with pytest.raises(InvalidFiberFileError, match=r"bad\.h5"):
            ASDFV1().get_metadata(path)
