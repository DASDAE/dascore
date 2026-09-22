"""Misc. tests for xml binary."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from upath import UPath

import dascore as dc
from dascore.constants import STORAGE_PROVENANCE_ATTRS
from dascore.exceptions import (
    InvalidFiberFileError,
    ParameterError,
    UnknownFiberFormatError,
)
from dascore.io import core as io_core
from dascore.io.xml_binary import XMLBinaryV1
from dascore.io.xml_binary.utils import _read_xml_metadata
from dascore.utils.time import to_float

sampling_rate = 1000
expected_duration = 1

metadata = f"""<?xml version='1.0' encoding='utf-8'?>
<Metadata>
  <FileFormat>RAW</FileFormat>
  <DateTime>2024-05-30T01:15:00Z</DateTime>
  <DasInterrogatorSerial>
    <Interrogator_1>CRI-4400_A001</Interrogator_1>
  </DasInterrogatorSerial>
  <GaugeLengthM>1</GaugeLengthM>
  <PulseWidthNs>100</PulseWidthNs>
  <DataType>uint16</DataType>
  <NumberOfLasers>2</NumberOfLasers>
  <ITUChannels>
    <Laser_1>15</Laser_1>
    <Laser_2>10</Laser_2>
  </ITUChannels>
  <OriginalTemporalSamplingRate>1000</OriginalTemporalSamplingRate>
  <OutputTemporalSamplingRate>{sampling_rate}</OutputTemporalSamplingRate>
  <OriginalSpatialSamplingInterval>1</OriginalSpatialSamplingInterval>
  <Units>m</Units>
  <Zones>
    <Zone>
      <StartChannel>1</StartChannel>
      <EndChannel>10</EndChannel>
      <Stride>1</Stride>
      <NumberOfChannels>10</NumberOfChannels>
    </Zone>
  </Zones>
  <NumberOfChannels>10</NumberOfChannels>
  <NumberOfFrames>1000</NumberOfFrames>
  <UseRelativeStrain>False</UseRelativeStrain>
  <TransposedData>False</TransposedData>
</Metadata>
"""


@pytest.fixture(scope="session")
def binary_xml_directory(tmp_path_factory):
    """Creates a directory of binary files and an xml metadata."""
    new = tmp_path_factory.mktemp("xml_binary_test_data")
    metadata_path = new / "metadata.xml"
    data_1_path = new / "DAS_20240530T011500_000000Z.raw"
    data_2_path = new / "DAS_20240530T011501_000000Z.raw"
    with open(metadata_path, "w") as fi:
        fi.write(metadata)
    with open(data_1_path, "wb") as fi:
        ar = np.arange(1000 * 10, dtype=np.dtype("uint16"))
        ar.tofile(fi)
    with open(data_2_path, "wb") as fi:
        ar = np.arange(1000 * 10, dtype=np.dtype("uint16"))
        ar.tofile(fi)
    return new


@pytest.fixture(scope="session")
def binary_xml_with_other_files(
    binary_xml_directory,
    tmp_path_factory,
):
    """Ensure other files can be included at the top level."""
    new = tmp_path_factory.mktemp("binary_xml_and_others")
    shutil.copytree(binary_xml_directory, new / "xml_binary")
    sp = dc.get_example_spool()
    dc.examples.spool_to_directory(sp, new)
    return new


@pytest.fixture(scope="session")
def xml_directory_no_data(tmp_path_factory):
    """Creates a directory of xml metadata with no binary files."""
    new = tmp_path_factory.mktemp("binary_xml_no_data")
    metadata_path = new / "metadata.xml"
    with open(metadata_path, "w") as fi:
        fi.write(metadata)
    return new


@pytest.fixture(scope="session")
def directory_bad_xml(tmp_path_factory):
    """Create a directory with bad xml metadata."""
    new = tmp_path_factory.mktemp("xml_binary_bad_xml")
    metadata_path = new / "metadata.xml"
    with open(metadata_path, "w") as fi:
        fi.write("The bathhouse makes him crazy.")
    return new


@pytest.fixture(scope="session")
def remote_binary_xml_directory(binary_xml_directory):
    """Create an in-memory remote XMLBinary directory."""
    root = UPath("memory://dascore/xml_binary_remote")
    root.mkdir(parents=True, exist_ok=True)
    for path in Path(binary_xml_directory).iterdir():
        (root / path.name).write_bytes(path.read_bytes())
    return root


class TestReadXMLMetadata:
    """Misc tests reading xml metadata."""

    def test_read_metadata_contents(self, binary_xml_directory):
        """Test reading xml metadata contents and metadata types."""
        metadata_path_1 = binary_xml_directory / "metadata.xml"
        metadata = _read_xml_metadata(metadata_path_1)
        assert metadata is not None
        # Just test a couple of attrs, pydantic should handle the rest.
        expected_attrs = ["file_format", "date_time", "units"]
        for attr in expected_attrs:
            assert hasattr(metadata, attr)


class TestGetFormat:
    """Test suite for xml binary get format."""

    def test_returns_name_format(self, binary_xml_directory):
        """Ensure the proper name/format are returned from working folder."""
        name, version = dc.get_format(binary_xml_directory)
        fiber_io = XMLBinaryV1()
        assert name == fiber_io.name
        assert version == fiber_io.version

    def test_bad_xml_metadata(self, directory_bad_xml):
        """Ensure get format raises with bad xml metadata."""
        # dc.get_format should raise, as per its docs
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(directory_bad_xml)
        # but the specific class get_format should return false.
        fiber_io = XMLBinaryV1()
        out = fiber_io.get_format(directory_bad_xml)
        assert not out

    def test_file_path_uses_parent_directory(self, binary_xml_directory):
        """A data-file path should resolve format metadata from its parent."""
        fiber_io = XMLBinaryV1()
        raw_path = next(binary_xml_directory.glob("*.raw"))
        assert fiber_io.get_format(raw_path) == (fiber_io.name, fiber_io.version)

    def test_remote_directory_upath(self, remote_binary_xml_directory):
        """Remote UPath directories should be format-detectable."""
        fiber_io = XMLBinaryV1()
        assert fiber_io.get_format(remote_binary_xml_directory) == (
            fiber_io.name,
            fiber_io.version,
        )


class TestScanContents:
    """Test scanning contents of xml binary directory."""

    def test_scan_keys_select_one_member(self, binary_xml_directory):
        """Each public scan key reloads exactly its original directory member."""
        summaries = dc.scan(binary_xml_directory)
        keys = [item.source_patch_key for item in summaries]
        assert keys == ["0", "1"]
        for summary in summaries:
            result = dc.read(
                binary_xml_directory, source_patch_key=summary.source_patch_key
            )
            assert len(result) == 1
            patch = result[0]
            assert patch.attrs.origin_id == summary.attrs.origin_id
            assert patch.summary.coords == summary.coords
            np.testing.assert_array_equal(
                patch.data, np.arange(10_000, dtype="uint16").reshape(1000, 10)
            )

    @pytest.mark.parametrize("survivors", [0, 1, 2])
    @pytest.mark.parametrize("directory_mtime_offset", [-2, 2])
    def test_timestamp_filter_preserves_member_identity(
        self, tmp_path, monkeypatch, survivors, directory_mtime_offset
    ):
        """Incremental scans keep original keys, IDs and independently known data."""
        (tmp_path / "metadata.xml").write_text(metadata)
        expected = {}
        for index in range(3):
            data = np.arange(10_000, dtype="uint16").reshape(1000, 10)
            data = data + index * 20_000
            path = tmp_path / f"DAS_20240530T01150{index}_000000Z.raw"
            path.write_bytes(data.tobytes())
            start = np.datetime64("2024-05-30T01:15:00", "ns")
            expected[start + np.timedelta64(index, "s")] = data

        # Assign mtimes in the reader's order, without assuming glob sorting.
        members = XMLBinaryV1().get_metadata(tmp_path)
        timestamp = 1_700_000_000
        for index, member in enumerate(members):
            mtime = timestamp + (1 if index >= 3 - survivors else -1)
            os.utime(member._source.path, (mtime, mtime))
        # Writing existing raw files need not change their directory's mtime.
        directory_mtime = timestamp + directory_mtime_offset
        os.utime(tmp_path, (directory_mtime, directory_mtime))

        def no_samples(*args, **kwargs):
            pytest.fail("Scanning directory metadata must not read sample arrays")

        derived_ordinals = []
        origin_id_for = io_core.origin_id_for

        def record_derivation(*args, **kwargs):
            derived_ordinals.append(kwargs["ordinal"])
            return origin_id_for(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(XMLBinaryV1, "read_array", no_samples)
            full = dc.scan(tmp_path)
            context.setattr(io_core, "origin_id_for", record_derivation)
            selected = dc.scan(tmp_path, timestamp=timestamp)
            payloads = dc.scan_payloads(tmp_path, timestamp=timestamp)
            direct = XMLBinaryV1().scan(tmp_path, timestamp=timestamp)

        assert len(selected) == len(payloads) == len(direct) == survivors
        assert derived_ordinals == list(range(3 - survivors, 3)) * 2
        original = full[len(full) - survivors :]
        assert [item.source_patch_key for item in selected] == [
            item.source_patch_key for item in original
        ]
        assert [item.attrs.origin_id for item in selected] == [
            item.attrs.origin_id for item in original
        ]
        assert [item.summary for item in payloads] == selected
        for summary in selected:
            patches = dc.read(
                summary.source_path, source_patch_key=summary.source_patch_key
            )
            assert len(patches) == 1
            patch = patches[0]
            assert patch.attrs.origin_id == summary.attrs.origin_id
            assert patch.summary.coords == summary.coords
            np.testing.assert_array_equal(
                patch.data, expected[summary.coords["time"].min]
            )

    def test_two_patches(self, binary_xml_directory):
        """Ensure the default test case has two patches."""
        fiber = XMLBinaryV1()
        out = fiber.scan(binary_xml_directory)
        assert len(out) == 2

    def test_scan_file_path_uses_parent_directory(self, binary_xml_directory):
        """Scanning a raw file should resolve metadata from the parent dir."""
        fiber = XMLBinaryV1()
        raw_path = next(binary_xml_directory.glob("*.raw"))
        out = fiber.scan(raw_path)
        assert len(out) == 2

    def test_mtime(self, binary_xml_directory):
        """Ensure scan returns contents appropriate to mtime."""
        # With no time specified, all contents should be scanned.
        scan1 = dc.scan_to_df(binary_xml_directory)
        assert len(scan1) == 2
        # When time is specified only those after should be returned.
        mtime = Path(binary_xml_directory).stat().st_mtime
        scan2 = dc.scan(binary_xml_directory, timestamp=mtime + 50)
        assert not len(scan2)
        scan3 = dc.scan(binary_xml_directory, timestamp=mtime - 50)
        assert len(scan3) == 2

    def test_direct_scan_filters_all_by_mtime(self, binary_xml_directory):
        """The FiberIO scan contract returns empty after filtering every file."""
        fiber = XMLBinaryV1()
        newest = max(
            path.stat().st_mtime for path in binary_xml_directory.glob("*.raw")
        )
        assert fiber.scan(binary_xml_directory, timestamp=newest + 1) == []

    def test_remote_directory(self, remote_binary_xml_directory):
        """Remote XMLBinary directories should be scannable."""
        fiber = XMLBinaryV1()
        out = fiber.scan(remote_binary_xml_directory)
        assert len(out) == 2

    def test_remote_directory_timestamp(self, remote_binary_xml_directory):
        """Timestamp-filtered remote directory scans should not crash."""
        out = dc.scan(remote_binary_xml_directory, timestamp=0)
        assert len(out) == 2

    def test_remote_file_path_uses_parent_directory(self, remote_binary_xml_directory):
        """Remote raw file scans should resolve metadata from the parent dir."""
        fiber = XMLBinaryV1()
        raw_path = next(remote_binary_xml_directory.glob("*.raw"))
        out = fiber.scan(raw_path)
        assert len(out) == 2


class TestRead:
    """Tests for reading contents into Patches."""

    @pytest.mark.parametrize("size_change", [-2, 1, 2])
    @pytest.mark.parametrize("windows", [(), ((0, 5),)])
    def test_reject_wrong_file_size(
        self, binary_xml_directory, tmp_path, size_change, windows
    ):
        """Full and bounded reads reject both trailing and missing raw bytes."""
        shutil.copy2(binary_xml_directory / "metadata.xml", tmp_path)
        source = next(binary_xml_directory.glob("*.raw"))
        path = tmp_path / source.name
        data = source.read_bytes()
        data = data[:size_change] if size_change < 0 else data + bytes(size_change)
        path.write_bytes(data)
        with pytest.raises(InvalidFiberFileError, match="exactly"):
            XMLBinaryV1().read_array(path, windows)

    def test_directory_order_cannot_swap_samples(
        self, binary_xml_directory, tmp_path, monkeypatch
    ):
        """Metadata and samples stay paired when listings return different orders."""
        shutil.copytree(binary_xml_directory, tmp_path, dirs_exist_ok=True)
        paths = sorted(tmp_path.glob("*.raw"))
        expected = {}
        for index, path in enumerate(paths):
            data = np.full(10000, index + 1, dtype="uint16")
            path.write_bytes(data.tobytes())
            patch = XMLBinaryV1().read(path)[0]
            expected[patch.get_coord("time").min()] = data.reshape(patch.shape)
        path_type = type(UPath(tmp_path))
        glob = path_type.glob
        calls = []

        def alternating_glob(path, pattern, **kwargs):
            items = list(glob(path, pattern, **kwargs))
            if str(path) == str(tmp_path) and pattern == "*.raw":
                calls.append(True)
                items = sorted(items, reverse=len(calls) % 2 == 0)
            return iter(items)

        monkeypatch.setattr(path_type, "glob", alternating_glob)
        patches = dc.read(tmp_path, "XMLBinary", "1")
        assert len(patches) == 2
        assert len(calls) >= 2
        assert [p._source.key for p in patches] == ["0", "1"]
        for patch in patches:
            np.testing.assert_array_equal(
                patch.data, expected[patch.get_coord("time").min()]
            )

    def test_read_single_file(self, binary_xml_directory):
        """Ensure we can read a single binary file in the directory."""
        fiber_io = XMLBinaryV1()
        path = next(binary_xml_directory.glob("*.raw"))
        out = fiber_io.read(path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) == 1

    @pytest.mark.parametrize("single", [False, True])
    def test_reader_trim_records_selection_once(
        self, binary_xml_directory, tmp_path, single
    ):
        """Directory and single-file sources match direct selection provenance."""
        directory = binary_xml_directory
        if single:
            directory = tmp_path
            shutil.copy2(binary_xml_directory / "metadata.xml", directory)
            shutil.copy2(next(binary_xml_directory.glob("*.raw")), directory)
        spool = dc.spool(directory).update()
        source = spool[0]
        time = source.get_coord("time")
        bounds = (time.min() + 10 * time.step, time.min() + 20 * time.step)
        expected = source.select(time=bounds)
        out = spool.select(time=bounds)[0]
        assert out.attrs.data_id == expected.attrs.data_id
        assert out.attrs.history == expected.attrs.history
        assert np.array_equal(out.data, expected.data)
        assert out.coords == expected.coords

    def test_read_whole_directory(self, binary_xml_directory):
        """Ensure the simple path can be read by fiber io instance."""
        fiber_io = XMLBinaryV1()
        spool = fiber_io.read(binary_xml_directory)
        assert len(spool) == 2
        for patch in spool:
            assert isinstance(patch, dc.Patch)
            # test time axis
            time_coord = patch.get_coord("time")
            time_step = to_float(time_coord.step)
            assert np.allclose(time_step, 1 / sampling_rate)
            # Expecting duration to be 1 second for each file.
            duration = (time_coord.max() - time_coord.min()) / dc.to_timedelta64(1)
            assert np.isclose(duration + time_step, 1)

    def test_doesnt_reindex(self, binary_xml_directory):
        """Indexing twice shouldn't double length of spool."""
        spool = dc.spool(binary_xml_directory)
        assert len(spool) == 2
        # Before, the spool would re-index exactly once doubling size.
        # We need to ensure this doesn't happen.
        new_spool = spool.update()
        assert len(new_spool) == 2

    def test_read_with_other_files(self, binary_xml_with_other_files):
        """Ensure other files are also included/indexed."""
        spool = dc.spool(binary_xml_with_other_files).update()
        assert len(spool) == 5
        for patch in spool:
            assert isinstance(patch, dc.Patch)

    def test_read_empty_data_dir(self, xml_directory_no_data):
        """Try reading the directory without data."""
        fiberio = XMLBinaryV1()
        path = Path(xml_directory_no_data)
        out = fiberio.read(path)
        assert not len(out)

    def test_read_remote_directory(self, remote_binary_xml_directory):
        """Remote XMLBinary directories should read into patches."""
        fiber_io = XMLBinaryV1()
        spool = fiber_io.read(remote_binary_xml_directory)
        assert len(spool) == 2
        assert all(isinstance(patch, dc.Patch) for patch in spool)

    def test_read_remote_single_file(self, remote_binary_xml_directory):
        """Remote raw-file inputs should resolve and read through the parent dir."""
        fiber_io = XMLBinaryV1()
        path = next(remote_binary_xml_directory.glob("*.raw"))
        out = fiber_io.read(path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) == 1


class TestStorageProvenance:
    """Where the bytes live belongs to the spool, not to patch attrs."""

    def test_read_omits_provenance(self, binary_xml_directory):
        """A read patch carries no path, format, or version attr."""
        patch = dc.read(binary_xml_directory)[0]
        names = set(dict(patch.attrs))
        assert not names & set(STORAGE_PROVENANCE_ATTRS)

    def test_scan_omits_provenance(self, binary_xml_directory):
        """Neither does a scanned summary's attrs."""
        summary = dc.scan(binary_xml_directory)[0]
        names = set(dict(summary.attrs))
        assert not names & set(STORAGE_PROVENANCE_ATTRS)


class TestDirectorySource:
    """A directory's arrays are pinned by member path, which only read holds."""

    def test_source_not_loadable(self, binary_xml_directory):
        """Reads still work, and the source names the directory alone."""
        patch = dc.read(binary_xml_directory)[0]
        source = patch._source
        assert patch.shape and patch.data.size
        assert source.format == XMLBinaryV1().name
        assert not source.loadable
        with pytest.raises(ParameterError, match="does not say enough"):
            source.load()


class TestDirectoryUnitStaleCheck:
    """A directory source is compared the way the index recorded it."""

    @pytest.fixture
    def indexed_unit(self, binary_xml_directory, tmp_path):
        """An indexed spool over a copy of the archive, and the copy."""
        unit = tmp_path / "root" / "xb"
        shutil.copytree(binary_xml_directory, unit)
        # Members copied within one clock tick would swap to the same manifest.
        for num, path in enumerate(sorted(unit.glob("*.raw"))):
            moved = path.stat().st_mtime_ns + num * 10**9
            os.utime(path, ns=(moved, moved))
        return dc.spool(tmp_path / "root").update(), unit

    def _resolver(self, spool):
        """The plan resolver of a whole-spool merge."""
        return spool.chunk(time=None)._catalog.resolver

    def test_an_untouched_unit_is_what_the_index_recorded(self, indexed_unit):
        """Its manifest, not its members' sizes, is what the index holds."""
        spool, unit = indexed_unit
        resolver = self._resolver(spool)
        rows = resolver.member_rows
        assert set(rows["source_path"]) == {str(unit)}
        assert resolver._sources_unchanged(rows)

    def test_a_changed_member_abandons_the_recipe(self, indexed_unit):
        """A member rewritten in place changes the unit's manifest."""
        spool, unit = indexed_unit
        resolver = self._resolver(spool)
        rows = resolver.member_rows
        path = sorted(unit.glob("*.raw"))[0]
        (np.arange(1000 * 10, dtype="uint16") + 1).tofile(path)
        moved = path.stat().st_mtime_ns + 10**9
        os.utime(path, ns=(moved, moved))
        assert not resolver._sources_unchanged(rows)

    def test_a_renamed_member_abandons_the_recipe(self, indexed_unit):
        """Which file holds which samples is part of what was recorded."""
        spool, unit = indexed_unit
        resolver = self._resolver(spool)
        rows = resolver.member_rows
        first, second = sorted(unit.glob("*.raw"))
        spare = unit / "spare.raw"
        # a swap: every member keeps its size and its modification time
        stats = [os.stat(x) for x in (first, second)]
        first.rename(spare)
        second.rename(first)
        spare.rename(second)
        # A rename need not keep an mtime everywhere; put the swapped ones back.
        for path, status in zip((second, first), stats):
            os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
        assert not resolver._sources_unchanged(rows)

    def test_a_merge_reads_the_same_patch(self, indexed_unit):
        """However the members load, the merged patch is the same one."""
        spool, _ = indexed_unit
        merged = spool.chunk(time=None)[0]
        expected = np.concatenate([x.data for x in spool], axis=0)
        assert np.array_equal(merged.data, expected)
