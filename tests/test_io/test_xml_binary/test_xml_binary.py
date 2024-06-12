"""Misc. tests for xml binary."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnknownFiberFormatError
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


class TestScanContents:
    """Test scanning contents of xml binary directory."""

    def test_two_patches(self, binary_xml_directory):
        """Ensure the default test case has two patches."""
        fiber = XMLBinaryV1()
        out = fiber.scan(binary_xml_directory)
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


class TestRead:
    """Tests for reading contents into Patches."""

    def test_read_single_file(self, binary_xml_directory):
        """Ensure we can read a single binary file in the directory."""
        fiber_io = XMLBinaryV1()
        path = next(binary_xml_directory.glob("*.raw"))
        out = fiber_io.read(path)
        assert isinstance(out, dc.BaseSpool)
        assert len(out) == 1

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

    def test_simple_spool(self, binary_xml_directory):
        """Ensure the simple path can be read into a spool."""
        spool = dc.spool(binary_xml_directory).update()
        assert isinstance(spool, dc.BaseSpool)
        assert len(spool) == 2

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
