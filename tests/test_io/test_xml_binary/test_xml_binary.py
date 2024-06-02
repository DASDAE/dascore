"""Misc. tests for xml binary."""
from __future__ import annotations

import numpy as np
import pytest

from dascore.io.xml_binary.utils import read_xml_metadata

metadata = """<?xml version='1.0' encoding='utf-8'?>
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
  <OutputTemporalSamplingRate>1000</OutputTemporalSamplingRate>
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
    data_2_path = new / "DAS_20240530T011501.raw"
    with open(metadata_path, "w") as fi:
        fi.write(metadata)
    with open(data_1_path, "wb") as fi:
        ar = np.arange(1000 * 10, dtype=np.dtype("uint16"))
        ar.tofile(fi)
    with open(data_2_path, "wb") as fi:
        ar = np.arange(1000 * 10, dtype=np.dtype("uint16"))
        ar.tofile(fi)
    return new


class TestReadXMLMetadata:
    """Misc tests reading xml metadata."""

    def test_read_metadata_contents(self, binary_xml_directory):
        """Test reading xml metadata contents and metadata types."""
        metadata_path_1 = binary_xml_directory / "metadata.xml"
        metadata = read_xml_metadata(metadata_path_1)
        assert metadata is not None
        # Just test a couple of attrs, pydantic should handle the rest.
        expected_attrs = ["file_format", "date_time", "units"]
        for attr in expected_attrs:
            assert hasattr(metadata, attr)
