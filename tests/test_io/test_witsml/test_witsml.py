"""Tests for WITSML 1.4 DTS logs, built from trimmed copies of real files."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.witsml import WITSMLV1400, WITSMLV1411

_NS = 'xmlns="http://www.witsml.org/schemas/1series"'
_CUSTOM = """  <customData>
   <acquisitionTime>150.181</acquisitionTime>
   <SystemSettings softwareVersion="6.0"><Card>0</Card></SystemSettings>
  </customData>
"""
_START = "2016-03-17T21:05:51.000Z"


def _curve(name, unit=""):
    """Return one logCurveInfo element."""
    unit = f"<unit>{unit}</unit>" if unit else ""
    return f"""  <logCurveInfo uid="{name}">
   <mnemonic>{name}</mnemonic>{unit}<typeLogData>double</typeLogData>
  </logCurveInfo>
"""


def brady_like(start=_START, end="2016-03-17T21:08:21.181Z"):
    """
    A 1.4.1.1 log as the Brady 2016 interrogator wrote it: a mnemonicList
    with stray spaces, customData after the data, and a header end index
    more precise than the stored index values.
    """
    names = ("LAF", "ST", "AST", "REV-ST", "REV-AST", "TMP")
    distances = ("-2", "-0.984", "0.032", "1.048", "2.064", "3.08")
    rows = "".join(
        f"<data>\n{x},{10 + i},{20 + i},{30 + i},{40 + i},{5.5 + i}\n</data>\n"
        for i, x in enumerate(distances)
    )
    curves = "".join(_curve(x, {"LAF": "m", "TMP": "C"}.get(x, "")) for x in names)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<logs version="1.4.1.1" {_NS}>
 <log uid="measurement">
  <nameWell>Ultima 1</nameWell>
  <startIndex uom="m">-2</startIndex>
  <endIndex uom="m">3.0795</endIndex>
  <stepIncrement uom="m">1.016</stepIncrement>
  <startDateTimeIndex>{start}</startDateTimeIndex>
  <endDateTimeIndex>{end}</endDateTimeIndex>
  <indexCurve>LAF</indexCurve>
{curves}  <logData>
   <mnemonicList>LAF, ST, AST, REV-ST, REV-AST ,TMP</mnemonicList>
   <unitList>m, none, none, none, none, degC</unitList>
   {rows}  </logData>
{_CUSTOM} </log>
</logs>
"""


def cranfield_like():
    """
    A 1.4.0.0 wellLog: customData first, columns given only by
    blockCurveInfo, and Celsius written "C".
    """
    distances = ("-80.2503", "-80.1238", "-79.9973", "-79.8709", "-79.7444")
    rows = "".join(
        f'<data uid="measurement">{x},{100 + i},{200 + i},{50 + i}</data>\n'
        for i, x in enumerate(distances)
    )
    curves = "".join(
        _curve(x, {"LAF": "m", "TMP": "C"}.get(x, ""))
        for x in ("LAF", "ST", "AST", "TMP")
    )
    blocks = "".join(
        f"""   <blockCurveInfo uid="measurement - {x}">
    <curveId>{x}</curveId><columnIndex>{i}</columnIndex>
    <minIndex uom="m">-80.2503</minIndex>
   </blockCurveInfo>
"""
        for i, x in ((4, "TMP"), (1, "LAF"), (3, "AST"), (2, "ST"))
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<wellLogs version="1.4.0.0" {_NS}>
 <wellLog>
  <minIndex uom="m">-80.2503</minIndex>
  <maxIndex uom="m">-79.7444</maxIndex>
  <minDateTimeIndex>2015-01-25T00:16:28.468Z</minDateTimeIndex>
  <maxDateTimeIndex>2015-01-25T00:16:58.929Z</maxDateTimeIndex>
{_CUSTOM}{curves}  <blockInfo uid="measurement">
   <startIndex uom="m">-80.25033</startIndex>
   <endIndex uom="m">-79.74437</endIndex>
   <stepIncrement uom="m">0.12649</stepIncrement>
   <indexCurve columnIndex="1">LAF</indexCurve>
{blocks}  </blockInfo>
  <logData>{rows}</logData>
 </wellLog>
</wellLogs>
"""


@pytest.fixture(scope="module")
def brady_path(tmp_path_factory):
    """A Brady-like 1.4.1.1 file."""
    path = tmp_path_factory.mktemp("witsml") / "brady.xml"
    path.write_text(brady_like())
    return path


@pytest.fixture(scope="module")
def cranfield_path(tmp_path_factory):
    """A Cranfield-like 1.4.0.0 file."""
    path = tmp_path_factory.mktemp("witsml") / "cranfield.xml"
    path.write_text(cranfield_like())
    return path


def _by_tag(spool):
    """Map each patch's tag to the patch."""
    return {x.attrs.tag: x for x in spool}


class TestRead:
    """Each variant reads one patch per curve."""

    def test_brady_like(self, brady_path):
        """Five curves, units from the unitList, and the stated times."""
        assert dc.get_format(brady_path) == ("WITSML", "1.4.1.1")
        patches = _by_tag(dc.read(brady_path))
        assert list(patches) == ["ST", "AST", "REV-ST", "REV-AST", "TMP"]
        tmp, rev_ast = patches["TMP"], patches["REV-AST"]
        assert tmp.dims == ("distance", "time")
        assert tmp.shape == (6, 1)
        assert tmp.data[[0, -1], 0].tolist() == [5.5, 10.5]
        assert rev_ast.data[[0, -1], 0].tolist() == [40, 45]
        assert tmp.attrs.data_type == "temperature"
        assert tmp.attrs.data_category == rev_ast.attrs.data_category == "DTS"
        assert dc.get_quantity(tmp.attrs.data_units) == dc.get_quantity("degC")
        assert rev_ast.attrs.data_type == ""
        assert not rev_ast.attrs.data_units
        time, span = tmp.coords.get_array("time"), tmp.coords.get_array("sample_span")
        assert time == [np.datetime64("2016-03-17T21:05:51")]
        assert span == [np.timedelta64(150_181, "ms")]

    def test_cranfield_like(self, cranfield_path):
        """Columns follow columnIndex; TMP's "C" is Celsius."""
        assert dc.get_format(cranfield_path) == ("WITSML", "1.4.0.0")
        patches = _by_tag(dc.read(cranfield_path))
        assert list(patches) == ["ST", "AST", "TMP"]
        assert patches["ST"].data[:, 0].tolist() == [100, 101, 102, 103, 104]
        assert patches["TMP"].data[[0, -1], 0].tolist() == [50, 54]
        units = dc.get_quantity(patches["TMP"].attrs.data_units)
        assert units == dc.get_quantity("degC")
        assert patches["TMP"].coords.get_array("time")[0] == np.datetime64(
            "2015-01-25T00:16:28.468"
        )
        span = patches["AST"].coords.get_array("sample_span")[0]
        assert span == np.timedelta64(30_461, "ms")


class TestDistance:
    """Distance runs over the header's ends by default, as stored on request."""

    @pytest.mark.parametrize(
        ("name", "ends"),
        [("brady_path", (-2, 3.0795)), ("cranfield_path", (-80.25033, -79.74437))],
    )
    def test_snapped(self, name, ends, request):
        """The default grid is even and spans the header's start to end."""
        patch = dc.read(request.getfixturevalue(name))[0]
        coord = patch.get_coord("distance")
        assert coord.evenly_sampled
        assert np.allclose((coord.min(), coord.max()), ends, rtol=1e-12)
        assert dc.get_quantity(coord.units) == dc.get_quantity("m")

    def test_snap_false(self, brady_path):
        """The stored index labels are kept exactly."""
        coord = dc.read(brady_path, snap=False)[0].get_coord("distance")
        expected = [-2, -0.984, 0.032, 1.048, 2.064, 3.08]
        assert coord.values.tolist() == expected


class TestSelect:
    """Selections return the right part of the keyed curve."""

    def test_spool_select(self, brady_path):
        """A distance range trims every curve alike."""
        spool = dc.spool(brady_path).select(distance=(0, 2.5))
        patches = _by_tag(spool)
        assert len(patches) == 5
        assert patches["TMP"].data[:, 0].tolist() == [7.5, 8.5, 9.5]

    def test_read_key(self, cranfield_path):
        """A key reads only its curve."""
        out = dc.read(cranfield_path, source_patch_key="AST", distance=(-80.0, None))
        assert len(out) == 1
        assert out[0].data[:, 0].tolist() == [202, 203, 204]


class TestInvalid:
    """Files whose header and data disagree are refused, naming the file."""

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("40,5.5\n", "40\n"),
            ("REV-AST ,TMP", "OTHER ,TMP"),
            (f"<startDateTimeIndex>{_START}</startDateTimeIndex>", ""),
            ("</logs>", ""),
        ],
    )
    def test_invalid(self, tmp_path, old, new):
        """A short row, an unlisted curve, no start time, truncated XML."""
        path = tmp_path / "bad.xml"
        text = brady_like()
        assert old in text
        path.write_text(text.replace(old, new))
        with pytest.raises(InvalidFiberFileError, match=r"bad\.xml"):
            WITSMLV1411().scan(path)

    def test_null_value(self, tmp_path):
        """A curve's nullValue and blank fields read as NaN."""
        path = tmp_path / "nulls.xml"
        text = brady_like().replace(",11,", ",-999.25,").replace(",22,", ",,")
        null = "<mnemonic>ST</mnemonic><nullValue>-999.25</nullValue>"
        path.write_text(text.replace("<mnemonic>ST</mnemonic>", null))
        patches = _by_tag(dc.read(path))
        assert np.isnan(patches["ST"].data[:, 0]).tolist() == [0, 1, 0, 0, 0, 0]
        assert np.isnan(patches["AST"].data[:, 0]).tolist() == [0, 0, 1, 0, 0, 0]


class TestFormat:
    """Detection claims only the WITSML objects and versions it reads."""

    @pytest.mark.parametrize(
        "text",
        [
            '<?xml version="1.0"?>\n<Metadata><Zone>1</Zone></Metadata>',
            f'<wells version="1.4.1.1" {_NS}><well/></wells>',
            f'<logs version="1.3.1.1" {_NS}><log/></logs>',
            f'<logs version="1.4.0.0" {_NS}><log/></logs>',
            '<logs version="1.4.1.1"><log/></logs>',
            "not xml at all",
        ],
    )
    def test_not_claimed(self, tmp_path, text):
        """Other XML, other WITSML objects or versions, no namespace, non-XML."""
        path = tmp_path / "other.xml"
        path.write_text(text)
        assert WITSMLV1411().get_format(path) is False
        assert WITSMLV1400().get_format(path) is False


class TestSpool:
    """Scans of one fiber at different times."""

    def test_chunk_keeps_curves_apart(self, tmp_path):
        """Chunking along time never merges different curves."""
        times = ("2016-03-17T21:08:21.181Z", "2016-03-17T21:10:51.362Z")
        (tmp_path / "a.xml").write_text(brady_like())
        (tmp_path / "b.xml").write_text(brady_like(*times))
        chunked = _by_tag(dc.spool(tmp_path).update().chunk(time=None))
        assert sorted(chunked) == ["AST", "REV-AST", "REV-ST", "ST", "TMP"]
        assert np.all(chunked["TMP"].data[0] == 5.5)
