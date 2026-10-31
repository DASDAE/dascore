"""
Tests for WITSML 1.4 DTS logs, built from small synthetic logs laid out
like real files.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.witsml import WITSMLV1400, WITSMLV1411

_NS = 'xmlns="http://www.witsml.org/schemas/1series"'
_XSI = (
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:schemaLocation="http://www.witsml.org/schemas/1series '
    '../xsd_schemas/obj_log.xsd"'
)
_CUSTOM = """  <customData>
   <acquisitionTime>150.181</acquisitionTime>
   <SystemSettings softwareVersion="6.0"><Card>0</Card></SystemSettings>
  </customData>
"""
_START = "2016-03-17T21:05:51.000Z"


def _curve(name, unit="", uid=None):
    """Return one logCurveInfo element."""
    unit = f"<unit>{unit}</unit>" if unit else ""
    return f"""  <logCurveInfo uid="{uid or name}">
   <mnemonic>{name}</mnemonic>{unit}<typeLogData>double</typeLogData>
  </logCurveInfo>
"""


def _edit(text, *pairs):
    """Return text with each (old, new) pair replaced, checking old is there."""
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new)
    return text


def brady_like(start=_START, end="2016-03-17T21:08:21.181Z"):
    """
    A 1.4.1.1 log as the Brady 2016 interrogator wrote it: a mnemonicList
    with stray spaces, customData after the data, and a header end index
    more precise than the stored index values. The unitList gives ST a
    unit its logCurveInfo does not.
    """
    names = ("LAF", "ST", "AST", "REV-ST", "REV-AST", "TMP")
    distances = ("-2", "-0.984", "0.032", "1.048", "2.064", "3.08")
    rows = "".join(
        f"<data>\n{x},{10 + i},{20 + i},{30 + i},{40 + i},{5.5 + i}\n</data>\n"
        for i, x in enumerate(distances)
    )
    units = {"LAF": "m", "ST": "V", "TMP": "C"}
    curves = "".join(_curve(x, units.get(x, "")) for x in names)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<logs {_XSI} version="1.4.1.1" {_NS}>
 <log uid="measurement">
  <nameWell>Ultima 1</nameWell>
  <indexType>length</indexType>
  <startIndex uom="m">-2</startIndex>
  <endIndex uom="m">3.0795</endIndex>
  <stepIncrement uom="m">1.016</stepIncrement>
  <startDateTimeIndex>{start}</startDateTimeIndex>
  <endDateTimeIndex>{end}</endDateTimeIndex>
  <indexCurve>LAF</indexCurve>
{curves}  <logData>
   <mnemonicList>LAF, ST, AST, REV-ST, REV-AST ,TMP</mnemonicList>
   <unitList>m, mW, none, none, none, degC</unitList>
   {rows}  </logData>
{_CUSTOM} </log>
</logs>
"""


def cranfield_like():
    """
    A 1.4.0.0 wellLog: customData before the curves, columns given only by
    out-of-order blockCurveInfo naming curves by uid, blockInfo ends finer
    than min/maxIndex, and Celsius written "C".
    """
    distances = ("-80.2503", "-80.1238", "-79.9973", "-79.8709", "-79.7444")
    rows = "".join(
        f'<data uid="measurement">{x},{100 + i},{200 + i},{50 + i}</data>\n'
        for i, x in enumerate(distances)
    )
    units = {"LAF": "m", "TMP": "C"}
    uids = {"ST": "st-id"}
    curves = "".join(
        _curve(x, units.get(x, ""), uids.get(x)) for x in ("LAF", "ST", "AST", "TMP")
    )
    blocks = "".join(
        f"""   <blockCurveInfo uid="measurement - {x}">
    <curveId>{x}</curveId><columnIndex>{i}</columnIndex>
    <minIndex uom="m">-80.2503</minIndex>
   </blockCurveInfo>
"""
        for i, x in ((4, "TMP"), (1, "LAF"), (3, "AST"), (2, "st-id"))
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<wellLogs {_XSI} version="1.4.0.0" {_NS}>
 <wellLog>
  <indexType>length along fiber</indexType>
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


_BRADY, _CRANFIELD = brady_like(), cranfield_like()


@pytest.fixture(scope="module")
def brady_path(tmp_path_factory):
    """A Brady-like 1.4.1.1 file."""
    path = tmp_path_factory.mktemp("witsml") / "brady.xml"
    path.write_text(_BRADY)
    return path


@pytest.fixture(scope="module")
def cranfield_path(tmp_path_factory):
    """A Cranfield-like 1.4.0.0 file."""
    path = tmp_path_factory.mktemp("witsml") / "cranfield.xml"
    path.write_text(_CRANFIELD)
    return path


def _by_tag(spool):
    """Map each patch's tag to the patch."""
    return {x.attrs.tag: x for x in spool}


class TestRead:
    """Each variant reads one patch per curve."""

    def test_brady_like(self, brady_path):
        """Five curves, unitList units over logCurveInfo, and the stated times."""
        assert dc.get_format(brady_path) == ("WITSML", "1.4.1.1")
        spool = dc.read(brady_path)
        patches = _by_tag(spool)
        assert len(spool) == 5
        assert list(patches) == ["ST", "AST", "REV-ST", "REV-AST", "TMP"]
        tmp, rev_ast = patches["TMP"], patches["REV-AST"]
        assert tmp.dims == ("distance", "time")
        assert tmp.shape == (6, 1)
        assert tmp.data[[0, -1], 0].tolist() == [5.5, 10.5]
        assert rev_ast.data[[0, -1], 0].tolist() == [40, 45]
        assert tmp.attrs.data_type == "temperature"
        assert tmp.attrs.data_category == rev_ast.attrs.data_category == "DTS"
        assert dc.get_quantity(tmp.attrs.data_units) == dc.get_quantity("degC")
        assert dc.get_quantity(patches["ST"].attrs.data_units) == dc.get_quantity("mW")
        assert rev_ast.attrs.data_type == ""
        assert not rev_ast.attrs.data_units
        time, span = tmp.coords.get_array("time"), tmp.coords.get_array("sample_span")
        assert time == [np.datetime64("2016-03-17T21:05:51")]
        assert span == [np.timedelta64(150_181, "ms")]

    def test_cranfield_like(self, cranfield_path):
        """Columns follow columnIndex by curve uid; TMP's "C" is Celsius."""
        assert dc.get_format(cranfield_path) == ("WITSML", "1.4.0.0")
        spool = dc.read(cranfield_path)
        patches = _by_tag(spool)
        assert len(spool) == 3
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

    def test_null_value(self, tmp_path):
        """The log's nullValue, a curve's own nullValue, and blanks read as NaN."""
        path = tmp_path / "nulls.xml"
        tmp_null = "<mnemonic>TMP</mnemonic><nullValue>9.5</nullValue>"
        text = _edit(
            _BRADY,
            ("<indexCurve>", "<nullValue>-999.25</nullValue><indexCurve>"),
            ("<mnemonic>TMP</mnemonic>", tmp_null),
            (",11,", ",-999.25,"),
            (",22,", ",,"),
            (",7.5\n", ",-999.25\n"),
        )
        path.write_text(text)
        patches = _by_tag(dc.read(path))
        assert np.isnan(patches["ST"].data[:, 0]).tolist() == [0, 1, 0, 0, 0, 0]
        assert np.isnan(patches["AST"].data[:, 0]).tolist() == [0, 0, 1, 0, 0, 0]
        # TMP's own nullValue replaces the log's
        assert np.isnan(patches["TMP"].data[:, 0]).tolist() == [0, 0, 0, 0, 1, 0]

    def test_no_unit_list_and_delimiter(self, tmp_path):
        """Without a unitList curves keep their own units; dataDelimiter splits rows."""
        path = tmp_path / "plain.xml"
        text = _edit(
            _BRADY,
            ("<unitList>m, mW, none, none, none, degC</unitList>", ""),
            ("<indexCurve>", "<dataDelimiter>;</dataDelimiter><indexCurve>"),
        )
        path.write_text(
            re.sub(r"(<data>[^<]*)", lambda m: m[1].replace(",", ";"), text)
        )
        patch = _by_tag(dc.read(path))["TMP"]
        assert dc.get_quantity(patch.get_coord("distance").units) == dc.get_quantity(
            "m"
        )
        assert patch.data[[0, -1], 0].tolist() == [5.5, 10.5]


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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("brady_path", [-2, -0.984, 0.032, 1.048, 2.064, 3.08]),
            ("cranfield_path", [-80.2503, -80.1238, -79.9973, -79.8709, -79.7444]),
        ],
    )
    def test_snap_false(self, name, expected, request):
        """The stored index labels are kept exactly."""
        path = request.getfixturevalue(name)
        coord = dc.read(path, snap=False)[0].get_coord("distance")
        assert coord.values.tolist() == expected

    def test_stored_ends(self, tmp_path):
        """Without a stated start and end index, the grid spans the stored ends."""
        path = tmp_path / "no_ends.xml"
        path.write_text(
            _edit(_CRANFIELD, ("<startIndex", "<x"), ("</startIndex", "</x"))
        )
        coord = dc.read(path)[0].get_coord("distance")
        assert coord.evenly_sampled
        assert (coord.min(), coord.max()) == (-80.2503, -79.7444)


_INVALID = {
    "short-row": _edit(_BRADY, ("40,5.5\n", "40\n")),
    "unlisted-curve": _edit(_BRADY, ("REV-AST ,TMP", "OTHER ,TMP")),
    "no-start-time": _edit(
        _BRADY, (f"<startDateTimeIndex>{_START}</startDateTimeIndex>", "")
    ),
    "truncated": _edit(_BRADY, ("</logs>", "")),
    "two-logs": _edit(_BRADY, ("</logs>", "<log/></logs>")),
    "two-log-data": _edit(_BRADY, (" </log>", "<logData/></log>")),
    "no-index-curve": _edit(_BRADY, ("<indexCurve>LAF</indexCurve>", "")),
    "time-index": _edit(_BRADY, ("<indexType>length<", "<indexType>date time<")),
    "no-rows": re.sub(r"<data>.*</data>\n", "", _BRADY, flags=re.DOTALL),
    "not-a-number": _edit(_BRADY, (",11,", ",abc,")),
    "two-block-info": _edit(_CRANFIELD, ("  <logData>", "  <blockInfo/><logData>")),
    "no-column-index": _edit(_CRANFIELD, ("<columnIndex>2</columnIndex>", "")),
    "unknown-curve-id": _edit(_CRANFIELD, ("<curveId>st-id<", "<curveId>XX<")),
    "repeated-column": _edit(
        _BRADY, ("ST, AST,", "ST, ST,"), ("<mnemonic>AST<", "<mnemonic>ST<")
    ),
    "short-unit-list": _edit(_BRADY, ("m, mW, none", "m, none")),
}


class TestInvalid:
    """Malformed files are refused, naming the file."""

    @pytest.mark.parametrize("text", list(_INVALID.values()), ids=list(_INVALID))
    def test_invalid(self, tmp_path, text):
        """Each malformation raises InvalidFiberFileError naming the file."""
        path = tmp_path / "bad.xml"
        path.write_text(text)
        io = WITSMLV1400() if "wellLogs" in text else WITSMLV1411()
        with pytest.raises(InvalidFiberFileError, match=r"bad\.xml"):
            spool = io.read(path)
            [x.data for x in spool]


class TestFormat:
    """Detection claims only the WITSML objects and versions it reads."""

    @pytest.mark.parametrize(
        "text",
        [
            f'<logs version="1.3.1.1" {_NS}><log/></logs>',
            f'<logs version="1.4.0.0" {_NS}><log/></logs>',
            '<logs version="1.4.1.1"><log/></logs>',
        ],
    )
    def test_not_claimed(self, tmp_path, text):
        """Another version, a root of the other version, or no namespace."""
        path = tmp_path / "other.xml"
        path.write_text(text)
        assert WITSMLV1411().get_format(path) is False
        assert WITSMLV1400().get_format(path) is False

    def test_long_root_tag(self, tmp_path):
        """A root start tag ending thousands of bytes in is still found."""
        path = tmp_path / "long_root.xml"
        location = "../" + "schemas/" * 400 + "obj_log.xsd"
        path.write_text(_edit(_BRADY, ("../xsd_schemas/obj_log.xsd", location)))
        assert path.read_bytes().index(b'version="1.4.1.1"') > 3000
        assert dc.get_format(path) == ("WITSML", "1.4.1.1")


class TestSpool:
    """Scans of one fiber at different times."""

    def test_chunk_keeps_curves_apart(self, tmp_path):
        """Chunking along time joins each curve and never merges two curves."""
        times = ("2016-03-17T21:08:21.181Z", "2016-03-17T21:10:51.362Z")
        (tmp_path / "a.xml").write_text(_BRADY)
        (tmp_path / "b.xml").write_text(brady_like(*times))
        chunked = dc.spool(tmp_path).update().chunk(time=None)
        assert len(chunked) == 5
        assert all(x.shape == (6, 2) for x in chunked)
        patches = _by_tag(chunked)
        assert sorted(patches) == ["AST", "REV-AST", "REV-ST", "ST", "TMP"]
        assert np.all(patches["TMP"].data[0] == 5.5)
