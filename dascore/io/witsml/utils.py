"""Utilities for reading WITSML 1.4 DTS logs."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from xml.etree import ElementTree

import numpy as np

import dascore as dc
from dascore.core.coords import get_coord
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import get_gridded_coord, parse_units, should_snap

_W = {"w": "http://www.witsml.org/schemas/1series"}
NS = f"{{{_W['w']}}}"
# Enough of the file to hold the XML declaration and the root's start tag.
_HEAD_BYTES = 4096
# Header elements, stated on the log, its blockInfo or its logData.
_HEADER = (
    "dataDelimiter",
    "endDateTimeIndex",
    "endIndex",
    "indexCurve",
    "indexType",
    "maxDateTimeIndex",
    "minDateTimeIndex",
    "mnemonicList",
    "nullValue",
    "startDateTimeIndex",
    "startIndex",
    "unitList",
)
_LENGTH_INDEX = ("length", "length along fiber")
# DTS logs write Celsius as "C", which would parse as coulombs.
_UNITS = {"c": "°C", "degc": "°C"}


def get_root(resource) -> tuple[str, str]:
    """Return the root element's tag and declared version from the file's head."""
    head = BytesIO(resource.read(_HEAD_BYTES))
    try:
        for _, elem in ElementTree.iterparse(head, events=("start",)):
            return elem.tag, elem.get("version", "")
    except ElementTree.ParseError:
        pass
    return "", ""


def _text(elem, path: str) -> str:
    """Return the stripped text at a WITSML path, blank when absent."""
    return elem.findtext(path, "", _W).strip()


def _children(elem) -> dict[str, str]:
    """Map an element's children's local names to their stripped text."""
    return {x.tag.removeprefix(NS): (x.text or "").strip() for x in elem}


def _get_units(raw: str) -> str:
    """Return the dascore form of a WITSML unit, blank for "none" or no unit."""
    raw = raw.strip()
    return parse_units(_UNITS.get(raw.lower(), raw)) or ""


@dataclass
class WitsmlLog:
    """The parts of one WITSML log a DTS reader uses."""

    where: str
    header: dict[str, str]
    columns: list[str]
    units: dict[str, str]
    nulls: dict[str, str]
    rows: list[list[str]]

    @property
    def index(self) -> str:
        """The mnemonic of the index (distance) curve."""
        return self.get("indexCurve")

    @property
    def curves(self) -> list[str]:
        """The mnemonics of the data curves, in column order."""
        return [x for x in self.columns if x != self.index]

    def get(self, *names: str) -> str:
        """Return the value of the first name the header states; raise if none."""
        for name in names:
            if value := self.header[name]:
                return value
        msg = f"{self.where} states none of {names}."
        raise InvalidFiberFileError(msg)

    def column(self, name: str) -> np.ndarray:
        """Return one curve's values; blanks and its nullValue are NaN."""
        pos = self.columns.index(name)
        try:
            values = np.array(
                [float(x) if x.strip() else np.nan for x in (r[pos] for r in self.rows)]
            )
            if null := self.nulls[name]:
                values[values == float(null)] = np.nan
        except ValueError as exc:
            msg = f"Curve {name} of {self.where} holds a value that is not a number."
            raise InvalidFiberFileError(msg) from exc
        return values


def _get_columns(log, where: str) -> tuple[list[str], list[str]]:
    """
    Return the column mnemonics and the units listed beside them.

    1.4.1.1 lists both in ``logData``; 1.4.0.0 places each curve, by its
    ``logCurveInfo`` uid, at a ``blockCurveInfo/columnIndex``.
    """
    if names := _text(log, "w:logData/w:mnemonicList"):
        units = _text(log, "w:logData/w:unitList")
        return [x.strip() for x in names.split(",")], units.split(",") if units else []
    mnemonics = {
        x.get("uid"): _text(x, "w:mnemonic") for x in log.iterfind("w:logCurveInfo", _W)
    }
    positions = {}
    for elem in log.iterfind("w:blockInfo/w:blockCurveInfo", _W):
        info = _children(elem)
        try:
            positions[int(info["columnIndex"])] = mnemonics[info["curveId"]]
        except (KeyError, ValueError) as exc:
            msg = f"A blockCurveInfo of {where} names no column or no curve: {info}"
            raise InvalidFiberFileError(msg) from exc
    return [positions[x] for x in sorted(positions)], []


def read_log(resource) -> WitsmlLog:
    """
    Parse a file's one WITSML log, keeping each data row's fields as text.

    Values are converted per curve, so a scan or a one-curve read converts
    only what it needs.
    """
    where = str(getattr(resource, "name", "the resource"))
    try:
        root = ElementTree.parse(resource).getroot()
    except ElementTree.ParseError as exc:
        raise InvalidFiberFileError(f"{where} is not well-formed XML: {exc}") from exc
    logs = root.findall("w:log", _W) + root.findall("w:wellLog", _W)
    log = logs[0] if len(logs) == 1 else None
    if (
        log is None
        or len(log.findall("w:logData", _W)) != 1
        or len(log.findall("w:blockInfo", _W)) > 1
    ):
        msg = f"{where} must hold one log with one block of data."
        raise InvalidFiberFileError(msg)
    places = ("w:", "w:blockInfo/w:", "w:logData/w:")
    header = {
        name: next((x for p in places if (x := _text(log, p + name))), "")
        for name in _HEADER
    }
    if header["indexType"] and header["indexType"].lower() not in _LENGTH_INDEX:
        msg = f"{where} is not indexed by length along the fiber."
        raise InvalidFiberFileError(msg)
    curves = {
        _text(x, "w:mnemonic"): _children(x) for x in log.iterfind("w:logCurveInfo", _W)
    }
    columns, listed = _get_columns(log, where)
    if "" in columns or len(set(columns)) < len(columns):
        msg = f"{where} names a data column twice or not at all: {columns}"
        raise InvalidFiberFileError(msg)
    units = {x: curves.get(x, {}).get("unit", "") for x in columns}
    if listed and len(listed) != len(columns):
        msg = f"{where} lists {len(listed)} units for {len(columns)} columns."
        raise InvalidFiberFileError(msg)
    units.update(zip(columns, listed))
    nulls = {
        x: curves.get(x, {}).get("nullValue") or header["nullValue"] for x in columns
    }
    sep = header["dataDelimiter"] or ","
    rows = [str(x.text or "").split(sep) for x in log.iterfind("w:logData/w:data", _W)]
    out = WitsmlLog(where, header, columns, units, nulls, rows)
    if missing := (set(curves) | {out.index}) - set(columns):
        msg = f"{where} declares curves {sorted(missing)} its data columns lack."
        raise InvalidFiberFileError(msg)
    if not rows:
        raise InvalidFiberFileError(f"{where} holds no data rows.")
    for num, row in enumerate(rows):
        if len(row) != len(columns):
            msg = (
                f"Row {num} of {where} holds {len(row)} values but the file "
                f"declares {len(columns)} columns."
            )
            raise InvalidFiberFileError(msg)
    return out


def get_coords(log: WitsmlLog, snap) -> dc.CoordManager:
    """
    Return the (distance, time) coords every curve of the log shares.

    The interrogator samples distance at a constant step, but the header's
    step and the stored index values are both rounded, so the snapped grid
    spans the header's start to end index over the row count, or the stored
    ends where the header states none. Time holds one label, the start, and
    ``sample_span`` how long the reading ran.
    """
    units = _get_units(log.units[log.index])
    if not should_snap(snap, "distance"):
        distance = get_coord(data=log.column(log.index), units=units, snap=False)
    elif log.header["startIndex"] and log.header["endIndex"]:
        start, end = float(log.header["startIndex"]), float(log.header["endIndex"])
        grid = np.linspace(start, end, len(log.rows))
        distance = get_gridded_coord(grid, units=units)
    else:
        distance = get_gridded_coord(log.column(log.index), units=units)
    start = dc.to_datetime64(log.get("startDateTimeIndex", "minDateTimeIndex"))
    end = dc.to_datetime64(log.get("endDateTimeIndex", "maxDateTimeIndex"))
    coords = {
        "distance": distance,
        "time": get_coord(data=np.array([start])),
        "sample_span": ("time", get_coord(data=np.array([end - start]))),
    }
    return dc.get_coord_manager(coords, dims=("distance", "time"))


def get_attrs(log: WitsmlLog, curve: str) -> dict:
    """Return one curve's patch attrs; only TMP states a data type."""
    return {
        "data_category": "DTS",
        "data_type": "temperature" if curve == "TMP" else "",
        "data_units": _get_units(log.units[curve]),
        "tag": curve,
    }
