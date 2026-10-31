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

NS = "{http://www.witsml.org/schemas/1series}"
# Enough of the file to hold the XML declaration and the root's start tag.
_HEAD_BYTES = 4096
# Header elements read from the log, its blockInfo or its logData.
_HEADER = frozenset(
    {
        "endDateTimeIndex",
        "endIndex",
        "indexCurve",
        "maxDateTimeIndex",
        "maxIndex",
        "minDateTimeIndex",
        "minIndex",
        "mnemonicList",
        "startDateTimeIndex",
        "startIndex",
        "unitList",
    }
)
_HEADER_PARENTS = frozenset({"log", "wellLog", "blockInfo", "logData"})
# DTS logs write Celsius as "C", which pint would read as coulombs.
_UNITS = {"c": "°C", "degc": "°C", "none": ""}


def get_root(resource) -> tuple[str, str]:
    """Return the root element's tag and declared version from the file's head."""
    head = BytesIO(resource.read(_HEAD_BYTES))
    try:
        for _, elem in ElementTree.iterparse(head, events=("start",)):
            return elem.tag, elem.get("version", "")
    except ElementTree.ParseError:
        pass
    return "", ""


def _children(elem) -> dict[str, str]:
    """Map an element's children's local names to their stripped text."""
    return {x.tag.removeprefix(NS): (x.text or "").strip() for x in elem}


def _get_units(raw: str) -> str:
    """Return the dascore form of a WITSML unit, blank when it states none."""
    raw = raw.strip()
    return parse_units(_UNITS.get(raw.lower(), raw)) or ""


@dataclass
class WitsmlLog:
    """The parts of one WITSML log a DTS reader uses."""

    where: str
    header: dict[str, str]
    index: str
    columns: list[str]
    units: dict[str, str]
    nulls: dict[str, str]
    rows: list[list[str]]

    @property
    def curves(self) -> list[str]:
        """The mnemonics of the data curves, in column order."""
        return [x for x in self.columns if x != self.index]

    def get(self, *names: str) -> str:
        """Return the first header value of names the log states."""
        for name in names:
            if value := self.header.get(name):
                return value
        msg = f"{self.where} states none of {names}."
        raise InvalidFiberFileError(msg)

    def column(self, name: str) -> np.ndarray:
        """Return one curve's values; blanks and its nullValue are NaN."""
        pos = self.columns.index(name)
        values = np.array(
            [float(x) if x.strip() else np.nan for x in (r[pos] for r in self.rows)]
        )
        if null := self.nulls.get(name):
            values[values == float(null)] = np.nan
        return values


def read_log(resource) -> WitsmlLog:
    """
    Parse a WITSML log, keeping each data row's fields as text.

    Columns follow ``logData/mnemonicList`` (1.4.1.1) or, without one,
    ``blockCurveInfo/columnIndex`` (1.4.0.0). Values are converted per
    curve, so a scan or a one-curve read converts only what it needs.
    """
    where = str(getattr(resource, "name", "the resource"))
    header, curves, positions, rows, parents = {}, {}, {}, [], []
    try:
        for event, elem in ElementTree.iterparse(resource, events=("start", "end")):
            name = elem.tag.removeprefix(NS)
            if event == "start":
                parents.append(name)
                continue
            parents.pop()
            parent = parents[-1] if parents else ""
            if name == "data" and parent == "logData":
                rows.append((elem.text or "").split(","))
                elem.clear()
            elif name == "customData":
                elem.clear()
            elif name == "logCurveInfo":
                info = _children(elem)
                curves[info.get("mnemonic", "")] = info
            elif name == "blockCurveInfo":
                info = _children(elem)
                positions[int(info["columnIndex"])] = info["curveId"]
            elif name in _HEADER and parent in _HEADER_PARENTS:
                header.setdefault(name, (elem.text or "").strip())
    except ElementTree.ParseError as exc:
        raise InvalidFiberFileError(f"{where} is not well-formed XML: {exc}") from exc
    if "mnemonicList" in header:
        columns = [x.strip() for x in header["mnemonicList"].split(",")]
        listed = header.get("unitList", "").split(",")
    else:
        columns, listed = [positions[x] for x in sorted(positions)], []
    units = {x: curves.get(x, {}).get("unit", "") for x in columns}
    units.update(zip(columns, listed, strict=False))
    nulls = {x: curves.get(x, {}).get("nullValue", "") for x in columns}
    index = header.get("indexCurve") or next(iter(columns), "")
    if missing := (set(curves) | {index}) - set(columns):
        msg = f"{where} declares curves {sorted(missing)} its data columns lack."
        raise InvalidFiberFileError(msg)
    for num, row in enumerate(rows):
        if len(row) != len(columns):
            msg = (
                f"Row {num} of {where} holds {len(row)} values but the file "
                f"declares {len(columns)} columns."
            )
            raise InvalidFiberFileError(msg)
    return WitsmlLog(where, header, index, columns, units, nulls, rows)


def get_coords(log: WitsmlLog, snap) -> dc.CoordManager:
    """
    Return the (distance, time) coords every curve of the log shares.

    The interrogator fixes the spatial sampling, so the snapped distance
    grid runs from the header's start to end index over the row count; the
    header's step and the stored index values are both rounded. Time holds
    one label, the start, and ``sample_span`` how long the reading ran.
    """
    units = _get_units(log.units[log.index])
    if should_snap(snap, "distance"):
        start = float(log.get("startIndex", "minIndex"))
        end = float(log.get("endIndex", "maxIndex"))
        grid = np.linspace(start, end, len(log.rows))
        distance = get_gridded_coord(grid, units=units)
    else:
        distance = get_coord(data=log.column(log.index), units=units, snap=False)
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
