"""
The arithmetic of a run table, one kernel for each kind of label.

A [`NumericND`](`dascore.core.coords.NumericND`) holds its runs as a record
array and leaves every question about what a row *means* to one of the two
kernels here, picked by the coordinate's dtype:

- `TickKernel` for labels counted in whole ticks (times and integers). Sample
  ``k`` of a run is ``start + floor((offset + k * num) / den)``; the row is
  exact rational arithmetic in int64.
- `FloatKernel` for float labels. Sample ``k`` of a run is
  ``start + step * (k0 + k * stride)`` in plain float64, which is the
  expression ``start + np.arange(n) * step`` builds an axis with, or
  ``start + (k0 + k * stride) / step`` for an axis built by dividing, as
  ``np.arange(n) / rate`` is. Slicing and reversal change only the integers
  ``k0`` and ``stride``, so no label of a slice can differ from its parent's.

Both kernels use one record layout, ``(start, length, num, den, offset)``,
so a row travels through a summary, the spool index, and a file the same way
whatever it labels. A float row spells its terms in those fields: ``num``
holds the bits of the float64 ``step``, ``den`` the ``stride`` (negative
for a run which divides by its step), and ``offset`` the grid index ``k0``;
`float_terms` reads them back. In either layout ``den == 0`` marks a
*stored* run, whose labels the table holds itself, and ``start`` is then
its first label.
"""

from __future__ import annotations

import math
from contextlib import suppress
from functools import cache
from typing import Any, NoReturn, cast

import numpy as np

from dascore.exceptions import CoordError

_INT64_MAX = 2**63
# The run record's integer fields; ``start`` leads them, as a tick or a float.
_RECORD_TAIL = ("length", "num", "den", "offset")
# The content id of the array a stored run reads its labels from, empty on a
# grid row. Fixed width ascii bytes, so the table stays one numpy record and
# a run's identity travels with it.
SOURCE_ID = "source_id"
SOURCE_ID_DTYPE = "S32"
_COLUMNS = ("start", *_RECORD_TAIL, SOURCE_ID)
# Index arithmetic of a float row passes through float64, which counts
# whole numbers exactly only this far.
_FLOAT_INDEX_MAX = 2**53


@cache
def _ticked(dtype) -> bool:
    """Whether labels are whole ticks (times and integers) rather than floats."""
    return np.dtype(dtype).kind in "iuMm"


@cache
def _tick_bounds(dtype) -> tuple[int, int]:
    """
    The lowest and highest tick a coordinate of this dtype can label.

    Every tick is counted in int64, so a uint64 coordinate is bounded by
    int64's ceiling rather than its own.
    """
    nd = np.dtype(dtype)
    info = np.iinfo(cast("Any", np.int64 if nd.kind in "mM" else nd))
    return max(int(info.min), -_INT64_MAX), min(int(info.max), _INT64_MAX - 1)


@cache
def _record_dtype_cached(start: str) -> np.dtype:
    """One of the two record layouts a run table can have."""
    fields = [("start", start), *[(x, "i8") for x in _RECORD_TAIL]]
    return np.dtype([*fields, (SOURCE_ID, SOURCE_ID_DTYPE)])


def _record_dtype(dtype) -> np.dtype:
    """The record of one run; only the start field varies with the coordinate."""
    return _record_dtype_cached("i8" if _ticked(dtype) else "f8")


def _one(value):
    """The single value a column of one row holds, however it was spelled."""
    if isinstance(value, list | tuple):
        return value[0]
    if isinstance(value, np.ndarray):
        return value[0] if value.ndim else value[()]
    return value


def _row_count(length) -> int:
    """How many runs a length column states."""
    if isinstance(length, np.ndarray):
        return len(length) if length.ndim else 1
    if isinstance(length, list | tuple):
        return len(length)
    return 1


def _rows(dtype, start, length, num, den, offset, source_id=b"") -> np.ndarray:
    """A run table from its columns."""
    columns = (start, length, num, den, offset, source_id)
    record = _record_dtype(dtype)
    if _row_count(length) == 1:
        # One run is the common table, and stating it as a row is four
        # times cheaper than six assignments into an empty array. A value
        # no row can hold -- a NaN or a float past int64 in a tick column --
        # falls through to the assignments, which cast it instead.
        with suppress(OverflowError, ValueError):
            return np.array([tuple(_one(x) for x in columns)], record)
    out = np.empty(_row_count(length), record)
    for name, column in zip(_COLUMNS, columns):
        out[name] = column
    return out


def step_bits(step) -> np.ndarray:
    """Float64 steps as the int64 words a float row's ``num`` field holds."""
    # Adding zero turns -0.0 into 0.0, so a flat run has one spelling.
    return (np.asarray(step, np.float64) + 0.0).view(np.int64)


def float_terms(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The ``step``, ``stride``, and ``k0`` columns of a float run table.

    The stride is signed: a negative one marks a run whose labels divide
    their grid index by the step rather than multiply it.
    """
    step = np.ascontiguousarray(rows["num"]).view(np.float64)
    return step, rows["den"], rows["offset"]


def float_rows(dtype, start, length, step, stride=1, k0=0, source_id=b"") -> np.ndarray:
    """A float run table from its own terms rather than the record's fields."""
    return _rows(dtype, start, length, step_bits(step), stride, k0, source_id)


def _stored_window(new: np.ndarray, rows: np.ndarray, k, stride: int) -> None:
    """
    Move each stored row's window through the array its labels live in.

    A stored row is ``(source_id, offset, num, length)``: ``offset`` indexes
    the source array and ``num`` strides through it, so slicing and reversal
    are arithmetic on those two and the source is never touched.
    """
    stored = rows["den"] == 0
    if not np.any(stored):
        return
    steps = rows["num"][stored]
    start = np.asarray(k, np.int64)
    start = start[stored] if start.ndim else start
    new["offset"][stored] = rows["offset"][stored] + start * steps
    new["num"][stored] = steps * stride


def _refuse(msg: str) -> NoReturn:
    raise CoordError(msg)


def _span_ok(count: int, den: int) -> bool:
    """Whether ``count`` samples of a grid over ``den`` stay inside int64."""
    return (abs(count) + 2) * den < _INT64_MAX


class TickKernel:
    """Rows of whole ticks: ``start + floor((offset + k * num) / den)``."""

    @staticmethod
    def labels(rows: np.ndarray, run, k, reach: int | None = None) -> np.ndarray:
        """
        The ticks of rows ``run`` at their own sample indices ``k``.

        ``reach`` is the largest magnitude in ``k`` where the caller knows it.

        The product ``k * num`` is taken whole while it fits int64. Past
        that the whole ticks of the step are taken out first, which gives
        the same integer and needs only ``k * den`` to fit: with
        ``q, r = divmod(num, den)`` the label is
        ``start + k * q + (offset + k * r) // den``.
        """
        start, num = rows["start"][run], rows["num"][run]
        stated = rows["den"][run]
        offset = rows["offset"][run]
        if np.any(stated == 0):
            # A stored row's num and offset index its source array, not a
            # grid; its labels are the caller's to fill in, so the row gives
            # its start for every sample rather than arithmetic on them.
            num = np.where(stated == 0, 0, num)
            offset = np.where(stated == 0, 0, offset)
        den = np.maximum(stated, 1)
        k = np.asarray(k, np.int64)
        if not k.size:
            return np.zeros(k.shape, np.int64)
        if reach is None:
            reach = max(abs(int(k.min())), abs(int(k.max())))
        top_num, top_den = int(np.max(np.abs(num))), int(np.max(den))
        if reach * top_num + top_den < _INT64_MAX:
            return start + (offset + k * num) // den
        if not _span_ok(reach, top_den):
            _refuse(
                f"A run of {reach} samples on a grid of denominator {top_den} "
                "is too long for int64 arithmetic."
            )
        whole, part = np.divmod(num, den)
        return start + k * whole + (offset + k * part) // den

    @staticmethod
    def heads(rows: np.ndarray) -> np.ndarray:
        """The first label of each run; a tick row states it outright."""
        return rows["start"]

    @staticmethod
    def sliced(rows: np.ndarray, k, stride: int) -> np.ndarray:
        """Each row re-anchored at its own sample ``k`` and re-strided."""
        new = rows.copy()
        grid = rows["den"] > 0
        den = np.maximum(rows["den"], 1)
        whole, part = np.divmod(rows["num"], den)
        carry, offset = np.divmod(rows["offset"] + k * part, den)
        new["start"] = np.where(grid, rows["start"] + k * whole + carry, rows["start"])
        new["offset"] = np.where(grid, offset, rows["offset"])
        if stride != 1:
            top = int(np.max(np.abs(rows["num"][grid]))) if np.any(grid) else 0
            if top * abs(stride) >= _INT64_MAX:
                _refuse(f"A stride of {stride} takes the step of a run past int64.")
            new["num"] = np.where(grid, rows["num"] * stride, rows["num"])
        _stored_window(new, rows, k, stride)
        return new

    @classmethod
    def reversed(cls, rows: np.ndarray) -> np.ndarray:
        """Each row re-anchored on its last sample, running the other way."""
        new = cls.sliced(rows, rows["length"] - 1, 1)
        new["num"] = -rows["num"]
        return new

    @staticmethod
    def continues(rows: np.ndarray) -> np.ndarray:
        """Whether each run begins where the run before puts its next sample."""
        before, after = rows[:-1], rows[1:]
        same = (
            (before["num"] == after["num"])
            & (before["den"] == after["den"])
            & (before["den"] > 0)
        )
        den = np.maximum(before["den"], 1)
        whole, part = np.divmod(before["num"], den)
        length = before["length"]
        wide = length.astype(np.float64)
        with np.errstate(over="ignore"):
            apart = after["start"] - before["start"]
            # A difference of starts which wrapped int64 changed sign; runs
            # that far apart do not meet, and neither do runs on opposite
            # sides of where the first is heading.
            same &= (apart >= 0) == (after["start"] >= before["start"])
            same &= (apart == 0) | ((apart > 0) == (before["num"] > 0))
            same &= wide * np.abs(whole.astype(np.float64)) < _INT64_MAX / 2
            same &= (wide + 2) * den.astype(np.float64) < _INT64_MAX
            # What is left of the difference once the run's whole ticks are
            # out of it is at most the run's length where the two meet.
            shift = apart - length * whole
            same &= (shift >= -1) & (shift <= length + 1)
            meets = shift * den + after["offset"] == before["offset"] + length * part
        return same & meets

    @staticmethod
    def reduced(rows: np.ndarray) -> np.ndarray:
        """Each grid in lowest terms, its phase inside its denominator."""
        grid = rows["den"] > 0
        den = np.maximum(rows["den"], 1)
        # A phase past its denominator is the next tick's phase.
        carry = np.where(grid, rows["offset"] // den, 0)
        moved = rows["start"].astype(np.float64) + carry.astype(np.float64)
        if np.any(np.abs(moved) >= _INT64_MAX):
            _refuse("A run's phase carries its first label past int64.")
        rows["start"] += carry
        rows["offset"] -= carry * rows["den"]
        # Lowest terms: the offset's remainder under gcd(num, den) can never
        # carry a floor over an integer boundary, so dividing it is label
        # preserving and leaves one table per set of labels.
        common = np.where(grid, np.gcd(np.abs(rows["num"]), rows["den"]), 1)
        common = np.maximum(common, 1)
        rows["num"] //= common
        rows["den"] //= common
        rows["offset"] //= common
        return rows

    @staticmethod
    def reduced_one(row: tuple) -> tuple:
        """`reduced` for a single row, in python integers."""
        start, length, num, den, offset = row[:5]
        if den > 0:
            carry, offset = divmod(offset, den)
            if not -_INT64_MAX <= start + carry < _INT64_MAX:
                _refuse("A run's phase carries its first label past int64.")
            common = max(math.gcd(abs(num), den), 1)
            start, num, den = start + carry, num // common, den // common
            offset //= common
        return start, length, num, den, offset

    @staticmethod
    def _check_one(row: tuple, dtype) -> None:
        """`check_range` for one row, in python integers."""
        start, length, num, den, offset = row[:5]
        if den <= 0:  # a stored run has no tick arithmetic to leave
            return
        if not _span_ok(length, den):
            _refuse(
                f"A run of {length} samples on a grid of denominator {den} "
                "is too long for int64 arithmetic."
            )
        low, high = _tick_bounds(dtype)
        stop = start + (offset + length * num) // den
        if min(start, stop) < low or max(start, stop) > high:
            _refuse(
                f"A grid of {length} samples with step {num}/{den} ticks "
                f"from {start} reaches {stop}, which exceeds the {dtype} range."
            )

    @classmethod
    def check_range(cls, rows: np.ndarray, dtype) -> None:
        """
        Refuse a run whose labels leave its dtype or whose arithmetic leaves int64.

        A run is limited by its labels. What int64 asks beyond that is only
        that ``length * den`` fits, which a nanosecond grid reaches after
        thousands of years.
        """
        if len(rows) <= 8:
            for row in rows:
                cls._check_one(row.item(), dtype)
            return
        grid = rows[rows["den"] > 0]
        length = grid["length"].astype(np.float64)
        den = grid["den"].astype(np.float64)
        reach = np.abs(grid["start"].astype(np.float64)) + length * np.abs(
            grid["num"].astype(np.float64)
        ) / np.maximum(den, 1)
        low, high = _tick_bounds(dtype)
        # Floats only screen: a run anywhere near a limit is judged exactly.
        near = ((length + 2) * den >= _INT64_MAX / 2) | (
            reach >= min(abs(low), high) / 2
        )
        for row in grid[near]:
            cls._check_one(row.item(), dtype)

    @staticmethod
    def index_of(row: tuple, anchor, forward: bool, narrow=None) -> int:
        """
        The sample index of one grid row a tick maps to, in python integers.

        Forward: the first index whose label has reached the tick; else the
        last index whose label has not passed it, each in the run's own
        direction of travel. The index may lie outside the run, which is how
        a caller reads an open bound.
        """
        start, length, num, den, offset = row[:5]
        if not num:  # every label of a flat run is its start
            if forward:
                return 0 if anchor <= start else length
            return length - 1 if anchor >= start else -1
        rel = (anchor - start) * den - offset
        # A rising run reaches the tick where offset + k num >= rel, and has
        # not passed it while offset + k num < rel + den; a falling run wants
        # each the other way round, which turns every bound over with it.
        if num > 0:
            return -((-rel) // num) if forward else -((-(rel + den)) // num) - 1
        return ((rel + den) // num) + 1 if forward else rel // num

    @staticmethod
    def step_of(row: tuple) -> int:
        """The whole-tick spacing a row's grid rounds to."""
        _, _, num, den, _ = row[:5]
        # round half to even, as a Fraction of these terms rounds
        whole, part = divmod(num, den)
        twice = 2 * part
        if twice > den or (twice == den and whole % 2):
            whole += 1
        return whole

    @staticmethod
    def same_grid(rows: np.ndarray, num: int, den: int) -> bool:
        """Whether every run begins on the one grid a spacing of num/den makes."""
        before, after = rows[:-1], rows[1:]
        if not num:
            return bool(np.all(after["start"] == before["start"]))
        # In python integers, and one run at a time: runs far apart on a
        # fine grid have a tick difference which leaves int64 the moment
        # it is multiplied by the denominator, and there are few of them.
        for prior, later in zip(before.tolist(), after.tolist()):
            rel = (later[0] - prior[0]) * den + later[4] - prior[4]
            if rel % num:
                return False
        return True


class FloatKernel:
    """
    Rows of float labels, counted along an integer grid index.

    The module docstring gives the expression and the fields it is spelled
    in. The two spellings are the two ways float axes are built, and they
    round differently, so a row keeps the one its labels were made with.
    """

    @staticmethod
    def labels(rows: np.ndarray, run, k, reach: int | None = None) -> np.ndarray:
        """The labels of rows ``run`` at their own sample indices ``k``."""
        step, den, k0 = float_terms(rows)
        step, den, k0 = step[run], den[run], k0[run]
        if np.any(den == 0):
            # A stored row's num and offset index its source array; read as
            # a step they are not even a number, so the row gives its start.
            step = np.where(den == 0, 0.0, step)
            k0 = np.where(den == 0, 0, k0)
        index = k0 + np.asarray(k, np.int64) * np.abs(den)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            moved = np.where(
                den < 0, index / np.where(den < 0, step, 1.0), step * index
            )
        return rows["start"][run] + moved

    @classmethod
    def heads(cls, rows: np.ndarray) -> np.ndarray:
        """The first label of each run, which a float row has to compute."""
        return cls.labels(rows, slice(None), 0)

    @staticmethod
    def sliced(rows: np.ndarray, k, stride: int) -> np.ndarray:
        """Each row moved to its own sample ``k`` and re-strided; no label moves."""
        new = rows.copy()
        new["offset"] = rows["offset"] + k * np.abs(rows["den"])
        _stored_window(new, rows, k, stride)
        wide = (
            np.abs(rows["offset"].astype(np.float64))
            + np.abs(np.asarray(k, np.float64)) * np.abs(rows["den"].astype(np.float64))
            + rows["length"].astype(np.float64)
            * np.abs(rows["den"].astype(np.float64))
            * abs(stride)
        )
        if len(new) and wide.max() >= _FLOAT_INDEX_MAX:
            _refuse("A float run's grid index has left the range a float64 counts.")
        new["den"] = rows["den"] * stride
        return new

    @staticmethod
    def reversed(rows: np.ndarray) -> np.ndarray:
        """
        Each row running the other way, every label the same double.

        Negating both the step and the grid index leaves each product or
        quotient, and so each label, exactly what it was.
        """
        new = rows.copy()
        step, den, k0 = float_terms(rows)
        grid = den != 0
        last = k0 + (rows["length"] - 1) * np.abs(den)
        new["offset"] = np.where(grid, -last, rows["offset"])
        new["num"] = np.where(grid, step_bits(-step), rows["num"])
        _stored_window(new, rows, rows["length"] - 1, -1)
        return new

    @staticmethod
    def continues(rows: np.ndarray) -> np.ndarray:
        """Whether each run carries on the grid index of the run before."""
        before, after = rows[:-1], rows[1:]
        same = (
            (before["start"] == after["start"])
            & (before["num"] == after["num"])
            & (before["den"] == after["den"])
            & (before["den"] != 0)
        )
        ahead = before["offset"] + before["length"] * np.abs(before["den"])
        return same & (after["offset"] == ahead)

    @staticmethod
    def reduced(rows: np.ndarray) -> np.ndarray:
        """A float row has one spelling already; even a signed zero is a label."""
        return rows

    @staticmethod
    def reduced_one(row: tuple) -> tuple:
        """`reduced` for a single row."""
        return row

    @classmethod
    def check_range(cls, rows: np.ndarray, dtype) -> None:
        """Refuse a run whose step or labels are not finite numbers."""
        grid = rows[rows["den"] != 0]
        if not len(grid):
            return
        step, den, k0 = float_terms(grid)
        reach = np.abs(k0.astype(np.float64)) + grid["length"].astype(
            np.float64
        ) * np.abs(den.astype(np.float64))
        if reach.max() >= _FLOAT_INDEX_MAX:
            _refuse("A float run's grid index has left the range a float64 counts.")
        ends = cls.labels(grid, slice(None), grid["length"])
        with np.errstate(over="ignore", invalid="ignore"):
            # in the coordinate's own dtype: a narrower float overflows to
            # an infinity the row's own arithmetic never reaches
            narrow = np.asarray(ends).astype(dtype)
            first = np.asarray(cls.heads(grid)).astype(dtype)
        bad = ~(np.isfinite(grid["start"]) & np.isfinite(step) & np.isfinite(ends))
        bad |= ~(np.isfinite(narrow) & np.isfinite(first))
        bad |= (den < 0) & (step == 0)
        if np.any(bad):
            row = grid[np.argmax(bad)]
            _refuse(
                f"A float grid of {row['length']} samples with step "
                f"{float(step[np.argmax(bad)])} from {row['start']} is not finite."
            )

    @staticmethod
    def _spacing(bits: int, den: int) -> float:
        """The distance between neighbouring labels these terms make."""
        step = float(np.int64(bits).view(np.float64))
        return abs(den) / step if den < 0 else step * den

    @classmethod
    def index_of(cls, row: tuple, value, forward: bool, narrow=None) -> int:
        """
        The sample index of one grid row a value maps to.

        Forward: the first index whose label is at or past the value; else
        the last index whose label is at or before it. The position is
        rounded before it is taken up or down, as a float range has always
        done it: a bound a rounding away from a label is that label, not
        the one past it. The index may lie outside the run, which is how a
        caller reads an open bound. ``narrow`` is the dtype the labels are
        held in where that is not the float64 a row counts in, so that the
        labels compared here are the ones the coordinate hands out.
        """
        start, length, bits, den, k0 = row[:5]
        step = float(np.int64(bits).view(np.float64))
        if not step:  # every label of a flat run is its start
            if forward:
                return 0 if value <= start else length
            return length - 1 if value >= start else -1
        index = (value - start) * step if den < 0 else (value - start) / step
        position = (index - k0) / abs(den)
        if not math.isfinite(position):
            return length if (position > 0) else -1
        if abs(position) >= _FLOAT_INDEX_MAX:
            return math.ceil(position) if forward else math.floor(position)
        spacing = cls._spacing(bits, den)
        rising = spacing > 0
        # A label within a rounding of its own arithmetic is the value.
        near = abs(spacing) * 1e-9

        def label(k: int) -> float:
            index = k0 + k * abs(den)
            out = start + (index / step if den < 0 else step * index)
            # In the dtype the labels are held in: a narrower float rounds
            # the row's arithmetic again, and two of its labels can lie
            # closer together than the row's own spacing suggests.
            return out if narrow is None else float(np.asarray(out, narrow))

        # The division only estimates where the value sits, and far from
        # the origin by a good deal more than a rounding; the labels either
        # side of the estimate decide, exactly.
        #
        # Rounding into a narrower float can make two neighbouring labels
        # one label, and stepping onto a label the run already holds gains
        # nothing, so the search moves on only where the labels progress.
        k = math.floor(position)
        for _ in range(4):
            if (label(k) > value) == rising and label(k) != value:
                k -= 1
            elif label(k + 1) != label(k) and (
                (label(k + 1) <= value) == rising or label(k + 1) == value
            ):
                k += 1
            else:
                break
        # Now label(k) is at or before the value and label(k + 1) past it.
        if forward:
            return k if abs(label(k) - value) <= near else k + 1
        if label(k + 1) != label(k) and abs(label(k + 1) - value) <= near:
            return k + 1
        return k

    @classmethod
    def step_of(cls, row: tuple) -> float:
        """The spacing between a row's neighbouring labels."""
        _, _, bits, den, _ = row[:5]
        return cls._spacing(bits, den)

    @classmethod
    def same_grid(cls, rows: np.ndarray, num: int, den: int) -> bool:
        """Whether every run begins on the one grid these terms make."""
        spacing = cls._spacing(num, den)
        heads = cls.heads(rows)
        if not spacing:
            return bool(np.all(heads[1:] == heads[:-1]))
        steps = np.diff(heads) / spacing
        return bool(np.allclose(steps, np.round(steps), rtol=1e-9, atol=1e-9))

    # --- reading a grid out of labels

    @classmethod
    def _candidates(cls, values: np.ndarray):
        """
        Direct multiplication grids the labels might have been built as.

        The first spacing and endpoint spacing are each tried from the first
        label and from zero. Exact reproduction decides whether one is the
        grid; no decimal, reciprocal-rate, or time-scale search is done.
        """
        count, first = len(values), float(values[0])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            steps = [
                float(values[1] - values[0]),
                float((values[-1] - values[0]) / (count - 1)),
            ]
        for step in dict.fromkeys(steps):
            if not step or not math.isfinite(step):
                continue
            yield first, step, 1, 0
            index = first / step
            if math.isfinite(index):
                yield 0.0, step, 1, round(index)

    @classmethod
    def _reproduces(cls, row: np.ndarray, values: np.ndarray, dtype) -> bool:
        """Whether a row gives back every one of these labels, bit for bit."""
        count = len(values)
        # A spread of samples refuses nearly every wrong row for the price
        # of a few labels; only a row which passes is checked in full.
        if count > 4096:
            probe = np.unique(np.linspace(0, count - 1, 1024).astype(np.int64))
            got = cls.labels(row, 0, probe).astype(dtype, copy=False)
            if not _same_labels(got, values[probe]):
                return False
        for first in range(0, count, 1 << 20):
            k = np.arange(first, min(first + (1 << 20), count), dtype=np.int64)
            got = cls.labels(row, 0, k).astype(dtype, copy=False)
            if not _same_labels(got, values[first : first + len(k)]):
                return False
        return True

    @classmethod
    def fit(cls, values, dtype, origin=None) -> np.ndarray | None:
        """
        The one row which reproduces every label exactly, or None.

        ``origin`` is a ``(start, step, den)`` to try first, normally the
        grid found for a neighbouring run or an explicitly declared step.
        """
        count = len(values)
        wide = np.asarray(values, np.float64)
        if count < 2 or not np.all(np.isfinite(wide)):
            return None
        tries = []
        if origin is not None:
            start, step, den = origin
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                index = (
                    (wide[0] - start) * step if den < 0 else (wide[0] - start) / step
                )
            if math.isfinite(index):
                tries.append((start, step, den, round(index)))
        for start, step, den, k0 in dict.fromkeys([*tries, *cls._candidates(wide)]):
            if abs(k0) + count * abs(den) >= _FLOAT_INDEX_MAX:
                continue
            row = float_rows(dtype, [start], [count], [step], [den], [k0])
            if cls._reproduces(row, values, dtype):
                return row
        return None


def _same_labels(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare labels exactly, including the sign of a floating zero."""
    if not np.array_equal(left, right):
        return False
    if left.dtype.kind != "f":
        return True
    return bool(np.array_equal(np.signbit(left), np.signbit(right)))


def get_kernel(dtype) -> type[TickKernel] | type[FloatKernel]:
    """The kernel which reads rows of this coordinate dtype."""
    return TickKernel if _ticked(dtype) else FloatKernel
