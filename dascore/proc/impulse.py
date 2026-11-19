"""
Impulse windows: an operation's reach measured from its response to an impulse.

[`Patch.impulse_window`](`dascore.Patch.impulse_window`) declares a tolerance
for each dimension a linear operation reaches along, and returns a
[`PatchImpulseWindow`](`dascore.proc.impulse.PatchImpulseWindow`); an
operation called on that is run by
[`Impulsed`](`dascore.proc.impulse.Impulsed`).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from pydantic import ConfigDict

from dascore.core.processor import PatchProcessor, captured_operation
from dascore.exceptions import ParameterError
from dascore.lazy.contract import Halo
from dascore.utils.array_api import _as_numpy_dtype, backend_name
from dascore.utils.misc import optional_import
from dascore.utils.patch import _call_str

__all__ = ("Impulsed", "PatchImpulseWindow")

# The probe's first length along an axis, and the most samples a probe
# may hold (64 MB of doubles), past which the response is unsettled.
_FIRST_PROBE = 1024
_PROBE_BUDGET = 1 << 23
# The reach must lie within this fraction of the probe on either side of
# the impulse, so the probe's own ends play no part in what is measured.
_WELL_INSIDE = 0.25
_UNSETTLED = ("raise", "whole")


def _plan_key(value) -> Any:
    """Return a hashable stand-in for a plan, arrays by their contents."""
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _plan_key(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_plan_key(x) for x in value)
    return value


def _tail_reach(response: np.ndarray, tolerance: float) -> int:
    """
    Return the fewest samples after the impulse beyond which the rest is small.

    `response` holds the absolute response from the impulse on; the reach
    is the smallest distance beyond which what is left sums to less than
    `tolerance` times the whole.
    """
    total = response.sum()
    if not total:
        return 0
    # What lies beyond each distance: the sum of everything after it.
    beyond = np.cumsum(response[::-1])[::-1]
    beyond = np.append(beyond[1:], 0.0)
    return int(np.argmax(beyond < tolerance * total))


# What `_measured` has found, by what it depends on; bounded, oldest out.
_MEASURED: dict[tuple, Any] = {}
_MEASURED_LIMIT = 256
# How long an axis the operation also reaches along is made in a probe of
# another axis: long enough for a filter's own padding of its ends.
_ACROSS = 64


def _kept(key, compute):
    """Return what `compute` gives, kept by `key` for the next ask."""
    if key in _MEASURED:
        return _MEASURED[key]
    if len(_MEASURED) >= _MEASURED_LIMIT:
        _MEASURED.pop(next(iter(_MEASURED)))
    _MEASURED[key] = out = compute()
    return out


def _impulsed(operation, plan, shape, dtype, room=None) -> np.ndarray:
    """
    Return what the operation makes of an impulse in the middle of a probe.

    A filter may need an axis longer than the probe gives it, to pad its
    ends: the axes `room` names are then lengthened, up to their length in
    the patch, until the kernel runs or none can be lengthened further.
    """
    kernel = type(operation).kernel_for("numpy")
    shape = list(shape)
    while True:
        probe = np.zeros(shape, dtype)
        probe[tuple(n // 2 for n in shape)] = 1
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.asarray(kernel(operation, probe, **plan))
        except ValueError:
            grown = [
                min(2 * n + 1, room[x]) if room and x in room else n
                for x, n in enumerate(shape)
            ]
            if grown == shape:
                raise
            shape = grown


def _response(operation, plan, shape, axis, dtype, room=None) -> np.ndarray:
    """
    Return the absolute response along an axis to an impulse in a probe.

    The impulse sits in the middle of every axis of the probe, and the
    response is read along `axis` through it.
    """
    out = _impulsed(operation, plan, shape, dtype, room)
    middle = tuple(n // 2 for n in out.shape)
    line = tuple(slice(None) if x == axis else m for x, m in enumerate(middle))
    return np.nan_to_num(np.abs(out[line]).astype(np.float64))


def _made_dtype(operation, plan, meta) -> np.dtype:
    """Return the dtype the operation's kernel makes of data of the patch's."""
    lengths = tuple(min(n, 2 * _ACROSS + 1) for n in meta.shape)
    dtype = _as_numpy_dtype(meta.dtype)
    room = dict(enumerate(meta.shape))

    def compute():
        return _impulsed(operation, plan, lengths, dtype, room).dtype

    key = ("dtype", type(operation), repr(operation), _plan_key(plan))
    return _kept((*key, lengths, dtype.str), compute)


def _reached(operation, plan, meta) -> tuple[int, ...]:
    """
    Return the axes the operation reaches along: those it spreads an impulse on.

    Every axis is a few dozen samples long, or its own length if shorter,
    and longer where a filter along it needs more to pad its ends; the
    impulse spreads to a neighbour along an axis the operation reaches.
    """
    lengths = [min(n, 2 * _ACROSS + 1) for n in meta.shape]
    dtype = _as_numpy_dtype(meta.dtype)
    room = dict(enumerate(meta.shape))

    def compute():
        out = []
        for axis, length in enumerate(lengths):
            if length < 3:
                continue
            response = _response(operation, plan, tuple(lengths), axis, dtype, room)
            middle = len(response) // 2
            spread = response[middle - 1] + response[middle + 1]
            if spread > 1e-12 * max(response[middle], 1e-300):
                out.append(axis)
        return tuple(out)

    key = ("reached", type(operation), repr(operation), _plan_key(plan))
    return _kept((*key, tuple(lengths), dtype.str), compute)


def _measured(operation, plan, meta, axis, reached, tolerance):
    """
    Return how far before and after a sample the operation reaches, or None.

    Kept for the plan and everything else it depends on: the reach of a
    linear operation which acts the same at every position is its
    response to one impulse, whatever the data. The probe grows along
    `axis`; the other axes the operation reaches along are a few dozen
    samples long, longer where a filter needs more to pad its ends, and
    the rest one.

    A reach as long as the axis is the whole axis, which a chunk then
    needs, so the probe never grows past four times the axis. None says
    the response did not fall under the tolerance before the probe grew
    past what memory allows it.
    """
    dtype = _as_numpy_dtype(meta.dtype)
    across = [min(n, _ACROSS) if x in reached else 1 for x, n in enumerate(meta.shape)]
    room = {x: n for x, n in enumerate(meta.shape) if x in reached and x != axis}
    whole = meta.shape[axis]

    def compute():
        length = _FIRST_PROBE
        while True:
            shape = list(across)
            shape[axis] = length
            if np.prod(shape) > _PROBE_BUDGET:
                return None
            response = _response(operation, plan, tuple(shape), axis, dtype, room)
            middle = length // 2
            # A response after the impulse is output which reads input
            # before it, so the reach is the response turned round.
            spread_before = _tail_reach(response[: middle + 1][::-1], tolerance)
            spread_after = _tail_reach(response[middle:], tolerance)
            if max(spread_before, spread_after) <= _WELL_INSIDE * length:
                return min(spread_after, whole), min(spread_before, whole)
            if length >= 4 * whole:
                # It reaches past the axis: a chunk needs all of it.
                return whole, whole
            length *= 2

    key = ("reach", type(operation), repr(operation), _plan_key(plan))
    return _kept((*key, axis, whole, tuple(across), dtype.str, tolerance), compute)


class Impulsed(PatchProcessor):
    """
    Run a linear operation in pieces, each read with the reach it measured.

    The operation behind a patch method called on
    [`Patch.impulse_window`](`dascore.Patch.impulse_window`), where the
    tolerances are described; they are given as keywords, such as
    ``time=1e-6``.

    Parameters
    ----------
    operation
        The operation to run, as a
        [`PatchProcessor`](`dascore.core.processor.PatchProcessor`). Its
        class must say it is `linear` and act the same wherever a patch's
        coordinates start (`position_free`).
    on_unsettled
        What to do when the response along a dimension does not fall under
        its tolerance within the longest probe: ``"raise"``, the default,
        or ``"whole"``, which needs that dimension whole as the operation
        alone does.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.proc.impulse import Impulsed
    >>> patch = dc.get_example_patch()
    >>> out = patch.impulse_window(time=1e-6).pass_filter(time=(1, 20))
    >>> same = Impulsed(dc.proc.PassFilter(time=(1, 20)), time=1e-6)(patch)
    >>> assert out.equals(same)

    Notes
    -----
    - The reach along each dimension is measured by running the
      operation's own kernel on an impulse, with the plan it will be run
      with: it is the smallest distance beyond which the summed absolute
      response is under the tolerance times the whole, before and after
      the impulse separately, so a causal filter reaches one way. The
      probe grows until the reach sits well inside it. The measurement
      depends on the plan alone and is kept for it.
    - Data in memory are computed whole, exactly as the operation alone
      computes them. A lazy or dask patch is computed a chunk at a time,
      each chunk read with the measured reach around it and that reach
      discarded, so it differs from the exact result by at most about the
      tolerance.
    """

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    # Reached through `Patch.impulse_window`, so it has no patch method.
    name = None

    operation: Any
    on_unsettled: str = "raise"

    _positional_fields = ("operation",)

    def _spelled(self, name: str) -> str:
        """Return the call as it is written: the window, then the operation."""
        given = self._inputs()[0]
        operation = given.pop("operation")
        called = operation.name or type(operation).__name__
        spelled = operation._spelled(called)
        spelled = spelled if spelled.endswith(")") else f"{spelled}()"
        return f"{_call_str('impulse_window', given)}.{spelled}"

    def _refuse(self, why: str):
        """Raise for an operation which cannot run in an impulse window."""
        operation = self.operation
        name = operation.name or type(operation).__name__
        msg = f"{name} cannot run in an impulse window: {why}"
        raise ParameterError(msg)

    def get_metadata(self, meta):
        """Return the operation's metadata, and the reach along each axis."""
        operation = self.operation
        if not isinstance(operation, PatchProcessor):
            msg = f"An impulse window runs a PatchProcessor; got {operation!r}."
            raise ParameterError(msg)
        if self.on_unsettled not in _UNSETTLED:
            msg = f"on_unsettled is one of {_UNSETTLED}; got {self.on_unsettled!r}."
            raise ParameterError(msg)
        if not (type(operation).linear and operation._is_position_free(meta.dims)):
            self._refuse(
                "its response to an impulse says nothing of what it does to "
                "other data unless it is linear and acts the same at every "
                "position, which its class says by `linear` and "
                "`position_free`."
            )
        if operation._inputs()[1]:
            self._refuse("it takes another patch.")
        tolerances = dict(self.model_extra or {})
        if not tolerances:
            msg = "An impulse window needs a dimension and its tolerance: time=1e-6."
            raise ParameterError(msg)
        unknown = set(tolerances) - set(meta.dims)
        if unknown:
            missing = sorted(unknown)
            msg = f"An impulse window names dimensions the patch lacks: {missing}."
            raise ParameterError(msg)
        for dim, value in tolerances.items():
            real = isinstance(value, Real) and not isinstance(value, bool)
            if not (real and 0 < value < 1):
                msg = (
                    f"The tolerance along {dim} is a fraction of the whole "
                    f"response, between 0 and 1; got {value!r}."
                )
                raise ParameterError(msg)
        out, inner = operation.get_metadata(meta)
        if out.shape != meta.shape or out.dims != meta.dims:
            self._refuse("it changes the patch's shape, which a reach cannot cut.")
        widths: dict[int, tuple[int, int]] = {}
        reached = _reached(operation, inner, meta)
        lacking = [meta.dims[x] for x in reached if meta.dims[x] not in tolerances]
        if lacking:
            self._refuse(
                f"it reaches along {lacking}, which the impulse window gives no "
                "tolerance for. Name every dimension it filters."
            )
        for axis, dim in enumerate(meta.dims):
            if (tolerance := tolerances.get(dim)) is None:
                continue
            if axis not in reached:
                # Reaching nowhere along it, it needs nothing around a chunk.
                widths[axis] = (0, 0)
                continue
            reach = _measured(operation, inner, meta, axis, reached, float(tolerance))
            if reach is None and self.on_unsettled == "raise":
                name = operation.name or type(operation).__name__
                msg = (
                    f"The response of {name} along {dim} does not fall under "
                    f"{tolerance} within a probe of {_PROBE_BUDGET} samples. "
                    "Give a larger tolerance, or on_unsettled='whole' to need "
                    "the dimension whole."
                )
                raise ParameterError(msg)
            # Unsettled, it needs the whole axis: a reach as long as it.
            whole = meta.shape[axis]
            widths[axis] = (whole, whole) if reach is None else reach
        # A plan holds no int-keyed dicts: the reach of each axis as a row,
        # a reach as long as its axis standing for all of it.
        halo = tuple((x, *r) for x, r in widths.items())
        made = _made_dtype(operation, inner, meta)
        # In the dtype the kernel makes, which a lazy patch states ahead.
        out = out.new(dtype=made)
        return out, {"inner": inner, "halo": halo, "like": np.empty(0, made)}

    def chunk_contract(self, dtype, plan):
        """Return the measured reach around a chunk; it may be its whole axis."""
        return Halo({axis: (before, after) for axis, before, after in plan["halo"]})

    def numpy_kernel(self, data, *, inner, halo, like):
        """Return the operation's result on the data, computed as it computes it."""
        kernel = type(self.operation).kernel_for("numpy")
        return kernel(self.operation, data, **inner)

    def kernel(self, data, *, inner, halo, like):
        """
        Return the operation's result on the data's own backend.

        A dask array is computed a chunk at a time, each chunk given the
        measured reach around it, which is then cut off; any other array
        is computed whole by the operation's kernel for its library.
        """
        operation = self.operation
        if backend_name(data) == "dask":
            array = optional_import("dask.array")
            kernel = type(operation).kernel_for("numpy")
            depth, whole = {}, {}
            for axis, before, after in halo:
                if max(before, after) >= data.shape[axis]:
                    # Reaching across the axis: it is one chunk, read whole.
                    whole[axis] = -1
                    depth[axis] = 0
                else:
                    depth[axis] = (before, after)
            joined = data.rechunk(whole) if whole else data
            out = array.map_overlap(
                lambda block: kernel(operation, block, **inner),
                joined,
                depth=depth,
                boundary="none",
                trim=True,
                dtype=like.dtype,
                meta=data._meta.astype(like.dtype),
            )
            # Chunks shorter than the reach are joined to it; the result is
            # cut back into the chunks the data came in.
            return out.rechunk(data.chunks)
        kernel = type(operation).kernel_for(backend_name(data))
        return kernel(operation, data, **inner)


@dataclass(frozen=True)
class PatchImpulseWindow:
    """
    A patch and the tolerances a linear operation is to be cut by.

    See [`Patch.impulse_window`](`dascore.Patch.impulse_window`). A patch
    operation called on it is run by
    [`Impulsed`](`dascore.proc.impulse.Impulsed`), and gives back a patch.
    """

    patch: Any
    tolerances: Mapping[str, float] = field(default_factory=dict)
    on_unsettled: str = "raise"

    def __getattr__(self, name: str):
        """Return a patch operation which runs with the measured reach."""
        if name.startswith("_"):
            raise AttributeError(name)
        method = getattr(type(self.patch), name, None)
        if method is None or getattr(method, "__processor__", None) is None:
            msg = (
                f"{name} is not a patch operation an impulse window can run. "
                "It runs linear operations written as processors, such as "
                "pass_filter or notch_filter."
            )
            raise AttributeError(msg)

        def impulsed(*args, **kwargs):
            """Run the operation with the reach it measured."""
            operation = captured_operation(method, self.patch, *args, **kwargs)
            return Impulsed(
                operation=operation,
                on_unsettled=self.on_unsettled,
                **self.tolerances,
            ).run(self.patch)

        impulsed.__name__ = impulsed.__qualname__ = name
        return impulsed
