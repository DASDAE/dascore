"""A description of an array stored in a resource, which can load it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite, isnan, prod
from typing import Any

import numpy as np

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.proc.coords import _fill_scalar
from dascore.utils.identity import H

# Where an array is, which names it when nothing better does.
_LOCATION_FIELDS = ("path", "format", "version", "key")

# The dtype kinds a constant may take: bool, uint, int and float.
_REAL_KINDS = frozenset("buif")


@dataclass(frozen=True, slots=True)
class ArraySource:
    """
    Where an array is stored, or the constant which fills it, and enough
    to load it without its patch.

    Holds no data and opens nothing until `load`. A reader sets `key`;
    the I/O framework fills in the rest.

    Parameters
    ----------
    path
        The resource holding the array.
    format
        The name of the FiberIO which reads the resource.
    version
        The version of that format.
    key
        Which array in the resource: the logical patch of a multi-patch
        resource, or an absolute path ("/...") to any stored array, such as
        a dense coordinate.
    windows
        A half-open `(start, stop)` sample range for each axis.
    shape
        The shape of the array the windows select.
    dtype
        The dtype of the loaded array.
    base_id
        The id of the whole array, which a window's id builds on. The
        framework gives a patch's data array the patch's `origin_id`; a
        caller which knows better, such as a hash of the contents, gives
        that. Empty means "by location": path, format, version and key.
    extent
        The shape of the whole array, which says when the windows select
        all of it.
    value
        A constant which fills the array. Such a source has no path and
        reads nothing; see [`full`](`dascore.core.source.ArraySource.full`).
        None means the array is stored rather than constant.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.core.source import ArraySource
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> patch = dc.read(fetch("example_dasdae_event_1.h5"))[0]
    >>> source = patch._source
    >>> # Slicing reads nothing and gives another source.
    >>> sub = source[10:20]
    >>> assert sub.shape[0] == 10 and sub.id != source.id
    >>> assert sub.load().shape == sub.shape
    >>>
    >>> # A constant source generates its array instead of reading one.
    >>> constant = ArraySource.full((2, 3), np.nan)
    >>> assert np.isnan(constant.load()).all()
    """

    path: str = ""
    format: str = ""
    version: str = ""
    key: str = ""
    windows: tuple[tuple[int, int], ...] = ()
    shape: tuple[int, ...] = ()
    dtype: Any = None
    base_id: str = ""
    extent: tuple[int, ...] = ()
    value: Any = None

    @classmethod
    def full(
        cls, shape: int | tuple[int, ...], value: bool | int | float, dtype: Any = None
    ) -> ArraySource:
        """
        Return a source for a constant array, as `np.full` would build one.

        Parameters
        ----------
        shape
            The shape of the array.
        value
            The real scalar which fills it; one the dtype would change is
            refused.
        dtype
            The dtype of the array; the value's own when not given.

        Examples
        --------
        >>> from dascore.core.source import ArraySource
        >>> assert ArraySource.full((3,), 0).load().dtype.kind == "i"
        """
        dtype = np.asarray(value).dtype if dtype is None else np.dtype(dtype)
        # A longdouble has no python scalar, which the id and JSON need.
        if dtype.kind not in _REAL_KINDS or dtype.itemsize > 8:
            msg = f"A constant source takes a real scalar, not {value!r} of {dtype}."
            raise ParameterError(msg)
        value = _fill_scalar(value, dtype).item()
        shape = shape if isinstance(shape, tuple | list) else (shape,)
        return cls(value=value).describe(shape, dtype)

    @property
    def constant(self) -> bool:
        """Whether this generates its array rather than reading one."""
        return self.value is not None

    @property
    def loadable(self) -> bool:
        """Whether this says enough to load the array."""
        described = self.dtype is not None and len(self.windows) == self.ndim
        return bool((self.constant or (self.path and self.format)) and described)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return prod(self.shape)

    @property
    def _dtype(self) -> str | None:
        """The canonical spelling of the dtype, if there is one."""
        return None if self.dtype is None else np.dtype(self.dtype).str

    @property
    def id(self) -> str:
        """
        The id of the array this selects; nothing is read to work it out.

        The whole array's id is its `base_id`. A window's is derived from
        the base and the absolute windows, so it does not depend on the
        slices which led to it, nor -- given a base -- on where the array
        is kept. A constant is its contents alone, so any two constant
        blocks of the same value, dtype and shape are one array.
        """
        if self.constant:
            content = {"value": self.value, "dtype": self._dtype, "shape": self.shape}
            return H("constant", content)
        location = {name: getattr(self, name) for name in _LOCATION_FIELDS}
        base = self.base_id or H("location", location)
        whole = tuple((0, size) for size in self.extent)
        if not self.windows or self.windows == whole:
            return base
        return H("window", [base, self.windows])

    def describe(self, shape, dtype) -> ArraySource:
        """Return a source for the whole of an array of this shape and dtype."""
        shape = tuple(int(x) for x in shape)
        windows = tuple((0, x) for x in shape)
        dtype = np.dtype(dtype)
        return replace(self, windows=windows, shape=shape, dtype=dtype, extent=shape)

    def detach(self) -> ArraySource:
        """Return the provenance alone, for an array this no longer loads."""
        # A constant's value was the array, not its origin.
        return replace(self, windows=(), shape=(), dtype=None, extent=(), value=None)

    def narrow(self, indexer) -> ArraySource:
        """Return the source `indexer` selects, detached if not contiguous."""
        try:
            return self[indexer]
        except (IndexError, TypeError):
            return self.detach()

    def __getitem__(self, index) -> ArraySource:
        """Compose contiguous slices onto the windows; nothing is read."""
        index = index if isinstance(index, tuple) else (index,)
        if not self.loadable:
            msg = f"{self} describes no array to index."
            raise IndexError(msg)
        if len(index) > self.ndim:
            msg = f"Cannot index a {self.ndim}-dimensional source with {index}."
            raise IndexError(msg)
        index = index + (slice(None),) * (self.ndim - len(index))
        windows = []
        for item, (start, _), size in zip(index, self.windows, self.shape):
            span = range(size)[item] if isinstance(item, slice) else None
            if span is None or span.step != 1:
                msg = "An ArraySource takes only slices with a step of one."
                raise TypeError(msg)
            windows.append((start + span.start, start + span.start + len(span)))
        shape = tuple(stop - start for start, stop in windows)
        return replace(self, windows=tuple(windows), shape=shape)

    def load(self) -> np.ndarray:
        """Read and return the array."""
        if not self.loadable:
            msg = f"{self} does not say enough to load an array."
            raise ParameterError(msg)
        if self.constant:
            return np.full(self.shape, self.value, dtype=self.dtype)
        return dc.io.core._load_array_source(self)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Load the array for numpy; each call reads the resource."""
        out = self.load()
        return out if dtype is None else out.astype(dtype, copy=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict which `from_dict` reads back."""
        out = asdict(self)
        out["dtype"] = self._dtype
        # Strict JSON has no nan or inf, so they are written as strings.
        value = self.value
        if isinstance(value, float) and not isfinite(value):
            out["value"] = "nan" if isnan(value) else ("inf" if value > 0 else "-inf")
        return out

    @classmethod
    def from_dict(cls, contents: dict[str, Any]) -> ArraySource:
        """Return the source `to_dict` wrote."""
        out = dict(contents)
        out["shape"] = tuple(out.get("shape", ()))
        out["extent"] = tuple(out.get("extent", ()))
        out["windows"] = tuple((a, b) for a, b in out.get("windows", ()))
        if out.get("dtype") is not None:
            out["dtype"] = np.dtype(out["dtype"])
        if isinstance(out.get("value"), str):
            out["value"] = float(out["value"])
        return cls(**out)
