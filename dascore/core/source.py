"""A description of an array stored in a resource, which can load it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import prod
from typing import Any

import numpy as np

from dascore.exceptions import ParameterError
from dascore.utils.serialize import digest

# The fields which say which array this is; `shape` and `dtype` follow from them.
_ID_FIELDS = ("path", "format", "version", "key", "address", "dims", "windows")


@dataclass(frozen=True, slots=True)
class ArraySource:
    """
    Where an array is stored, and enough to load it without its patch.

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
        The logical patch within a multi-patch resource.
    address
        The array's location inside the resource, such as an HDF5 dataset
        path. Empty means the data array of the patch named by `key`.
    dims
        The stored order of the array's dimensions.
    windows
        A half-open `(start, stop)` sample range for each axis.
    shape
        The shape of the array the windows select.
    dtype
        The dtype of the loaded array.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> patch = dc.read(fetch("example_dasdae_event_1.h5"))[0]
    >>> source = patch._source
    >>> # Slicing reads nothing and gives another source.
    >>> sub = source[10:20]
    >>> assert sub.shape[0] == 10 and sub.id != source.id
    >>> assert sub.load().shape == sub.shape
    """

    path: str = ""
    format: str = ""
    version: str = ""
    key: str = ""
    address: str = ""
    dims: tuple[str, ...] = ()
    windows: tuple[tuple[int, int], ...] = ()
    shape: tuple[int, ...] = ()
    dtype: Any = None

    @property
    def loadable(self) -> bool:
        """Whether this says enough to load the array."""
        described = self.dtype is not None and len(self.windows) == self.ndim
        # `read_array` takes its windows by dimension name.
        named = bool(self.address) or len(self.dims) == self.ndim
        return bool(self.path and self.format and described and named)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return prod(self.shape)

    @property
    def id(self) -> str:
        """A digest of which array this is; equal sources share it."""
        return digest({name: getattr(self, name) for name in _ID_FIELDS})

    def describe(self, shape, dtype, dims=()) -> ArraySource:
        """Return a source for the whole of an array of this shape and dtype."""
        shape = tuple(int(x) for x in shape)
        return replace(
            self,
            dims=tuple(dims),
            windows=tuple((0, x) for x in shape),
            shape=shape,
            dtype=np.dtype(dtype),
        )

    def detach(self) -> ArraySource:
        """Return the provenance alone, for an array this no longer loads."""
        return replace(self, dims=(), windows=(), shape=(), dtype=None)

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
        # The reader registry imports the patch, which imports this module.
        from dascore.io.core import _load_array_source  # noqa: PLC0415

        if not self.loadable:
            msg = f"{self} does not say enough to load an array."
            raise ParameterError(msg)
        return _load_array_source(self)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Load the array for numpy; each call reads the resource."""
        out = self.load()
        return out if dtype is None else out.astype(dtype, copy=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict which `from_dict` reads back."""
        out = asdict(self)
        out["dtype"] = None if self.dtype is None else np.dtype(self.dtype).str
        return out

    @classmethod
    def from_dict(cls, contents: dict[str, Any]) -> ArraySource:
        """Return the source `to_dict` wrote."""
        out = dict(contents)
        out["dims"] = tuple(out.get("dims", ()))
        out["shape"] = tuple(out.get("shape", ()))
        out["windows"] = tuple((a, b) for a, b in out.get("windows", ()))
        if out.get("dtype") is not None:
            out["dtype"] = np.dtype(out["dtype"])
        return cls(**out)
