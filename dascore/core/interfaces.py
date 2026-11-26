"""
Interfaces for the objects another package's classes can stand in for.

Each interface states the members dascore uses on an object it did not
build. dascore's own classes subclass their interface; another package's
classes register with it, as `BaseNumericCoord.register(MyCoord)`. A check
or annotation that accepts any implementation names the interface; one
naming a concrete class, such as `NumericCoord` or `Spool`, accepts only
dascore's. See [Interfaces](`dascore/docs/notes/interfaces.qmd`).

This module imports nothing else from dascore, so any module may import it.

Examples
--------
>>> import dascore as dc
>>> from dascore.core.interfaces import BaseCoordManager, BaseNumericCoord
>>>
>>> patch = dc.get_example_patch()
>>> assert isinstance(patch.get_coord("time"), BaseNumericCoord)
>>> assert isinstance(patch.coords, BaseCoordManager)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Self


class BaseCoord(ABC):
    """
    A coordinate: the labels along one or more dimensions of data.

    Register a class with a kind, `BaseNumericCoord` or `BaseStringCoord`,
    rather than with this.
    """

    __slots__ = ()

    # Fields of dascore's coordinate models, so stated here as annotations.
    units: Any
    step: Any
    shape: tuple[int, ...]
    dtype: Any
    # The one evenly sampled run of a numeric coordinate, else None; whether
    # the coordinate states a shape but no labels; the style it displays in.
    _grid: Any
    _partial: bool
    _rich_style: Any

    if TYPE_CHECKING:
        # A field of some coordinates and a property of others; a field
        # cannot override a property, so it is stated for type checkers.
        @property
        def values(self) -> Any:
            """The coordinate's labels."""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of labels."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """The number of dimensions."""

    @property
    @abstractmethod
    def data_id(self) -> str:
        """The id of the labels."""

    @property
    @abstractmethod
    def evenly_sampled(self) -> bool:
        """Whether the labels are one evenly sampled run."""

    @property
    @abstractmethod
    def step_exact(self) -> Any:
        """The step as an exact fraction, or None."""

    @property
    @abstractmethod
    def degenerate(self) -> bool:
        """Whether the coordinate holds no labels."""

    @property
    @abstractmethod
    def sorted(self) -> bool:
        """Whether the labels increase."""

    @property
    @abstractmethod
    def reverse_sorted(self) -> bool:
        """Whether the labels decrease."""

    @abstractmethod
    def min(self) -> Any:
        """The smallest label."""

    @abstractmethod
    def max(self) -> Any:
        """The largest label."""

    @abstractmethod
    def coord_range(self, extend: bool = True) -> Any:
        """The distance the labels span."""

    @abstractmethod
    def get_sample_count(
        self, value, samples: bool = False, enforce_lt_coord: bool = False
    ) -> int:
        """The number of samples a value spans."""

    @abstractmethod
    def get_next_index(
        self, value, samples=False, allow_out_of_bounds=False, relative=False
    ) -> Any:
        """The index a value has, or would have, in the coordinate."""

    @abstractmethod
    def approx_equal(self, other) -> bool:
        """Whether two coordinates hold nearly the same labels."""

    @abstractmethod
    def select(self, args, relative=False, samples=False) -> tuple[BaseCoord, Any]:
        """The selected coordinate and the indexer which selects it."""

    @abstractmethod
    def order(self, array, relative=False, samples=False) -> tuple[BaseCoord, Any]:
        """The coordinate in the order of some labels, and its indexer."""

    @abstractmethod
    def index(self, indexer, axis: int | None = None) -> BaseCoord:
        """The coordinate indexed by position."""

    @abstractmethod
    def sort(self, reverse=False) -> tuple[BaseCoord, Any]:
        """The coordinate sorted, and the indexer which sorts it."""

    @abstractmethod
    def new(self, **kwargs) -> BaseCoord:
        """A coordinate rebuilt by `get_coord` from its fields, some replaced."""

    @abstractmethod
    def update(self, **kwargs) -> BaseCoord:
        """A coordinate with new limits, step or labels, or in other units."""

    @abstractmethod
    def update_limits(self, min=None, max=None, step=None, **kwargs) -> BaseCoord:
        """A coordinate with new limits or a new step."""

    @abstractmethod
    def snap(self, step=None, max_shift=None, warn=False) -> BaseCoord:
        """The coordinate on one evenly sampled grid."""

    @abstractmethod
    def set_units(self, units) -> Self:
        """The coordinate with new units, its labels unchanged."""

    @abstractmethod
    def convert_units(self, units) -> Self:
        """The coordinate in other units."""

    @abstractmethod
    def simplify_units(self) -> Self:
        """The coordinate in simplified units."""

    @abstractmethod
    def change_length(self, length: int) -> Self:
        """The coordinate with a new length, its end moved."""

    @abstractmethod
    def reduce_coord(self, dim_reduce="empty") -> BaseCoord | None:
        """The coordinate left after reducing along its dimension."""

    @abstractmethod
    def align_to(self, other) -> tuple[BaseCoord, BaseCoord, Any, Any]:
        """Both coordinates on their shared labels, and their indexers."""

    @abstractmethod
    def to_summary(self, dims=()) -> Any:
        """A summary of the coordinate."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, item): ...

    @abstractmethod
    def __eq__(self, other) -> bool: ...

    @abstractmethod
    def _get_compatible_value(self, value, relative=False, from_end=True) -> Any:
        """A value in the coordinate's dtype and units."""

    @abstractmethod
    def _get_index_values(self, indices) -> Any:
        """The labels at some indices."""

    @abstractmethod
    def _get_index(self, value, forward=True) -> Any:
        """The index a value would occupy in the coordinate."""

    @abstractmethod
    def _get_slice_tuple(self, select, relative=False) -> tuple[Any, Any]:
        """The checked (start, stop) of a selection."""

    @abstractmethod
    def _samples_in(self, duration) -> Any:
        """How many steps a duration spans, unrounded."""

    @abstractmethod
    def _gap_tolerance(self, tolerance) -> Any:
        """The gap tolerance a public argument spells."""

    @abstractmethod
    def _identity(self) -> tuple[str, str]:
        """The id the coordinate has inside another id."""

    @abstractmethod
    def _repr_fields(self) -> tuple:
        """The facts a display of the coordinate states."""

    @abstractmethod
    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """The fields which rebuild the coordinate with `get_coord`."""


class BaseNumericCoord(BaseCoord):
    """A numeric or time-like coordinate, its labels held as runs."""

    __slots__ = ()

    # dascore's run classes (`Grid`, `Labels`, `Blank`) and the arrays the
    # stored runs window, by id.
    runs: tuple
    sources: Mapping[str, Any]

    @property
    @abstractmethod
    def runs_count(self) -> int:
        """The number of runs."""

    @property
    @abstractmethod
    def segments(self) -> tuple[BaseNumericCoord, ...]:
        """One coordinate per run."""

    @abstractmethod
    def _with_runs(self, runs, dtype=None) -> Self:
        """The coordinate holding other runs."""

    @abstractmethod
    def _run_offsets(self) -> Any:
        """The index each run starts at."""

    @abstractmethod
    def _run_labels(self, run, indices=None) -> Any:
        """The labels of one run."""

    @abstractmethod
    def _run_records(self) -> tuple:
        """Each segment as a spool index stores it, with a grid's step."""

    @abstractmethod
    def _direction(self) -> int:
        """1 for ascending labels, -1 for descending, 0 for neither."""

    @abstractmethod
    def _slice_runs(self, indices: range) -> BaseCoord:
        """The coordinate over a range of indices, its runs windowed."""


class BaseStringCoord(BaseCoord):
    """A coordinate of text labels."""

    __slots__ = ()


class BaseCoordManager(ABC):
    """The coordinates of a patch, and the dimensions they run along."""

    __slots__ = ()

    # Fields of dascore's manager model, so stated here as annotations.
    dims: tuple[str, ...]
    scalar: bool
    coord_map: Mapping[str, BaseCoord]
    dim_map: Mapping[str, tuple[str, ...]]

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the data the coordinates describe."""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of samples in the data."""

    @property
    @abstractmethod
    def coord_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Each coordinate's shape."""

    @property
    @abstractmethod
    def dim_to_coord_map(self) -> Mapping[str, tuple[str, ...]]:
        """The coordinates along each dimension."""

    @abstractmethod
    def get_coord(self, coord_name: str) -> BaseCoord:
        """A coordinate by name."""

    @abstractmethod
    def get_array(self, coord_name: str) -> Any:
        """A coordinate's labels."""

    @abstractmethod
    def get_axis(self, dim: str) -> int:
        """The axis of a dimension."""

    @abstractmethod
    def coord_range(self, coord_name: str) -> Any:
        """The distance a coordinate's labels span."""

    @abstractmethod
    def coord_size(self, coord_name: str) -> int:
        """The number of a coordinate's labels."""

    @abstractmethod
    def min(self, coord_name: str) -> Any:
        """A coordinate's smallest label."""

    @abstractmethod
    def max(self, coord_name: str) -> Any:
        """A coordinate's largest label."""

    @abstractmethod
    def step(self, coord_name: str) -> Any:
        """A coordinate's step."""

    @abstractmethod
    def get_coord_tuple_map(self) -> dict[str, tuple[tuple[str, ...], BaseCoord]]:
        """Each coordinate with the dimensions it runs along."""

    @abstractmethod
    def to_summary_dict(self) -> dict[str, Any]:
        """A summary of each coordinate."""

    @abstractmethod
    def validate_data(self, data) -> Any:
        """The data, if their shape is the coordinates'."""

    @abstractmethod
    def select(self, *args, **kwargs) -> tuple[BaseCoordManager, Any]:
        """The selected coordinates, and the selected array if given one."""

    @abstractmethod
    def isel(self, *args, **kwargs) -> tuple[BaseCoordManager, Any]:
        """The coordinates selected by position, and the array if given one."""

    @abstractmethod
    def select_indexers(self, **kwargs) -> tuple[BaseCoordManager, dict[str, Any]]:
        """The selected coordinates, and each dimension's indexer."""

    @abstractmethod
    def update(self, /, **kwargs) -> BaseCoordManager:
        """The coordinates with some added, replaced or dropped."""

    @abstractmethod
    def drop_coords(self, *coords, array=None) -> tuple[BaseCoordManager, Any]:
        """The coordinates without some, and the array without their axes."""

    @abstractmethod
    def drop_private_coords(self, array=None) -> tuple[BaseCoordManager, Any]:
        """The coordinates without the private ones."""

    @abstractmethod
    def drop_associated(self, dim: str, action: str | None) -> BaseCoordManager:
        """The coordinates without the non-dimensional ones along a dimension."""

    @abstractmethod
    def disassociate_coord(self, *coord: str) -> BaseCoordManager:
        """The coordinates with some no longer on their dimensions."""

    @abstractmethod
    def rename_coord(self, /, **kwargs) -> BaseCoordManager:
        """The coordinates with some renamed."""

    @abstractmethod
    def set_dims(self, **kwargs: str) -> BaseCoordManager:
        """The coordinates with other coordinates as dimensions."""

    @abstractmethod
    def transpose(self, *dims) -> BaseCoordManager:
        """The coordinates with the dimensions reordered."""

    @abstractmethod
    def squeeze(self, dim=None) -> BaseCoordManager:
        """The coordinates without dimensions of length one."""

    @abstractmethod
    def flip(self, *dims) -> BaseCoordManager:
        """The coordinates with some dimensions reversed."""

    @abstractmethod
    def sort(self, *coords, array=None, reverse=False) -> tuple[BaseCoordManager, Any]:
        """The coordinates sorted, and the array sorted with them."""

    @abstractmethod
    def snap(self, *coords, **kwargs) -> tuple[BaseCoordManager, Any]:
        """The coordinates snapped to grids, and the array with them."""

    @abstractmethod
    def decimate(self, **kwargs) -> tuple[BaseCoordManager, tuple[slice, ...]]:
        """The coordinates decimated, and the slices which decimate."""

    @abstractmethod
    def set_units(self, **kwargs) -> BaseCoordManager:
        """The coordinates with new units, their labels unchanged."""

    @abstractmethod
    def convert_units(self, **kwargs) -> BaseCoordManager:
        """The coordinates in other units."""

    @abstractmethod
    def simplify_units(self) -> BaseCoordManager:
        """The coordinates in simplified units."""

    @abstractmethod
    def make_broadcastable_to(self, shape, *, drop_coords=False) -> BaseCoordManager:
        """The coordinates of data broadcast to a shape."""

    @abstractmethod
    def __getitem__(self, item) -> BaseCoord: ...

    @abstractmethod
    def __contains__(self, key) -> bool: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[str, BaseCoord]]: ...

    @abstractmethod
    def __eq__(self, other) -> bool: ...

    @abstractmethod
    def _update_grid(self, *dims, **kwargs) -> BaseCoordManager:
        """The coordinates with a grid replaced, its cell edges dropped."""

    @abstractmethod
    def _select(self, *args, **kwargs) -> Any:
        """Select by queries; what `select` and `isel` share."""

    @abstractmethod
    def _get_coord_dims_tuple(self) -> tuple:
        """Each coordinate's name followed by its dimensions."""

    @abstractmethod
    def _get_dim_summary(self) -> dict:
        """A summary of each dimension."""

    @abstractmethod
    def _repr_section(self, depth: int = 0) -> Any:
        """The section a display of the coordinates draws."""


class BaseLazyArray(ABC):
    """
    An array whose values are read only when asked for.

    A patch holding one is lazy, so `load` reads it. Indexing one, which
    dascore does to select from it, should give a lazy array too, still
    unread. dascore's own selects through its recipe instead.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The array's shape."""

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """The array's dtype."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """The number of dimensions."""

    @abstractmethod
    def __array__(self, dtype=None, copy=None) -> Any:
        """Read the values; the array protocol."""


class BaseSpool(ABC):
    """
    A container of patches.

    `isinstance(x, BaseSpool)` asks whether x is any spool;
    `isinstance(x, Spool)` asks whether it is dascore's.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def spool_path(self) -> Any:
        """The directory or file the spool derives from, or None."""

    @property
    @abstractmethod
    def backend(self) -> Any:
        """The backend the spool's patches hold arrays of, or None."""

    @abstractmethod
    def set_backend(self, backend) -> BaseSpool:
        """A spool whose patches hold arrays of a backend."""

    @abstractmethod
    def select(self, *args, **kwargs) -> BaseSpool:
        """A spool of the patches, or parts of patches, selected."""

    @abstractmethod
    def chunk(self, *args, **kwargs) -> BaseSpool:
        """A spool of the patches joined and cut into windows."""

    @abstractmethod
    def concatenate(self, *args, **kwargs) -> BaseSpool:
        """A spool of the patches joined along a dimension."""

    @abstractmethod
    def update(self, *args, **kwargs) -> BaseSpool:
        """The spool with its files indexed again."""

    @abstractmethod
    def split(self, *args, **kwargs) -> Iterator[BaseSpool]:
        """Spools of the patches, by patches per spool or by spool count."""

    @abstractmethod
    def map(self, *args, **kwargs) -> list:
        """A function's result on each patch."""

    @abstractmethod
    def get_contents(self) -> Any:
        """A dataframe describing each patch."""

    @abstractmethod
    def get_gaps(self, *args, **kwargs) -> Any:
        """A dataframe of the gaps between patches."""

    @abstractmethod
    def get_coverage(self, *args, **kwargs) -> Any:
        """A dataframe of the stretches the patches cover."""

    @abstractmethod
    def get_patch_names(self, *args, **kwargs) -> Any:
        """A name for each patch."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator: ...

    @abstractmethod
    def __getitem__(self, item): ...

    @abstractmethod
    def __add__(self, other) -> BaseSpool: ...
