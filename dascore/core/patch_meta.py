"""The metadata half of a patch: what the data are, without holding them."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from functools import cached_property
from typing import Any, Self
from uuid import uuid4

import numpy as np
import pandas as pd

import dascore as dc
import dascore.proc
import dascore.proc.coords
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.source import PatchSource
from dascore.core.summary import PatchSummary
from dascore.exceptions import ParameterError
from dascore.utils.attrs import _values_equal
from dascore.utils.display import (
    NodeRepr,
    Repr,
    attrs_to_text,
    dataless_to_text,
    get_header_text,
    split_block,
)
from dascore.utils.identity import with_patch_id
from dascore.utils.patch import (
    check_patch_attrs,
    check_patch_coords,
    get_patch_names,
)
from dascore.utils.time import to_float

# The lineage fields an equality check leaves out, shared with the patch's
# own `equals` so the two cannot drift apart.
_LINEAGE = dascore.proc.basic._LINEAGE


def _attr_values_equal(one, two) -> bool:
    """Whether two attr values say the same thing, two nulls included."""
    if _values_equal(one, two):
        return True
    # A NaN equals nothing, itself included, but two of them agree. An
    # array answers `isnull` elementwise, which is not the question here.
    with suppress(TypeError, ValueError):
        return bool(pd.isnull(one) and pd.isnull(two))
    return False


def _as_dtype(dtype):
    """Return a numpy dtype where there is one, else the backend's own."""
    try:
        return np.dtype(dtype)
    except TypeError:
        # A string is a spelling numpy should know; a misspelling is an error.
        if isinstance(dtype, str):
            raise
        return dtype


class PatchMeta(NodeRepr):
    """
    Everything a patch is apart from its data.

    A `PatchMeta` states the coordinates, attributes, dtype and array
    backend of data it does not hold, so an operation can work out what a
    result would look like before anything is read or computed.
    [`to_patch`](`dascore.PatchMeta.to_patch`) gives it data, making the
    [`Patch`](`dascore.Patch`) it describes; `new` and `update` change the
    description and hand back another `PatchMeta`.

    Parameters
    ----------
    coords
        The coordinates, or dimensional labels, of the described data.
        Takes the same input as [`Patch`](`dascore.Patch`).
    dims
        A sequence of dimension strings, in array order. Read from `coords`
        when it is a
        [`CoordManager`](`dascore.core.coordmanager.CoordManager`).
    attrs
        Optional attributes (non-coordinate metadata) passed as a dict or
        [PatchAttrs](`dascore.core.attrs.PatchAttrs`).
    dtype
        The dtype of the described data.
    backend
        The array backend the data are in, as
        [`backend_name`](`dascore.utils.array_api.backend_name`) spells it.
        Kept so an operation can resolve its kernel without the data.
    source
        Internal source metadata supplied by the I/O framework.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> meta = patch.drop_data()
    >>> assert meta.shape == patch.shape
    >>> assert meta.dtype == patch.dtype
    >>>
    >>> # Data make it a patch again.
    >>> assert isinstance(meta.to_patch(patch.data), dc.Patch)
    """

    coords: CoordManager
    attrs: PatchAttrs
    dims: tuple[str, ...]
    dtype: Any
    backend: str
    # The class default also covers objects restored from older pickles.
    _source: PatchSource | None = None
    # The class `to_patch` builds; None means a plain `Patch`. Set by
    # `Patch.drop_data`, so an operation on a subclass returns that subclass.
    _patch_type: type | None = None

    def __init__(
        self,
        coords: Mapping[str, Any] | CoordManager | None = None,
        dims: Sequence[str] | None = None,
        attrs: Mapping | PatchAttrs | None = None,
        dtype: Any = None,
        backend: str = "numpy",
        source: PatchSource | None = None,
    ):
        if dims is None and isinstance(coords, CoordManager):
            dims = coords.dims
        if coords is None or dims is None or dtype is None:
            msg = (
                "coords, dims and dtype must be defined to init PatchMeta; "
                "it describes data it does not hold."
            )
            raise ValueError(msg)
        self._dtype = _as_dtype(dtype)
        self._backend = backend
        self._set_state(get_coord_manager(coords, dims=dims), attrs, source)

    def _set_state(self, coords, attrs, source) -> None:
        """Store the state every patch object holds, and mint its identity."""
        self._coords = coords
        # Data which names no source still says which data it is, so that
        # everything downstream has something to carry forward.
        self._attrs = with_patch_id(PatchAttrs.from_dict(attrs))
        self._source = source
        # Lineage identity: minted eagerly so copies made at any point
        # (deepcopy/pickle carry __dict__) share it deterministically, and
        # so a spool, which keys patches by it, tells two of them apart.
        self._instance_id = uuid4().hex

    def _repr_node(self) -> Repr:
        """The banner, the coordinates, the described data and the attributes."""
        attrs = self.attrs
        return Repr(
            header=get_header_text("PatchMeta ⚡"),
            body=(
                self.coords._repr_section(),
                split_block(
                    dataless_to_text(self.dtype, units=attrs.get("data_units"))
                ),
                split_block(attrs_to_text(attrs)),
            ),
        )

    @property
    def coords(self) -> CoordManager:
        """
        Return the coordinates of the described data.

        Examples
        --------
        >>> import dascore as dc
        >>> meta = dc.get_example_patch().drop_data()
        >>> assert 'time' in meta.coords
        """
        return self._coords

    @property
    def attrs(self) -> PatchAttrs:
        """
        Return the non-coordinate metadata.

        Examples
        --------
        >>> import dascore as dc
        >>> meta = dc.get_example_patch().drop_data()
        >>> assert hasattr(meta.attrs, 'data_type')
        """
        return self._attrs

    @property
    def dims(self) -> tuple[str, ...]:
        """
        Return the dimensions of the described data.

        Examples
        --------
        >>> import dascore as dc
        >>> meta = dc.get_example_patch().drop_data()
        >>> assert 'time' in meta.dims
        """
        return self.coords.dims

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the described data."""
        return self._dtype

    @property
    def backend(self) -> str:
        """Return the array backend the described data are in."""
        return self._backend

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the described data."""
        return len(self.coords.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the described data."""
        return self.coords.shape

    @property
    def size(self) -> int:
        """Return the size of the described data."""
        return self.coords.size

    @property
    def coord_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Return an immutable mapping of {coordinate: (shape, ...)}."""
        return self.coords.coord_shapes

    @property
    def seconds(self) -> float:
        """Return number of seconds in the time coordinate."""
        return to_float(self.coords.coord_range("time"))

    @property
    def channel_count(self) -> int:
        """Return number of channels in the distance coordinate."""
        return self.coords.coord_size("distance")

    @cached_property
    def summary(self) -> PatchSummary:
        """Return a metadata-only summary."""
        return PatchSummary.from_patch(self)

    def flat_dump(self, exclude=None) -> dict:
        """Return a flat summary dict for dataframe-oriented helpers."""
        return self.summary.flat_dump(exclude=exclude)

    def __eq__(self, other):
        """Compare against another object; see `equals`."""
        return self.equals(other)

    def equals(self, other: Any, only_required_attrs: bool = True) -> bool:
        """
        Whether another object describes the same data.

        A patch is not equal to metadata describing it: the one holds the
        data the other only names.

        Parameters
        ----------
        other
            The object to compare against.
        only_required_attrs
            If True, compare only the declared attrs, which leaves out
            history and anything a caller attached.
        """
        if type(other) is not type(self):
            return False
        if self.dtype != other.dtype:
            return False
        return self._metadata_equals(other, only_required_attrs)

    def _metadata_equals(self, other, only_required_attrs: bool) -> bool:
        """
        Whether two objects carry the same coords and attrs.

        The half a patch and its metadata compare alike, so the two cannot
        drift: `Patch.equals` adds the data to this, and `PatchMeta.equals`
        adds the dtype, which is all a description has of them.
        """
        if self.coords != other.coords:
            return False
        keep = set(PatchAttrs.model_fields) - {"history"} - _LINEAGE
        # Annotated because these are keyword arguments for `model_dump`, not
        # a mapping of one type; without it each set is checked against every
        # parameter the method takes.
        dump: dict[str, Any] = (
            dict(include=keep) if only_required_attrs else dict(exclude=_LINEAGE)
        )
        mine, theirs = self.attrs.model_dump(**dump), other.attrs.model_dump(**dump)
        if set(mine) != set(theirs):
            return False
        return all(_attr_values_equal(mine[x], theirs[x]) for x in mine)

    def _like(self, coords: CoordManager, attrs: PatchAttrs, dtype=None) -> PatchMeta:
        """Return metadata like this one's, describing the same kind of patch."""
        out = type(self)(
            coords=coords,
            attrs=attrs,
            dtype=self.dtype if dtype is None else dtype,
            backend=self.backend,
            source=self._source,
        )
        out._patch_type = self._patch_type
        return out

    def _new_like(self, data, coords: CoordManager, attrs: PatchAttrs, dtype=None):
        """Return the metadata `new` builds; metadata has nowhere to put data."""
        if data is not None:
            msg = (
                "A PatchMeta describes data rather than holding any, so `new` "
                "cannot take data. Use meta.to_patch(data) for the patch it "
                "describes."
            )
            raise ParameterError(msg)
        return self._like(coords, attrs, dtype)

    def to_patch(self, data) -> dc.Patch:
        """
        Return the patch this describes, holding `data`.

        The inverse of [`drop_data`](`dascore.Patch.drop_data`): one takes
        a patch's data away and keeps what described them, this gives that
        description its data back. The patch is of the class the metadata
        was dropped from, so a `Patch` subclass round-trips.

        Parameters
        ----------
        data
            The array this metadata describes. Its shape is checked against
            the coords, as any patch's is; its dtype and backend are not. A
            patch's are its data's, so a description they contradict is
            merely out of date -- `real()` derives complex metadata from a
            complex patch and computes a float array through here.

        Examples
        --------
        >>> import dascore as dc
        >>> patch = dc.get_example_patch()
        >>> meta = patch.drop_data()
        >>> assert meta.to_patch(patch.data).equals(patch)
        """
        patch_class = self._patch_type or dc.Patch
        out = patch_class(data=data, coords=self.coords, attrs=self.attrs)
        out._source = self._source
        return out

    def _reattach(self, out: PatchMeta, attrs: PatchAttrs) -> PatchMeta:
        """Return `out` under `attrs`; there is no data to put back."""
        return out.new(attrs=attrs)

    # --- metadata operations, which every patch object can run.

    # The operations written as `PatchProcessor` subclasses which compute no
    # data. Written here rather than attached at import so that a reader, an
    # IDE and a type checker all see what metadata can do; each body builds
    # its processor and runs it, and the framework refuses a method which is
    # missing, on the wrong class, or whose parameters have drifted from the
    # processor's fields. The operation is documented once, with its class,
    # and that docstring replaces the summary line below at import.

    def rename_coords(self, /, **kwargs) -> Self:
        """Rename coordinates (or dimensions) of the patch."""
        return dascore.proc.coords.RenameCoords(**kwargs).run(self)

    def update_coords(self, /, **kwargs) -> Self:
        """Update the coordinates of the patch."""
        return dascore.proc.coords.UpdateCoords(**kwargs).run(self)

    def drop_coords(self, *coords: str | Iterable[str]) -> Self:
        """Drop coordinates from the patch."""
        # By name, never positionally: `coords` is one field holding them
        # all, and `DropCoords(*coords)` would hand the first to the first
        # field and refuse the rest.
        return dascore.proc.coords.DropCoords(coords=coords).run(self)

    def coords_from_df(
        self,
        dataframe: pd.DataFrame,
        units: dict[str, Any] | None = None,
        extrapolate: bool = False,
    ) -> Self:
        """Update non-dimensional coordinates from a dataframe."""
        return dascore.proc.coords.CoordsFromDf(
            dataframe=dataframe, units=units, extrapolate=extrapolate
        ).run(self)

    update = dascore.proc.update
    # Before 0.1.0 update was called new, this is for backwards compatibility.
    new = dascore.proc.update
    update_attrs = dascore.proc.update_attrs
    check_coords = check_patch_coords
    check_attrs = check_patch_attrs
    # Names a coordinate a dimension and the old dimension a coordinate,
    # which reaches no sample: undecorated, so no data guard stands in the
    # way of metadata running it.
    set_dims = dascore.proc.set_dims
    get_coord = dascore.proc.get_coord
    get_axis = dascore.proc.get_axis
    pipe = dascore.proc.pipe
    get_patch_names = get_patch_names

    def get_patch_name(self, *args, **kwargs) -> str:
        """
        Return the name of the patch.

        See [`get_patch_names`](`dascore.utils.patch.get_patch_names`)
        for argument details.
        """
        return get_patch_names(self, *args, **kwargs).iloc[0]
