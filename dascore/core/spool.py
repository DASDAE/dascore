"""Module for spools, containers of patches."""

from __future__ import annotations

import abc
import inspect
import warnings
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar, overload

import numpy as np
import pandas as pd
from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.compat import UPath, is_array
from dascore.constants import (
    PROGRESS_LEVELS,
    WARN_LEVELS,
    ExecutorType,
    PatchType,
    attr_conflict_description,
    enrich_attrs_description,
    enrich_conflicts_description,
    enrich_coords_description,
    enrich_on_missing_description,
    namespace_select_type,
    numeric_types,
    path_types,
    timeable_types,
)
from dascore.core.inventory import Inventory
from dascore.exceptions import (
    InvalidSpoolError,
    InvalidSpoolQueryError,
    MissingPatchError,
    ParameterError,
    UnresolvedPatchError,
)
from dascore.utils.chunk_plan import (
    _SOURCE_COLUMNS,
    ChunkPlan,
    _combined_dtype,
    _ensure_patch_id,
    build_chunk_plan,
    samples_adjusted_envelopes,
)
from dascore.utils.display import get_dascore_text, get_nice_text
from dascore.utils.docs import compose_docstring, get_docstring
from dascore.utils.misc import (
    _spool_map,
    deep_equality_check,
)
from dascore.utils.namespace import NamespaceOwner
from dascore.utils.patch import (
    concatenate_patches,
    get_patch_names,
    stack_patches,
)
from dascore.utils.paths import coerce_to_upath, requires_local_directory
from dascore.utils.pd import present_units_columns, resolve_selector_namespaces

if TYPE_CHECKING:
    from dascore.io.index.catalog import PatchCatalog

T = TypeVar("T")


# Copy-on-write is always on from pandas 3, which also deprecates the
# option: reading it there warns on every access, so settle it by version
# once and only consult the option on pandas 2.
_COPY_ON_WRITE_ALWAYS = int(pd.__version__.split(".", maxsplit=1)[0]) >= 3


def _copy_public_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return a caller-owned view of an internally cached dataframe.

    Copy-on-write makes a shallow copy enough, since the frames detach on
    the first write. Only the literal True enables it; pandas 2 also
    accepts "warn", which keeps the old sharing semantics.
    """
    copy_on_write = _COPY_ON_WRITE_ALWAYS or pd.options.mode.copy_on_write is True
    return frame.copy(deep=not copy_on_write)


_VALID_ON_UNRESOLVED = ("warn", "raise", "ignore")

_UNRESOLVED_WARNING = (
    "The attached inventory does not describe every patch in this spool, and "
    "those it does not describe were not enriched. Use on_unresolved='raise' "
    "to see which, 'ignore' to silence this, or remove them from the spool "
    "once Spool.prune_to_inventory exists."
)


def _unstated(values) -> np.ndarray:
    """
    Return a mask of the entries which state no value.

    A patch says it does not know a name by leaving it null, which a
    string column spells as the empty string; both are what an attached
    inventory is asked to fill in.
    """
    series = pd.Series(np.asarray(values, dtype=object))
    return (series.isna() | series.eq("")).to_numpy()


def _match_resolved(values, name: str, selector) -> np.ndarray:
    """Return which of the values an inventory states match a selector."""
    from dascore.io.index.query import evaluate_attr_predicate  # noqa: PLC0415

    values = np.asarray(values, dtype=object)
    out = np.zeros(len(values), dtype=bool)
    # A name the inventory has no answer for is not one it can select on,
    # and the predicate would be comparing against the missing marker.
    stated = ~_unstated(values)
    if stated.any():
        out[stated] = evaluate_attr_predicate(values[stated], name, selector)
    return out


def _normalize_enrich_kwargs(kwargs) -> dict:
    """
    Canonicalize enrich arguments, rejecting any Patch.enrich would.

    Two spools which enrich identically have to compare equal, so an
    argument stated explicitly at its own default, or given as a list
    where a tuple would do, must reach the same stored form.
    """
    signature = inspect.signature(dc.Patch.enrich)
    valid = set(signature.parameters) - {"patch", "inventory"}
    if bad := sorted(set(kwargs) - valid):
        msg = (
            f"Spool.enrich got unknown argument(s) {bad}; it passes "
            f"{sorted(valid)} through to Patch.enrich."
        )
        raise ParameterError(msg)
    bound = signature.bind_partial(**kwargs)
    bound.apply_defaults()
    # A collection of names means what it holds, not which container holds it.
    return {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in bound.arguments.items()
        if name in valid
    }


def _combine_state(values, label):
    """Return the one value two spools agree on, or None if neither has one."""
    present = [x for x in values if x is not None]
    if not present:
        return None
    if len(present) == 2 and present[0] != present[1]:
        msg = (
            f"The spools carry different {label}, which have no combined "
            "meaning. Attach one inventory to the combined spool instead."
        )
        raise InvalidSpoolError(msg)
    return present[0]


def _combine_inventories(first, second) -> tuple:
    """
    Return the (inventory, enrich kwargs) a union of two spools carries.

    The two halves carry over independently: an inventory attached to one
    operand still describes the patches it came with, and so does the
    enrichment set up from it — which is why attaching the same inventory
    to the other operand cannot turn a working union into an error. Two
    operands answering either question differently have no single answer.
    """
    inventory = _combine_state(
        [getattr(x, "_inventory", None) for x in (first, second)], "inventories"
    )
    enrichment = _combine_state(
        [x._enrichment() if isinstance(x, Spool) else None for x in (first, second)],
        "enrich arguments",
    )
    # Enrichment is only reachable through an inventory, so it cannot
    # outlive one; a union carrying arguments and nothing to apply them
    # from would enrich nothing while claiming to.
    assert inventory is not None or enrichment is None
    return inventory, enrichment


class BaseSpool(NamespaceOwner, abc.ABC):
    """Spool Abstract Base Class (ABC) for defining Spool interface."""

    _rich_style = "bold"
    _namespace_entry_point_group = "dascore.spool_namespace"
    _namespace_attr_errors: ClassVar[dict[str, str]] = {
        "viz": (
            "'Spool' has no 'viz' namespace. "
            "Apply 'viz' on a Patch object. "
            "(you can merge a subset of the spool into a single patch using "
            "the Chunk function. i.e., spool.chunk(time=None)[0].viz.waterfall())"
        )
    }

    # An int selects one patch; a slice or array selects a sub-spool.
    @overload
    def __getitem__(self, item: int) -> dc.Patch: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray) -> BaseSpool: ...

    @abc.abstractmethod
    def __getitem__(self, item: int | slice | np.ndarray) -> dc.Patch | BaseSpool:
        """Return a patch, or a spool for a slice or array of indices."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[dc.Patch]:
        """
        Iterate through the Patches in the spool.

        Notes
        -----
        Iteration may skip patches in certain cases (e.g., when coordinate
        mismatches occur as described in issue #583). Therefore, the number
        of patches yielded during iteration may differ from len(spool).
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return len of spool."""

    def __rich__(self):
        """Rich rep. of spool."""
        text = get_dascore_text() + Text(" ")
        text += Text(self.__class__.__name__, style=self._rich_style)
        text += Text(" 🧵 ")
        patch_len = len(self)
        text += Text(f"({patch_len:d}")
        text += Text(" Patches)") if patch_len != 1 else Text(" Patch)")
        return text

    def __str__(self):
        return str(self.__rich__())

    __repr__ = __str__

    def __eq__(self, other) -> bool:
        """Simple equality checks on spools."""
        my_dict = self.__dict__
        other_dict = getattr(other, "__dict__", {})
        return deep_equality_check(my_dict, other_dict)

    def __add__(self, other) -> BaseSpool:
        """
        Combine two spools into one containing the patches of both.

        The result is a lazy spool over the union of both spools'
        metadata: file-backed patches stay unloaded, in-memory patches
        are shared (not copied), and the same source appearing in both
        spools keeps a single entry. Selections on the inputs carry over
        by row membership.

        Examples
        --------
        >>> import dascore as dc
        >>> sp1 = dc.get_example_spool("random_das")
        >>> sp2 = dc.get_example_spool("diverse_das")
        >>> combined = sp1 + sp2
        >>> assert len(combined) == len(sp1) + len(sp2)
        """
        if not isinstance(other, BaseSpool):
            return NotImplemented
        from dascore.io.index.catalog import PatchCatalog  # noqa: PLC0415

        members = [self._as_catalog_member(), other._as_catalog_member()]
        union = PatchCatalog.union(members)
        new = Spool()
        new._catalog = union
        # An attached inventory is part of what a spool yields, so it must
        # survive the union; two different ones have no combined answer.
        new._inventory, enrichment = _combine_inventories(self, other)
        if enrichment is not None:
            new._enrich_kwargs, new._on_unresolved = enrichment
        return new

    def _as_catalog_member(self):
        """
        Return (catalog, patch_ids) describing this spool for a union.

        `patch_ids` limits membership to the spool's current rows; None
        means the whole catalog (or the catalog view itself carries the
        selection). The base implementation materializes the patches.
        """
        from dascore.io.index.catalog import PatchCatalog  # noqa: PLC0415

        return PatchCatalog.from_patches(list(self)), None

    @abc.abstractmethod
    @compose_docstring(conflict_desc=attr_conflict_description)
    def chunk(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float = 1.5,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        group: str | Sequence[str] | None = None,
        missing_dim: Literal["raise", "drop"] = "raise",
        **kwargs,
    ) -> Self:
        """
        Chunk the data in the spool along specified dimension.

        Parameters
        ----------
        overlap
            The amount of overlap between each segment, starting with the end of
            first patch. Negative values can be used to create gaps.
        keep_partial
            If True, keep the segments which are smaller than chunk size.
            This often occurs because of data gaps or at end of chunks.
        snap_coords
            If True (default), simplify the coordinates of joined patches to
            an evenly sampled range when doing so moves no coordinate value
            by more than `tolerance` samples. Merges whose gaps exceed that
            keep an exact segmented coordinate instead.
        tolerance
            The maximum number of samples a block of data can be spaced (gap)
            and still be considered contiguous.
        conflict
            {conflict_desc}
        group
            Attributes which partition patches into separate outputs (their
            values differing is never an error). Defaults to the config
            option `groupby_attrs`; unlike the default, explicitly passed
            names must exist on at least one patch. Dimensions and
            coordinate identities always partition implicitly.
        missing_dim
            What to do when patches lack the chunked dimension: "raise"
            (default) or "drop" (exclude them from the output).
        kwargs
            kwargs are used to specify the dimension along which to chunk, eg:
            `time=10` chunks along the time axis in 10 second increments.
            The value may also be a quantity: one of the coordinate's own
            units (`time=10 * s`) or a data size (`time=25 * megabytes`),
            which chunks so each patch's data array is about that large.
            `overlap` accepts the same forms.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.units import s, megabytes
        >>>
        >>> spool = dc.get_example_spool("random_das")
        >>> # get spools with time duration of 10 seconds
        >>> time_chunked = spool.chunk(time=10, overlap=1)
        >>> # the same, with the units stated explicitly
        >>> unit_chunked = spool.chunk(time=10 * s)
        >>> # get patches whose data arrays are at most ~1 MB
        >>> size_chunked = spool.chunk(time=1 * megabytes)
        >>> # merge along time axis
        >>> time_merged = spool.chunk(time=...)

        Notes
        -----
        A data size measures the patch's data array only; coordinates and
        attrs are extra, as are any copies a later processing step makes,
        so the patch as a whole is somewhat larger. The sample count is
        rounded down, so the data never exceeds the requested size, and a
        merge of patches with different dtypes is sized against the dtype
        they upcast to.

        [`Spool.concatenate`](`dascore.BaseSpool.concatenate`) performs a
        similar operation but disregards the coordinate values.

        To inspect what a chunk call will do before running it — which
        output patches it produces and which slice of which source patch
        feeds each one — use
        [`Spool.chunk_plan`](`dascore.core.spool.Spool.chunk_plan`),
        which takes the same arguments and returns the plan without
        touching any data.
        """

    @abc.abstractmethod
    def select(
        self,
        *,
        _attrs: namespace_select_type = None,
        _coords: namespace_select_type = None,
        samples: bool = False,
        relative: bool = False,
        **kwargs,
    ) -> Self:
        """
        Sub-select parts of the spool.

        Can be used to specify dimension ranges, or unix-style matches
        on string attributes. Bare keyword names resolve against
        attributes first, then coordinates; unknown names raise.

        Parameters
        ----------
        _attrs
            Attribute selections: a dict of ``name -> selector`` (the
            general form — required when a name cannot be a Python
            keyword) or a name/collection of names tagging bare kwargs
            as attributes (disambiguates names shared with coordinates).
        _coords
            Coordinate selections; same forms as ``_attrs``, validating
            names as coordinates only.
        samples
            If True, selections are coordinate-only and given in sample
            indices; they never exclude patches, but are applied to each
            patch as it loads.
        relative
            If True, range bounds are relative to the spool's coordinate
            envelope: positive from the start, negative from the end.
        **kwargs
            Specifies query. Can be of the form {dim_name=(start, stop)}
            or {attr_name=query}.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("diverse_das")
        >>> # subselect data in a particular time range
        >>> time = ('2020-01-03', '2020-01-03T00:00:10')
        >>> time_spool = spool.select(time=time)
        >>> # subselect based on matching tag parameter
        >>> tag_spool = spool.select(tag='some*')
        """

    @abc.abstractmethod
    def get_contents(self) -> pd.DataFrame:
        """
        Get a dataframe of the spool contents.

        Notes
        -----
        Each call returns a caller-owned dataframe; mutating it never
        changes the spool. Use ``frame.copy(deep=True)`` when an eager
        block copy is needed.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("random_das")
        >>> df = spool.get_contents()
        """

    # Bind get_patch names as a spool method.
    get_patch_names = get_patch_names

    # --- optional methods

    def sort(self, attribute) -> Self:
        """
        Sort the Spool based on a specific attribute.

        Parameters
        ----------
        attribute
            The attribute or coordinate used for sorting. If a coordinate name
            is used, the sorting will be based on the minimum value.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool()
        >>> # sort spool based on values in time coordinate.
        >>> spool_time_sorted = spool.sort("time")
        >>> # sort spool based on values in tag
        >>> spool_tag_sorted = spool.sort("tag")
        """
        msg = f"spool of type {self.__class__} has no sort implementation"
        raise NotImplementedError(msg)

    def split(
        self,
        size: int | None = None,
        count: int | None = None,
    ) -> Generator[BaseSpool, None, None]:
        """
        Yield sub-patches based on specified parameters.

        Parameters
        ----------
        size
            The number of patches desired in each output spool. The last
            spool may have fewer patches. Must be greater than zero.
        count
            The number of spools to include. If count is greater than
            the length of the spool then the output will be smaller than
            count, with one patch per spool. Must be greater than zero.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("diverse_das")
        >>> # split spool into list of spools each with 3 patches.
        >>> split = spool.split(size=3)
        >>> # split spool into 3 evenly sized (if possible) spools
        >>> split = spool.split(count=3)
        """
        msg = f"spool of type {self.__class__} has no split implementation"
        raise NotImplementedError(msg)

    def update(self, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        Updates the contents of the spool, return the updated spool.

        Parameters
        ----------
        progress
            Controls the progress bar. "standard" produces the standard
            progress bar. "basic" is a simplified version with lower refresh
            rates, best for high-latency environments, and None disables
            the progress bar.
        """
        return self

    @compose_docstring(desc=get_docstring(concatenate_patches))
    def concatenate(self, check_behavior: WARN_LEVELS = "warn", **kwargs):
        """{desc}"""
        msg = f"spool of type {self.__class__} has no concatenate implementation"
        raise NotImplementedError(msg)

    def map(
        self,
        func: Callable[..., T],
        *,
        client: ExecutorType | None = None,
        size: int | None = None,
        progress: bool = True,
        **kwargs,
    ) -> list[T]:
        """
        Map a function of all the contents of the spool.

        Parameters
        ----------
        func
            A callable which takes a patch as its first argument.
        client
            A client, or executor, which has a `map` method.
        size
            The number of patches in each spool mapped to a client.
            If not set, defaults to the number of processors on the host.
            Does nothing unless client is defined.
        progress
            If True, display a progress bar.
        **kwargs
            kwargs passed to func.

        Notes
        -----
        When a client is specified, the spool is split then passed to the
        client's map method. This is to avoid serializing loaded patches.
        See [`Spool.split`](`dascore.core.spool.BaseSpool.split`) for more
        details about the `spool_count` and `spool_size` parameters.

        Examples
        --------
        import numpy as np
        import dascore as dc

        spool = dc.get_example_spool("random_das")

        # Calculate the std for each channel in 5 second chunks
        results = (
             spool.chunk(time=5)
             .map(lambda x: np.std(x.data, axis=0))
        )
        # stack back into array. dims are (distance, time chunk)
        out = np.stack(results, axis=-1)
        """
        return _spool_map(
            self,
            func,
            client=client,
            size=size,
            progress=progress,
            **kwargs,
        )

    # Add method for stacking (adding the data arrays) patches in spool.
    stack = stack_patches


class Spool(BaseSpool):
    """
    The concrete spool: a view over a `PatchCatalog`.

    Constructed from in-memory patches directly (or via
    [`dascore.spool`](`dascore.spool`)), from a directory of files with
    [`Spool.from_directory`](`dascore.core.spool.Spool.from_directory`),
    or from a single file with
    [`Spool.from_file`](`dascore.core.spool.Spool.from_file`).

    Parameters
    ----------
    data
        A patch, sequence of patches, or another spool whose (in-memory)
        patches this spool should hold; None creates an empty spool.

    Notes
    -----
    The catalog is the spool's entire state: live patches sit in its
    resolver registry, file-backed patches in its index tables, and
    restructured views (chunk/concat) are derived in-memory catalogs
    whose rows are the plan outputs. Selection, ordering, and windowing
    are lazy specs composed on the catalog; one engine serves every
    construction path.
    """

    # synthetic catalog identity columns must not join patch kwargs
    # comparisons or chunk merge-compatibility checks
    _drop_columns = (
        "patch",
        "source_path",
        "source_format",
        "source_version",
        "source_patch_id",
    )
    # The catalog backing this spool; every construction path sets one.
    _catalog: PatchCatalog
    # An attached inventory, the enrich kwargs to apply on extraction
    # (None means attached without automatic enrichment), and what to do
    # with a patch the inventory does not describe.
    _inventory = None
    _enrich_kwargs: dict | None = None
    _on_unresolved: str = "warn"
    # Whether this spool has already said its inventory covers only part of
    # it; the warning is worth making once, not once per patch.
    _warned_unresolved: bool = False
    # single-file provenance (set by from_file; drives update())
    _file_path = None
    _file_format = None
    _file_version = None

    def __init__(
        self,
        data: PatchType | Sequence[PatchType] | BaseSpool | None = None,
    ):
        from dascore.io.index.catalog import PatchCatalog  # noqa: PLC0415

        if isinstance(data, Spool):
            # copy-construction: share the catalog and provenance
            self.__dict__.update(data.__dict__)
            return
        if data is None:
            patches = ()
        elif isinstance(data, dc.Patch):
            patches = (data,)
        elif isinstance(data, BaseSpool):
            # e.g. wrapping dc.read output; the patches are in memory
            patches = tuple(data)
        elif isinstance(data, Sequence) and all(isinstance(x, dc.Patch) for x in data):
            patches = data
        else:
            msg = (
                "Spool accepts a Patch, a sequence of patches, or a "
                f"spool; got {type(data)}."
            )
            raise InvalidSpoolError(msg)
        self._catalog = PatchCatalog.from_patches(patches)

    # --- presented relation --------------------------------------------

    @property
    def _df(self) -> pd.DataFrame:
        """The realized flat relation (cached by the catalog)."""
        return self._catalog.to_df()

    @compose_docstring(doc=get_docstring(BaseSpool.get_contents))
    def get_contents(self) -> pd.DataFrame:
        """{doc}."""
        return present_units_columns(_copy_public_dataframe(self._df))

    def __len__(self):
        # counting pushes to SQL (or the cold live registry); the flat
        # relation is never realized just for a length
        return len(self._catalog)

    @overload
    def __getitem__(self, item: int) -> dc.Patch: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray) -> BaseSpool: ...

    def __getitem__(self, item) -> dc.Patch | BaseSpool:
        if isinstance(item, slice):
            # a lazy id-membership window (D2); never realizes the flat
            # relation, and keeps split()/map() parts cheap
            return self._new_from_catalog(self._catalog.window(item))
        if is_array(item):
            array = np.asarray(item)
            if not (
                np.issubdtype(array.dtype, np.bool_)
                or np.issubdtype(array.dtype, np.integer)
            ):
                msg = "Only bool or int dtypes are supported for spool array selection."
                raise ValueError(msg)
            return self._new_from_catalog(self._catalog.restrict(array))
        try:
            return self._maybe_enrich(self._catalog.get_patch(int(item)))
        except MissingPatchError:
            # MissingPatchError subclasses IndexError for backwards
            # compatibility; it must never masquerade as out-of-bounds
            raise
        except IndexError:
            msg = f"index of [{item}] is out of bounds for spool."
            raise IndexError(msg) from None

    def __iter__(self):
        # The catalog snapshots the relation once and skips patches which
        # cannot be resolved (see #583).
        for patch in self._catalog:
            yield self._maybe_enrich(patch)

    # --- selection and presentation specs -------------------------------

    @compose_docstring(doc=get_docstring(BaseSpool.select))
    def select(
        self,
        *,
        _attrs: namespace_select_type = None,
        _coords: namespace_select_type = None,
        samples: bool = False,
        relative: bool = False,
        **kwargs,
    ) -> Self:
        """{doc}."""
        inventory_query, _attrs, _coords, kwargs = self._split_inventory_query(
            _attrs, _coords, kwargs, samples
        )
        catalog = self._catalog.select(
            _attrs=_attrs,
            _coords=_coords,
            samples=samples,
            relative=relative,
            **kwargs,
        )
        out = self._new_from_catalog(catalog)
        if inventory_query:
            out = out._select_from_inventory(inventory_query)
        return out

    def _split_inventory_query(self, _attrs, _coords, kwargs, samples):
        """
        Split selectors into the ones the index answers and the rest.

        A name the attached inventory could contribute is evaluated per
        row rather than pushed into SQL: the index states it for some rows
        and the inventory only fills in the others.
        """
        if self._inventory is None:
            return {}, _attrs, _coords, kwargs
        names = self._inventory.get_names()
        backend = self._catalog.backend
        known_attrs, known_coords = (
            set(backend.attr_names()),
            set(backend.coord_names()),
        )
        # A tag-form _attrs/_coords names bare kwargs, so every requested
        # name is either a bare kwarg or a key of a mapping form.
        requested = set(kwargs)
        for spec in (_attrs, _coords):
            if isinstance(spec, Mapping):
                requested |= set(spec)
        if channel_level := sorted(
            requested & set(names.coords) - known_attrs - known_coords
        ):
            msg = (
                f"{channel_level} name coordinates the attached inventory "
                "defines along the fiber, which selection cannot trim to "
                "yet. Enrich the patches and select on each one instead."
            )
            raise InvalidSpoolQueryError(msg)
        # samples=True selections are coordinate-only, so an attr among
        # them is an error the index states better than this can.
        selectable = set(names.attrs) - known_coords
        if samples or not requested & selectable:
            return {}, _attrs, _coords, kwargs
        attrs, coords = resolve_selector_namespaces(
            known_attrs | selectable,
            known_coords,
            _attrs=_attrs,
            _coords=_coords,
            kwargs=kwargs,
        )
        query = {x: attrs.pop(x) for x in list(attrs) if x in selectable}
        return query, attrs, coords, {}

    def _select_from_inventory(self, query: dict) -> Self:
        """
        Keep the rows whose inventory-backed values match.

        Precedence is per row: a row which states the name is judged by
        the index, exactly as it would be without an inventory, and only
        the rows leaving it unstated are resolved. A spool whose headers
        state everything therefore never touches the inventory, and one
        which states nothing resolves once per epoch rather than per row.
        A row the inventory has no answer for is not selected, as a patch
        lacking the attr entirely is not.
        """
        from dascore.proc.inventory import (  # noqa: PLC0415
            get_attr_values,
            resolve_contexts,
        )

        df = self._df
        if not len(df):
            return self
        # Resolution needs an identity and a time, which the relation
        # always carries: they are structural columns of the index.
        assert {"acquisition_key", "time_min", "time_max"} <= set(df.columns)
        ids = df["_patch_id"].to_numpy()
        contexts = None
        mask = np.ones(len(df), dtype=bool)
        for name, selector in query.items():
            stated = (
                ~_unstated(df[name])
                if name in df.columns
                else np.zeros(len(df), dtype=bool)
            )
            # SQL never matches a row which states nothing, so this is the
            # verdict for the stated rows and False everywhere else.
            matched = np.isin(ids, self._index_matches(name, selector))
            if not stated.all():
                if contexts is None:
                    contexts = resolve_contexts(
                        self._inventory,
                        df["acquisition_key"],
                        df["time_min"],
                        df["time_max"],
                    )
                matched[~stated] = _match_resolved(
                    get_attr_values(self._inventory, contexts[~stated], name),
                    name,
                    selector,
                )
            mask &= matched
        return self._new_from_catalog(self._catalog.restrict(mask))

    def _index_matches(self, name: str, selector) -> np.ndarray:
        """Return the ids of the rows the index itself selects for one name."""
        try:
            catalog = self._catalog.select(_attrs={name: selector})
        except InvalidSpoolQueryError:
            # No patch in this spool states the name, so the index selects
            # none of them and the inventory answers for every row.
            return np.empty(0, dtype=np.int64)
        return np.asarray(catalog._ordered_ids(), dtype=np.int64)

    def attach_inventory(self, inventory) -> Self:
        """
        Attach a DASDAE inventory to this spool.

        The spool carries the reference and nothing else: attaching costs
        no work per patch and adds nothing to the patches it yields. Call
        [`Spool.enrich`](`dascore.core.spool.Spool.enrich`) to copy the
        inventory's metadata onto the patches as they are extracted.

        Attaching replaces whatever the spool carried before, and clears
        enrichment set up from it — swapping the inventory silently under
        a configured enrichment would change every patch's metadata, so
        the new one has to be asked for. `enrich()` resumes with defaults.

        Parameters
        ----------
        inventory
            The inventory to carry.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).attach_inventory(inventory)
        >>> assert "gauge_length" not in dict(spool[0].attrs)
        >>> assert spool.enrich()[0].attrs.gauge_length == 10.0

        Notes
        -----
        This is the metadata half of the workflow: resolving the index
        against the inventory, subdividing it at epoch boundaries, and
        selecting on inventory tracks are not implemented yet.
        """
        if not isinstance(inventory, Inventory):
            msg = f"attach_inventory needs an Inventory, got {type(inventory)}."
            raise ParameterError(msg)
        new = self.__class__(self)
        new._inventory = inventory
        new._enrich_kwargs = None
        return new

    def remove_inventory(self) -> Self:
        """
        Return a spool carrying no inventory.

        Any enrichment set up by [`Spool.enrich`](`dascore.core.spool.Spool.enrich`)
        goes with it, since there is nothing left to enrich from. A spool
        with no inventory is returned unchanged in substance; as everywhere
        else, the original spool is left alone.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).enrich(inventory)
        >>> plain = spool.remove_inventory()
        >>> assert "gauge_length" not in dict(plain[0].attrs)
        >>> assert spool[0].attrs.gauge_length == 10.0  # the original stands
        """
        new = self.__class__(self)
        new._inventory = None
        new._enrich_kwargs = None
        return new

    @compose_docstring(
        attrs_desc=enrich_attrs_description,
        coords_desc=enrich_coords_description,
        on_missing_desc=enrich_on_missing_description,
        conflicts_desc=enrich_conflicts_description,
    )
    def enrich(
        self,
        inventory=None,
        *,
        on_unresolved: Literal["warn", "raise", "ignore"] = "warn",
        **kwargs,
    ) -> Self:
        """
        Enrich each patch this spool yields from an inventory.

        The work happens as each patch is extracted, not now, so this is
        cheap on a large spool and costs one
        [`Patch.enrich`](`dascore.proc.inventory.enrich`) per patch which
        actually comes out. Enrichment survives `select`, `sort`, `chunk`
        and friends; `Spool.remove_inventory` undoes it.

        Enriching never removes a patch: one the inventory does not
        describe comes out unchanged rather than missing, so an inventory
        covering part of an archive needs no pruning first. Deciding
        membership will be `prune_to_inventory`'s job, and leaving it there is
        what keeps this lazy — nothing resolves until a patch is pulled.

        Parameters
        ----------
        inventory
            The inventory to enrich from. Defaults to the spool's attached
            inventory; given one, it is attached as well.
        on_unresolved
            What to do with a patch the inventory does not describe — one
            naming no entry, or naming one the inventory does not resolve
            to exactly one of. "warn" (the default) leaves it un-enriched
            and says so, "ignore" leaves it silently, and "raise" fails.
            A patch which *straddles* two epochs is described twice rather
            than not at all, and raises regardless: it needs subdividing.
        **kwargs
            Held and passed to
            [`Patch.enrich`](`dascore.proc.inventory.enrich`) for each
            extracted patch. The names accepted are read from its
            signature, so the two cannot disagree, and only the names are
            checked at this point — the values each patch's own enrichment
            checks as it is extracted. Calling `enrich` again replaces
            these rather than adding to them. They are:

        Other Parameters
        ----------------
        {attrs_desc}
        {coords_desc}
        acquisition_key
            The inventory identity to resolve, for patches which do not
            carry one. Given both, each patch and this argument must agree.
        time
            The instant to resolve at, for patches whose time axis is not
            physical. A patch with a real time coordinate resolves at its
            own time and passing this raises.
        {on_missing_desc}
        {conflicts_desc}

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).enrich(inventory)
        >>> assert spool[0].attrs.gauge_length == 10.0
        >>>
        >>> # Or name what is wanted, as with Patch.enrich.
        >>> spool = dc.spool(patch).enrich(inventory, coords=False)
        """
        # Settled now rather than on extraction: a misspelled argument
        # should be an error here, not on some patch pulled much later.
        enrich_kwargs = _normalize_enrich_kwargs(kwargs)
        if on_unresolved not in _VALID_ON_UNRESOLVED:
            msg = (
                f"on_unresolved must be one of {_VALID_ON_UNRESOLVED}, "
                f"got {on_unresolved!r}."
            )
            raise ParameterError(msg)
        if inventory is None and self._inventory is None:
            msg = (
                "Spool.enrich needs an inventory: pass one, or attach one "
                "first with Spool.attach_inventory."
            )
            raise ParameterError(msg)
        new = (
            self.__class__(self)
            if inventory is None
            else self.attach_inventory(inventory)
        )
        new._enrich_kwargs = enrich_kwargs
        new._on_unresolved = on_unresolved
        return new

    def _enrichment(self):
        """Return how this spool enriches, or None if it does not."""
        if self._inventory is None or self._enrich_kwargs is None:
            return None
        return self._enrich_kwargs, self._on_unresolved

    def _maybe_enrich(self, patch):
        """Enrich an extracted patch when enrichment is set up."""
        if (enrichment := self._enrichment()) is None:
            return patch
        kwargs, on_unresolved = enrichment
        try:
            return patch.enrich(self._inventory, **kwargs)
        except UnresolvedPatchError:
            # The inventory does not describe this patch. Dropping it is
            # prune_to_inventory's job, so it comes out as it went in.
            if on_unresolved == "raise":
                raise
            if on_unresolved == "warn" and not self._warned_unresolved:
                self._warned_unresolved = True
                # Deduped per spool, which is the granularity that means
                # something. Naming the patch would warn once per distinct
                # key -- thousands of them on exactly the partly-covered
                # archive this default exists to serve -- while leaving a
                # constant message to Python's registry would silence every
                # spool after the first in a session.
                warnings.warn_explicit(
                    _UNRESOLVED_WARNING, UserWarning, __file__, 0, registry=None
                )
            return patch

    @compose_docstring(doc=get_docstring(BaseSpool.sort))
    def sort(self, attribute) -> Self:
        """{doc}."""
        # a lazy ORDER BY spec (D2): no copy, no realization; the
        # ordinal contract supplies the deterministic tiebreak
        return self._new_from_catalog(self._catalog.order_by(attribute))

    @compose_docstring(doc=get_docstring(BaseSpool.split))
    def split(
        self,
        size: int | None = None,
        count: int | None = None,
    ) -> Generator[BaseSpool, None, None]:
        """{doc}."""
        if not ((count is not None) ^ (size is not None)):
            msg = "Spool.split requires either spool_count or spool_size."
            raise ParameterError(msg)
        value = count if count is not None else size
        assert value is not None  # the check above sets exactly one of them
        # A step of zero or less never advances start, so the loop below
        # would yield forever.
        if value <= 0:
            msg = f"Spool.split requires a positive size or count, got {value}."
            raise ParameterError(msg)
        start = 0
        if count is not None:
            step = int(np.ceil(len(self) / value))
        else:
            step = int(np.ceil(value))  # tolerate a non-integral size
        while start < len(self):
            yield self[start : start + step]
            start += step

    def _new_from_catalog(self, catalog) -> Self:
        """Create a spool view over a (possibly derived) catalog."""
        new = self.__class__(self)
        new._catalog = catalog
        return new

    def _as_catalog_member(self):
        """
        Return (catalog, patch_ids) describing this spool for a union.

        Row membership (attr predicates, windows, id arrays) survives a
        table union as-is, but residual trims and order specs live
        Python-side and would silently vanish; a spool carrying those
        first bakes them into a derived catalog (tables only — no patch
        data is loaded). A catalog default order (directory time
        presentation) bakes only when the source-record transfer would
        actually present rows differently — an interleaved multi-patch
        file — so ordinary archives keep record-grain transfer and its
        same-source deduplication.
        """
        catalog = self._catalog
        if catalog._residuals or catalog._order is not None:
            return self._materialize_lossy(), None
        if catalog._default_order is not None and not self._transfer_keeps_order():
            return self._materialize_lossy(), None
        return catalog, None

    def _transfer_keeps_order(self) -> bool:
        """True when ordinal-grain transfer matches the presented order."""
        catalog = self._catalog
        presented = catalog._ordered_ids()
        by_ordinal = tuple(
            catalog.backend.query_ids(
                list(catalog._queries) or None,
                order_by=None,
                patch_ids=catalog._ids,
            )
        )
        return tuple(presented) == by_ordinal

    def _materialize_lossy(self):
        """
        Bake residual trims and presentation order into a derived catalog.

        An identity plan over the view's presented rows: one output per
        row (in presentation order, so ordinals record the order spec),
        with trimmed envelopes as the output envelopes and the trims
        themselves re-applied at load through the plan resolver.
        """
        from dascore.io.index.planned import derived_catalog  # noqa: PLC0415

        rows = self._df.reset_index(drop=True)
        working = samples_adjusted_envelopes(rows, self._catalog._residuals)
        working = working.reset_index(drop=True)
        ids = np.arange(len(working), dtype=np.int64)
        # outputs are not file rows: source bookkeeping stays on the
        # members (where loading needs it), never on the derived rows
        outputs = working.drop(
            columns=["_patch_id", *_SOURCE_COLUMNS], errors="ignore"
        ).assign(output_id=ids)
        members = pd.DataFrame(
            {
                "output_id": ids,
                "_patch_id": working.get("_patch_id", pd.Series(dtype=object)).values,
                "_modified": False,
            }
        )
        plan = ChunkPlan(outputs, members, "", None, {})
        return derived_catalog(
            source_rows=working,
            plan=plan,
            parent=self._catalog,
            merge_kwargs={},
            mode="identity",
            origin_path=self.spool_path,
        )

    # --- restructuring (materializing) operations -----------------------

    def _plan_frames(self, dim: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (source_rows, working) frames for planning along ``dim``.

        Re-planning the *same* dimension collapses (never nests): a
        derived catalog re-plans from its members — the trimmed source
        rows — restricted to the outputs the current view presents.
        Planning a *different* dimension must keep the already-assembled
        boundaries, so it plans over the current output rows themselves
        (loaded through the plan resolver). Patch-local samples
        residuals adjust the working envelopes so plans reflect the
        loading truth.
        """
        from dascore.io.index.planned import (  # noqa: PLC0415
            PlanResolver,
            collapse_working_df,
        )

        resolver = self._catalog.resolver
        same_dim = isinstance(resolver, PlanResolver) and resolver.dim == dim
        base = collapse_working_df(self._catalog) if same_dim else None
        if base is None:
            base = self._catalog.to_df().reset_index(drop=True)
        base = _ensure_patch_id(base)
        working = base.drop(columns=list(self._drop_columns), errors="ignore")
        working = samples_adjusted_envelopes(working, self._catalog._residuals)
        base = base[base["_patch_id"].isin(working["_patch_id"])]
        return base.reset_index(drop=True), working.reset_index(drop=True)

    def chunk_plan(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float = 1.5,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        group: str | Sequence[str] | None = None,
        missing_dim: Literal["raise", "drop"] = "raise",
        **kwargs,
    ):
        """
        Return the plan `chunk` would execute, without touching any data.

        The returned [`ChunkPlan`](`dascore.utils.chunk_plan.ChunkPlan`) is a
        read-only diagnostic: its `outputs` table describes each patch the
        chunked spool would contain (envelopes, step, carried attributes),
        its `members` table shows exactly which slice of which source patch
        feeds each output, and `params` records every resolved parameter
        (including the group attributes and sampling tolerance in effect).
        Accepts the same arguments as
        [`chunk`](`dascore.BaseSpool.chunk`).

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("random_das")
        >>> plan = spool.chunk_plan(time=3)
        >>> assert len(plan.outputs) == len(spool.chunk(time=3))
        >>> # See which sources contribute to the first output patch.
        >>> members = plan.members
        >>> first = members[members["output_id"] == 0]
        """
        _, working = self._plan_frames(next(iter(kwargs), None))
        return build_chunk_plan(
            working,
            overlap=overlap,
            keep_partial=keep_partial,
            snap_coords=snap_coords,
            tolerance=tolerance,
            conflict=conflict,
            group=group,
            missing_dim=missing_dim,
            **kwargs,
        )

    @compose_docstring(doc=get_docstring(BaseSpool.chunk))
    def chunk(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float = 1.5,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        group: str | Sequence[str] | None = None,
        missing_dim: Literal["raise", "drop"] = "raise",
        **kwargs,
    ) -> Self:
        """{doc}"""
        from dascore.io.index.planned import derived_catalog  # noqa: PLC0415

        source_rows, working = self._plan_frames(next(iter(kwargs), None))
        plan = build_chunk_plan(
            working,
            overlap=overlap,
            keep_partial=keep_partial,
            snap_coords=snap_coords,
            tolerance=tolerance,
            conflict=conflict,
            group=group,
            missing_dim=missing_dim,
            **kwargs,
        )
        merge_kwargs = {
            "conflicts": conflict,
            "snap_coords": snap_coords,
            "tolerance": tolerance,
        }
        catalog = derived_catalog(
            source_rows=source_rows,
            plan=plan,
            parent=self._catalog,
            merge_kwargs=merge_kwargs,
            mode="chunk",
            origin_path=self.spool_path,
        )
        return self._new_from_catalog(catalog)

    @compose_docstring(desc=get_docstring(concatenate_patches))
    def concatenate(self, check_behavior: WARN_LEVELS = "warn", **kwargs) -> Self:
        """{desc}"""
        from dascore.io.index.planned import derived_catalog  # noqa: PLC0415

        if len(kwargs) != 1:
            msg = (
                "concatenate requires exactly one dimension keyword, "
                f"got {sorted(kwargs)}"
            )
            raise ParameterError(msg)
        ((dim, value),) = kwargs.items()
        value = None if value is Ellipsis else value
        source_rows, working = self._plan_frames(dim)
        # a dim absent from the metadata envelopes is legal: concatenate
        # can stack patches along a brand-new dimension
        has_envelope = f"{dim}_min" in working.columns
        count = len(working) if value in (None,) else int(value)
        count = max(count, 1)
        rows = working.reset_index(drop=True)
        member_frames = []
        output_rows = []
        for output_id, start in enumerate(range(0, len(rows), count)):
            group_rows = rows.iloc[start : start + count]
            members = pd.DataFrame(
                {
                    "output_id": output_id,
                    "_patch_id": group_rows["_patch_id"].values,
                    "_modified": False,
                }
            )
            member_frames.append(members)
            first = group_rows.iloc[0].to_dict()
            if "_dtype" in group_rows.columns:
                # concatenation upcasts like a merge does, so the group's
                # dtype is what the members combine to, not the first row's
                combined = _combined_dtype(group_rows["_dtype"])
                first["_dtype"] = "" if combined is None else str(combined)
            if has_envelope:
                first[f"{dim}_min"] = group_rows[f"{dim}_min"].min()
                first[f"{dim}_max"] = group_rows[f"{dim}_max"].max()
            first["output_id"] = output_id
            first.pop("_patch_id", None)
            output_rows.append(first)
        outputs = pd.DataFrame(output_rows)
        if member_frames:
            members = pd.concat(member_frames, ignore_index=True)
        else:  # nothing to concatenate: an empty spool stays empty
            members = pd.DataFrame(
                {
                    "output_id": pd.Series(dtype=np.int64),
                    "_patch_id": pd.Series(dtype=object),
                    "_modified": pd.Series(dtype=bool),
                }
            )
        plan = ChunkPlan(outputs, members, dim, None, {})
        catalog = derived_catalog(
            source_rows=source_rows,
            plan=plan,
            parent=self._catalog,
            merge_kwargs={},
            mode="concat",
            check_behavior=check_behavior,
            origin_path=self.spool_path,
        )
        return self._new_from_catalog(catalog)

    # --- construction --------------------------------------------------

    @classmethod
    def from_directory(cls, path, index_path=None) -> Self:
        """
        Create a spool over a directory of fiber files.

        The directory's index (created/updated via ``update()``) backs
        the catalog; ``path`` may also be an existing directory indexer.
        """
        from dascore.io.index.catalog import FileResolver, PatchCatalog  # noqa: PLC0415
        from dascore.io.index.indexer import DBDirectoryIndexer  # noqa: PLC0415

        out = cls()
        if isinstance(path, DBDirectoryIndexer):
            from dascore.io.index.catalog import _DIRECTORY_ORDER  # noqa: PLC0415

            out._catalog = PatchCatalog(
                backend=path._backend,
                resolver=FileResolver(root=path.path),
                syncer=path,
                default_order=_DIRECTORY_ORDER,
            )
        else:
            out._catalog = PatchCatalog.from_directory(path, index_path=index_path)
        return out

    @classmethod
    def from_file(
        cls,
        path,
        file_format: str | None = None,
        file_version: str | None = None,
    ) -> Self:
        """
        Create a spool over a single (multi-patch capable) fiber file.

        The file is scanned once; patches load lazily per row.
        """
        path = path if isinstance(path, UPath) else Path(path)
        if not path.exists() or path.is_dir():
            msg = f"{path} does not exist or is a directory"
            raise FileNotFoundError(msg)
        from dascore.io.index.catalog import PatchCatalog  # noqa: PLC0415

        _format, _version = dc.get_format(path, file_format, file_version)
        out = cls()
        out._catalog = PatchCatalog.from_file(
            path, file_format=_format, file_version=_version
        )
        out._file_path = path
        out._file_format = _format
        out._file_version = _version
        return out

    # --- capabilities --------------------------------------------------

    @property
    def indexer(self):
        """The directory syncer, or None for non-directory spools."""
        return self._catalog._syncer

    @property
    def spool_path(self):
        """The directory or file path this spool derives from, or None."""
        indexer = self.indexer
        if indexer is not None:
            return indexer.path
        if self._file_path is not None:
            return self._file_path
        return getattr(self._catalog.resolver, "origin_path", None)

    @property
    def has_live_patches(self) -> bool:
        """True when any of this spool's patches live in memory."""
        return bool(self._catalog.resolver.live_entries())

    @compose_docstring(doc=get_docstring(BaseSpool.update))
    def update(self, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        {doc}

        Update is allowed only on a root spool — one no operation has
        been applied to. Directory roots re-index their directory,
        single-file roots rescan the file, and purely in-memory roots
        are trivially current (no-op). Any derived spool (the result of
        select, slicing, sort, chunk, concatenate, or combining spools)
        raises: update the root and re-apply the operations.
        """
        from dascore.io.index.catalog import LiveResolver  # noqa: PLC0415

        catalog = self._catalog
        derived_msg = (
            "update() is only allowed on a root spool; this spool is the "
            "result of an operation (select/slice/sort/chunk/combine). "
            "Update the root spool and re-apply the operations, e.g. "
            "root = root.update(); view = root.select(...)."
        )
        if catalog.is_view:
            raise InvalidSpoolError(derived_msg)
        if catalog._syncer is not None:
            catalog.update(progress=progress)
            return self._new_from_catalog(catalog)
        if self._file_path is not None:
            from dascore.io.core import FiberIO  # noqa: PLC0415

            formatter = FiberIO.manager.get_fiberio(
                format=self._file_format, version=self._file_version
            )
            getattr(formatter, "index", lambda _: None)(self._file_path)
            refreshed = self.from_file(
                self._file_path, self._file_format, self._file_version
            )
            # from_file builds a spool from the file alone, but an attached
            # inventory is the caller's state rather than the file's, and
            # re-reading the file is no reason to stop enriching.
            refreshed._inventory = self._inventory
            refreshed._enrich_kwargs = self._enrich_kwargs
            refreshed._on_unresolved = self._on_unresolved
            return refreshed
        if isinstance(catalog.resolver, LiveResolver):
            return self  # in-memory contents are trivially current
        # composite/plan roots are computed spools (unions, chunks)
        raise InvalidSpoolError(derived_msg)

    # --- equality ------------------------------------------------------

    def __eq__(self, other) -> bool:
        """
        Equality check which ignores the state of lazy realization.

        The flat relations are built and compared directly so that
        equality does not depend on whether they were realized yet.
        """
        if self is other:
            return True
        if not isinstance(other, Spool):
            return super().__eq__(other)
        # Models compare with ==; deep_equality_check walks their fields,
        # where an unset (NaT) time never equals itself.
        if (self._inventory, self._enrichment()) != (
            other._inventory,
            other._enrichment(),
        ):
            return False
        # views over the same catalog state are equal without realizing
        # the relations (a 200k-row archive must not materialize for ==)
        mine, theirs = self._catalog, other._catalog
        if (
            mine is not None
            and theirs is not None
            and (
                mine is theirs
                or (mine._backend is not None and mine._backend is theirs._backend)
            )
            and mine._queries == theirs._queries
            and mine._residuals == theirs._residuals
            and mine._order == theirs._order
            and mine._ids == theirs._ids
        ):
            return True
        return deep_equality_check(self._eq_state(), other._eq_state())

    def _eq_state(self) -> dict:
        """
        The spool's semantic state, explicitly enumerated for equality.

        Equality is over *effective* rows, never backends or
        representation: same length and order of patch rows, row-wise
        equal semantic columns (source identity like paths and
        live-vs-file backing stripped), with pending residual
        selections folded into the envelopes — a trimmed view equals
        its materialized twin, and spools differing only by a samples
        trim differ in their adjusted envelopes. Whether rows come from
        a live registry, an index file, or a plan is invisible; data
        arrays are never compared (metadata-level, like everything
        here). Because the state is enumerated — never ``__dict__`` —
        new instance attributes cannot silently join equality.
        """

        def _strip_identity(df):
            # synthetic per-catalog identities (memory:// paths, ids) and
            # backend provenance (format/version) are not content; equal
            # spools must compare equal without them, and column order
            # (a construction artifact) must not matter. Coordinate def
            # keys are representation artifacts too: a residual-trimmed
            # view cannot know its trimmed fingerprint without loading,
            # and data values are never compared here anyway.
            drop = [
                "source_path",
                "_patch_id",
                "source_patch_id",
                "source_format",
                "source_version",
                "_modified",
                *[c for c in df.columns if str(c).endswith("_def_key")],
            ]
            out = df.drop(columns=drop, errors="ignore")
            return out[sorted(out.columns)]

        catalog = self._catalog
        rows = self._df
        # value residuals already trim the presented envelopes (to_df);
        # samples residuals fold in here. Presented-but-empty rows stay:
        # a spool exposing an emptied patch is not equal to one without.
        if catalog is not None and catalog._residuals:
            rows = samples_adjusted_envelopes(
                rows, catalog._residuals, drop_empty=False
            )
        return {"rows": _strip_identity(rows)}

    def __rich__(self):
        base = super().__rich__()
        path = self.spool_path
        if path is not None:
            base += Text(f"\n    Path: {path}")
        # Only render a time span when realization is cheap: live
        # contents, single files, and derived catalogs are in memory; a
        # huge directory index is not realized for a repr.
        cheap = self.indexer is None
        if cheap:
            df = self._df
            if df is not None and len(df) and "time_min" in df.columns:
                t1, t2 = df["time_min"].min(), df["time_min"].max()
                if pd.notna(t1) and pd.notna(t2):
                    duration = get_nice_text(t2 - t1)
                    base += Text(
                        f"\n    Time Span: <{duration}> "
                        f"{get_nice_text(t1)} to {get_nice_text(t2)}"
                    )
        return base


@singledispatch
def spool(obj: path_types | BaseSpool | Sequence[PatchType], **kwargs) -> BaseSpool:
    """
    Create a spool from a data source.

    This is the main function for loading in DASCore.

    Parameters
    ----------
    obj
        An object from which a spool can be derived.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> # Get a spool from a single file
    >>> single_file_path = fetch("example_dasdae_event_1.h5")
    >>> file_spool = dc.spool(single_file_path)
    >>>
    >>> # get a spool from a directory of files
    >>> directory_path = fetch("example_dasdae_event_1.h5").parent
    >>> directory_spool = dc.spool(directory_path)
    >>>
    >>> # get a spool from a single patch
    >>> patch = dc.get_example_patch()
    >>> spool = dc.spool(patch)
    """
    msg = f"Could not get spool from: {obj}"
    raise ValueError(msg)


@spool.register(str)
@spool.register(Path)
@spool.register(UPath)
def _spool_from_str(path, **kwargs):
    """Get a spool from a path."""
    path = coerce_to_upath(path)
    # A directory was passed; index it.
    if path.is_dir():
        requires_local_directory(path, label="Directory spool")
        return Spool.from_directory(path, **kwargs)
    # A single file was passed. If the file format supports quick
    # scanning build a lazy file-backed spool, else read it into memory.
    elif path.exists():  # a single file path was passed.
        _format, _version = dc.get_format(path, **kwargs)
        formatter = dc.io.FiberIO.manager.get_fiberio(format=_format, version=_version)
        if formatter.implements_scan:
            return Spool.from_file(path, _format, _version)
        else:
            return Spool(dc.read(path, _format, _version))
    else:
        msg = (
            f"could not get spool from argument: {path}. "
            f"If it is a path, it may not exist."
        )
        raise InvalidSpoolError(msg)


@spool.register(BaseSpool)
def _spool_from_spool(spool, **kwargs):
    """Return a spool from a spool."""
    return spool


@spool.register(list)
@spool.register(tuple)
def _spool_from_patch_list(patch_list, **kwargs):
    """Return a spool from a sequence of patches."""
    return Spool(patch_list)


@spool.register(dc.Patch)
def _spool_from_patch(patch):
    """Get a spool from a single patch."""
    return Spool([patch])
