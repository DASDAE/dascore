"""Module for spools, containers of patches."""

from __future__ import annotations

import abc
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import ClassVar, Literal, TypeVar

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
    numeric_types,
    path_types,
    timeable_types,
)
from dascore.exceptions import (
    InvalidSpoolError,
    InvalidSpoolQueryError,
    MissingPatchError,
    ParameterError,
)
from dascore.utils.display import get_dascore_text, get_nice_text
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _spool_map,
    deep_equality_check,
)
from dascore.utils.namespace import NamespaceOwner
from dascore.utils.patch import (
    _spool_up,
    concatenate_patches,
    get_patch_names,
    stack_patches,
)
from dascore.utils.paths import coerce_to_upath, requires_local_directory
from dascore.utils.pd import (
    _column_or_value,
    adjust_segments,
    get_column_names_from_dim,
    get_dim_names_from_columns,
    resolve_selector_namespaces,
)

T = TypeVar("T")


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

    @abc.abstractmethod
    def __getitem__(self, item: int | slice | np.ndarray) -> PatchType:
        """Returns a patch from the spool."""

    @abc.abstractmethod
    def __iter__(self) -> PatchType:
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
        from dascore.io.index.catalog import PatchCatalog

        members = [self._as_catalog_member(), other._as_catalog_member()]
        union = PatchCatalog.union(members)
        new = Spool()
        new._catalog = union
        return new

    def _as_catalog_member(self):
        """
        Return (catalog, patch_ids) describing this spool for a union.

        `patch_ids` limits membership to the spool's current rows; None
        means the whole catalog (or the catalog view itself carries the
        selection). The base implementation materializes the patches.
        """
        from dascore.io.index.catalog import PatchCatalog

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

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.units import s
        >>>
        >>> spool = dc.get_example_spool("random_das")
        >>> # get spools with time duration of 10 seconds
        >>> time_chunked = spool.chunk(time=10, overlap=1)
        >>> # merge along time axis
        >>> time_merged = spool.chunk(time=...)

        Notes
        -----
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
    def select(self, **kwargs) -> Self:
        """
        Sub-select parts of the spool.

        Can be used to specify dimension ranges, or unix-style matches
        on string attributes. Bare keyword names resolve against
        attributes first, then coordinates; unknown names raise.

        Parameters
        ----------
        _attrs
            A dict of attribute selections; names validate as attributes
            only (disambiguates names shared with coordinates).
        _coords
            A dict of coordinate selections; names validate as
            coordinates only.
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
    ) -> Generator[Self, None, None]:
        """
        Yield sub-patches based on specified parameters.

        Parameters
        ----------
        size
            The number of patches desired in each output spool. The last
            spool may have fewer patches.
        count
            The number of spools to include. If count is greater than
            the length of the spool then the output will be smaller than
            count, with one patch per spool.

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

    @compose_docstring(desc=concatenate_patches.__doc__)
    def concatenate(self, check_behavior: WARN_LEVELS = "warn", **kwargs):
        """{desc}"""
        msg = f"spool of type {self.__class__} has no concatenate implementation"
        raise NotImplementedError(msg)

    def map(
        self,
        func: Callable[[dc.Patch, ...], T],
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

    @property
    def viz(self):
        """Raise AttributeError when Spool.viz is accessed."""
        msg = (
            "'Spool' has no 'viz' namespace. "
            "Apply 'viz' on a Patch object. "
            "(you can merge a subset of the spool into a single patch using "
            "the Chunk function. i.e., spool.chunk(time=None)[0].viz.waterfall())"
        )
        raise AttributeError(msg)


@dataclass(eq=False, frozen=True)
class SpoolView:
    """
    The derived relation a restructured spool presents.

    ``outputs`` are the rows the spool shows (one per patch it yields),
    ``members`` bind each output to the source rows that feed it (the
    instruction frame), and ``sources`` are those source rows. A spool
    without a view presents its catalog's rows directly; operations
    that restructure or reorder rows (chunk, sort, slice) attach a view
    instead of replacing the backing store.
    """

    outputs: pd.DataFrame
    members: pd.DataFrame
    sources: pd.DataFrame


class Spool(BaseSpool):
    """
    The concrete spool: a `PatchCatalog` plus an optional derived view.

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
    merge_kwargs
        Kwargs controlling how member patches merge when assembled.

    Notes
    -----
    The catalog is the single store — live patches sit in its resolver
    registry, file-backed patches in its index tables — regardless of
    how the spool was constructed. A spool presents rows from exactly
    one of two derivations:

    - **catalog-backed** (``_plan is None``): rows map one-to-one to a
      ``PatchCatalog`` query, so metadata operations (length,
      selection) stay lazy and push down to the index. Use
      ``_is_catalog_backed()`` to test this.
    - **planned** (``_plan`` is a :class:`SpoolView`): the view's
      outputs/members/sources frames are the presented relation.
      The catalog remains attached for patch resolution.
    """

    # kwargs for merging patches
    _merge_kwargs: Mapping | None = FrozenDict()
    # synthetic catalog identity columns must not join patch kwargs
    # comparisons or chunk merge-compatibility checks
    _drop_columns = ("patch", "path", "file_format", "file_version", "source_patch_id")
    # patch-local selections (samples=True) applied as patches load
    _post_selects: tuple = ()
    # The catalog backing this spool (None until one is built).
    _catalog = None
    # The derived relation for restructured views (None = catalog rows).
    _plan: SpoolView | None = None
    # single-file provenance (set by from_file; drives update())
    _file_path = None
    _file_format = None
    _file_version = None

    def _is_catalog_backed(self) -> bool:
        """True when rows map one-to-one to a live catalog query."""
        return self._plan is None and self._catalog is not None

    @property
    def _catalog_native(self) -> bool:
        """Derived state: presented rows are the catalog's own rows."""
        return self._is_catalog_backed()

    @property
    def _df(self) -> pd.DataFrame | None:
        """The dataframe of contents as they will be output."""
        if self._plan is not None:
            return self._plan.outputs
        if "_df" not in self._cache:
            self._cache["_df"] = self._get_df()
        return self._cache["_df"]

    @property
    def _source_df(self) -> pd.DataFrame | None:
        """The dataframe of source patch rows."""
        if self._plan is not None:
            return self._plan.sources
        if "_source_df" not in self._cache:
            self._cache["_source_df"] = self._get_source_df()
        return self._cache["_source_df"]

    @property
    def _instruction_df(self) -> pd.DataFrame | None:
        """The instructions for going from source_df to df."""
        if self._plan is not None:
            return self._plan.members
        if "_instruction_df" not in self._cache:
            self._cache["_instruction_df"] = self._get_instruction_df()
        return self._cache["_instruction_df"]

    def _get_df(self):
        """Realize the flat relation from the catalog."""
        current = self._catalog.to_df().reset_index(drop=True)
        df, source, instruction = self._get_dummy_dataframes(current)
        self._cache["_source_df"] = source
        self._cache["_instruction_df"] = instruction
        return df

    def _get_source_df(self):
        """Build the source df (happens as part of building current df)."""
        _ = self._df
        return self._cache.get("_source_df")

    def _get_instruction_df(self):
        """Build the instruction df (happens as part of building current df)."""
        _ = self._df
        return self._cache.get("_instruction_df")

    def __init__(
        self,
        data: PatchType | Sequence[PatchType] | BaseSpool | None = None,
        merge_kwargs: dict | None = None,
    ):
        from dascore.io.index.catalog import PatchCatalog

        self._cache = {}
        self._merge_kwargs = {} if merge_kwargs is None else merge_kwargs
        self._post_selects = ()
        if isinstance(data, Spool):
            # copy-construction (the new_from_df convention): share the
            # catalog, take fresh derived state
            self.__dict__.update(data.__dict__)
            self._cache = {}
            self._merge_kwargs = dict(data._merge_kwargs)
            if merge_kwargs:
                self._merge_kwargs.update(merge_kwargs)
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

    def _select_from_array(self, array) -> Self:
        """Create new spool with contents changed from array input."""
        if not (
            np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.integer)
        ):
            msg = "Only bool or int dtypes are supported for spool array selection."
            raise ValueError(msg)
        if self._rows_are_catalog():
            return self._new_from_catalog(self._catalog.restrict(array))
        if np.issubdtype(array.dtype, np.bool_):  # boolean select
            df = self._df[array]
        else:
            df = self._df.iloc[array]
        source = self._source_df
        inst = self._instruction_df
        new = self.new_from_df(
            df,
            source_df=source,
            instruction_df=inst,
            merge_kwargs=self._merge_kwargs,
        )
        return new

    def _rows_are_catalog(self) -> bool:
        """
        True when patch access can go straight through the catalog.

        Holds for catalog-backed views with no spool-level row filtering
        or patch-local selections layered outside the catalog (which
        carries its own selection as queries/residuals).
        """
        return self._is_catalog_backed() and not self._post_selects

    def __getitem__(self, item) -> PatchType | BaseSpool:
        if isinstance(item, slice):  # a slice was used, return a sub-spool
            if self._rows_are_catalog():
                # a lazy id-membership window (D2); never realizes the
                # flat relation, and keeps split()/map() parts cheap
                return self._new_from_catalog(self._catalog.window(item))
            new_df = self._df.iloc[item]
            inst, source = self._instruction_df, self._source_df
            new_inst = inst[inst["current_index"].isin(new_df.index)]
            # unique labels only: member rows repeat source labels, and
            # label indexing would multiply rows on every slice
            new_source = source.loc[new_inst.index.unique()]
            out = self.new_from_df(
                df=new_df,
                instruction_df=new_inst,
                source_df=new_source,
            )
        elif is_array(item):  # An array was passed use np type selection.
            return self._select_from_array(np.asarray(item))
        elif self._rows_are_catalog() and isinstance(item, int | np.integer):
            # catalog rows are 1:1 with patches; skip the instruction join
            try:
                return self._catalog.get_patch(int(item))
            except IndexError:
                msg = f"index of [{item}] is out of bounds for spool."
                raise IndexError(msg) from None
        else:  # a single index was used, should return a single patch
            out = self._assembler.get_patch(item)
        return out

    def __len__(self):
        # A catalog-native view can count in SQL, skipping the full flat
        # realization (query + attr expansion + coordinate pivot) a plain
        # len(self._df) would force. Fall back to the realized frame once
        # it is cached or on the dataframe path.
        if self._is_catalog_backed() and "_df" not in self._cache:
            return len(self._catalog)
        return len(self._df)

    def __iter__(self):
        if self._rows_are_catalog():
            for ind in range(len(self._catalog)):
                try:
                    yield self._catalog.get_patch(ind)
                except MissingPatchError as e:
                    msg = f"Skipping patch at index {ind} (see #583): {e}"
                    warnings.warn(msg, UserWarning, stacklevel=2)
            return
        for ind in range(len(self._df)):
            try:
                yield self._assembler.get_patch(ind)
            except MissingPatchError as e:
                # The patch couldn't be produced, usually because a
                # coordinate mismatch trimmed it to nothing (see #583).
                msg = f"Skipping patch at index {ind} (see #583): {e}"
                warnings.warn(msg, UserWarning, stacklevel=2)

    @property
    def _assembler(self):
        """The (cached per view) executor turning member rows into patches."""
        from dascore.utils.patch_assembly import PatchAssembler

        if "_assembler" not in self._cache:
            self._cache["_assembler"] = PatchAssembler(
                df=self._df,
                source_df=self._source_df,
                instruction_df=self._instruction_df,
                load_patch=self._load_patch,
                merge_kwargs=self._merge_kwargs,
                post_selects=self._post_selects,
                drop_columns=self._drop_columns,
            )
        return self._cache["_assembler"]

    def _get_dummy_dataframes(self, current):
        """
        Return dummy current, source, and instruction dataframes.

        Dummy because the source and current df are the same, so the
        instruction df is a straight mapping between the two.
        """
        source = current.copy(deep=False)  # shallow to not copy patches
        dims = get_dim_names_from_columns(source)
        cols2keep = get_column_names_from_dim(dims)
        instruction = (
            current.copy(deep=False)[cols2keep]
            .assign(
                source_index=source.index,
                # This tracks the current spool row after spool operations.
                # It is not the source patch identity within a file.
                current_index=source.index,
                _modified=lambda x: _column_or_value(x, "_modified", False),
            )
            .set_index("source_index")
            .sort_values("current_index")
        )
        return current, source, instruction

    def _load_patch(self, kwargs) -> dc.Patch:
        """Given a row from the managed dataframe, return a patch."""
        # Push trims into the reader only when the instruction row narrows
        # the source (chunk/select); otherwise the whole source is wanted
        # and selection is wasted. Live patches ignore trim hints;
        # exactness is re-applied above (catalog residuals).
        trim = {}
        if kwargs.get("_modified"):
            trim = {
                k: v
                for k, v in kwargs.items()
                if k not in self._drop_columns and not k.startswith("_")
            }
        return self._catalog.resolve_row(kwargs, extra_trim=trim)

    def _as_catalog_member(self):
        """
        Return (catalog, patch_ids) describing this spool for a union.

        Catalog-native spools contribute their catalog view directly.
        Dataframe-layer selections narrow rows without touching the
        catalog, so their membership carries over as patch ids.
        Restructured rows (e.g. chunked views) no longer map to sources
        and contribute their materialized patches instead.
        """
        if self._catalog_native:
            return self._catalog, None
        df = self._df
        if "_patch_id" in df.columns:
            return self._catalog, df["_patch_id"].tolist()
        return super()._as_catalog_member()

    def _chunk_working_df(self) -> pd.DataFrame:
        """Return the source rows the chunk planner consumes."""
        from dascore.utils.chunk_plan import _ensure_patch_id

        # _patch_id is never in _drop_columns, so it survives the drop when
        # present; _ensure_patch_id supplies a positional fallback otherwise.
        working = self._source_df.drop(
            columns=list(self._drop_columns), errors="ignore"
        )
        return _ensure_patch_id(working)

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
        from dascore.utils.chunk_plan import build_chunk_plan

        return build_chunk_plan(
            self._chunk_working_df(),
            overlap=overlap,
            keep_partial=keep_partial,
            snap_coords=snap_coords,
            tolerance=tolerance,
            conflict=conflict,
            group=group,
            missing_dim=missing_dim,
            **kwargs,
        )

    @compose_docstring(doc=BaseSpool.chunk.__doc__)
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
        source = self._source_df
        working = self._chunk_working_df()
        plan = self.chunk_plan(
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
        if plan.outputs.empty:
            empty = source.iloc[0:0]
            return self.new_from_df(empty, merge_kwargs=merge_kwargs)
        out_df = plan.outputs.drop(columns=["output_id"]).reset_index(drop=True)
        # Instructions bind plan members back to source rows by patch id.
        pid_to_index = pd.Series(source.index.values, index=working["_patch_id"].values)
        names = [f"{plan.dim}_min", f"{plan.dim}_max", f"{plan.dim}_step"]
        instructions = (
            plan.members.assign(
                source_index=lambda x: x["_patch_id"].map(pid_to_index),
                current_index=lambda x: x["output_id"],
            )
            .drop(columns=["output_id", "_patch_id"])
            .loc[:, ["source_index", "current_index", *names, "_modified"]]
            .set_index("source_index")
            .sort_values("current_index")
        )
        return self.new_from_df(
            out_df,
            source_df=source,
            instruction_df=instructions,
            merge_kwargs=merge_kwargs,
        )

    def new_from_df(
        self,
        df,
        source_df=None,
        instruction_df=None,
        merge_kwargs=None,
    ):
        """Create a new instance from dataframes."""
        new = self.__class__(self)
        if source_df is None or instruction_df is None:
            _, source_, inst_ = self._get_dummy_dataframes(df)
            source_df = source_df if source_df is not None else source_
            instruction_df = instruction_df if instruction_df is not None else inst_
        # Dataframe-producing operations (chunk, sort, slice) define their
        # own row/instruction plan; the catalog stays attached for patch
        # resolution but no longer defines the presented rows.
        new._plan = SpoolView(outputs=df, members=instruction_df, sources=source_df)
        new._cache = {}
        new._merge_kwargs = dict(self._merge_kwargs)
        new._merge_kwargs.update(merge_kwargs or {})
        new._post_selects = self._post_selects
        return new

    def _select_namespaces(self) -> tuple[set[str], set[str]]:
        """Return (attr names, coord names) selectable on this spool."""
        columns = set(self._df.columns)
        coords = {
            c.removesuffix("_min")
            for c in columns
            if c.endswith("_min") and f"{c.removesuffix('_min')}_max" in columns
        }
        skip = set(self._drop_columns) | {"coord_names", "dims"}
        attrs = {
            c
            for c in columns
            if not c.startswith("_")
            and not c.endswith(("_min", "_max", "_step", "_units"))
            and c not in skip
        }
        return attrs, coords

    def _resolve_select_kwargs(self, _attrs, _coords, kwargs) -> tuple[dict, dict]:
        """
        Split select kwargs into (attrs, coords) per the selector spec.

        Name resolution is shared with the catalog path, so a name means
        the same thing whether or not this spool is catalog-backed; only
        how the predicate is applied differs.
        """
        attrs, coords = self._select_namespaces()
        return resolve_selector_namespaces(
            attrs, coords, _attrs=_attrs, _coords=_coords, kwargs=kwargs
        )

    def _relative_select_kwargs(self, kwargs: dict) -> dict:
        """Resolve relative bounds against the spool's global envelopes."""
        from dascore.utils.pd import relative_ranges_to_absolute

        return relative_ranges_to_absolute(self._df, kwargs)

    @compose_docstring(doc=BaseSpool.select.__doc__)
    def select(
        self,
        *,
        _attrs: dict | None = None,
        _coords: dict | None = None,
        samples: bool = False,
        relative: bool = False,
        **kwargs,
    ) -> Self:
        """{doc}."""
        # The catalog path owns the full selector semantics (e.g. unit
        # canonicalization) and stays lazy on cold spools.
        if self._catalog_native:
            catalog = self._catalog.select(
                _attrs=_attrs,
                _coords=_coords,
                samples=samples,
                relative=relative,
                **kwargs,
            )
            return self._new_from_catalog(catalog)
        attr_kwargs, coord_kwargs = self._resolve_select_kwargs(_attrs, _coords, kwargs)
        if samples:
            # sample indices are patch-local: never filter the spool,
            # record the selection and apply it as patches load (#447).
            if attr_kwargs:
                msg = (
                    f"samples=True selections are coordinate-only; got "
                    f"{sorted(attr_kwargs)}."
                )
                raise InvalidSpoolQueryError(msg)
            new = self.new_from_df(
                self._df,
                source_df=self._source_df,
                instruction_df=self._instruction_df,
            )
            new._post_selects = (*self._post_selects, (coord_kwargs, True))
            return new
        if relative and coord_kwargs:
            coord_kwargs = self._relative_select_kwargs(coord_kwargs)
        kwargs = {**attr_kwargs, **coord_kwargs}
        filtered_df = adjust_segments(self._df, ignore_bad_kwargs=True, **kwargs)
        inst = adjust_segments(
            self._instruction_df,
            ignore_bad_kwargs=True,
            **kwargs,
        ).loc[lambda x: x["current_index"].isin(filtered_df.index)]
        source = adjust_segments(
            self._source_df.loc[inst.index], ignore_bad_kwargs=True, **kwargs
        )
        out = self.new_from_df(
            filtered_df,
            # Drop rows that are no longer needed.
            source_df=source,
            instruction_df=inst,
        )
        return out

    def _new_from_catalog(self, catalog) -> Self:
        """Create a lazy catalog-native view of this spool."""
        new = self.__class__(self)
        new._catalog = catalog
        new._plan = None
        new._cache = {}
        new._post_selects = ()
        return new

    @compose_docstring(doc=BaseSpool.sort.__doc__)
    def sort(self, attribute) -> Self:
        """{doc}."""
        if self._rows_are_catalog():
            # a lazy ORDER BY spec (D2): no copy, no realization; the
            # ordinal contract supplies the deterministic tiebreak
            return self._new_from_catalog(self._catalog.order_by(attribute))
        df = self._df
        inst_df = self._instruction_df

        # make sure a suitable attribute is entered
        attrs = set(df.columns)
        if attribute not in attrs:
            # make sure we can also cover coordinate names instead of the attribute
            if f"{attribute}_min" in attrs:
                attribute = f"{attribute}_min"
            else:
                msg = "Invalid attribute. Please use a valid attribute such as: 'time'"
                raise IndexError(msg)

        # get a mapping from the old current index to the sorted ones
        sorted_df = df.sort_values(attribute)
        sorted_original_indices = sorted_df.index
        sorted_df = sorted_df.reset_index(drop=True)
        mapper = pd.Series(np.arange(len(sorted_df)), index=sorted_original_indices)
        # swap out all the old values with new ones
        new_current_index = inst_df["current_index"].map(mapper)
        new_instruction_df = inst_df.assign(current_index=new_current_index)
        # create new spool from new dataframes
        return self.new_from_df(
            df=sorted_df,
            source_df=self._source_df,
            instruction_df=new_instruction_df,
        )

    @compose_docstring(doc=BaseSpool.split.__doc__)
    def split(
        self,
        size: int | None = None,
        count: int | None = None,
    ) -> Generator[Self, None, None]:
        """{doc}."""
        if not ((count is not None) ^ (size is not None)):
            msg = "Spool.split requires either spool_count or spool_size."
            raise ParameterError(msg)
        start = 0
        step = int(np.ceil(len(self) / count if count else size))
        while start < len(self):
            yield self[start : start + step]
            start += step

    @compose_docstring(doc=BaseSpool.get_contents.__doc__)
    def get_contents(self) -> pd.DataFrame:
        """{doc}."""
        return self._df

    # --- construction --------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        path,
        index_path=None,
        select_kwargs: dict | None = None,
        merge_kwargs: dict | None = None,
    ) -> Self:
        """
        Create a spool over a directory of fiber files.

        The directory's index (created/updated via ``update()``) backs
        the catalog; ``path`` may also be an existing directory indexer.
        ``select_kwargs`` compose a selection into the catalog exactly
        like ``.select(**select_kwargs)`` — validating the names
        triggers the initial directory index if it doesn't exist yet.
        """
        from dascore.io.index.catalog import FileResolver, PatchCatalog
        from dascore.io.indexer import AbstractIndexer

        out = cls(merge_kwargs=merge_kwargs)
        if isinstance(path, AbstractIndexer):
            catalog = PatchCatalog(
                backend=path._backend,
                resolver=FileResolver(root=path.path),
                syncer=path,
            )
        else:
            catalog = PatchCatalog.from_directory(path, index_path=index_path)
        if select_kwargs:
            catalog = catalog.select(**select_kwargs)
        out._catalog = catalog
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
        from dascore.io.index.catalog import PatchCatalog

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
        return None if self._catalog is None else self._catalog._syncer

    @property
    def spool_path(self):
        """Return the path in which the spool contents are found."""
        return self.indexer.path

    @property
    def has_live_patches(self) -> bool:
        """True when any of this spool's patches live in memory."""
        catalog = self._catalog
        return catalog is not None and bool(catalog.resolver.live_entries())

    def _has_file_rows(self) -> bool:
        """True when any catalog row is backed by a file."""
        from dascore.io.index.catalog import LiveResolver
        from dascore.utils.paths import is_memory_uri

        if isinstance(self._catalog.resolver, LiveResolver):
            return False
        paths = self._catalog.backend.get_sources()["source_path"]
        return not paths.map(is_memory_uri).all()

    @compose_docstring(doc=BaseSpool.update.__doc__)
    def update(self, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        {doc}

        Update means syncing contents with the backing source: a
        directory-backed spool re-indexes its directory, a single-file
        spool rescans the file, and a purely in-memory spool is
        trivially current (no-op). A spool with file-backed contents
        but no update source (e.g. the result of combining spools)
        raises — recreate it from its directory instead.
        """
        catalog = self._catalog
        if catalog is not None and catalog._syncer is not None:
            catalog.update(progress=progress)
            return self._new_from_catalog(catalog)
        if self._file_path is not None:
            from dascore.io.core import FiberIO

            formatter = FiberIO.manager.get_fiberio(
                format=self._file_format, version=self._file_version
            )
            getattr(formatter, "index", lambda _: None)(self._file_path)
            return self.from_file(
                self._file_path, self._file_format, self._file_version
            )
        if not self._has_file_rows():
            return self  # in-memory contents are trivially current
        msg = (
            "This spool has file-backed contents but no update source "
            "(e.g. it combines several spools); recreate it from its "
            "directory to pick up new files."
        )
        raise InvalidSpoolError(msg)

    # --- equality ------------------------------------------------------

    def __eq__(self, other) -> bool:
        """
        Equality check which ignores the state of the lazy dataframes.

        The managed dataframes are built and compared directly so that
        equality does not depend on whether they were constructed yet.
        """
        if self is other:
            return True
        if not isinstance(other, Spool):
            return super().__eq__(other)
        return deep_equality_check(self._eq_state(), other._eq_state())

    def _eq_state(self) -> dict:
        """
        The spool's semantic state, explicitly enumerated for equality.

        Equality is over rows, never backends: same length and order of
        patch rows, row-wise equal semantic columns (source identity
        like paths and live-vs-file backing stripped), plus equal
        pending residual selections and policy. Whether rows come from
        a live registry, an index file, or a plan is invisible; data
        arrays are never compared (metadata-level, like everything
        here). Because the state is enumerated — never ``__dict__`` —
        new instance attributes cannot silently join equality.
        """

        def _strip_identity(df):
            # synthetic per-catalog identities (memory:// paths, ids) and
            # backend provenance (format/version) are not content; equal
            # spools must compare equal without them, and column order
            # (a construction artifact) must not matter.
            drop = (
                "path",
                "_patch_id",
                "source_patch_id",
                "file_format",
                "file_version",
            )
            out = df.drop(columns=list(drop), errors="ignore")
            return out[sorted(out.columns)]

        catalog = self._catalog
        return {
            # the presented relation (row content and order), plus the
            # source rows and member bindings that define patch assembly;
            # the plan's frames surface through the same accessors, so
            # planned and identity views with equal contents compare equal
            "rows": _strip_identity(self._df),
            "sources": _strip_identity(self._source_df),
            "members": _strip_identity(self._instruction_df),
            # residuals (e.g. samples trims) change what patches load
            # without changing the visible rows
            "residuals": None if catalog is None else catalog._residuals,
            "post_selects": self._post_selects,
            "merge_kwargs": dict(self._merge_kwargs),
        }

    def __rich__(self):
        base = super().__rich__()
        indexer = self.indexer
        path = getattr(indexer, "path", None) or self._file_path
        if path is not None:
            base += Text(f"\n    Path: {path}")
        # Only render a time span when the relation is (or is nearly)
        # realized: planned views carry their frames and live spools are
        # in memory; a huge directory index is not realized for a repr.
        if self._plan is not None or self.has_live_patches:
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

    # Add specific implementation of concatenate patches.
    concatenate = _spool_up(concatenate_patches)

    get_patch_names = get_patch_names


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
