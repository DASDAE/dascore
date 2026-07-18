"""Module for spools, containers of patches."""

from __future__ import annotations

import abc
import warnings
from collections.abc import Callable, Generator, Sequence
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
    MissingPatchError,
    ParameterError,
)
from dascore.utils.display import get_dascore_text, get_nice_text
from dascore.utils.docs import compose_docstring
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
    _drop_columns = ("patch", "path", "file_format", "file_version", "source_patch_id")
    # The catalog backing this spool.
    _catalog = None
    # single-file provenance (set by from_file; drives update())
    _file_path = None
    _file_format = None
    _file_version = None

    def __init__(
        self,
        data: PatchType | Sequence[PatchType] | BaseSpool | None = None,
    ):
        from dascore.io.index.catalog import PatchCatalog

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

    @compose_docstring(doc=BaseSpool.get_contents.__doc__)
    def get_contents(self) -> pd.DataFrame:
        """{doc}."""
        return self._df

    def __len__(self):
        # counting pushes to SQL (or the cold live registry); the flat
        # relation is never realized just for a length
        return len(self._catalog)

    def __getitem__(self, item) -> PatchType | BaseSpool:
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
                msg = (
                    "Only bool or int dtypes are supported for spool "
                    "array selection."
                )
                raise ValueError(msg)
            return self._new_from_catalog(self._catalog.restrict(array))
        try:
            return self._catalog.get_patch(int(item))
        except MissingPatchError:
            # MissingPatchError subclasses IndexError for backwards
            # compatibility; it must never masquerade as out-of-bounds
            raise
        except IndexError:
            msg = f"index of [{item}] is out of bounds for spool."
            raise IndexError(msg) from None

    def __iter__(self):
        for ind in range(len(self._catalog)):
            try:
                yield self._catalog.get_patch(ind)
            except MissingPatchError as e:
                # The patch couldn't be produced, usually because a
                # coordinate mismatch trimmed it to nothing (see #583).
                msg = f"Skipping patch at index {ind} (see #583): {e}"
                warnings.warn(msg, UserWarning, stacklevel=2)

    # --- selection and presentation specs -------------------------------

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
        catalog = self._catalog.select(
            _attrs=_attrs,
            _coords=_coords,
            samples=samples,
            relative=relative,
            **kwargs,
        )
        return self._new_from_catalog(catalog)

    @compose_docstring(doc=BaseSpool.sort.__doc__)
    def sort(self, attribute) -> Self:
        """{doc}."""
        # a lazy ORDER BY spec (D2): no copy, no realization; the
        # ordinal contract supplies the deterministic tiebreak
        return self._new_from_catalog(self._catalog.order_by(attribute))

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
        data is loaded).
        """
        catalog = self._catalog
        if catalog._residuals or catalog._order is not None:
            return self._materialize_lossy(), None
        return catalog, None

    def _materialize_lossy(self):
        """
        Bake residual trims and presentation order into a derived catalog.

        An identity plan over the view's presented rows: one output per
        row (in presentation order, so ordinals record the order spec),
        with trimmed envelopes as the output envelopes and the trims
        themselves re-applied at load through the plan resolver.
        """
        from dascore.io.index.planned import derived_catalog
        from dascore.utils.chunk_plan import (
            _SOURCE_COLUMNS,
            ChunkPlan,
            samples_adjusted_envelopes,
        )

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
        from dascore.io.index.planned import PlanResolver, collapse_working_df
        from dascore.utils.chunk_plan import (
            _ensure_patch_id,
            samples_adjusted_envelopes,
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
        from dascore.utils.chunk_plan import build_chunk_plan

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
        from dascore.io.index.planned import derived_catalog
        from dascore.utils.chunk_plan import build_chunk_plan

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

    @compose_docstring(desc=concatenate_patches.__doc__)
    def concatenate(self, check_behavior: WARN_LEVELS = "warn", **kwargs) -> Self:
        """{desc}"""
        from dascore.io.index.planned import derived_catalog
        from dascore.utils.chunk_plan import ChunkPlan

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
        from dascore.io.index.catalog import FileResolver, PatchCatalog
        from dascore.io.indexer import AbstractIndexer

        out = cls()
        if isinstance(path, AbstractIndexer):
            from dascore.io.index.catalog import _DIRECTORY_ORDER

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
        catalog = self._catalog
        return catalog is not None and bool(catalog.resolver.live_entries())

    @compose_docstring(doc=BaseSpool.update.__doc__)
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
        from dascore.io.index.catalog import LiveResolver

        catalog = self._catalog
        derived_msg = (
            "update() is only allowed on a root spool; this spool is the "
            "result of an operation (select/slice/sort/chunk/combine). "
            "Update the root spool and re-apply the operations, e.g. "
            "root = root.update(); view = root.select(...)."
        )
        if catalog is None or catalog.is_view:
            raise InvalidSpoolError(derived_msg)
        if catalog._syncer is not None:
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

        Equality is over rows, never backends: same length and order of
        patch rows, row-wise equal semantic columns (source identity
        like paths and live-vs-file backing stripped), plus equal
        pending residual selections. Whether rows come from a live
        registry, an index file, or a plan is invisible; data arrays
        are never compared (metadata-level, like everything here).
        Because the state is enumerated — never ``__dict__`` — new
        instance attributes cannot silently join equality.
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
            "rows": _strip_identity(self._df),
            "residuals": None if catalog is None else catalog._residuals,
        }

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
