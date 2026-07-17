"""Module for spools, containers of patches."""

from __future__ import annotations

import abc
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
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
    CoordMergeError,
    InvalidSpoolError,
    InvalidSpoolQueryError,
    MissingPatchError,
    ParameterError,
)
from dascore.utils.attrs import combine_patch_attrs
from dascore.utils.display import get_dascore_text, get_nice_text
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    CacheDescriptor,
    _spool_map,
    broadcast_for_index,
    deep_equality_check,
)
from dascore.utils.namespace import NamespaceOwner
from dascore.utils.patch import (
    _force_patch_merge,
    _get_merge_dim,
    _get_merged_coord,
    _split_coord_merge_kwargs,
    _spool_up,
    concatenate_patches,
    get_patch_names,
    patches_to_df,
    stack_patches,
)
from dascore.utils.paths import coerce_to_upath, requires_local_directory
from dascore.utils.pd import (
    _column_or_value,
    _convert_min_max_in_kwargs,
    adjust_segments,
    filter_df,
    get_column_names_from_dim,
    get_dim_names_from_columns,
    resolve_selector_namespaces,
)

T = TypeVar("T")


def _get_varying_dim(df) -> str | None:
    """
    Get the single dimension whose range varies across rows of df.

    Returns None when no dimension varies, several do, or the dataframe
    doesn't carry range columns for the varying dimension; those cases
    need the fully materialized merge to sort out.
    """
    dims = get_dim_names_from_columns(df)
    varying = []
    for dim in dims:
        mins, maxs = df.get(f"{dim}_min"), df.get(f"{dim}_max")
        if mins.nunique(dropna=False) > 1 or maxs.nunique(dropna=False) > 1:
            varying.append(dim)
    return varying[0] if len(varying) == 1 else None


def _estimate_merge_samples(df, dim) -> int | None:
    """
    Estimate the total number of samples along dim of the merged rows.

    Returns None if the estimate cannot be made (eg unknown steps), in
    which case streaming the merge isn't possible.
    """
    if dim is None:
        return None
    cols = [f"{dim}_min", f"{dim}_max", f"{dim}_step"]
    if not set(cols).issubset(df.columns):
        return None
    mins, maxs, steps = (df[x] for x in cols)
    if mins.isnull().any() or maxs.isnull().any() or steps.isnull().any():
        return None
    ratios = (maxs - mins) / steps
    # Degenerate steps (eg 0) make the sample counts meaningless.
    if not np.isfinite(ratios.astype(np.float64)).all():
        return None
    counts = np.round(ratios).astype(np.int64) + 1
    if (counts < 0).any():
        return None
    return int(counts.sum())


def _coord_only_kwargs(patch, kwargs) -> dict:
    """Keep only the kwargs naming a dim or coordinate of patch."""
    return {
        k: v
        for k, v in kwargs.items()
        if k in patch.dims or k in patch.coords.coord_map
    }


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
        new = MemorySpool()
        new._catalog = union
        new._catalog_native = True
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
        [`Spool.chunk_plan`](`dascore.core.spool.DataFrameSpool.chunk_plan`),
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


class DataFrameSpool(BaseSpool):
    """
    An abstract class for spools whose contents are managed by a dataframe.

    A spool is in one of two internal states:

    - **catalog-backed** (``_catalog_native`` and ``_catalog is not None``):
      rows map one-to-one to a ``PatchCatalog`` query, so metadata
      operations (length, selection) can stay lazy and push down to the
      index. Use ``_is_catalog_backed()`` to test this and
      ``_ensure_catalog()`` to enter it without realizing the relation.
    - **materialized**: the managed dataframe/instruction frames are the
      authoritative contents. Operations that restructure or reorder rows
      (chunk, sort, slice) leave catalog-backed mode via
      ``new_from_df`` (which clears ``_catalog_native``).
    """

    # A dataframe which represents contents as they will be output
    _df: pd.DataFrame = CacheDescriptor("_cache", "_get_df")
    # A dataframe which shows patches in the source
    _source_df: pd.DataFrame = CacheDescriptor("_cache", "_get_source_df")
    # A dataframe of instructions for going from source_df to df
    _instruction_df: pd.DataFrame = CacheDescriptor("_cache", "_get_instruction_df")
    # kwargs for filtering contents
    _select_kwargs: Mapping | None = FrozenDict()
    # kwargs for merging patches
    _merge_kwargs: Mapping | None = FrozenDict()
    _drop_columns = ("patch",)
    # patch-local selections (samples=True) applied as patches load
    _post_selects: tuple = ()
    # The catalog backing this spool (None until one is built).
    _catalog = None
    # True while rows directly represent a PatchCatalog query. Operations
    # which restructure/order rows switch back to the dataframe machinery.
    _catalog_native = False

    def _is_catalog_backed(self) -> bool:
        """True when rows map one-to-one to a live catalog query."""
        return self._catalog_native and self._catalog is not None

    def _get_df(self):
        """Function to get the current df."""

    def _get_source_df(self):
        """Function to get the current df."""

    def _get_instruction_df(self):
        """Function to get the current df."""

    def __init__(
        self, select_kwargs: dict | None = None, merge_kwargs: dict | None = None
    ):
        self._cache = {}
        self._select_kwargs = {} if select_kwargs is None else select_kwargs
        self._merge_kwargs = {} if merge_kwargs is None else merge_kwargs
        self._post_selects = ()

    def _select_from_array(self, array) -> Self:
        """Create new spool with contents changed from array input."""
        if np.issubdtype(array.dtype, np.bool_):  # boolean select
            df = self._df[array]
        elif np.issubdtype(array.dtype, np.integer):
            df = self._df.iloc[array]
        else:
            msg = "Only bool or int dtypes are supported for spool array selection."
            raise ValueError(msg)
        source = self._source_df
        inst = self._instruction_df
        select_kwargs, merge_kwargs = self._select_kwargs, self._merge_kwargs
        new = self.new_from_df(
            df,
            source_df=source,
            instruction_df=inst,
            select_kwargs=select_kwargs,
            merge_kwargs=merge_kwargs,
        )
        return new

    def __getitem__(self, item) -> PatchType | BaseSpool:
        if isinstance(item, slice):  # a slice was used, return a sub-spool
            new_df = self._df.iloc[item]
            inst, source = self._instruction_df, self._source_df
            new_inst = inst[inst["current_index"].isin(new_df.index)]
            new_source = source.loc[new_inst.index]
            out = self.new_from_df(
                df=new_df,
                instruction_df=new_inst,
                source_df=new_source,
            )
        elif is_array(item):  # An array was passed use np type selection.
            return self._select_from_array(np.asarray(item))
        else:  # a single index was used, should return a single patch
            out = self._unbox_patch(self._get_patches_from_index(item))
        return out

    def __len__(self):
        # A catalog-native view can count in SQL, skipping the full flat
        # realization (query + attr expansion + coordinate pivot) a plain
        # len(self._df) would force. Fall back to the realized frame once
        # it is cached, on the dataframe path, or when constructor
        # select_kwargs post-filter rows outside the catalog's queries.
        if (
            self._is_catalog_backed()
            and not self._select_kwargs
            and "_df" not in self._cache
        ):
            return len(self._catalog)
        df = self._df
        # An empty spool with no patches, data, or catalog has no frame.
        return 0 if df is None else len(df)

    def __iter__(self):
        if self._df is None:  # an empty spool has nothing to yield
            return
        for ind in range(len(self._df)):
            try:
                yield self._unbox_patch(self._get_patches_from_index(ind))
            except MissingPatchError as e:
                # The patch couldn't be produced, usually because a
                # coordinate mismatch trimmed it to nothing (see #583).
                msg = f"Skipping patch at index {ind} (see #583): {e}"
                warnings.warn(msg, UserWarning, stacklevel=2)

    def _unbox_patch(self, patch_list):
        """Unbox a single patch from a patch list, check len."""
        assert len(patch_list) == 1
        return patch_list[0]

    def _get_patches_from_index(self, df_ind):
        """Given an index (from current df), return the corresponding patch."""
        source = self._source_df
        instruction = self._instruction_df
        # handle negative index.
        df_ind = df_ind if df_ind >= 0 else len(self._df) + df_ind
        try:
            inds = self._df.index[df_ind]
        except IndexError:
            msg = f"index of [{df_ind}] is out of bounds for spool."
            raise IndexError(msg) from None
        # Group positional instruction rows by current index (and cache) to
        # avoid a full instruction df scan for each requested patch.
        indices = self._cache.get("_instruction_indices")
        if indices is None:
            indices = instruction.groupby("current_index").indices
            self._cache["_instruction_indices"] = indices
        positions = indices.get(inds)
        assert positions is not None and len(positions), "no instructions found"
        df1 = instruction.iloc[positions]
        joined = df1.join(source.drop(columns=df1.columns, errors="ignore"))
        # Occasionally, duplicates can creep into the source_df,
        # but it costs a bit to check for duplicates, so only check and drop
        # duplicates on large joined dataframes where performance might be
        # affected.
        if len(joined) > 10:
            cols = set(joined.columns) - set(self._drop_columns)
            joined = joined.drop_duplicates(subset=list(cols), keep="first")
        return self._patch_from_instruction_df(joined)

    def _patch_from_instruction_df(self, joined):
        """Get the patches joined columns of instruction df."""
        df_dict_list = self._df_to_dict_list(joined)
        expected_len = len(joined["current_index"].unique())
        if len(df_dict_list) > expected_len:
            # Several sources merge into one patch. When the output size can
            # be determined from the instructions, stream the sources into a
            # pre-allocated array so they don't all need to be in memory with
            # the merged output at once.
            merge_dim = _get_varying_dim(joined)
            samples = _estimate_merge_samples(joined, merge_dim)
            if samples is not None:
                patch = self._merge_patches_streaming(
                    joined, df_dict_list, merge_dim, samples
                )
                return [patch]
        out = []
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined)
            # The index doesn't carry all the dimensional info, so get what
            # merging needs from the patch coords (cheaper than attr dumps).
            info = patch.coords._get_dim_summary()
            info["patch"] = patch
            out.append(info)
        if len(out) > expected_len:
            out = _force_patch_merge(out, merge_kwargs=self._merge_kwargs)
        return [x["patch"] for x in out]

    def _load_trimmed_patch(self, patch_kwargs, joined) -> dc.Patch:
        """Load a single patch and trim it to its instruction range."""
        # convert kwargs to format understood by parser/patch.select
        kwargs = _convert_min_max_in_kwargs(patch_kwargs, joined)
        patch = self._load_patch(kwargs)
        # If the limits of the source patch were not modified, we can just
        # use the select kwargs. This is important for missing coordinates
        # (NaN values) to not get trimmed out.
        source_kwargs = kwargs if kwargs.get("_modified") else self._select_kwargs
        # attr-style entries (e.g. constructor select_kwargs) filter rows
        # above; only coordinate entries are valid patch selections.
        if select_kwargs := _coord_only_kwargs(patch, source_kwargs):
            patch = patch.select(**select_kwargs)
        # patch-local selections (samples=True) recorded by spool.select
        for post_kwargs, samples in self._post_selects:
            if usable := _coord_only_kwargs(patch, post_kwargs):
                patch = patch.select(**usable, samples=samples)
        return patch

    def _merge_patches_streaming(self, joined, df_dict_list, merge_dim, samples):
        """
        Merge the patches described by the instructions along merge_dim.

        Each patch is copied into a pre-allocated output array as it is
        loaded, then released; this avoids holding all source patches and
        the merged output in memory at the same time, as concatenating
        would.
        """
        buffer, offset, axis, dims = None, 0, None, None
        coords, attrs, summaries = [], [], []
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined)
            if dims is None:
                dims = patch.dims
                axis = patch.get_axis(merge_dim)
            elif patch.dims != dims:
                patch = patch.transpose(*dims)
            data = patch.data
            if buffer is None:
                shape = list(data.shape)
                shape[axis] = samples
                buffer = np.empty(shape, dtype=data.dtype)
            # Mixed dtypes upcast, mirroring np.concatenate behavior.
            dtype = np.result_type(buffer.dtype, data.dtype)
            if dtype != buffer.dtype:
                buffer = buffer.astype(dtype)
            end = offset + data.shape[axis]
            if end > buffer.shape[axis]:
                # The estimate came up short (eg from slightly uneven
                # sampling); grow the buffer to fit.
                shape = list(buffer.shape)
                shape[axis] = end
                new_buffer = np.empty(shape, dtype=buffer.dtype)
                head = broadcast_for_index(buffer.ndim, axis, slice(0, offset))
                new_buffer[head] = buffer[head]
                buffer = new_buffer
            try:
                index = broadcast_for_index(buffer.ndim, axis, slice(offset, end))
                buffer[index] = data
            except ValueError as e:
                msg = (
                    f"Cannot merge patches; their shapes are incompatible "
                    f"along the dimensions not being merged ({merge_dim})."
                )
                raise CoordMergeError(msg) from e
            offset = end
            coords.append(patch.coords)
            attrs.append(patch.attrs)
            summaries.append(patch.coords._get_dim_summary())
        if offset != buffer.shape[axis]:  # over-estimated; trim excess.
            buffer = buffer[broadcast_for_index(buffer.ndim, axis, slice(0, offset))]
        # Ensure the loaded patches only vary along the expected dimension,
        # the same requirement _force_patch_merge enforces.
        summary_df = pd.DataFrame(summaries)
        found_dim = _get_merge_dim(summary_df)
        if found_dim != merge_dim:
            msg = (
                f"Cannot merge patches; expected them to vary along "
                f"{merge_dim} but found {found_dim}."
            )
            raise CoordMergeError(msg)
        attr_kwargs, coord_kwargs = _split_coord_merge_kwargs(self._merge_kwargs)
        conf = attr_kwargs.get("conflicts", None)
        drop_conflicting = conf in {"drop", "keep_first"}
        new_coord = _get_merged_coord(
            summary_df, merge_dim, coords, drop_conflicting, **coord_kwargs
        )
        new_attrs = combine_patch_attrs(attrs, **attr_kwargs)
        return dc.Patch(data=buffer, coords=new_coord, attrs=new_attrs, dims=list(dims))

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

    def _df_to_dict_list(self, df):
        """
        Convert the dataframe to a list of dicts for iteration.

        This is significantly faster than iterating rows.
        """
        return df.to_dict("records")

    @abc.abstractmethod
    def _load_patch(self, kwargs) -> dc.Patch:
        """Given a row from the managed dataframe, return a patch."""

    def _read_and_resolve_patch(self, final_kwargs) -> dc.Patch:
        """Read patches for one instruction row and resolve to one patch."""
        from dascore.io.core import _resolve_read_spool

        source_patch_id = final_kwargs.get("source_patch_id", "")
        spool = dc.read(**final_kwargs)
        return _resolve_read_spool(spool, source_patch_id)

    def _as_catalog_member(self):
        """
        Return (catalog, patch_ids) describing this spool for a union.

        Catalog-native spools contribute their catalog view directly.
        Dataframe-layer selections narrow rows without touching the
        catalog, so their membership carries over as patch ids.
        Restructured rows (e.g. chunked views) no longer map to sources
        and contribute their materialized patches instead.
        """
        if self._catalog is None:
            return super()._as_catalog_member()
        # A catalog-native spool is the whole catalog only when nothing
        # narrows it; constructor select_kwargs (DirectorySpool) restrict
        # the visible rows without touching the catalog, so carry only the
        # surviving patch ids rather than the entire catalog.
        if self._catalog_native and not self._select_kwargs:
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
        select_kwargs=None,
        merge_kwargs=None,
    ):
        """Create a new instance from dataframes."""
        new = self.__class__(self)
        if source_df is None or instruction_df is None:
            _, source_, inst_ = self._get_dummy_dataframes(df)
            source_df = source_df if source_df is not None else source_
            instruction_df = instruction_df if instruction_df is not None else inst_
        new._df = df
        new._source_df = source_df
        new._instruction_df = instruction_df
        # Discard stale instruction indices (eg from copied caches).
        new._cache.pop("_instruction_indices", None)
        new._select_kwargs = dict(self._select_kwargs)
        new._select_kwargs.update(select_kwargs or {})
        new._merge_kwargs = dict(self._merge_kwargs)
        new._merge_kwargs.update(merge_kwargs or {})
        new._post_selects = self._post_selects
        # Dataframe-producing operations (chunk, sort, slice) define their
        # own row/instruction plan and must not bypass it through the catalog.
        new._catalog_native = False
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

    def _ensure_catalog(self) -> None:
        """
        Switch to catalog-native mode if this spool supports it.

        Must not realize the flat relation: it exists so selection can
        route through the catalog on cold spools while staying lazy.
        Dataframe-backed spools (post chunk/sort/slice) stay put.
        """
        return

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
        # canonicalization); adopt it where possible without realizing
        # the flat relation (selection must stay lazy on cold spools).
        self._ensure_catalog()
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
        new._catalog_native = True
        new._cache = {}
        # selection composed into the catalog is dropped, but constructor
        # select_kwargs (DirectorySpool contract) persist across views.
        new._select_kwargs = dict(self._select_kwargs)
        new._post_selects = ()
        return new

    @compose_docstring(doc=BaseSpool.sort.__doc__)
    def sort(self, attribute) -> Self:
        """{doc}."""
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
        return self._df[filter_df(self._df, **self._select_kwargs)]

    get_patch_names = get_patch_names


class MemorySpool(DataFrameSpool):
    """
    A Spool for storing patches in memory.

    When created from patches, the managing dataframes are built lazily
    (on first access by an operation which needs them, such as chunk or
    select) and simple operations (len, integer access, iteration) are
    served straight from the patch tuple. This makes creating a spool
    from patches nearly free, which matters when reading many files.
    """

    # synthetic catalog identity columns must not join patch kwargs
    # comparisons or chunk merge-compatibility checks
    _drop_columns = ("patch", "path", "file_format", "file_version", "source_patch_id")

    def __init__(self, data: PatchType | Sequence[PatchType] | None = None):
        super().__init__()
        self._patches: tuple[PatchType, ...] | None = None
        self._data = None
        self._catalog = None
        if data is not None:
            if isinstance(data, dc.Patch):
                self._patches = (data,)
            elif isinstance(data, Sequence) and all(
                isinstance(x, dc.Patch) for x in data
            ):
                # The same patch instance (by lineage: copies share an
                # identity) appears once; spools have set semantics for
                # identical in-memory patches.
                unique = {x._instance_id: x for x in data}
                self._patches = tuple(unique.values())
            else:  # eg a spool or dataframe; needs the dataframe machinery.
                self._data = data

    def _get_df(self):
        """Build the managing dataframes from the input patches."""
        if self._is_catalog_backed():
            current = self._catalog.to_df()
        else:
            data = self._patches if self._patches is not None else self._data
            if data is None:
                return None
            if self._patches is not None:
                # patch-list spools run on the index catalog: one metadata
                # engine (and one select semantics) for every spool type.
                self._ensure_catalog()
                current = self._catalog.to_df()
            else:  # spools/dataframes: legacy flat-dump path (patch column)
                current = patches_to_df(data)
        df, source, instruction = self._get_dummy_dataframes(current)
        self._source_df = source
        self._instruction_df = instruction
        return df

    def _get_catalog(self):
        """Get (lazily creating) the catalog for patch-list spools.

        Pure accessor: never flips ``_catalog_native``; the transition
        into catalog-backed state belongs to ``_ensure_catalog``.
        """
        from dascore.io.index.catalog import PatchCatalog

        if self._catalog is None:
            self._catalog = PatchCatalog.from_patches(self._patches)
        return self._catalog

    def _ensure_catalog(self) -> None:
        """Patch-list spools ingest into a catalog; no flat realization."""
        if self._patches is not None and not self._catalog_native:
            self._get_catalog()
            self._catalog_native = True

    def _as_catalog_member(self):
        """
        Return (catalog, patch_ids) describing this spool for a union.

        Patch-list spools contribute their own (lazily created) catalog,
        reusing any summaries the patches already computed.
        """
        self._ensure_catalog()
        return super()._as_catalog_member()

    def _get_source_df(self):
        """Build the source df (happens as part of building current df)."""
        _ = self._df
        return self._cache.get("_source_df")

    def _get_instruction_df(self):
        """Build the instruction df (happens as part of building current df)."""
        _ = self._df
        return self._cache.get("_instruction_df")

    def __len__(self) -> int:
        if self._patches is not None:
            return len(self._patches)
        return super().__len__()

    def __getitem__(self, item) -> PatchType | BaseSpool:
        # Fast path: a spool created directly from patches (which never has
        # select kwargs) can serve integer requests from the patch list.
        patches = self._patches
        if (
            patches is not None
            and not self._select_kwargs
            and isinstance(item, int | np.integer)
        ):
            try:
                return patches[item]
            except IndexError:
                msg = f"index of [{item}] is out of bounds for spool."
                raise IndexError(msg) from None
        return super().__getitem__(item)

    def __iter__(self) -> PatchType:
        patches = self._patches
        if patches is not None and not self._select_kwargs:
            yield from patches
        else:
            yield from super().__iter__()

    def __eq__(self, other) -> bool:
        """
        Equality check which ignores the state of the lazy dataframes.

        The managed dataframes are built and compared directly so that
        equality does not depend on whether they were constructed yet.
        """
        if self is other:
            return True
        if not isinstance(other, MemorySpool):
            return super().__eq__(other)
        return deep_equality_check(self._eq_dict(), other._eq_dict())

    def _eq_dict(self) -> dict:
        """Get a dict for equality checks, normalizing lazy state."""

        def _strip_identity(df):
            # synthetic per-catalog identities (memory:// paths, ids) are
            # not content; equal spools must compare equal without them.
            drop = ("path", "_patch_id", "source_patch_id", "file_format")
            if df is None:
                return df
            return df.drop(columns=list(drop), errors="ignore")

        out = dict(self.__dict__)
        # Build (if needed) and compare the dataframes; drop the inputs
        # they were built from, whose form can differ for equal contents.
        out["_cache"] = {
            "_df": _strip_identity(self._df),
            "_source_df": _strip_identity(self._source_df),
            "_instruction_df": _strip_identity(self._instruction_df),
        }
        out.pop("_patches", None)
        out.pop("_data", None)
        out.pop("_catalog", None)
        out.pop("_catalog_native", None)
        return out

    def __rich__(self):
        base = super().__rich__()
        df = self._df
        # An empty MemorySpool() has no dataframe, and patches without a
        # time coordinate have a null time_min; only render a time span
        # when the spool actually carries one.
        if df is not None and len(df) and "time_min" in df.columns:
            t1, t2 = df["time_min"].min(), df["time_min"].max()
            if pd.notna(t1) and pd.notna(t2):
                duration = get_nice_text(t2 - t1)
                base += Text(
                    f"\n    Time Span: <{duration}> "
                    f"{get_nice_text(t1)} to {get_nice_text(t2)}"
                )
        return base

    def _load_patch(self, kwargs) -> Self:
        """Load the patch into memory."""
        if (patch := kwargs.get("patch")) is not None:
            return patch
        return self._catalog.resolve_row(kwargs)

    def _new_from_catalog(self, catalog) -> Self:
        """Create a lazy memory-spool view backed by a catalog query."""
        new = self.__class__()
        new._catalog = catalog
        new._catalog_native = True
        return new

    @compose_docstring(doc=DataFrameSpool.new_from_df.__doc__)
    def new_from_df(self, *args, **kwargs):
        """{doc}."""
        new = super().new_from_df(*args, **kwargs)
        # The provided dataframes fully define the new spool; drop the
        # construction input so derived spools don't retain their parents.
        new._data = None
        new._patches = None
        # derived spools resolve patches through the shared catalog
        new._catalog = self._catalog
        new._catalog_native = False
        return new

    # Add specific implementation of concatenate patches.
    concatenate = _spool_up(concatenate_patches)


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
    # A directory was passed, create Directory Spool
    if path.is_dir():
        requires_local_directory(path, label="Directory spool")
        from dascore.clients.dirspool import DirectorySpool

        return DirectorySpool(path, **kwargs)
    # A single file was passed. If the file format supports quick scanning
    # Return a FileSpool (lazy file reader), else return DirectorySpool.
    elif path.exists():  # a single file path was passed.
        _format, _version = dc.get_format(path, **kwargs)
        formatter = dc.io.FiberIO.manager.get_fiberio(format=_format, version=_version)
        if formatter.implements_scan:
            from dascore.clients.filespool import FileSpool

            return FileSpool(path, _format, _version)

        else:
            return MemorySpool(dc.read(path, _format, _version))
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
    return MemorySpool(patch_list)


@spool.register(dc.Patch)
def _spool_from_patch(patch):
    """Get a spool from a single patch."""
    return MemorySpool([patch])
