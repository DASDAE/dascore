"""Module for spools, containers of patches."""
from __future__ import annotations

import abc
from collections.abc import Callable, Generator, Mapping, Sequence
from functools import singledispatch
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.constants import (
    PROGRESS_LEVELS,
    WARN_LEVELS,
    ExecutorType,
    PatchType,
    attr_conflict_description,
    numeric_types,
    timeable_types,
)
from dascore.core.attrs import check_coords, check_dims
from dascore.core.patch import Patch
from dascore.exceptions import InvalidSpoolError, ParameterError, PatchDimError
from dascore.utils.chunk import ChunkManager
from dascore.utils.display import get_dascore_text, get_nice_text
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import CacheDescriptor, _spool_map
from dascore.utils.patch import _force_patch_merge, patches_to_df
from dascore.utils.pd import (
    _convert_min_max_in_kwargs,
    adjust_segments,
    filter_df,
    get_column_names_from_dim,
    get_dim_names_from_columns,
    split_df_query,
)

T = TypeVar("T")


class BaseSpool(abc.ABC):
    """Spool Abstract Base Class (ABC) for defining Spool interface."""

    _rich_style = "bold"

    @abc.abstractmethod
    def __getitem__(self, item: int) -> PatchType:
        """Returns a patch from the spool."""

    @abc.abstractmethod
    def __iter__(self) -> PatchType:
        """Iterate through the Patches in the spool."""

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

        def _vals_equal(dict1, dict2):
            if (set1 := set(dict1)) != set(dict2):
                return False
            for key in set1:
                val1, val2 = dict1[key], dict2[key]
                if isinstance(val1, dict):
                    if not _vals_equal(val1, val2):
                        return
                # this is primarily for dataframes which have equals method.
                elif hasattr(val1, "equals"):
                    if not val1.equals(val2):
                        return False
                elif val1 != val2:
                    return False
            return True

        my_dict = self.__dict__
        other_dict = getattr(other, "__dict__", {})
        return _vals_equal(my_dict, other_dict)

    @abc.abstractmethod
    @compose_docstring(conflict_desc=attr_conflict_description)
    def chunk(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float = 1.5,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        **kwargs,
    ) -> Self:
        """
        Chunk the data in the spool along specified dimensions.

        Parameters
        ----------
        overlap
            The amount of overlap between each segment, starting with the end of
            first patch. Negative values can be used to create gaps.
        keep_partial
            If True, keep the segments which are smaller than chunk size.
            This often occurs because of data gaps or at end of chunks.
        snap_coords
            If True, snap the coords on joined patches such that the spacing
            remains constant.
        tolerance
            The number of samples a block of data can be spaced and still be
            considered contiguous.
        conflict
            {conflict_desc}
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
        """

    @abc.abstractmethod
    def select(self, **kwargs) -> Self:
        """
        Sub-select parts of the spool.

        Can be used to specify dimension ranges, or unix-style matches
        on string attributes.

        Parameters
        ----------
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

    def stack(self, dim_vary=None, check_behavior: WARN_LEVELS = "warn") -> PatchType:
        """
        Stack (add) all patches compatible with first patch together.

        Parameters
        ----------
        dim_vary
            The name of the dimension which can be different in values
            (but not shape) and patches still added together.
        check_behavior
            Indicates what to do when an incompatible patch is found in the
            spool. `None` will silently skip any incompatible patches,
            'warn' will issue a warning and then skip incompatible patches,
            'raise' will raise an
            [`IncompatiblePatchError`](`dascore.exceptions.IncompatiblePatchError`)
            if any incompatible patches are found.

        Examples
        --------
        >>> import dascore as dc
        >>> # add a spool with equal sized patches but progressing time dim
        >>> spool = dc.get_example_spool()
        >>> stacked_patch = spool.stack(dim_vary='time')
        """
        # check the dims/coords of first patch (considered to be standard for rest)
        init_patch = self[0]
        stack_arr = np.zeros_like(init_patch.data)

        # ensure dim_vary is in dims
        if dim_vary is not None and dim_vary not in init_patch.dims:
            msg = f"Dimension {dim_vary} is not in first patch."
            raise PatchDimError(msg)

        for p in self:
            # check dimensions of patch compared to init_patch
            dims_ok = check_dims(init_patch, p, check_behavior)
            coords_ok = check_coords(init_patch, p, check_behavior, dim_vary)
            # actually do the stacking of data
            if dims_ok and coords_ok:
                stack_arr = stack_arr + p.data

        # create attributes for the stack with adjusted history
        stack_attrs = init_patch.attrs
        new_history = list(init_patch.attrs.history)
        new_history.append("stack")
        stack_attrs = stack_attrs.update(history=new_history)

        # create coords array for the stack
        stack_coords = init_patch.coords
        if dim_vary:  # adjust dim_vary to start at 0 for junk dimension indicator
            coord_to_change = stack_coords.coord_map[dim_vary]
            new_dim = coord_to_change.update_limits(min=0)
            stack_coords = stack_coords.update_coords(**{dim_vary: new_dim})
        return Patch(stack_arr, stack_coords, init_patch.dims, stack_attrs)


class DataFrameSpool(BaseSpool):
    """An abstract class for spools whose contents are managed by a dataframe."""

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
    # attributes which effect merge groups for internal patches
    _group_columns = ("network", "station", "dims", "data_type", "tag")
    _drop_columns = ("patch",)

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
        else:  # a single index was used, should return a single patch
            out = self._unbox_patch(self._get_patches_from_index(item))
        return out

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        for ind in range(len(self._df)):
            yield self._unbox_patch(self._get_patches_from_index(ind))

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
            raise IndexError(msg)
        df1 = instruction[instruction["current_index"] == inds]
        assert not df1.empty
        joined = df1.join(source.drop(columns=df1.columns, errors="ignore"))
        return self._patch_from_instruction_df(joined)

    def _patch_from_instruction_df(self, joined):
        """Get the patches joined columns of instruction df."""
        out = []
        df_dict_list = self._df_to_dict_list(joined)
        expected_len = len(joined["current_index"].unique())
        for patch_kwargs in df_dict_list:
            # convert kwargs to format understood by parser/patch.select
            kwargs = _convert_min_max_in_kwargs(patch_kwargs, joined)
            patch = self._load_patch(kwargs)
            # apply any trimming needed on patch
            select_kwargs = {
                i: v
                for i, v in kwargs.items()
                if i in patch.dims or i in patch.coords.coord_map
            }
            trimmed_patch: dc.Patch = patch.select(**select_kwargs)
            # its unfortunate, but currently we need to regenerate the patch
            # dict because the index doesn't carry all the dimensional info
            info = trimmed_patch.attrs.flat_dump(exclude=["history"])
            info["patch"] = trimmed_patch
            out.append(info)
        if len(out) > expected_len:
            out = _force_patch_merge(out, merge_kwargs=self._merge_kwargs)
        return [x["patch"] for x in out]

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
            current.copy()[cols2keep]
            .assign(source_index=source.index, current_index=source.index)
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
    def _load_patch(self, kwargs) -> Self:
        """Given a row from the managed dataframe, return a patch."""

    @compose_docstring(doc=BaseSpool.chunk.__doc__)
    def chunk(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float = 1.5,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        **kwargs,
    ) -> Self:
        """{doc}."""
        df = self._df.drop(columns=list(self._drop_columns), errors="ignore")
        chunker = ChunkManager(
            overlap=overlap,
            keep_partial=keep_partial,
            group_columns=self._group_columns,
            tolerance=tolerance,
            conflict=conflict,
            **kwargs,
        )
        in_df, out_df = chunker.chunk(df)
        if df.empty:
            instructions = None
        else:
            instructions = chunker.get_instruction_df(in_df, out_df)
        return self.new_from_df(
            out_df,
            source_df=self._df,
            instruction_df=instructions,
            merge_kwargs={"conflicts": conflict},
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
        df_, source_, inst_ = self._get_dummy_dataframes(df)
        new._df = df
        new._source_df = source_df if source_df is not None else source_
        new._instruction_df = instruction_df if instruction_df is not None else inst_
        new._select_kwargs = dict(self._select_kwargs)
        new._select_kwargs.update(select_kwargs or {})
        new._merge_kwargs = dict(self._merge_kwargs)
        new._merge_kwargs.update(merge_kwargs or {})
        return new

    @compose_docstring(doc=BaseSpool.select.__doc__)
    def select(self, **kwargs) -> Self:
        """{doc}."""
        _, _, extra_kwargs = split_df_query(kwargs, self._df, ignore_bad_kwargs=True)
        filtered_df = adjust_segments(self._df, ignore_bad_kwargs=True, **kwargs)
        inst = adjust_segments(
            self._instruction_df,
            ignore_bad_kwargs=True,
            **kwargs,
        ).loc[lambda x: x["current_index"].isin(filtered_df.index)]
        out = self.new_from_df(
            filtered_df,
            source_df=self._source_df,
            instruction_df=inst,
            select_kwargs=extra_kwargs,
        )
        return out

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
                msg = (
                    "Invalid attribute. " "Please use a valid attribute such as: 'time'"
                )
                raise IndexError(msg)

        # get a mapping from the old current index to the sorted ones
        sorted_df = df.sort_values(attribute).reset_index(drop=True)
        old_indices = df.index
        new_indices = np.arange(len(df))
        mapper = pd.Series(new_indices, index=old_indices)
        # swap out all the old values with new ones
        new_current_index = inst_df["current_index"].map(mapper)
        new_instruction_df = inst_df.assign(current_index=new_current_index)
        # create new spool from new dataframes
        return self.new_from_df(df=sorted_df, instruction_df=new_instruction_df)

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


class MemorySpool(DataFrameSpool):
    """A Spool for storing patches in memory."""

    # tuple of attributes to remove from table

    def __init__(self, data: PatchType | Sequence[PatchType] | None = None):
        super().__init__()
        if data is not None:
            dfs = self._get_dummy_dataframes(patches_to_df(data))
            self._df, self._source_df, self._instruction_df = dfs

    def __rich__(self):
        base = super().__rich__()
        df = self._df
        if len(df):
            t1, t2 = df["time_min"].min(), df["time_max"].max()
            tmin = get_nice_text(t1)
            tmax = get_nice_text(t2)
            duration = get_nice_text(t2 - t1)
            base += Text(f"\n    Time Span: <{duration}> {tmin} to {tmax}")
        return base

    def _load_patch(self, kwargs) -> Self:
        """Load the patch into memory."""
        return kwargs["patch"]


@singledispatch
def spool(obj: Path | str | BaseSpool | Sequence[PatchType], **kwargs) -> BaseSpool:
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
def _spool_from_str(path, **kwargs):
    """Get a spool from a path."""
    path = Path(path)
    # A directory was passed, create Directory Spool
    if path.is_dir():
        from dascore.clients.dirspool import DirectorySpool

        return DirectorySpool(path, **kwargs)
    # A single file was passed. If the file format supports quick scanning
    # Return a FileSpool (lazy file reader), else return DirectorySpool.
    elif path.exists():  # a single file path was passed.
        _format, _version = dc.get_format(path, **kwargs)
        formatter = dc.io.FiberIO.manager.get_fiberio(_format, _version)
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
