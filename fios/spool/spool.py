"""
A local database for waveform formats.
"""
import time
from collections import defaultdict
from concurrent.futures import Executor
from contextlib import suppress
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import numpy as np
import obspy
import pandas as pd
import tables
from obsplus.bank.core import _Bank
from obsplus.constants import (EMPTYTD64, WAVEFORM_DTYPES,
                               WAVEFORM_DTYPES_INPUT, WAVEFORM_NAME_STRUCTURE,
                               WAVEFORM_STRUCTURE, availability_type,
                               bank_subpaths_type, bar_parameter_description,
                               bulk_waveform_arg_type,
                               get_waveforms_parameters, paths_description,
                               utc_able_type, utc_time_type)
from obsplus.utils.bank import (_IndexCache, _remove_base_path,
                                _summarize_trace, _summarize_wave_file,
                                _try_read_stream, summarizing_functions)
from obsplus.utils.docs import compose_docstring
from obsplus.utils.misc import replace_null_nlsc_codes
from obsplus.utils.pd import (cast_dtypes, convert_bytestrings, filter_index,
                              get_seed_id_series, order_columns)
from obsplus.utils.time import (make_time_chunks, to_datetime64,
                                to_timedelta64, to_utc)
from obsplus.utils.waveforms import (_filter_index_to_bulk,
                                     get_waveform_bulk_df, merge_traces)
from obspy import Stream, UTCDateTime

# No idea why but this needs to be here to avoid problems with pandas
assert tables.get_hdf5_version()


# ------------------------ constants


class WaveBank(_Bank):
    """
    A class to interact with a directory of waveform files.

    WaveBank recursively reads each file in a directory and creates an index
    to allow the files to be efficiently queried.

    Implements a superset of the :class:`~obsplus.interfaces.WaveformClient`
    interface.

    Parameters
    ----------
    base_path : str
        The path to the directory containing waveform files. If it does not
        exist an empty directory will be created.
    path_structure : str
        Define the directory structure of the wavebank that will be used to
        put waveforms into the directory. Characters are separated by /,
        regardless of operating system. The following words can be used in
        curly braces as data specific variables:
            year, month, day, julday, hour, minute, second, network,
            station, location, channel, time
        example : streams/{year}/{month}/{day}/{network}/{station}
        If no structure is provided it will be read from the index, if no
        index exists the default is {net}/{sta}/{chan}/{year}/{month}/{day}
    name_structure : str
        The same as path structure but for the file name. Supports the same
        variables but requires a period as the separation character. The
        default extension (.mseed) will be added. The default is {time}
        example : {seedid}.{time}
    cache_size : int
        The number of queries to store. Avoids having to read the index of
        the spool multiple times for queries involving the same start and end
        times.
    format : str
        The expected format for the waveform files. Any format supported by
        obspy.read is permitted. The default is mseed. Other formats will be
        tried after the default parser fails.
    ext : str or None
        The extension of the waveform files. If provided, only files with
        this extension will be read.
    executor
        An executor with the same interface as concurrent.futures.Executor,
        the map method of the executor will be used for reading files and
        updating indices.

    Examples
    --------
    >>> # --- Create a `WaveBank` from a path to a directory with waveform files.
    >>> import obsplus
    >>> import obspy
    >>> waveform_path = obsplus.copy_dataset('default_test').waveform_path
    >>> # init a WaveBank and index the files.
    >>> wbank = obsplus.WaveBank(waveform_path).update_index()

    >>> # --- Retrieve a stream objects from the spool.
    >>> # Load all Z component data (dont do this for large datasets!)
    >>> st = wbank.get_waveforms(channel='*Z')
    >>> assert isinstance(st, obspy.Stream) and len(st) == 1

    >>> # --- Read the index used by WaveBank as a DataFrame.
    >>> df = wbank.read_index()
    >>> assert len(df) == 3, 'there should be 3 traces in the spool.'

    >>> # --- Get availability of archive as dataframe
    >>> avail = wbank.get_availability_df()

    >>> # --- Get table of gaps in the archive
    >>> gaps_df = wbank.get_gaps_df()

    >>> # --- yield 5 sec contiguous streams with 1 sec overlap (6 sec total)
    >>> # get input parameters
    >>> t1, t2 = avail.iloc[0]['starttime'], avail.iloc[0]['endtime']
    >>> kwargs = dict(starttime=t1, endtime=t2, duration=5, overlap=1)
    >>> # init list for storing output
    >>> out = []
    >>> for st in wbank.yield_waveforms(**kwargs):
    ...     out.append(st)
    >>> assert len(out) == 6

    >>> # --- Put a new stream and into the spool
    >>> # get an event from another dataset, keep track of its id
    >>> ds = obsplus.load_dataset('bingham_test')
    >>> query_kwargs = dict (station='NOQ', channel='*Z')
    >>> new_st = ds.waveform_client.get_waveforms(**query_kwargs)
    >>> assert len(new_st)
    >>> wbank.put_waveforms(new_st)
    >>> st2 = wbank.get_waveforms(channel='*Z')
    >>> assert len(new_st) + 2
    """

    # index columns and types
    metadata_columns = "last_updated path_structure name_structure".split()
    index_str = tuple(NSLC)
    index_ints = ("starttime", "endtime", "sampling_period")
    index_columns = tuple(list(index_str) + list(index_ints) + ["path"])
    columns_no_path = index_columns[:-1]
    _gap_columns = tuple(list(columns_no_path) + ["gap_duration"])
    namespace = "/waveforms"
    buffer = np.timedelta64(1_000_000_000, "ns")
    # dict defining lengths of str columns (after seed spec)
    # Note: Empty strings get their dtypes caste as S8, which means 8 is the min
    min_itemsize = {"path": 79, "station": 8, "network": 8, "location": 8, "channel": 8}
    _min_files_for_bar = 5000  # number of files before progress bar kicks in
    _dtypes_input = WAVEFORM_DTYPES_INPUT
    _dtypes_output = WAVEFORM_DTYPES

    # ----------------------------- setup stuff

    def __init__(
        self,
        base_path: Union[str, Path, "WaveBank"] = ".",
        path_structure: Optional[str] = None,
        name_structure: Optional[str] = None,
        cache_size: int = 5,
        format="mseed",
        ext=None,
        executor: Optional[Executor] = None,
    ):
        if isinstance(base_path, WaveBank):
            self.__dict__.update(base_path.__dict__)
            return
        self.format = format
        self.ext = ext
        self.bank_path = Path(base_path).absolute()
        # get waveforms structure based on structures of path and filename
        self.path_structure = (
            path_structure if path_structure is not None else WAVEFORM_STRUCTURE
        )
        self.name_structure = name_structure or WAVEFORM_NAME_STRUCTURE
        self.executor = executor
        # initialize cache
        self._index_cache = _IndexCache(self, cache_size=cache_size)
        # enforce min version or warn on newer version
        self._enforce_min_version()
        self._warn_on_newer_version()

    # ----------------------- index related stuff

    @property
    def last_updated_timestamp(self) -> Optional[float]:
        """
        Return the last modified time stored in the index, else None.
        """
        self.ensure_bank_path_exists()
        node = self._time_node
        try:
            out = pd.read_hdf(self.index_path, node)[0]
        except (IOError, IndexError, ValueError, KeyError, AttributeError):
            out = None
        return out

    @property
    def hdf_kwargs(self) -> dict:
        """A dict of hdf_kwargs to pass to PyTables"""
        return dict(
            complib=self._complib,
            complevel=self._complevel,
            format="table",
            data_columns=list(self.index_ints),
        )

    @compose_docstring(
        bar_description=bar_parameter_description, paths_description=paths_description
    )
    def update_index(
        self, bar: Optional = None, paths: Optional[bank_subpaths_type] = None
    ) -> "WaveBank":
        """
        Iterate files in spool and add any modified since last update to index.

        Parameters
        ----------
        {bar_description}
        {paths_description}
        """
        self._enforce_min_version()  # delete index if schema has changed
        update_time = time.time()
        # create a function for the mapping and apply
        func = partial(
            _summarize_wave_file,
            format=self.format,
            summarizer=summarizing_functions.get(self.format, None),
        )
        file_yielder = self._unindexed_iterator(paths=paths)
        iterable = self._measure_iterator(file_yielder, bar)
        updates = list(self._map(func, iterable))
        update_list = list(chain.from_iterable(updates))
        df = pd.DataFrame.from_dict(update_list)
        # push updates to index if any were found
        if not df.empty:
            self._write_update(df, update_time)
            # clear cache out when new traces are added
            self.clear_cache()
        return self

    def _write_update(self, update_df, update_time):
        """convert updates to dataframe, then append to index table"""
        # read in dataframe and prepare for input into hdf5 index
        df = self._prep_write_df(update_df)
        with pd.HDFStore(self.index_path) as store:
            node = self._index_node
            try:
                nrows = store.get_storer(node).nrows
            except (AttributeError, KeyError):
                store.append(
                    node, df, min_itemsize=self.min_itemsize, **self.hdf_kwargs
                )
            else:
                df.index += nrows
                store.append(node, df, append=True, **self.hdf_kwargs)
            # update timestamp
            update_time = time.time() if update_time is None else update_time
            store.put(self._time_node, pd.Series(update_time))
            # make sure meta table also exists.
            # Note this is hear to avoid opening the store again.
            if self._meta_node not in store:
                meta = self._make_meta_table()
                store.put(self._meta_node, meta, format="table")

    def _prep_write_df(self, df):
        """Prepare the dataframe to put it into the HDF5 store."""
        # ensure the spool path is not in the path column
        assert "path" in set(df.columns), f"{df} has no path column"
        df["path"] = _remove_base_path(df["path"], self.bank_path)
        dtype = WAVEFORM_DTYPES_INPUT
        df = (
            df.pipe(order_columns, required_columns=list(dtype))
            .pipe(cast_dtypes, dtype=dtype, inplace=True)
            .pipe(convert_bytestrings, columns=self.index_str, inplace=True)
        )
        # populate index store and update metadata
        assert not df.isnull().any().any(), "null values found in index"
        return df

    def _ensure_meta_table_exists(self):
        """
        If the spool path exists ensure it has a meta table, if not create it.
        """
        if not Path(self.index_path).exists():
            return
        with pd.HDFStore(self.index_path) as store:
            # add metadata if not in store
            if self._meta_node not in store:
                meta = self._make_meta_table()
                store.put(self._meta_node, meta, format="table")

    @compose_docstring(waveform_params=get_waveforms_parameters)
    def read_index(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[utc_time_type] = None,
        endtime: Optional[utc_time_type] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return a dataframe of the index, optionally applying filters.

        Parameters
        ----------
        {waveform_params}
        kwargs
            kwargs are passed to pandas.read_hdf function
        """
        self.ensure_bank_path_exists()
        if not self.index_path.exists():
            self.update_index()
        # if no file was created (dealing with empty spool) return empty index
        if not self.index_path.exists():
            return pd.DataFrame(columns=self.index_columns)
        # grab index from cache
        index = self._index_cache(starttime, endtime, buffer=self.buffer, **kwargs)
        # filter and return
        filt = filter_index(
            index, network=network, station=station, location=location, channel=channel
        )
        return index[filt]

    def _read_metadata(self):
        """
        Read the metadata table.
        """
        try:
            with pd.HDFStore(self.index_path, "r") as store:
                out = store.get(self._meta_node)
            store.close()
            return out
        except (FileNotFoundError, ValueError, KeyError, OSError):
            with suppress(UnboundLocalError):
                store.close()
            self._ensure_meta_table_exists()
            return pd.read_hdf(self.index_path, self._meta_node)

    # ------------------------ availability stuff

    @compose_docstring(get_waveform_params=get_waveforms_parameters)
    def get_availability_df(self, *args, **kwargs) -> pd.DataFrame:
        """
        Return a dataframe specifying the availability of the archive.

        Parameters
        ----------
        {get_waveform_params}

        """
        # no need to read in path, just read needed columns
        ind = self.read_index(*args, columns=self.columns_no_path, **kwargs)
        gro = ind.groupby(list(NSLC))
        min_start = gro.starttime.min().reset_index()
        max_end = gro.endtime.max().reset_index()
        return pd.merge(min_start, max_end)

    def availability(
        self,
        network: str = None,
        station: str = None,
        location: str = None,
        channel: str = None,
    ) -> availability_type:
        """
        Get availability for a given group of instruments.

        Parameters
        ----------
        network
            The network code.
        station
            The station code.
        location
            The location code
        channel
            The chanel code.
        """
        df = self.get_availability_df(network, station, location, channel)
        # convert timestamps to UTCDateTime objects
        df["starttime"] = df.starttime.apply(UTCDateTime)
        df["endtime"] = df.endtime.apply(UTCDateTime)
        # convert to list of tuples, return
        return df.to_records(index=False).tolist()

    # --------------------------- get gaps stuff

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def get_gaps_df(
        self, *args, min_gap: Optional[Union[float, np.timedelta64]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Return a dataframe containing an entry for every gap in the archive.

        Parameters
        ----------
        {get_waveforms_params}
        min_gap
            The minimum gap to report in seconds or as a timedelta64.
             If None, use 1.5 x sampling rate for
            each channel.
        """

        def _get_gap_dfs(df, min_gap):
            """function to apply to each group of seed_id dataframes"""
            # get the min gap
            if min_gap is None:
                min_gap = 1.5 * df["sampling_period"].iloc[0]
            else:
                min_gap = to_timedelta64(min_gap)
            # get df for determining gaps
            dd = (
                df.drop_duplicates()
                .sort_values(["starttime", "endtime"])
                .reset_index(drop=True)
            )
            shifted_starttimes = dd.starttime.shift(-1)
            cum_max = np.maximum.accumulate(dd["endtime"] + min_gap)
            gap_index = cum_max < shifted_starttimes
            # create a dataframe of gaps
            df = dd[gap_index]
            df["starttime"] = dd.endtime[gap_index]
            df["endtime"] = shifted_starttimes[gap_index]
            df["gap_duration"] = df["endtime"] - df["starttime"]
            return df

        # get index and group by NSLC and sampling_period
        index = self.read_index(*args, **kwargs)
        group_names = list(NSLC) + ["sampling_period"]  # include period
        group = index.groupby(group_names, as_index=False)
        out = group.apply(_get_gap_dfs, min_gap=min_gap)
        if out.empty:  # if not gaps return empty dataframe with needed cols
            return pd.DataFrame(columns=self._gap_columns)
        return out.reset_index(drop=True)

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def get_uptime_df(self, *args, **kwargs) -> pd.DataFrame:
        """
        Return a dataframe with uptime stats for selected channels.

        Parameters
        ----------
        {get_waveforms_params}

        """
        # get total number of seconds spool spans for each seed id
        avail = self.get_availability_df(*args, **kwargs)
        avail["duration"] = avail["endtime"] - avail["starttime"]
        # get total duration of gaps by seed id
        gaps_df = self.get_gaps_df(*args, **kwargs)
        if gaps_df.empty:
            gap_total_df = pd.DataFrame(avail[list(NSLC)])
            gap_total_df["gap_duration"] = EMPTYTD64
        else:
            gap_totals = gaps_df.groupby(list(NSLC)).gap_duration.sum()
            gap_total_df = pd.DataFrame(gap_totals).reset_index()
        # merge gap dataframe with availability dataframe, add uptime and %
        df = pd.merge(avail, gap_total_df, how="outer")
        # fill any Nan in gap_duration with empty timedelta
        df.loc[:, "gap_duration"] = df["gap_duration"].fillna(EMPTYTD64)
        df["uptime"] = df["duration"] - df["gap_duration"]
        df["availability"] = df["uptime"] / df["duration"]
        return df

    # ------------------------ get waveform related methods

    def get_waveforms_bulk(
        self,
        bulk: bulk_waveform_arg_type,
        index: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Stream:
        """
        Get a large number of waveforms with a bulk request.

        Parameters
        ----------
        bulk
            A list of any number of lists containing the following:
            (network, station, location, channel, starttime, endtime).
        index
            A dataframe returned by read_index. Enables calling code to only
            read the index from disk once for repetitive calls.
        """
        df = get_waveform_bulk_df(bulk)
        if not len(df):
            return obspy.Stream()
        # get index and filter to temporal extents of request.
        t_min, t_max = df["starttime"].min(), df["endtime"].max()
        if index is not None:
            ind = index[~((index.starttime > t_max) | (index.endtime < t_min))]
        else:
            ind = self.read_index(starttime=t_min, endtime=t_max)
        # for each unique time, apply other filtering conditions and get traces
        unique_times = np.unique(df[["starttime", "endtime"]].values, axis=0)
        traces = []
        for utime in unique_times:
            sub = _filter_index_to_bulk(utime, ind, df)
            traces += self._index2stream(sub, utime[0], utime[1], merge=False).traces
        return merge_traces(obspy.Stream(traces=traces), inplace=True)

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def get_waveforms(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[utc_able_type] = None,
        endtime: Optional[utc_able_type] = None,
    ) -> Stream:
        """
        Get waveforms from the spool.

        Parameters
        ----------
        {get_waveforms_params}

        Notes
        -----
        All string parameters can use posix style matching with * and ? chars.
        All datapoints between selected starttime and endtime will be returned.
        Consequently there may be gaps in the returned stream.
        """
        index = self.read_index(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
        return self._index2stream(index, starttime, endtime)

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def yield_waveforms(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[utc_able_type] = None,
        endtime: Optional[utc_able_type] = None,
        duration: float = 3600.0,
        overlap: Optional[float] = None,
    ) -> Stream:
        """
        Yield time-series segments.

        Parameters
        ----------
        {get_waveforms_params}
        duration : float
            The duration of the streams to yield. All channels selected
            channels will be included in the waveforms.
        overlap : float
            If duration is used, the amount of overlap in yielded streams,
            added to the end of the waveforms.

        Notes
        -----
        All string parameters can use posix style matching with * and ? chars.

        Total duration of yielded streams = duration + overlap.
        """
        # get times in float format
        starttime = to_datetime64(starttime, 0.0)
        endtime = to_datetime64(endtime, "2999-01-01")
        # read in the whole index df
        index = self.read_index(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
        # adjust start/end times
        starttime = max(starttime, index.starttime.min())
        endtime = min(endtime, index.endtime.max())
        # chunk time and iterate over chunks
        time_chunks = make_time_chunks(starttime, endtime, duration, overlap)
        for t1, t2 in time_chunks:
            t1, t2 = to_datetime64(t1), to_datetime64(t2)
            con1 = (index.starttime - self.buffer) > t2
            con2 = (index.endtime + self.buffer) < t1
            ind = index[~(con1 | con2)]
            if not len(ind):
                continue
            yield self._index2stream(ind, t1, t2)

    # ----------------------- deposit waveforms methods

    def put_waveforms(
        self, stream: Union[obspy.Stream, obspy.Trace], name=None, update_index=True
    ):
        """
        Add the waveforms in a waveforms to the spool.

        Parameters
        ----------
        stream
            An obspy waveforms object to add to the spool
        name
            Name of file, if None it will be determined based on contents
        update_index
            Flag to indicate whether or not to update the waveform index
            after writing the new events. Default is True.
        """
        self.ensure_bank_path_exists(create=True)
        st_dic = defaultdict(lambda: [])
        # make sure we have a trace iterable
        stream = [stream] if isinstance(stream, obspy.Trace) else stream
        # iter the waveforms and group by common paths
        paths = []
        for tr in stream:
            summary = _summarize_trace(
                tr,
                name=name,
                path_struct=self.path_structure,
                name_struct=self.name_structure,
            )
            path = self.bank_path / summary["path"]
            st_dic[path].append(tr)
        # iter all the unique paths and save
        for path, tr_list in st_dic.items():
            # make the parent directories if they dont exist
            path.parent.mkdir(exist_ok=True, parents=True)
            stream = obspy.Stream(traces=tr_list).split()
            # load the waveforms if the file already exists
            if path.exists():
                st_existing = obspy.read(str(path))
                stream += st_existing
            # polish streams and write
            stream = merge_traces(stream, inplace=True)
            stream.write(str(path), format="mseed")
            paths.append(path)
        # update the index as the contents have changed
        if st_dic and update_index:
            self.update_index(paths=paths)

    # ------------------------ misc methods

    def _index2stream(self, index, starttime=None, endtime=None, merge=True) -> Stream:
        """return the waveforms in the index"""
        # get abs path to each datafame
        files: pd.Series = (str(self.bank_path) + index.path).unique()
        # make sure start and endtimes are in UTCDateTime
        starttime = to_utc(starttime) if starttime else None
        endtime = to_utc(endtime) if endtime else None
        # iterate the files to read and try to load into waveforms
        kwargs = dict(format=self.format, starttime=starttime, endtime=endtime)
        func = partial(_try_read_stream, **kwargs)
        stt = obspy.Stream()
        chunksize = (len(files) // self._max_workers) or 1
        for st in self._map(func, files, chunksize=chunksize):
            if st is not None:
                stt += st
        # sort out nullish nslc codes
        stt = replace_null_nlsc_codes(stt)
        # filter out any traces not in index (this can happen when files hold
        # multiple traces).
        nslc = set(get_seed_id_series(index))
        stt.traces = [x for x in stt if x.id in nslc]
        # trim, merge, attach response
        stt = self._prep_output_stream(stt, starttime, endtime, merge=merge)
        return stt

    def _prep_output_stream(
        self, st, starttime=None, endtime=None, merge=True
    ) -> obspy.Stream:
        """
        Prepare waveforms object for output by trimming to desired times,
        merging channels, and attaching responses.
        """
        if not len(st):
            return st
        if starttime is not None or endtime is not None:
            starttime = starttime or min([x.stats.starttime for x in st])
            endtime = endtime or max([x.stats.endtime for x in st])
            st.trim(starttime=to_utc(starttime), endtime=to_utc(endtime))
        if merge:
            st = merge_traces(st, inplace=True)
        return st.sort()
