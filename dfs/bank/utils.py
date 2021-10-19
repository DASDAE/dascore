"""
Utils for banks
"""
import contextlib
import os
import re
import time
import warnings
from typing import Optional, Sequence, List

import pandas as pd
import numpy as np
from tables.exceptions import ClosedNodeError

from obsplus.constants import (
    NSLC,
    WAVEFORM_STRUCTURE,
    WAVEFORM_NAME_STRUCTURE,
    SMALLDT64,
    LARGEDT64,
    READ_HDF5_KWARGS,
)
from obsplus.exceptions import UnsupportedKeyword
from obsplus.utils.geodetics import map_longitudes
from obsplus.utils.misc import READ_DICT, _get_path
from obsplus.utils.mseed import summarize_mseed
from obsplus.utils.time import to_datetime64, _dict_times_to_ns

# functions for summarizing the various formats
summarizing_functions = dict(mseed=summarize_mseed)

# extensions
WAVEFORM_EXT = ".mseed"
EVENT_EXT = ".xml"
STATION_EXT = ".xml"


# name structures


def _get_time_values(time1, time2=None):
    """get the time values from a UTCDateTime object or two"""
    tvals = "year month day hour minute second microsecond".split()
    utc1 = time1
    split = re.split("-|:|T|[.]", str(utc1).replace("Z", ""))
    assert len(tvals) == len(split)
    out = {key: val for key, val in zip(tvals, split)}
    out["julday"] = "%03d" % utc1.julday
    out["starttime"] = utc1.timestamp
    if time2:
        out["endtime"] = time2.timestamp
    out["time"] = str(utc1).replace(":", "-").split(".")[0]
    return out


def summarize_generic_stream(path, format=None) -> List[dict]:
    """
    Return summary information for a stream stored on disk.

    Parameters
    ----------
    path
        The path to the stream.
    format
        The format code, as used by obspy.
    """
    st = _try_read_stream(path, format=format) or _try_read_stream(path) or []
    out = []
    for tr in st:
        summary = {
            "starttime": tr.stats.starttime._ns,
            "endtime": tr.stats.endtime._ns,
            "sampling_period": int(tr.stats.delta * 1_000_000_000),
            "path": path,
        }
        summary.update(dict((x, c) for x, c in zip(NSLC, tr.id.split("."))))
        out.append(summary)
    return out


def _summarize_wave_file(path, format, summarizer=None):
    """
    Summarize waveform files for indexing.

    Note: this function is a bit ugly, but it gets called *a lot* so be sure
    to profile before refactoring.
    """
    if summarizer is not None:
        try:
            return summarizer(path)
        except Exception:
            pass
    return summarize_generic_stream(path, format)


def _summarize_trace(
    trace: obspy.Trace,
    path: Optional[str] = None,
    name: Optional[str] = None,
    path_struct: Optional[str] = None,
    name_struct: Optional[str] = None,
) -> dict:
    """
    Function to extract info from traces for indexing.

    Parameters
    ----------
    trace
        The trace object
    path
        Other Parameters to the file
    name
        Name of the file
    path_struct
        directory structure to create
    name_struct
        structure of the file name if name is not used.
    """
    assert hasattr(trace, "stats"), "only a trace object is accepted"
    out = {"seedid": trace.id, "ext": WAVEFORM_EXT}
    t1, t2 = trace.stats.starttime, trace.stats.endtime
    out.update(_get_time_values(t1, t2))
    out.update(dict((x, c) for x, c in zip(NSLC, trace.id.split("."))))

    path_struct = path_struct if path_struct is not None else WAVEFORM_STRUCTURE
    name_struct = name_struct or WAVEFORM_NAME_STRUCTURE

    out.update(_get_path(out, path, name, path_struct, name_struct))
    return out


def _remove_base_path(series: pd.Series, base="") -> pd.Series:
    """
    Ensure paths stored in column name use unix style paths and have base
    path removed.
    """
    if series.empty:
        return series
    unix_paths = series.str.replace(os.sep, "/")
    unix_base_path = str(base).replace(os.sep, "/")
    return unix_paths.str.replace(unix_base_path, "")


def _natify_paths(series: pd.Series) -> pd.Series:
    """
    Natify paths in a series. IE, on windows replace / with \
    """
    if series.empty:
        return series
    return series.str.replace("/", os.sep)


class _IndexCache:
    """A simple class for caching indexes"""

    def __init__(self, bank, cache_size=5):
        self.max_size = cache_size
        self.bank = bank
        self.cache = pd.DataFrame(
            index=range(cache_size), columns="t1 t2 kwargs cindex".split()
        )
        self._current_index = 0
        # self.next_index = itertools.cycle(self.cache.index)

    def __call__(self, starttime, endtime, buffer, **kwargs):
        """get start and end times, perform in kernel lookup"""
        starttime, endtime = self._get_times(starttime, endtime)
        self._validate_kwargs(kwargs)
        # find out if the query falls within one cached times
        con1 = self.cache.t1 <= starttime
        con2 = self.cache.t2 >= endtime
        con3 = self.cache.kwargs == self._kwargs_to_str(kwargs)
        cached_index = self.cache[con1 & con2 & con3]
        if not len(cached_index):  # query is not cached get it from hdf5 file
            where = _get_kernel_query(
                starttime.astype(np.int64), endtime.astype(np.int64), int(buffer)
            )
            raw_index = self._get_index(where, **kwargs)
            # replace "None" with None
            ic = self.bank.index_str
            raw_index.loc[:, ic] = raw_index.loc[:, ic].replace(["None"], [None])
            # convert data types used by bank back to those seen by user
            index = raw_index.astype(dict(self.bank._dtypes_output))
            self._set_cache(index, starttime, endtime, kwargs)
        else:
            index = cached_index.iloc[0]["cindex"]
        # trim down index
        con1 = index["starttime"] >= (endtime + buffer)
        con2 = index["endtime"] <= (starttime - buffer)
        return index[~(con1 | con2)]

    @staticmethod
    def _get_times(starttime, endtime):
        """Return starttimes and endtimes."""
        # get defaults if starttime or endtime is none
        starttime = None if pd.isnull(starttime) else starttime
        endtime = None if pd.isnull(endtime) else endtime
        starttime = to_datetime64(starttime or SMALLDT64)
        endtime = to_datetime64(endtime or LARGEDT64)
        if starttime is not None and endtime is not None:
            if starttime > endtime:
                msg = "starttime cannot be greater than endtime."
                raise ValueError(msg)
        return starttime, endtime

    def _validate_kwargs(self, kwargs):
        """Ensure kwargs are supported."""
        kwarg_set = set(kwargs)
        if not kwarg_set.issubset(READ_HDF5_KWARGS):
            bad_kwargs = kwarg_set - set(READ_HDF5_KWARGS)
            msg = f"The following kwargs are not supported: {bad_kwargs}. "
            raise UnsupportedKeyword(msg)

    def _set_cache(self, index, starttime, endtime, kwargs):
        """Cache the current index"""
        ser = pd.Series(
            {
                "t1": starttime,
                "t2": endtime,
                "cindex": index,
                "kwargs": self._kwargs_to_str(kwargs),
            }
        )
        self.cache.loc[self._get_next_index()] = ser

    def _kwargs_to_str(self, kwargs):
        """convert kwargs to a string"""
        keys = sorted(list(kwargs.keys()))
        ou = str([(item, kwargs[item]) for item in keys])
        return ou

    def _get_index(self, where, fail_counts=0, **kwargs):
        """read the hdf5 file"""
        try:
            return pd.read_hdf(
                self.bank.index_path, self.bank._index_node, where=where, **kwargs
            )

        except (ClosedNodeError, Exception) as e:
            # Sometimes in concurrent updates the nodes need time to open/close
            if fail_counts > 10:
                raise e
            # Wait a bit and try again (up to 10 times)
            time.sleep(0.1)
            return self._get_index(where, fail_counts=fail_counts + 1, **kwargs)

    def clear_cache(self):
        """removes all cached dataframes."""
        self.cache = pd.DataFrame(
            index=range(self.max_size), columns="t1 t2 kwargs cindex".split()
        )

    def _get_next_index(self):
        """
        Get the next index value on cache.

        Note we can't use itertools.cycle here because it cant be pickled.
        """
        if self._current_index == len(self.cache.index) - 1:
            self._current_index = 0
        else:
            self._current_index += 1
        return self.cache.index[self._current_index]


@contextlib.contextmanager
def sql_connection(path, **kwargs):
    """
    A simple context manager for a sqlite connection.

    Parameters
    ----------
    path
        A path to the sqlite database.
    """
    con = sqlite3.connect(str(path), **kwargs)
    with con:
        yield con
    con.close()  # this is needed on windows but not linux, weird...


def _get_kernel_query(starttime: int, endtime: int, buffer: int):
    """
    Create a HDF5 kernel query based on start and end times.

    This is necessary because hdf5 doesnt accept inverted conditions.
    A slight buffer is applied to the ranges to make sure no edge files
    are excluded.
    """
    t1 = starttime - buffer
    t2 = endtime + buffer
    con = (
        f"(starttime>{t1:d} & starttime<{t2:d}) | "
        f"((endtime>{t1:d} & endtime<{t2:d}) | "
        f"(starttime<{t1:d} & endtime>{t2:d}))"
    )
    return con


# --- SQL stuff


def _str_of_params(value):
    """
    Make sure a list of params is returned.

    This allows user to specify a single parameter, a list, set, nparray, etc.
    to match on.
    """
    if isinstance(value, str):
        return value
    else:
        # try to coerce in a list of str
        try:
            return [str(x) for x in value]
        except TypeError:  # else fallback to str repr
            return str(value)


def _make_wheres(queries):
    """Create the where queries, join with AND clauses"""

    def _rename_keys(kwargs):
        """re-word some keys to make automatic sql generation easier"""
        if "eventid" in kwargs:
            kwargs["event_id"] = kwargs.pop("eventid")
        if "event_id" in kwargs:
            kwargs["event_id"] = _str_of_params(kwargs["event_id"])
        if "event_description" in kwargs:
            kwargs["event_description"] = _str_of_params(kwargs["event_description"])
        if "endtime" in kwargs:
            kwargs["maxtime"] = kwargs.pop("endtime")
        if "starttime" in kwargs:
            kwargs["mintime"] = kwargs.pop("starttime")
        return kwargs

    def _handle_nat(kwargs):
        """add a mintime that will exclude NaT values if endtime is used"""
        if "maxtime" in kwargs and "mintime" not in kwargs:
            kwargs["mintime"] = SMALLDT64.astype(np.int64) + 1
        return kwargs

    def _handle_dateline_transversal(kwargs, out):
        """Check if dateline should be transversed by query."""
        # if longitudes aren't being used bail out
        if not {"minlongitude", "maxlongitude"}.issubset(set(kwargs)):
            return kwargs, out
        # if dateline is not to be transversed by query bail out
        long_array = np.array([kwargs["minlongitude"], kwargs["maxlongitude"]])
        minlong, maxlong = map_longitudes(long_array)
        if not minlong > maxlong:
            return kwargs, out
        # remove min/max long from query dict and reform to two queries.
        kwargs.pop("minlongitude"), kwargs.pop("maxlongitude")
        cond = f"(( longitude > {minlong}) OR ( longitude < {maxlong})) "
        out.append(cond)
        return kwargs, out

    def _build_query(kwargs):
        """iterate each key/value and build query"""
        out = []
        kwargs, out = _handle_dateline_transversal(kwargs, out)
        for key, val in kwargs.items():
            # deal with simple min/max
            if key.startswith("min"):
                out.append(f"{key.replace('min', '')} > {val}")
            elif key.startswith("max"):
                out.append(f"{key.replace('max', '')} < {val}")
            # deal with equals or ins
            elif isinstance(val, Sequence):
                if isinstance(val, str):
                    val = [val]
                tup = str(tuple(val)).replace(",)", ")")  # no trailing comma
                out.append(f"{key} IN {tup}")
            else:
                out.append(f"{key} = {val}")
        return " AND ".join(out).replace("'", '"')

    kwargs = _dict_times_to_ns(queries)
    kwargs = _rename_keys(kwargs)
    kwargs = _handle_nat(kwargs)
    return _build_query(kwargs)


def _make_sql_command(cmd, table_name, columns=None, **kwargs) -> str:
    """build a sql command"""
    # get columns
    if columns:
        col = [columns] if isinstance(columns, str) else columns
        col += ["event_id"]  # event_id is used as index
        columns = ", ".join(col)
    elif cmd.upper() == "DELETE":
        columns = ""
    else:
        columns = "*"
    limit = kwargs.pop("limit", None)
    wheres = _make_wheres(kwargs)
    sql = f'{cmd.upper()} {columns} FROM "{table_name}"'
    if wheres:
        sql += f" WHERE {wheres}"
    if limit:
        sql += f" LIMIT {limit}"
    return sql + ";"


def _read_table(table_name, con, columns=None, **kwargs) -> pd.DataFrame:
    """
    Read a SQLite table.

    Parameters
    ----------
    table_name
    con
    columns

    Returns
    -------

    """
    # first ensure all times are ns (as ints)
    sql = _make_sql_command("select", table_name, columns=columns, **kwargs)
    # replace "None" with None
    return pd.read_sql(sql, con)


def _get_tables(con):
    """Return a list of table in sqlite database"""
    out = con.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return set(out)


def _drop_rows(table_name, con, columns=None, **kwargs):
    """Drop indicies in table"""
    sql = _make_sql_command("delete", table_name, columns=columns, **kwargs)
    con.execute(sql)


def _try_read_stream(stream_path, format=None, **kwargs):
    """Try to read a waveforms from file, if raises return None"""
    read = READ_DICT.get(format, obspy.read)
    stt = None
    try:
        stt = read(stream_path, **kwargs)
    except Exception:
        try:
            stt = obspy.read(stream_path, **kwargs)
        except Exception:
            warnings.warn("obspy failed to read %s" % stream_path, UserWarning)
        else:
            msg = f"{stream_path} was read but is not of format {format}"
            warnings.warn(msg, UserWarning)
    finally:
        return stt if stt else None
