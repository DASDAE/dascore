"""Base class for ObsPlus' in-process databases (aka banks)."""
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from inspect import isclass
from types import MappingProxyType as MapProxy
from typing import Optional, TypeVar, Mapping, Iterable, Union

import numpy as np
import pandas as pd
from pandas.io.sql import DatabaseError

import obsplus
from obsplus.constants import CPU_COUNT, bank_subpaths_type
from obsplus.exceptions import BankDoesNotExistError
from obsplus.interfaces import ProgressBar
from obsplus.utils.bank import _IndexCache
from obsplus.utils.misc import get_progressbar, iter_files, iterate, get_version_tuple
from obsplus.utils.time import to_datetime64

BankType = TypeVar("BankType", bound="_Bank")


class _Bank(ABC):
    """
    The abstract base class for ObsPlus' banks.

    Used to access local archives in a client-like fashion.
    """

    # hdf5 compression defaults
    _complib = "blosc"
    _complevel = 9
    # attributes subclasses need to define
    ext = ""
    bank_path: Path = ""
    namespace = ""
    index_name = ".index.h5"  # name of index file
    executor = None  # an executor for using parallelism
    # optional str defining the directory structure and file name schemes
    path_structure = None
    name_structure = None
    # the minimum obsplus version. If not met delete index and re-index
    # bump when database schema change.
    _min_version = "0.0.3"
    # status bar attributes
    _bar_update_interval = 50  # number of files before updating bar
    _min_files_for_bar = 100  # min number of files before using bar enabled
    _read_func: callable  # function for reading datatype
    # required dtypes for input to storage layer
    _dtypes_input: Mapping = MapProxy({})
    # required dtypes for output from bank
    _dtypes_output: Mapping = MapProxy({})
    # the index cache (can greatly reduce IO efforts)
    _index_cache: Optional[_IndexCache] = None

    @abstractmethod
    def read_index(self, **kwargs) -> pd.DataFrame:
        """Read the index filtering on various params."""

    @abstractmethod
    def update_index(self: BankType) -> BankType:
        """Update the index."""

    @abstractmethod
    def last_updated_timestamp(self) -> Optional[float]:
        """
        Get the last modified time stored in the index.

        If not available return None.
        """

    @property
    def last_updated(self) -> Optional[np.datetime64]:
        """
        Get the last time (UTC) that the bank was updated.
        """
        return to_datetime64(self.last_updated_timestamp)

    @abstractmethod
    def _read_metadata(self) -> pd.DataFrame:
        """Return a dictionary of metadata."""

    # --- path/node related objects

    @property
    def index_path(self):
        """Return the expected path to the index file."""
        return Path(self.bank_path) / self.index_name

    @property
    def _index_node(self):
        """Return the node/table where the index information is stored."""
        return "/".join([self.namespace, "index"])

    @property
    def _index_version(self) -> str:
        """Get the version of obsplus used to create the index."""
        return self._read_metadata()["obsplus_version"].iloc[0]

    @property
    def _time_node(self):
        """The node/table where the update time information is stored."""
        return "/".join([self.namespace, "last_updated"])

    @property
    def _meta_node(self):
        """The node/table where the update metadata is stored."""
        return "/".join([self.namespace, "metadata"])

    @property
    def _version_or_none(self) -> Optional[str]:
        """Return the version string or None if it doesn't yet exist."""
        try:
            version = self._index_version
        except (FileNotFoundError, DatabaseError):
            return
        return version

    def _enforce_min_version(self):
        """
        Check version of obsplus used to create index and delete index if the
        minimum version requirement is not met.
        """
        version = self._version_or_none
        if version is not None:
            min_version_tuple = get_version_tuple(self._min_version)
            version_tuple = get_version_tuple(version)
            if min_version_tuple > version_tuple:
                msg = (
                    f"The indexing schema has changed since {self._min_version} "
                    f"the index will be recreated."
                )
                warnings.warn(msg)
                os.remove(self.index_path)

    def _warn_on_newer_version(self):
        """
        Issue a warning if the bank was created by a newer version of obsplus.

        If this is the case, there is no guarantee it will work.
        """
        version = self._version_or_none
        if version is not None:
            obsplus_version = get_version_tuple(obsplus.__last_version__)
            bank_version = get_version_tuple(version)
            if bank_version > obsplus_version:
                msg = (
                    f"The bank was created with a newer version of ObsPlus ("
                    f"{version}), you are running ({obsplus.__last_version__}),"
                    f"You may encounter problems, consider updating ObsPlus."
                )
                warnings.warn(msg)

    def _unindexed_iterator(self, paths: Optional[bank_subpaths_type] = None):
        """Return an iterator of potential unindexed files."""
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        last_updated = self.last_updated_timestamp  # this needs db so only call once
        if last_updated is not None:
            mtime = last_updated - 0.001
        # get paths to iterate
        bank_path = self.bank_path
        if paths is None:
            paths = self.bank_path
        else:
            paths = [
                f"{self.bank_path}/{x}" if str(bank_path) not in str(x) else str(x)
                for x in iterate(paths)
            ]
        # return file iterator
        return iter_files(paths, ext=self.ext, mtime=mtime)

    def _measure_iterator(self, iterable: Iterable, bar: Optional[ProgressBar] = None):
        """
        A generator to yield un-indexed files and update progress bar.

        Parameters
        ----------
        iterable
            Any iterable to yield.
        bar
            Any object which has a 'update' method.
        """
        # get progress bar
        bar = self.get_progress_bar(bar)
        # get the iterator
        for num, obj in enumerate(iterable):
            # update bar if count is in update interval
            if bar is not None and num % self._bar_update_interval == 0:
                bar.update(num)
            yield obj
        # finish progress bar
        getattr(bar, "finish", lambda: None)()  # call finish if bar exists

    def _make_meta_table(self):
        """get a dataframe of meta info"""
        meta = dict(
            path_structure=self.path_structure,
            name_structure=self.name_structure,
            obsplus_version=obsplus.__last_version__,
        )
        return pd.DataFrame(meta, index=[0])

    def get_service_version(self):
        """Return the version of obsplus used to create index."""
        return self._index_version

    def ensure_bank_path_exists(self, create=False):
        """
        Ensure the bank_path exists else raise an BankDoesNotExistError.

        If create is True, simply create the bank.
        """
        path = Path(self.bank_path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            msg = f"{path} is not a directory, cant read bank"
            raise BankDoesNotExistError(msg)

    def get_progress_bar(self, bar=None) -> Optional[ProgressBar]:
        """
        Return a progress bar instance based on bar parameter.

        If bar is False, return None.
        If bar is None return default Bar
        If bar is a subclass of ProgressBar, init class and set max_values.
        If bar is an instance of ProgressBar, return it.
        """
        # conditions to bail out early
        if bar is False:  # False indicates no bar is to be used
            return None
        # bar is already instantiated
        elif isinstance(bar, ProgressBar) and not isclass(bar):
            return bar
        # next, count number of files
        num_files = sum([1 for _ in self._unindexed_iterator()])
        if num_files < self._min_files_for_bar:  # not enough files to use bar
            return None
        # instantiate bar and return
        kwargs = {"min_value": self._min_files_for_bar, "max_value": num_files}
        # an instance should be init'ed
        if isinstance(bar, type) and issubclass(bar, ProgressBar):
            return bar(**kwargs)
        elif bar is None:
            return get_progressbar(**kwargs)
        else:
            msg = f"{bar} is not a valid input for get_progress_bar"
            raise ValueError(msg)

    def clear_cache(self):
        """
        Clear the index cache if the bank is using one.
        """
        if self._index_cache is not None:
            self._index_cache.clear_cache()

    @property
    def _max_workers(self):
        """
        Return the max number of workers allowed by the executor.

        If the Executor has no attribute `_max_workers` use the number of
        CPUs instead. If there is no executor assigned to bank instance
        return 1.
        """
        executor = getattr(self, "executor", None)
        if executor is not None:
            return getattr(executor, "_max_workers", CPU_COUNT)
        return 1

    def _map(self, func, args, chunksize=1):
        """
        Map the args to function, using executor if defined, else perform
        in serial.
        """
        if self.executor is not None:
            return self.executor.map(func, args, chunksize=chunksize)
        else:
            return (func(x) for x in args)

    @classmethod
    def load_example_bank(
        cls: BankType,
        dataset: str = "default_test",
        path: Optional[Union[str, Path]] = None,
    ) -> BankType:
        """
        Create an example bank which is safe to modify.

        Copies relevant files from a dataset to a specified path, or a
        temporary directory if None is specified.

        Parameters
        ----------
        dataset
            The name of the dataset.
        path
            The path to which the dataset files will be copied. If None
            just create a temporary directory.
        """
        # determine which directory in the dataset this bank needs
        data_types = {
            obsplus.EventBank: "event_path",
            obsplus.StationBank: "station_path",
            obsplus.WaveBank: "waveform_path",
        }
        ds = obsplus.load_dataset(dataset)
        destination = Path(tempfile.mkdtemp() if path is None else path) / "temp"
        assert cls in data_types, f"{cls} Bank type not supported."
        path_to_copy = getattr(ds, data_types[cls])
        shutil.copytree(path_to_copy, destination)
        return cls(destination)

    def __repr__(self):
        """Return the class name with bank path."""
        name = type(self).__name__
        return f"{name}(base_path={self.bank_path})"

    __str__ = __repr__
