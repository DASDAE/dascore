"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""

from __future__ import annotations

import inspect
import os.path
import warnings
from collections import defaultdict
from collections.abc import Generator
from functools import cache, cached_property, wraps
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Literal, get_type_hints

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, model_validator

import dascore as dc
from dascore.compat import Progress
from dascore.constants import (
    PROGRESS_LEVELS,
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    PatchType,
    SpoolType,
    max_lens,
    timeable_types,
)
from dascore.core.attrs import str_validator
from dascore.core.spool import DataFrameSpool
from dascore.exceptions import (
    InvalidFiberFileError,
    InvalidFiberIOError,
    MissingOptionalDependencyError,
    UnknownFiberFormatError,
)
from dascore.utils.io import IOResourceManager, get_handle_from_resource
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import _iter_filesystem, cached_method, iterate, warn_or_raise
from dascore.utils.models import (
    CommaSeparatedStr,
    DascoreBaseModel,
    DateTime64,
    TimeDelta64,
)
from dascore.utils.pd import _model_list_to_df
from dascore.utils.progress import track


class PatchFileSummary(DascoreBaseModel):
    """
    The necessary attributes for indexing a fiber file.

    A subset of [PatchAttributes](`dascore.core.attrs.PatchAttrs`).
    """

    model_config = ConfigDict(
        title="Patch File Summary",
        extra="ignore",
    )

    data_type: Annotated[Literal[VALID_DATA_TYPES], str_validator] = ""
    data_category: Annotated[Literal[VALID_DATA_CATEGORIES], str_validator] = ""
    instrument_id: str = Field("", max_length=max_lens["instrument_id"])
    experiment_id: str = Field("", max_length=max_lens["experiment_id"])
    tag: str = Field("", max_length=max_lens["tag"])
    station: str = Field("", max_length=max_lens["station"])
    network: str = Field("", max_length=max_lens["network"])
    dims: CommaSeparatedStr = Field("", max_length=max_lens["dims"])
    time_min: DateTime64 = np.datetime64("NaT")
    time_max: DateTime64 = np.datetime64("NaT")
    time_step: TimeDelta64 = np.timedelta64("NaT")
    # the attributes to index on
    file_version: str = ""
    file_format: str = ""
    path: str | Path = ""

    @property
    def dim_tuple(self):
        """Return a tuple of dimensions (eg ("time", "distance"))."""
        return tuple(self.dims.split(","))

    @model_validator(mode="before")
    @classmethod
    def translate_d_to_step(cls, data):
        """Translate d_time and d_distance to time_step, distance_step."""
        if isinstance(data, dict):
            for name in ["time", "distance"]:
                step_name, d_name = f"{name}_step", f"d_{name}"
                if step_name not in data and d_name in data:
                    data[step_name] = data.pop(d_name)
        return data

    def flat_dump(self):
        """Alias for dump, for compatibility with PatchAttrs.flat_dump."""
        return self.model_dump()


class _FiberIOManager:
    """
    A structure for intelligently storing, loading, and return FiberIO objects.

    This should only be used in conjunction with `FiberIO`.
    """

    def __init__(self, entry_point: str):
        self._entry_point = entry_point
        self._loaded_eps: set[str] = set()
        self._format_version = defaultdict(dict)
        self._extension_list = defaultdict(list)
        # This is a dict of {input_type: (fiberio_name, version)}
        self._fiber_io_by_input_type = defaultdict(set)
        self._fiber_io_name_ver = set()

    @cached_property
    def _eps(self):
        """
        Get the unloaded entry points registered to this domain into a dict of
        {name: ep}.
        """
        fiber_io_eps = entry_points(group="dascore.fiber_io")
        out = {x.name: x.load for x in fiber_io_eps}
        return pd.Series(out)

    @cached_property
    def known_formats(self):
        """Return names of known formats."""
        formats = self._eps.index.str.split("__").str[0]
        return set(formats) | set(self._format_version)

    @property
    def unloaded_formats(self):
        """Return names of known formats."""
        return sorted(self.known_formats - set(self._format_version))

    @cached_method
    def _get_fiber_io_by_input_type(self, input_type) -> set[FiberIO]:
        """Get a set of FiberIO instances that meet input type."""
        if input_type not in self._fiber_io_by_input_type:
            out = set()
            for input_set in self._fiber_io_by_input_type.values():
                out |= input_set
        else:
            out = self._fiber_io_by_input_type[input_type]
        return out

    @cache
    def _get_prioritized_list(self, input_type="file"):
        """Yield a prioritized list of fiber_ios."""
        # must load all plugins before getting list
        self.load_plugins()
        priority_fiber_ios = []
        second_class_fiber_ios = []
        for format_name in self.known_formats:
            unsorted = self._format_version[format_name]
            keys = sorted(unsorted, reverse=True)
            fiber_ios = [unsorted[key] for key in keys]
            priority_fiber_ios.append(fiber_ios[0])
            if len(fiber_ios) > 1:
                second_class_fiber_ios.extend(fiber_ios[1:])
        maybe_ios = priority_fiber_ios + second_class_fiber_ios
        # Now filter to input_type
        valid_fiberio_by_type = self._get_fiber_io_by_input_type(input_type)
        out = tuple(x for x in maybe_ios if x in valid_fiberio_by_type)
        # And return fiberIOs that much the input type.
        return out

    @cached_method
    def load_plugins(self, format: str | None = None):
        """Load plugin for specific format or ensure all formats are loaded."""
        if format is not None and format in self._format_version:
            return  # already loaded
        if not (unloaded := self.unloaded_formats):
            return
        formats = {format} if format is not None else unloaded
        # Load one, or all, formats
        for form in formats:
            for eps in self._eps.loc[self._eps.index.str.startswith(form)]:
                self.register_fiberio(eps()())
        # The selected format(s) should now be loaded
        assert set(formats).isdisjoint(self.unloaded_formats)

    def register_fiberio(self, fiberio: FiberIO):
        """Register a new fiber IO to manage."""
        forma, ver = fiberio.name.upper(), fiberio.version
        id_tuple = (forma, ver)
        if id_tuple in self._fiber_io_name_ver:
            return
        self._loaded_eps.add(fiberio.name)
        for ext in iter(fiberio.preferred_extensions):
            self._extension_list[ext].append(fiberio)
        self._format_version[forma][ver] = fiberio
        self._fiber_io_by_input_type[fiberio.input_type].add(fiberio)
        self._fiber_io_name_ver.add(id_tuple)

    @cached_method
    def get_fiberio(
        self,
        *,
        format: str | None = None,
        version: str | None = None,
        extension: str | None = None,
    ) -> FiberIO:
        """
        Return the most likely fiber_io for given inputs.

        If no such fiber_io exists, raise UnknownFiberFormat error.

        Parameters
        ----------
        format
            The format string indicating the format name
        version
            The version string of the format
        extension
            The extension of the file.
        """
        iterator = self.yield_fiberio(
            format=format,
            version=version,
            extension=extension,
        )
        for fiber_io in iterator:
            return fiber_io

    def yield_fiberio(
        self,
        format: str | None = None,
        version: str | None = None,
        extension: str | None = None,
        fiber_io_hint: dict[str, FiberIO] | None = None,
        input_type: str | None = None,
    ) -> Generator[FiberIO, None, None]:
        """
        Yields fiber IO object based on input priorities.

        The order is sorted in likelihood of the fiber_io being correct. For
        example, if file format is specified but file_version is not, all
        fiber_ios for the format will be yielded with the newest versions
        first in the list.

        If neither version nor format are specified but extension is all fiber_ios
        specifying the extension will be first in the list, sorted by format name
        and format version.

        If nothing is specified, all fiber_ios will be returned starting with
        the newest (the highest version) of each fiber_io, followed by older
        versions.

        Parameters
        ----------
        format
            The format string indicating the format name.
        version
            The version string of the format
        extension
            The extension of the file.
        fiber_io_hint
            If not None, a suspected fiber_io to use first. This is an
            optimization for file archives which tend to have many files of
            the same format.
        """
        fiber_io_hint = {} if fiber_io_hint is None else fiber_io_hint
        if version and not format:
            msg = "Providing only a version is not sufficient to determine format"
            raise UnknownFiberFormatError(msg)
        elif format is not None:
            self.load_plugins(format)
            yield from self._yield_format_version(format, version)
            return
        if input_type is not None and (out := fiber_io_hint.get(input_type)):
            yield out
        if extension is not None:
            yield from self._yield_extensions(extension, input_type)
        else:
            yield from self._get_prioritized_list(input_type)

    def _yield_format_version(self, format, version):
        """Yield file format/version prioritized fiber_ios."""
        assert isinstance(format, str), "Only works once format is known."
        format = format.upper()
        self.load_plugins(format)
        fiber_ios = self._format_version.get(format, None)
        # no format found
        if not fiber_ios:
            format_list = list(self.known_formats)
            msg = f"Unknown format {format}, " f"known formats are {format_list}"
            raise UnknownFiberFormatError(msg)
        # a version is specified
        if version:
            fiber_io = fiber_ios.get(version, None)
            if fiber_io is None:
                msg = (
                    f"Format {format} has no version: [{version}] "
                    f"known versions of this format are: {list(fiber_ios)}"
                )
                raise UnknownFiberFormatError(msg)
            yield fiber_io
            return
        # reverse sort fiber_ios and yield latest version first.
        for fiber_io in dict(sorted(fiber_ios.items(), reverse=True)).values():
            yield fiber_io
        return

    def _yield_extensions(self, extension, input_type=None):
        """Generator to get fiber_io prioritized by preferred extensions."""
        has_yielded = set()
        self.load_plugins()
        potential_fiberios = self._get_fiber_io_by_input_type(input_type)
        for fiber_io in self._extension_list[extension]:
            if fiber_io in potential_fiberios:
                yield fiber_io
            has_yielded.add(fiber_io)
        for fiber_io in self._get_prioritized_list(input_type):
            if fiber_io not in has_yielded:
                yield fiber_io

    def _get_format(
        self,
        path: str | Path | IOResourceManager,
        file_format: str | None = None,
        file_version: str | None = None,
        fiber_io_hint: dict[str, FiberIO] | None = None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Return the name of the format contained in the file and version number.

        See [`dascore.io.core.get_format`](`dascore.io.core.get_format`)
        for docs.
        """
        with IOResourceManager(path) as man:
            path = man.source
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} does not exist.")
            # get extension (str minus .)
            suffix = Path(path).suffix
            ext = suffix[1:] if suffix else None
            input_type = self._get_input_type_name(path)
            iterator = self.yield_fiberio(
                file_format,
                file_version,
                extension=ext,
                fiber_io_hint=fiber_io_hint,
                input_type=input_type,
            )
            for fiber_io in iterator:
                # We need to wrap this in try except to make it robust to what
                # may happen in each fiber_ios get_format method, many of which
                # may be third party code.
                func = fiber_io.get_format
                required_type = func._required_type
                func_input = None
                try:
                    # Get resource has to be in the try block because it can also
                    # raise, in which case the format doesn't belong.
                    func_input = man.get_resource(required_type)
                    format_version = func(func_input, _pre_cast=True)
                # For robustness, we need to catch everything else here.
                except Exception:
                    continue
                finally:
                    # If file handle-like seek back to 0 so it can be reused.
                    getattr(func_input, "seek", lambda x: None)(0)
                if format_version:
                    return format_version
            else:
                msg = f"Could not determine file format of {man.source}"
                raise UnknownFiberFormatError(msg)

    def _get_input_type_name(self, obj):
        """Get the name of the IO type."""
        # This effectively acts as a dispatch to determine which type of
        # FiberIO could possibly read the obj.
        out = "file"
        if isinstance(obj, str | Path) and (path := Path(obj)).exists():
            out = "directory" if path.is_dir() else "file"
        return out


# ------------- Protocol for File Format support


def _type_caster(func, sig, required_type, arg_name):
    """A decorator for casting types for arguments of cast ind."""
    fun_name = func.__name__

    # this is a subclass of a FiberIO subclass and its key methods
    # have already been wrapped. Just return.
    if getattr(func, "_type_caster_wrapped", False):
        return func

    @wraps(func)
    def _wrapper(*args, _pre_cast=False, **kwargs):
        """Wraps args but performs coercion to get proper stream."""
        # TODO look at replacing this with pydantic's type_guard thing.

        # this allows us to fast-track calls from generic functions
        if required_type is None or _pre_cast:
            return func(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        new_kw = bound.arguments
        resource = new_kw.pop(arg_name)
        try:
            new_resource = get_handle_from_resource(resource, required_type)
            new_kw[arg_name] = new_resource
            # kwargs is included in bound arguments, need to re-attach
            new_kw.update(new_kw.pop("kwargs", {}))
            out = func(**new_kw)
        except Exception as e:  # get_format can't raise; must return false.
            if fun_name == "get_format":
                out = False
            else:
                raise e
        else:
            # if a new file handle was created we need to close it now. But it
            # shouldn't close any passed in, that should happen up the stack.
            if new_resource is not resource and hasattr(new_resource, "close"):
                new_resource.close()
        return out

    # attach the function and required type for later use
    _wrapper.func = func
    # subclasses of FIBERIO subclasses can wrap this twice, so we mark
    # it to avoid that scenario.
    _wrapper._type_caster_wrapped = True
    # also specify required type
    _wrapper._required_type = required_type

    return _wrapper


def _is_wrapped_func(func1, func2):
    """Small helper function to determine if func1 is func2, unwrapping decorators."""
    func = func1
    while hasattr(func, "func") or hasattr(func, "__func__"):
        func = getattr(func, "func", func)
        func = getattr(func, "__func__", func)
    return func is func2


class FiberIO:
    """
    An interface which adds support for a given filer format.

    This class should be subclassed when adding support for new formats.
    """

    name: str = ""
    version: str = ""
    preferred_extensions: tuple[str] = ()
    # Specifies if this fiber IO expects a directory or single file
    input_type: Literal["file", "directory"] = "file"

    manager = _FiberIOManager("dascore.fiber_io")

    # A dict of methods which should implement automatic type casting.
    # and the index of the parameter to type cast.
    _automatic_type_casters = FrozenDict(
        {
            "read": 1,
            "scan": 1,
            "write": 2,
            "get_format": 1,
        }
    )

    def read(self, resource, **kwargs) -> SpoolType:
        """
        Load data from a path.

        *kwargs should include support for selecting expected dimensions. For
        example, distance=(100, 200) would only read data with distance from
        100 to 200.
        """
        msg = f"FiberIO: {self.name} has no read method"
        raise NotImplementedError(msg)

    def scan(self, resource, **kwargs) -> list[dc.PatchAttrs]:
        """Returns a list of summary info for patches contained in file."""
        # default scan method reads in the file and returns required attributes
        # however, this can be very slow, so each parser should implement scan
        # when possible.
        try:
            spool = self.read(resource)
        except NotImplementedError:
            msg = f"FiberIO: {self.name} has no scan or read method"
            raise NotImplementedError(msg)
        out = []
        for pa in spool:
            new = pa.attrs.update(
                file_format=self.name,
                path=str(resource),
            )
            out.append(new)
        return out

    def write(self, spool: SpoolType, resource, **kwargs):
        """Write the spool to a resource (eg path, stream, etc.)."""
        msg = f"FiberIO: {self.name} has no write method"
        raise NotImplementedError(msg)

    def get_format(self, resource, **kwargs) -> tuple[str, str] | bool:
        """
        Return a tuple of (format_name, version_numbers).

        This should only work if path is the supported file format, otherwise
        raise UnknownFiberError or return False.
        """
        msg = f"FiberIO: {self.name} has no get_version method"
        raise NotImplementedError(msg)

    @property
    def implements_read(self) -> bool:
        """Returns True if the subclass implements its own scan method else False."""
        return not _is_wrapped_func(self.read, FiberIO.read)

    @property
    def implements_write(self) -> bool:
        """Returns True if the subclass implements its own scan method else False."""
        return not _is_wrapped_func(self.write, FiberIO.write)

    @property
    def implements_scan(self) -> bool:
        """Returns True if the subclass implements its own scan method else False."""
        return not _is_wrapped_func(self.scan, FiberIO.scan)

    @property
    def implements_get_format(self) -> bool:
        """Return True if the subclass implements its own get_format method."""
        return not _is_wrapped_func(self.get_format, FiberIO.get_format)

    @classmethod
    def get_supported_io_table(cls):
        """Make a table of all the supported formats and the methods."""
        # load all the plugins, so we know about all the FiberIO classes
        FiberIO.manager.load_plugins()
        out = []
        # iterate the dict _format_version_items,
        # which has the form {format_name: {version_str: FiberIO}}
        for format_name, version_dict in FiberIO.manager._format_version.items():
            for version_name, fiberio in version_dict.items():
                format_info = {
                    "name": format_name,
                    "version": version_name,
                    "scan": fiberio.implements_scan,
                    "get_format": fiberio.implements_get_format,
                    "read": fiberio.implements_read,
                    "write": fiberio.implements_write,
                }
                out.append(format_info)
        return pd.DataFrame(out)

    def _updated_after(self, resource, timestamp):
        """Determine if the resource was updated after specified mtime."""
        if not timestamp:
            return True
        return Path(resource).stat().st_mtime > timestamp

    def __hash__(self):
        """FiberIO instances should be uniquely defined by (format, version)."""
        return hash((self.name, self.version))

    def __init_subclass__(cls, **kwargs):
        """Hook for registering subclasses."""
        # check that the subclass is valid
        if not cls.name:
            msg = "You must specify the file format with the name field."
            raise InvalidFiberIOError(msg)
        # register fiber_io
        manager: _FiberIOManager = cls.__mro__[1].manager
        manager.register_fiberio(cls())
        # decorate methods for type-casting
        for name, param_ind in cls._automatic_type_casters.items():
            method = getattr(cls, name)
            sig = inspect.signature(method)
            arg_name = list(sig.parameters)[param_ind]
            required_type = get_type_hints(method).get(arg_name)
            method_wrapped = _type_caster(method, sig, required_type, arg_name)
            setattr(cls, name, method_wrapped)


def read(
    path: str | Path | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    time: tuple[timeable_types | None, timeable_types | None] | None = None,
    distance: tuple[float | None, float | None] | None = None,
    **kwargs,
) -> SpoolType:
    """
    Read a fiber file.

    For most cases, [`dascore.spool`](`dascore.spool`) is preferable to
    this function.

    Parameters
    ----------
    path
        A path to the file to read.
    file_format
        A string indicating the file format. If not provided dascore will
        try to estimate the format.
    file_version
        An optional string indicating the format version.
    time
        An optional tuple of time ranges.
    distance
        An optional tuple of distances.
    *kwargs
        All kwargs are passed to the format-specific read functions.

    Notes
    -----
    Unlike [`spool`](`dascore.spool`) this function reads the entire file
    into memory.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("prodml_2.1.h5")
    >>>
    >>> patch = dc.read(file_path)
    """
    with IOResourceManager(path) as man:
        if not file_format or not file_version:
            file_format, file_version = get_format(
                man,
                file_format=file_format,
                file_version=file_version,
            )
        fiber_io = FiberIO.manager.get_fiberio(format=file_format, version=file_version)
        required_type = fiber_io.read._required_type
        path = man.get_resource(required_type)
        out = fiber_io.read(
            path,
            file_version=file_version,
            time=time,
            distance=distance,
            _pre_cast=True,
            **kwargs,
        )
        # if resource has a seek go back to 0 so this stream can be re-used.
        getattr(path, "seek", lambda x: None)(0)
        return out


def scan_to_df(
    path: Path | str | PatchType | SpoolType | IOResourceManager | pd.DataFrame,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS = "standard",
    exclude=("history",),
) -> pd.DataFrame:
    """
    Scan a path, return a dataframe of contents.

    The columns of the dataframe depend on the attributes and coordinates
    found in the data files.

    Parameters
    ----------
    path
        The path to the to file to scan
    file_format
        Format of the file. If not provided DASCore will try to determine it.
    file_version
        The version string of the file.
    exclude
        A sequence of column names to exclude in the final dataframe.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("prodml_2.1.h5")
    >>>
    >>> df = dc.scan_to_df(file_path)
    """
    if isinstance(path, pd.DataFrame):
        return path
    if isinstance(path, DataFrameSpool):
        return path.get_contents()
    info = scan(
        path=path,
        file_format=file_format,
        file_version=file_version,
        ext=ext,
        timestamp=timestamp,
        progress=progress,
    )
    df = _model_list_to_df(info, exclude=exclude)
    return df


def _iterate_scan_inputs(patch_source, ext, mtime, include_directories=True, **kwargs):
    """Yield scan candidates."""
    for el in iterate(patch_source):
        if isinstance(el, str | Path) and (path := Path(el)).exists():
            generator = _iter_filesystem(
                path, ext=ext, timestamp=mtime, include_directories=include_directories
            )
            yield from generator
        else:
            yield el


def _get_fiber_io_and_req_type(
    manager,
    file_format: str | None = None,
    file_version: str | None = None,
    fiber_io_hint=None,
):
    """
    Get the fiber IO for a patch source.

    Raises
    ------
    UnknownFileFormatError if no format is determinable from the
    patch_source

    """
    if not file_format or not file_version:
        file_format_, file_version_ = FiberIO.manager._get_format(
            path=manager,
            file_format=file_format,
            file_version=file_version,
            fiber_io_hint=fiber_io_hint,
        )
    else:
        # we need separate loop variables so this doesn't get assumed
        # to be the version/format in all subsequent values for the loop.
        file_format_, file_version_ = file_format, file_version
    fiber_io_hint = FiberIO.manager.get_fiberio(
        format=file_format_, version=file_version_
    )
    req_type = getattr(fiber_io_hint.scan, "_required_type", None)
    resource = manager.get_resource(req_type)
    # this will get the required resource type to pass to scan.
    return fiber_io_hint, resource


def _count_generator(generator):
    """Estimate the number of updates needed."""
    # TODO: This is a but sloppy, need to think of a better way to do
    # this to avoid double iteration.
    # First get total number of possible update-able files
    entity_count = 0
    for _ in generator:
        entity_count += 1
    return entity_count


def _handle_missing_optionals(outputs, optional_dep_dict):
    """
    Inform the user there are files that can be read but the proper
    dependencies are not installed.

    If there are other readable files that were found, raise a warning.
    Otherwise, raise a MissingOptionalDependencyError.
    """
    msg = (
        f"DASCore found files that can be read if additional packages are "
        f"installed. The needed packages and the found number of files are: "
        f"{dict(optional_dep_dict)}"
    )
    warn_or_raise(
        msg,
        exception=MissingOptionalDependencyError,
        warning=UserWarning,
        behavior="warn" if len(outputs) else "raise",
    )


def scan(
    path: Path | str | PatchType | SpoolType | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    ext: str | None = None,
    timestamp: float | None = None,
    progress: PROGRESS_LEVELS | Progress = "standard",
) -> list[dc.PatchAttrs]:
    """
    Scan a potential patch source, return a list of PatchAttrs.

    Parameters
    ----------
    path
        A resource containing Fiber data.
    file_format
        Format of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    file_version
        Version of the file. If not provided DASCore will try to determine it.
        Only applicable for path-like inputs.
    ext : str or None
        The extensions to map.
    timestamp : int or float
        Time stamp indicating the minimum mtime.
    progress
        The type of progress bar to use. None disables progress bar and
        "basic" is best for low latency scenarios. Can also acceted a subclass
        of rich.progress.Progress.

    Returns
    -------
    A list of [`PatchAttrs`](`dascore.core.attrs.PatchAttrs`) or subclasses
    which may have extra fields.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("prodml_2.1.h5")
    >>>
    >>> attr_list = dc.scan(file_path)

    See also [`iter_fs_contents`](`dascore.utils.misc.iter_fs_contents`)
    """
    out = []
    fiber_io_hint: dict[str, FiberIO] = {}
    # A dict for keeping track of missing optional dependencies.
    missing_optional_deps = defaultdict(lambda: 0)
    # Unfortunately, we have to iterate the scan candidates twice to get
    # an estimate for the progress bar length. Maybe there is a better way...
    _generator = _iterate_scan_inputs(
        path, ext=ext, mtime=timestamp, include_directories=False
    )
    length = _count_generator(_generator)
    generator = _iterate_scan_inputs(path, ext=ext, mtime=timestamp)
    # We want to avoid printing long object str reprs, so only print paths.
    resource_str = path if isinstance(path, str | Path) else ""
    tracker = track(
        generator,
        f"scan {resource_str}",
        progress=progress,
        length=length,
        min_length=20,
    )
    try:
        for patch_source in tracker:
            # just pull attrs from patch
            if isinstance(patch_source, dc.Patch):
                out.append(patch_source.attrs)
                continue
            with IOResourceManager(patch_source) as man:
                try:
                    fiber_io, resource = _get_fiber_io_and_req_type(
                        man,
                        file_format=file_format,
                        file_version=file_version,
                        fiber_io_hint=fiber_io_hint,
                    )
                except UnknownFiberFormatError:  # skip bad entities
                    continue
                # Cache this fiber io to given preferential treatment next
                # iteration. This speeds up the common case of many files
                # with the same format.
                fiber_io_hint[fiber_io.input_type] = fiber_io
                # Special handling of directory FiberIOs.
                if fiber_io.input_type == "directory":
                    # Directory fiber_io should send skip signal back to generator
                    # so that no files/sub directories are scanned.
                    generator.send("skip")
                    if not fiber_io._updated_after(resource, timestamp):
                        continue
                    # Directory FiberIO may need to know the time after which
                    # contents should be returned.
                    source = fiber_io.scan(
                        resource, timestamp=timestamp, _pre_cast=True
                    )
                else:
                    try:
                        source = fiber_io.scan(resource, _pre_cast=True)
                    # This happens if the file is corrupt see #346.
                    except (OSError, InvalidFiberFileError, ValueError, TypeError):
                        warnings.warn(f"Failed to scan {resource}", UserWarning)
                        continue
                    except MissingOptionalDependencyError as ex:
                        missing_optional_deps[ex.msg.split(" ")[0]] += 1
                        continue
                for attr in source:
                    out.append(dc.PatchAttrs.from_dict(attr))
    # Ensure ctl + c exists scan.
    except KeyboardInterrupt:
        getattr(progress, "stop", lambda: None)()
        raise
    if missing_optional_deps:
        _handle_missing_optionals(out, missing_optional_deps)
    return out


def get_format(
    path: str | Path | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    fiber_io_hint: dict[str, FiberIO] | None = None,
    **kwargs,
) -> tuple[str, str]:
    """
    Return the name of the format contained in the file and version number.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The known file format.
    file_version
        The known file version.
    fiber_io_hint
        A dict of {input_type: fiber_io}. This is an optimization
        which assumes the last used fiberio (for a given input type)
        is likely to be the next one.

    Returns
    -------
    A tuple of (file_format_name, version) both as strings.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("prodml_2.1.h5")
    >>>
    >>> file_format, file_version = dc.get_format(file_path)
    """
    out = FiberIO.manager._get_format(
        path, file_format, file_version, fiber_io_hint, **kwargs
    )
    return out


def write(
    patch_or_spool,
    path: str | Path,
    file_format: str,
    file_version: str | None = None,
    **kwargs,
) -> Path:
    """
    Write a Patch or Spool to disk.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The string indicating the format to write.
    file_version
        Optionally specify the version of the file, else use the latest
        version for the format.

    Raises
    ------
    [`UnkownFiberFormatError`](`dascore.exceptions.UnknownFiberFormatError`)
        - Could not determine the fiber format.

    Examples
    --------
    >>> from pathlib import Path
    >>> import dascore as dc
    >>>
    >>> patch = dc.get_example_patch()
    >>> path = Path("output.h5")
    >>> _ = dc.write(patch, path, "dasdae")
    >>>
    >>> assert path.exists()
    >>> path.unlink()
    """
    fiber_io = FiberIO.manager.get_fiberio(format=file_format, version=file_version)
    if not isinstance(patch_or_spool, dc.BaseSpool):
        patch_or_spool = dc.spool([patch_or_spool])
    with IOResourceManager(path) as man:
        func = fiber_io.write
        required_type = func._required_type
        resource = man.get_resource(required_type)
        func(patch_or_spool, resource, _pre_cast=True, **kwargs)
    return path
