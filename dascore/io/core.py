"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""
from __future__ import annotations

import inspect
import os.path
from collections import defaultdict
from functools import cached_property, wraps
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Literal, get_type_hints

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

import dascore as dc
from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    PatchType,
    SpoolType,
    max_lens,
    timeable_types,
)
from dascore.core.attrs import str_validator
from dascore.exceptions import InvalidFiberIOError, UnknownFiberFormatError
from dascore.utils.io import IOResourceManager, get_handle_from_resource
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import cached_method, iterate
from dascore.utils.models import (
    CommaSeparatedStr,
    DascoreBaseModel,
    DateTime64,
    TimeDelta64,
)
from dascore.utils.pd import _model_list_to_df


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

    @cached_property
    def _prioritized_list(self):
        """Yield a prioritized list of formatters."""
        # must load all plugins before getting list
        self.load_plugins()
        priority_formatters = []
        second_class_formatters = []
        for format_name in self.known_formats:
            unsorted = self._format_version[format_name]
            keys = sorted(unsorted, reverse=True)
            formatters = [unsorted[key] for key in keys]
            priority_formatters.append(formatters[0])
            if len(formatters) > 1:
                second_class_formatters.extend(formatters[1:])
        return tuple(priority_formatters + second_class_formatters)

    @cached_method
    def load_plugins(self, format: str | None = None):
        """Load plugin for specific format or ensure all formats are loaded."""
        if format is not None and format in self._format_version:
            return  # already loaded
        if not (unloaded := self.unloaded_formats):
            return
        formats = {format} if format is not None else unloaded
        # load one, or all, formats
        for form in formats:
            for eps in self._eps.loc[self._eps.index.str.startswith(form)]:
                self.register_fiberio(eps()())
        # The selected format(s) should now be loaded
        assert set(formats).isdisjoint(self.unloaded_formats)

    def register_fiberio(self, fiberio: FiberIO):
        """Register a new fiber IO to manage."""
        forma, ver = fiberio.name.upper(), fiberio.version
        self._loaded_eps.add(fiberio.name)
        for ext in iter(fiberio.preferred_extensions):
            self._extension_list[ext].append(fiberio)
        self._format_version[forma][ver] = fiberio

    @cached_method
    def get_fiberio(
        self,
        format: str | None = None,
        version: str | None = None,
        extension: str | None = None,
    ) -> FiberIO:
        """
        Return the most likely formatter for given inputs.

        If no such formatter exists, raise UnknownFiberFormat error.

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
        for formatter in iterator:
            return formatter

    def yield_fiberio(
        self,
        format: str | None = None,
        version: str | None = None,
        extension: str | None = None,
        formatter_hint: FiberIO | None = None,
    ) -> Self:
        """
        Yields fiber IO object based on input priorities.

        The order is sorted in likelihood of the formatter being correct. For
        example, if file format is specified but file_version is not, all
        formatters for the format will be yielded with the newest versions
        first in the list.

        If neither version nor format are specified but extension is all formatters
        specifying the extension will be first in the list, sorted by format name
        and format version.

        If nothing is specified, all formatters will be returned starting with
        the newest (the highest version) of each formatter, followed by older
        versions.

        Parameters
        ----------
        format
            The format string indicating the format name.
        version
            The version string of the format
        extension
            The extension of the file.
        formatter_hint
            If not None, a suspected formatter to use first. This is an
            optimization for file archives which tend to have many files of
            the same format.

        """
        # TODO replace this with concise pattern matching once 3.9 is dropped
        if formatter_hint is not None:
            yield formatter_hint
        if version and not format:
            msg = "Providing only a version is not sufficient to determine format"
            raise UnknownFiberFormatError(msg)
        if format is not None:
            self.load_plugins(format)
            yield from self._yield_format_version(format, version)
        elif extension is not None:
            yield from self._yield_extensions(extension)
        else:
            yield from self._prioritized_list

    def _yield_format_version(self, format, version):
        """Yield file format/version prioritized formatters."""
        if format is not None:
            format = format.upper()
            self.load_plugins(format)
            formatters = self._format_version.get(format, None)
            # no format found
            if not formatters:
                format_list = list(self.known_formats)
                msg = f"Unknown format {format}, " f"known formats are {format_list}"
                raise UnknownFiberFormatError(msg)
            # a version is specified
            if version:
                formatter = formatters.get(version, None)
                if formatter is None:
                    msg = (
                        f"Format {format} has no version: [{version}] "
                        f"known versions of this format are: {list(formatters)}"
                    )
                    raise UnknownFiberFormatError(msg)
                yield formatter
                return
            # reverse sort formatters and yield latest version first.
            for formatter in dict(sorted(formatters.items(), reverse=True)).values():
                yield formatter
            return

    def _yield_extensions(self, extension):
        """Generator to get formatter prioritized by preferred extensions."""
        has_yielded = set()
        self.load_plugins()
        for formatter in self._extension_list[extension]:
            yield formatter
            has_yielded.add(formatter)
        for formatter in self._prioritized_list:
            if formatter not in has_yielded:
                yield formatter


# ------------- Protocol for File Format support


def _type_caster(func, sig, required_type, arg_name):
    """A decorator for casting types for arguments of cast ind."""
    fun_name = func.__name__

    # this is a subclass of a FiberIO subclass and its key methods
    # have already been wrapped. Just return.
    if getattr(func, "_type_caster_wrapped", False):
        return func

    @wraps(func)
    def _wraper(*args, _pre_cast=False, **kwargs):
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
    _wraper.func = func
    # subclasses of FIBERIO subclasses can wrap this twice, so we mark
    # it to avoid that scenario.
    _wraper._type_caster_wrapped = True
    # also specify required type
    _wraper._required_type = required_type

    return _wraper


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

    def scan(self, resource) -> list[dc.PatchAttrs]:
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

    def write(self, spool: SpoolType, resource):
        """Write the spool to a resource (eg path, stream, etc.)."""
        msg = f"FiberIO: {self.name} has no write method"
        raise NotImplementedError(msg)

    def get_format(self, resource) -> tuple[str, str] | bool:
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

    def __hash__(self):
        """FiberIO instances should be uniquely defined by (format, version)."""
        return hash((self.name, self.version))

    def __init_subclass__(cls, **kwargs):
        """Hook for registering subclasses."""
        # check that the subclass is valid
        if not cls.name:
            msg = "You must specify the file format with the name field."
            raise InvalidFiberIOError(msg)
        # register formatter
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
        formatter = FiberIO.manager.get_fiberio(file_format, file_version)
        required_type = formatter.read._required_type
        path = man.get_resource(required_type)
        out = formatter.read(
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
    path: Path | str | PatchType | SpoolType | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    exclude=("history",),
) -> pd.DataFrame:
    """
    Scan a path, return a dataframe of contents.

    The columns of the dataframe depend on the attributes and coordinates
    found in the data files.

    Parameters
    ----------
    path
        The path the to file to scan
    file_format
        Format of the file. If not provided DASCore will try to determine it.
    file_version
        The version string of the file.
    exclude
        A sequence of strings to exclude from the analysis.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.downloader import fetch
    >>>
    >>> file_path = fetch("prodml_2.1.h5")
    >>>
    >>> df = dc.scan_to_df(file_path)
    """
    info = scan(
        path=path,
        file_format=file_format,
        file_version=file_version,
    )
    df = _model_list_to_df(info, exclude=exclude)
    return df


def _iterate_scan_inputs(patch_source):
    """Yield scan candidates."""
    for el in iterate(patch_source):
        if isinstance(el, str | Path) and (path := Path(el)).exists():
            if path.is_dir():  # directory, yield contents
                for sub_path in path.rglob("*"):
                    if sub_path.is_dir():
                        continue
                    yield sub_path
            else:
                yield path
        else:
            yield el


def scan(
    path: Path | str | PatchType | SpoolType | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
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
    """
    out = []
    formatter = None
    for patch_source in _iterate_scan_inputs(path):
        # just pull attrs from patch
        if isinstance(patch_source, dc.Patch):
            out.append(patch_source.attrs)
            continue
        with IOResourceManager(patch_source) as man:
            # get fiberio
            if not file_format or not file_version:
                try:
                    file_format_, file_version_ = get_format(
                        man,
                        file_format=file_format,
                        file_version=file_version,
                        formatter_hint=formatter,
                    )
                except UnknownFiberFormatError:  # skip bad entities
                    continue
            else:
                # we need separate loop variables so this doesn't get assumed
                # to be the version/format in all subsequent values for the loop.
                file_format_, file_version_ = file_format, file_version
            formatter = FiberIO.manager.get_fiberio(file_format_, file_version_)
            req_type = getattr(formatter.scan, "_required_type", None)
            # this will get an open file handle to pass to get_resource
            patch_thing = man.get_resource(req_type)
            for attr in formatter.scan(patch_thing, _pre_cast=True):
                out.append(dc.PatchAttrs.from_dict(attr))
    return out


def get_format(
    path: str | Path | IOResourceManager,
    file_format: str | None = None,
    file_version: str | None = None,
    formatter_hint: FiberIO | None = None,
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
    formatter_hint
        A suspected formatter to try first. This is primarily an optimization
        for reading file archives where the formats usually are the same.

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
    with IOResourceManager(path) as man:
        path = man.source
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")
        # get extension (minus .)
        suffix = Path(path).suffix
        ext = suffix[1:] if suffix else None
        iterator = FiberIO.manager.yield_fiberio(
            file_format,
            file_version,
            extension=ext,
            formatter_hint=formatter_hint,
        )
        for formatter in iterator:
            # we need to wrap this in try except to make it robust to what
            # may happen in each formatters get_format method, many of which
            # may be third party code
            func = formatter.get_format
            required_type = func._required_type
            func_input = None
            try:
                # get resource has to be in the try block because it can also
                # raise, in which case the format doesn't belong.
                func_input = man.get_resource(required_type)
                format_version = func(func_input, _pre_cast=True)
            except Exception:  # we need to catch everythign here
                continue
            finally:
                # If file handle-like seek back to 0 so it can be reused.
                getattr(func_input, "seek", lambda x: None)(0)
            if format_version:
                return format_version
        else:
            msg = f"Could not determine file format of {man.source}"
            raise UnknownFiberFormatError(msg)


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
    formatter = FiberIO.manager.get_fiberio(file_format, file_version)
    if not isinstance(patch_or_spool, dc.BaseSpool):
        patch_or_spool = dc.spool([patch_or_spool])
    with IOResourceManager(path) as man:
        func = formatter.write
        required_type = func._required_type
        resource = man.get_resource(required_type)
        func(patch_or_spool, resource, _pre_cast=True, **kwargs)
    return path
