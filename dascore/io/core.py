"""
Base functionality for reading, writing, determining file formats, and scanning
Das Data.
"""
import inspect
import os.path
from collections import defaultdict
from functools import cached_property, wraps
from importlib.metadata import entry_points
from pathlib import Path
from typing import List, Optional, Union, get_type_hints

import pandas as pd
from typing_extensions import Self

import dascore as dc
from dascore.constants import PatchType, SpoolType, timeable_types
from dascore.core.schema import PatchFileSummary
from dascore.exceptions import InvalidFiberIO, UnknownFiberFormat
from dascore.utils.docs import compose_docstring
from dascore.utils.io import IOResourceManager, get_handle_from_resource
from dascore.utils.misc import cached_method, iterate, suppress_warnings
from dascore.utils.pd import _model_list_to_df


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
        Get the unlaoded entry points registered to this domain into a dict of
        {name: ep}
        """
        # TODO remove warning suppression and switch to select when 3.9 is dropped
        # see https://docs.python.org/3/library/importlib.metadata.html#entry-points
        with suppress_warnings(DeprecationWarning):
            out = {ep.name: ep.load for ep in entry_points()[self._entry_point]}
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
    def load_plugins(self, format: Optional[str] = None):
        """Load plugin for specific format or ensure all formats are loaded"""
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

    def register_fiberio(self, fiberio: "FiberIO"):
        """Register a new fiber IO to manage."""
        forma, ver = fiberio.name.upper(), fiberio.version
        self._loaded_eps.add(fiberio.name)
        for ext in iter(fiberio.preferred_extensions):
            self._extension_list[ext].append(fiberio)
        self._format_version[forma][ver] = fiberio

    @cached_method
    def get_fiberio(
        self,
        format: Optional[str] = None,
        version: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> "FiberIO":
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
        format: Optional[str] = None,
        version: Optional[str] = None,
        extension: Optional[str] = None,
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
            The format string indicating the format name
        version
            The version string of the format
        extension
            The extension of the file.
        """
        # TODO replace this with concise pattern matching once 3.9 is dropped
        if version and not format:
            msg = "Providing only a version is not sufficient to determine format"
            raise UnknownFiberFormat(msg)
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
                raise UnknownFiberFormat(msg)
            # a version is specified
            if version:
                formatter = formatters.get(version, None)
                if formatter is None:
                    msg = (
                        f"Format {format} has no version: [{version}] "
                        f"known versions of this format are: {list(formatters)}"
                    )
                    raise UnknownFiberFormat(msg)
                yield formatter
                return
            # reverse sort formatters and yield latest version first.
            for formatter in dict(sorted(formatters.items(), reverse=True)).values():
                yield formatter
            return

    def _yield_extensions(self, extension):
        """generator to get formatter prioritized by preferred extensions."""
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
    """
    A decorator for casting types for arguments of cast ind.
    """
    fun_name = func.__name__

    @wraps(func)
    def _wraper(*args, _pre_cast=False, **kwargs):
        """Wraps args but performs coercion to get proper stream"""
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
        except Exception as e:  # noqa get_format can't raise; must return false.
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
    _wraper._required_type = required_type

    return _wraper


def _is_wrapped_func(func1, func2):
    """
    Small helper function to determine if func1 is func2, unwrapping decorators.
    """
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
    _automatic_type_casters = {
        "read": 1,
        "scan": 1,
        "write": 2,
        "get_format": 1,
    }

    def read(self, resource, **kwargs) -> SpoolType:
        """
        Load data from a path.

        *kwargs should include support for selecting expected dimensions. For
        example, distance=(100, 200) would only read data with distance from
        100 to 200.
        """
        msg = f"FiberIO: {self.name} has no read method"
        raise NotImplementedError(msg)

    def scan(self, resource) -> List[PatchFileSummary]:
        """
        Returns a list of summary info for patches contained in file.
        """
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
            info = dict(pa.attrs)
            info["file_format"] = self.name
            info["path"] = str(resource)
            out.append(PatchFileSummary(**info))
        return out

    def write(self, spool: SpoolType, resource):
        """
        Write the spool to a resource (eg path, stream, etc.).
        """
        msg = f"FiberIO: {self.name} has no write method"
        raise NotImplementedError(msg)

    def get_format(self, resource) -> Union[tuple[str, str], bool]:
        """
        Return a tuple of (format_name, version_numbers).

        This should only work if path is the supported file format, otherwise
        raise UnknownFiberError or return False.
        """
        msg = f"FiberIO: {self.name} has no get_version method"
        raise NotImplementedError(msg)

    @property
    def implements_read(self) -> bool:
        """
        Returns True if the subclass implements its own scan method else False.
        """
        return not _is_wrapped_func(self.read, FiberIO.read)

    @property
    def implements_write(self) -> bool:
        """
        Returns True if the subclass implements its own scan method else False.
        """
        return not _is_wrapped_func(self.write, FiberIO.write)

    @property
    def implements_scan(self) -> bool:
        """
        Returns True if the subclass implements its own scan method else False.
        """
        return not _is_wrapped_func(self.scan, FiberIO.scan)

    @property
    def implements_get_format(self) -> bool:
        """Return True if the subclass implements its own get_format method."""
        return not _is_wrapped_func(self.get_format, FiberIO.get_format)

    def __hash__(self):
        """FiberIO instances should be uniquely defined by (format, version)"""
        return hash((self.name, self.version))

    def __init_subclass__(cls, **kwargs):
        """
        Hook for registering subclasses.
        """
        # check that the subclass is valid
        if not cls.name:
            msg = "You must specify the file format with the name field."
            raise InvalidFiberIO(msg)
        # register formatter
        manager: _FiberIOManager = getattr(cls.__mro__[1], "manager")
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
    path: Union[str, Path, IOResourceManager],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
    time: Optional[tuple[Optional[timeable_types], Optional[timeable_types]]] = None,
    distance: Optional[tuple[Optional[float], Optional[float]]] = None,
    **kwargs,
) -> SpoolType:
    """
    Read a fiber file.

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
    """
    with IOResourceManager(path) as man:
        if not file_format or not file_version:
            file_format, file_version = get_format(
                man,
                file_format=file_format,
                file_version=file_version,
            )
        formatter = FiberIO.manager.get_fiberio(file_format, file_version)
        required_type = getattr(formatter.read, "_required_type")
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


@compose_docstring(fields=list(PatchFileSummary.model_fields))
def scan_to_df(
    path: Union[Path, str, PatchType, SpoolType, IOResourceManager],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scan a path, return a dataframe of contents.

    Parameters
    ----------
    path
        The path the to file to scan
    file_format
        Format of the file. If not provided DASCore will try to determine it.

    Returns
    -------
    Return a dataframe with columns:
        {fields}
    """
    info = scan(
        path=path,
        file_format=file_format,
        file_version=file_version,
    )
    df = _model_list_to_df(info)
    return df[list(PatchFileSummary.get_index_columns())]


def _iterate_scan_inputs(patch_source):
    """Yield scan candidates."""
    for el in iterate(patch_source):
        if isinstance(el, (str, Path)) and (path := Path(el)).exists():
            if path.is_dir():  # directory, yield contents
                for sub_path in path.rglob("*"):
                    if sub_path.is_dir():
                        continue
                    yield sub_path
            else:
                yield path
        else:
            yield el


@compose_docstring(fields=list(PatchFileSummary.__annotations__))
def scan(
    path: Union[Path, str, PatchType, SpoolType, IOResourceManager],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
) -> list[PatchFileSummary]:
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
    A list of [PatchFileSummary](`dascore.core.schema.PatchFileSummary`) which
    have the following fields:
        {fields}
    """
    out = []
    for patch_source in _iterate_scan_inputs(path):
        # just pull attrs from patch
        if isinstance(patch_source, dc.Patch):
            out.append(PatchFileSummary(**dict(patch_source.attrs)))
            continue
        with IOResourceManager(patch_source) as man:
            # get fiberio
            if not file_format or not file_version:
                try:
                    file_format, file_version = get_format(
                        man,
                        file_format=file_format,
                        file_version=file_version,
                    )
                except UnknownFiberFormat:  # skip bad entities
                    continue
            formatter = FiberIO.manager.get_fiberio(file_format, file_version)
            req_type = getattr(formatter.scan, "_required_type", None)
            patch_thing = man.get_resource(req_type)
            for attr in formatter.scan(patch_thing, _pre_cast=True):
                out.append(attr)
    return out


def get_format(
    path: Union[str, Path, IOResourceManager],
    file_format: Optional[str] = None,
    file_version: Optional[str] = None,
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

    Returns
    -------
    A tuple of (file_format_name, version) both as strings.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.

    """
    with IOResourceManager(path) as man:
        path = man.source
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")
        ext = Path(path).suffix or None
        iterator = FiberIO.manager.yield_fiberio(
            file_format, file_version, extension=ext
        )
        for formatter in iterator:
            # we need to wrap this in try except to make it robust to what
            # may happen in each formatters get_format method, many of which
            # may be third party code
            func = formatter.get_format
            required_type = getattr(func, "_required_type")
            func_input = None
            try:
                # get resource has to be in the try block because it can also
                # raise, in which case the format doesn't belong.
                func_input = man.get_resource(required_type)
                format_version = func(func_input, _pre_cast=True)  # noqa
            except Exception:  # noqa we need to catch everythign here
                continue
            finally:
                # If file handle-like seek back to 0 so it can be reused.
                getattr(func_input, "seek", lambda x: None)(0)
            if format_version:
                return format_version
        else:
            msg = f"Could not determine file format of {man.source}"
            raise UnknownFiberFormat(msg)


def write(
    patch_or_spool,
    path: Union[str, Path],
    file_format: str,
    file_version: Optional[str] = None,
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
        version.

    Raises
    ------
    dascore.exceptions.UnknownFiberFormat - Could not determine the fiber format.
    """
    formatter = FiberIO.manager.get_fiberio(file_format, file_version)
    if not isinstance(patch_or_spool, dc.BaseSpool):
        patch_or_spool = dc.spool([patch_or_spool])
    with IOResourceManager(path) as man:
        func = formatter.write
        required_type = getattr(func, "_required_type")
        resource = man.get_resource(required_type)
        func(patch_or_spool, resource, _pre_cast=True, **kwargs)  # noqa
    return path
