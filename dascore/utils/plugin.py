"""
A module for handling dascore' plugins.
"""
from collections import defaultdict
from functools import cached_property
from typing import Optional

import pkg_resources

import dascore as dc
from dascore.exceptions import UnknownFiberFormat


class FiberIOManager:
    """
    A simple dict-like structure for managing IO.

    This will store all discovered plugins for the given entry point, and
    also allow for adding FiberIO instances.
    """

    def __init__(self, entry_point: str):
        self._entry_point = entry_point
        self._loaded_eps: dict[str, "dc.io.core.FiberIO"] = {}
        self._format_version = defaultdict(dict)
        self._extention_list = defaultdict(list)

    @cached_property
    def _eps(self):
        """
        Get the unlaoded entry points registered to this domain into a dict of
        {name: ep}
        """
        out = {
            ep.name: ep.load
            for ep in pkg_resources.iter_entry_points(self._entry_point)
        }
        return out

    def __iter__(self):
        names = sorted(set(self._eps) | set(self._loaded_eps))
        for name in names:
            yield name

    def items(self):
        """return items and content."""
        for name in sorted(set(self._eps) | set(self._loaded_eps)):
            yield name, self[name]

    def __getitem__(self, item):
        if item in self._eps or item in self._loaded_eps:
            if item not in self._loaded_eps:  # load unloaded entry points
                self._eps[item]()
                assert item in self._loaded_eps
            return self._loaded_eps[item]
        else:
            known_formats = set(self._loaded_eps) | set(self._eps)
            msg = (
                f"File format {item} is unknown to DASCore. Known formats "
                f"are: [{', '.join(sorted(known_formats))}]"
            )
            raise UnknownFiberFormat(msg)

    def __setitem__(self, key, value: "dc.io.core.FiberIO"):
        """Set the loaded (instances of) formatters."""
        self._loaded_eps[key] = value
        self._format_version[key][value.file_version] = value
        self._extention_list[key].append(value)

    def get_formatter_list(
        self,
        file_format: Optional[str] = None,
        file_version: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        """
        Get a prioritized list of formatters based on query info.
        """
        if file_format is not None:

            if file_version is not None:
                pass
