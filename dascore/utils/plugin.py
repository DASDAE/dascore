"""
A module for handling dascore' plugins.
"""
from collections.abc import MutableMapping

import pkg_resources

from dascore.exceptions import UnknownFiberFormat


class FiberIOManager(MutableMapping):
    """
    A simple dict-like structure for managing IO.

    This will store all discovered plugins for the given entry point, and
    also allow for adding FiberIO instances.
    """

    def __init__(self, entry_point: str):
        self.entry_point = entry_point
        self.eps = {
            ep.name: ep.load for ep in pkg_resources.iter_entry_points(entry_point)
        }
        self.loaded_eps = {}  # store for objects loaded from entry points

    def __len__(self):
        return len(set(self.eps) | set(self.loaded_eps))

    def __iter__(self):
        names = sorted(set(self.eps) | set(self.loaded_eps))
        for name in names:
            yield name

    def __getitem__(self, item):
        if item in self.eps or item in self.loaded_eps:
            if item not in self.loaded_eps:  # load unloaded entry points
                self.loaded_eps[item] = self.eps[item]()()
            return self.loaded_eps[item]
        else:
            known_formats = set(self.loaded_eps) | set(self.eps)
            msg = (
                f"File format {item} is unknown to DASCore. Known formats "
                f"are: [{', '.join(sorted(known_formats))}]"
            )
            raise UnknownFiberFormat(msg)

    def __setitem__(self, key, value):
        self.loaded_eps[key] = value

    def __delitem__(self, key):
        self.eps.pop(key, None)
        self.loaded_eps.pop(key, None)
