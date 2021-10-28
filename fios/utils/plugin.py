"""
A module for handling fios' plugins.
"""
from collections.abc import MutableMapping

import pkg_resources


class PluginManager(MutableMapping):
    """
    A simple dict like structure for storing and loading references to
    plugins.
    """

    def __len__(self):
        return max(len(self.eps), len(self.loaded_eps))

    def __iter__(self):
        for eps in list(self.eps.keys()):
            yield eps

    def __init__(self, entry_point: str):
        self.entry_point = entry_point
        self.eps = {
            ep.name: ep.load for ep in pkg_resources.iter_entry_points(entry_point)
        }
        self.loaded_eps = {}  # store for objects loaded from entry points

    def __getitem__(self, item):
        if callable(item):  # assume this is already the desired function
            return item
        elif item in self.eps or item in self.loaded_eps:
            if item not in self.loaded_eps:  # load unloaded entry points
                self.loaded_eps[item] = self.eps[item]()
            return self.loaded_eps[item]
        else:
            raise KeyError(f"{item} not in entry point {self.entry_point}")

    def __setitem__(self, key, value):
        assert callable(value), "can only store callables in manager"
        self.loaded_eps[key] = value

    def __delitem__(self, key):
        self.eps.pop(key, None)
        self.loaded_eps.pop(key, None)
