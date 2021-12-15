"""
A few mappings that might be useful.

We can't simply use types.MappingProxyType because it can't be pickled.
"""
import collections.abc


class FrozenDict(collections.abc.Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete
    :py:class:`collections.Mapping` interface. It can be used as a drop-in
    replacement for dictionaries where immutability is desired.

    Notes
    -----
    This implimentation was Inspired by the no-longer maintained package
    frozen-dict (https://github.com/slezica/python-frozendict)

    By design, changes in the original dict are not reflected in the frozen
    dict so that the hash doesn't break.
    """

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        """Perform a shallow copy on the dictionaries contents."""
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._dict)

    def _hash_contents(self):
        """Returns a hash of the dictionary"""
        out = 0
        for key, value in self._dict.items():
            out ^= hash((key, value))
        return out

    def __hash__(self):
        if self._hash is None:
            self._hash = self._hash_contents()
        return self._hash
