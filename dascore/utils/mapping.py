"""
A few mappings that might be useful.

We can't simply use types.MappingProxyType because it can't be pickled.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping as ABCMap
from typing import TypeVar, cast

K = TypeVar("K")
V = TypeVar("V")


class FrozenDict(ABCMap[K, V]):
    """
    An immutable wrapper around dictionaries that implements the complete
    :py:class:`collections.Mapping` interface. It can be used as a drop-in
    replacement for dictionaries where immutability is desired.

    Notes
    -----
    This implementation was Inspired by the no-longer maintained package
    frozen-dict (https://github.com/slezica/python-frozendict)

    By design, changes in the original dict are not reflected in the frozen
    dict so that the hash doesn't break.
    """

    # Declared rather than inferred. The constructor is overloaded in a way
    # one untyped signature cannot express -- FrozenDict(**kwargs) can only
    # produce str keys, FrozenDict(mapping) produces K keys -- which is why
    # typeshed gives dict.__init__ five overloads. Given **kwargs, the
    # checker picks the str-keyed one, and dict is invariant in its key, so
    # the cast is the only way to say what the class actually holds.
    _dict: dict[K, V]

    def __init__(self, *args, **kwargs):
        self._dict = cast("dict[K, V]", dict(*args, **kwargs))
        self._hash: int | None = None

    def __getitem__(self, key: K) -> V:
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def new(self, **kwargs):
        """Copy the contents  and update with new values."""
        # Merged in a literal rather than with update, whose typed overloads
        # all require str keys, and passed as a mapping rather than splatted
        # so keys that are not strings survive the round trip.
        contents = {**self._dict, **kwargs}
        return self.__class__(contents)

    def __iter__(self) -> Iterator[K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._dict!r}>"

    def _hash_contents(self):
        """Returns a hash of the dictionary."""
        out = 0
        for key, value in self._dict.items():
            out ^= hash((key, value))
        return out

    def __hash__(self):
        if self._hash is None:
            self._hash = self._hash_contents()
        return self._hash
