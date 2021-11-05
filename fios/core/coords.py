"""
Module managing named coordinates.
"""


class Coords:
    """A wrapper around xarray coords for a bit more intuitive access."""

    def __init__(self, coords):
        self._coords = coords

    def __getitem__(self, item):
        """Return the raw numpy array."""
        out = self._coords[item]
        return getattr(out, "values", out)

    def __str__(self):
        return str(self._coords)

    __repr__ = __str__

    def get(self, item):
        """Return item or None if not in coord. Same as dict.get"""
        return self._coords.get(item)

    def __iter__(self):
        return self._coords.__iter__()
