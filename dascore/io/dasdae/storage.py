"""Storage options for writing DASDAE files."""

from __future__ import annotations

from itertools import zip_longest
from typing import Any, ClassVar

import h5py
import numpy as np
from pydantic import PositiveInt

from dascore.exceptions import ParameterError
from dascore.io.core import BaseStorage


class DASDAEStorage(BaseStorage):
    """
    Storage options for writing DASDAE files.

    The fields are h5py ``create_dataset`` arguments, applied to the patch
    data and to each coordinate stored as values; empty and scalar arrays
    are written unfiltered. The ``"compressed"`` preset is gzip level 5 with
    shuffle.

    Parameters
    ----------
    chunks
        Samples per chunk for each dimension name. A dimension not named is
        one chunk along its axis; a size longer than an array is clamped.
    compression
        Any compression h5py accepts, such as "gzip", "lzf", a filter id, or
        an hdf5plugin filter.
    compression_opts
        Options for the compression, such as the gzip level; a list is
        passed as a tuple.
    shuffle
        If True, apply HDF5's shuffle filter, which often helps compression.

    Examples
    --------
    >>> from pathlib import Path
    >>> import dascore as dc
    >>> from dascore.io.dasdae import DASDAEStorage
    >>>
    >>> patch = dc.get_example_patch()
    >>> path = Path("compressed.h5")
    >>> # Use a preset,
    >>> _ = dc.write(patch, path, "DASDAE", storage="compressed")
    >>> # or say what is wanted.
    >>> storage = DASDAEStorage(compression="gzip", chunks={"time": 1000})
    >>> _ = dc.write(patch, path, "DASDAE", storage=storage)
    >>> path.unlink()
    """

    presets: ClassVar[dict[str, dict]] = {
        "compressed": {"compression": "gzip", "compression_opts": 5, "shuffle": True}
    }

    chunks: dict[str, PositiveInt] | None = None
    compression: Any = None
    compression_opts: Any = None
    shuffle: bool = False

    def _check(self, spool):
        """Refuse options h5py or the spool's dims reject, before writing."""
        # Only compression options can fail in h5py; a failure there would
        # otherwise come midway through a patch group.
        if self.compression is not None or self.compression_opts is not None:
            self._trial()
        if not self.chunks:
            return
        # The contents frame stores each patch's dims as "dim1,dim2".
        dims = {d for x in spool.get_contents()["dims"] for d in x.split(",")}
        if dims and (unknown := set(self.chunks) - dims):
            msg = (
                f"Chunk dimension(s) {sorted(unknown)} are not dimensions of "
                f"the patches being written, which are {sorted(dims)}."
            )
            raise ParameterError(msg)

    def _trial(self):
        """Create one small dataset in memory with these options."""
        with h5py.File("trial", "w", driver="core", backing_store=False) as h5:
            h5.create_dataset("x", data=np.zeros(2), **self._dataset_kwargs((), (2,)))

    def _dataset_kwargs(self, dims, shape) -> dict:
        """Return h5py create_dataset kwargs for an array of the given dims."""
        if not (shape and all(shape)):  # h5py refuses filters on these
            return {}
        chunks = None
        if self.chunks is not None:
            # A dimensionless coordinate has no dims to pair with its axis.
            pairs = zip_longest(dims, shape)
            chunks = tuple(min(self.chunks.get(d, n), n) for d, n in pairs)
        opts = self.compression_opts
        return {
            "chunks": chunks,
            "compression": self.compression,
            # JSON and YAML give lists; h5py filters need tuples.
            "compression_opts": tuple(opts) if isinstance(opts, list) else opts,
            "shuffle": self.shuffle,
        }
