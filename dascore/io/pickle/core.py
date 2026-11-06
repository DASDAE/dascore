"""Core module for reading and writing pickle format."""

from __future__ import annotations

import pickle
import pickletools

import numpy as np

import dascore
import dascore as dc
from dascore.config import get_config
from dascore.constants import snap_type, windows_type
from dascore.core.summary import normalize_source_patch_key
from dascore.exceptions import InvalidFiberFileError, PickleFormatDisabledError
from dascore.io import ArraySource, BinaryReader, BinaryWriter, FiberIO
from dascore.io.utils import (
    resolve_keyed_source,
    should_snap,
    slice_dataset,
    stored_coord,
)
from dascore.utils.io import IOResourceManager

# Enough of the stream to reach the first class a DASCore pickle names.
_HEADER_SIZE = 256
# Protocols 0 and 1 name copyreg's reconstructor before the pickled class.
_COPYREG_MODULES = frozenset({"copy_reg", "copyreg"})


def _check_opt_in():
    """Refuse to go on unless the caller has opted in to unpickling."""
    if not get_config().allow_pickle_format:
        msg = (
            "This looks like a DASCore pickle, but reading the PICKLE format "
            "is disabled by default because unpickling a file can run "
            "arbitrary code. If you trust this file, read it inside a "
            "'with dc.config_context(allow_pickle_format=True):' block, or "
            "enable it permanently with dc.set_config(allow_pickle_format=True)."
        )
        raise PickleFormatDisabledError(msg)


def _first_class(header: bytes) -> tuple[str, str] | None:
    """Name the first class a pickle header refers to, without unpickling."""
    strings = []
    # genops only parses opcodes; it never imports or calls anything. It
    # raises ValueError where the header stops being a pickle, or is cut off.
    try:
        for opcode, arg, _ in pickletools.genops(header):
            if opcode.name == "GLOBAL":
                module, _, name = str(arg).partition(" ")
            elif opcode.name == "STACK_GLOBAL" and len(strings) >= 2:
                module, name = strings[-2:]
            else:
                if isinstance(arg, str):
                    strings.append(arg)
                continue
            if module not in _COPYREG_MODULES:
                return module, name
    except ValueError:
        pass
    return None


def _header_is_dascore(header: bytes) -> bool:
    """Return True if a pickle header first names a DASCore Patch or Spool."""
    module, name = _first_class(header) or ("", "")
    return module.startswith("dascore.") and name.endswith(("Patch", "Spool"))


def _read_patches(resource):
    """Load positional entries, retaining unambiguous keys from legacy pickles."""
    # The gate comes before pickle.load: a payload runs while it is loaded.
    _check_opt_in()
    try:
        loaded = pickle.load(resource)
    except (pickle.UnpicklingError, EOFError) as exc:
        msg = f"Could not unpickle the DASCore pickle: {exc}"
        raise InvalidFiberFileError(msg) from exc
    patches = list(dascore.spool(loaded))
    # ArraySource describes the previous file, not positions in this container.
    keys = [
        normalize_source_patch_key(p.attrs.get("_source_patch_key", ""))
        for p in patches
    ]
    resolved = [key or str(index) for index, key in enumerate(keys)]
    if len(set(resolved)) != len(patches):
        keys = [""] * len(patches)
    return list(zip(patches, keys, strict=True))


class _PickleReader:
    """Keep decoded entries only for the lifetime of one I/O resource manager."""

    mode = "rb"

    def __init__(self, patches):
        self.patches = patches

    @classmethod
    def get_handle(cls, resource):
        """Decode once; preserve a caller-owned stream through the nested manager."""
        with IOResourceManager(resource) as manager:
            return cls(_read_patches(manager.get_resource(BinaryReader)))

    def close(self):
        """Release sample arrays when the scan or read finishes."""
        self.patches.clear()


class PickleIO(FiberIO):
    """
    Provides IO support for the pickle format.

    Warning
    -------
    Unpickling a file can run arbitrary code, so reading the PICKLE format
    is disabled by default. Enable it with the `allow_pickle_format` config
    setting, and only for files you trust: while it is on, any directory
    scan unpickles every file whose header names a DASCore Patch or Spool,
    whatever its extension. Detection reads only the header and never
    unpickles; reading or scanning a pickle without the setting raises
    [`PickleFormatDisabledError`](`dascore.exceptions.PickleFormatDisabledError`).
    Writing pickles needs no opt-in. Pickles also tie a file to the
    DASCore version which wrote it, so prefer DASDAE for storage.

    Examples
    --------
    >>> import dascore as dc
    >>> import tempfile
    >>> from pathlib import Path
    >>>
    >>> patch = dc.get_example_patch()
    >>> path = Path(tempfile.mkdtemp()) / "patch.pkl"
    >>> _ = dc.write(patch, path, "pickle")
    >>> # Only for a file you trust; unpickling it can run any code.
    >>> with dc.config_context(allow_pickle_format=True):
    ...     spool = dc.read(path)
    """

    name = "PICKLE"
    preferred_extensions = ("pkl", "pickle")
    multi_patch_write = True

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """
        Return the version if the header names a DASCore Patch or Spool.

        This parses only the header's opcodes and never unpickles. A
        DASCore pickle raises PickleFormatDisabledError unless
        `allow_pickle_format` is on; format detection treats that as an
        unknown format, and reports it only if no other format matches.
        """
        if not _header_is_dascore(resource.read(_HEADER_SIZE)):
            return None
        _check_opt_in()
        return self.version

    def get_metadata(
        self, resource: _PickleReader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Decode a pickle to describe its patches without retaining their arrays."""
        return [
            dc.PatchMeta(
                coords=_snap_dims(p, snap),
                attrs=p.attrs.drop("_source_patch_key"),
                dtype=p.dtype,
                source=ArraySource(key=key),
            )
            for p, key in resource.patches
        ]

    def read_array(
        self,
        resource: _PickleReader,
        windows: windows_type = (),
        key: str = "",
    ) -> np.ndarray:
        """Slice one logical patch from the entries decoded for this operation."""
        patch = resolve_keyed_source(
            [(native or str(i), p) for i, (p, native) in enumerate(resource.patches)],
            key,
        )
        return slice_dataset(patch.data, windows)

    def write(self, spool, resource: BinaryWriter, **kwargs):
        """Write a Patch/Spool to disk."""
        pickle.dump(spool, resource)


def _snap_dims(patch, snap):
    """A pickled patch's coordinates, with the dimensions `snap` names gridded."""
    grids = {
        dim: stored_coord(coord.values, True, name=dim, units=coord.units)
        for dim in patch.dims
        if should_snap(snap, dim) and not (coord := patch.get_coord(dim)).evenly_sampled
    }
    return patch.coords.update(**grids)
