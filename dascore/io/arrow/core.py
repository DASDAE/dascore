"""Core module for reading and writing the DASARROW patch format."""

from __future__ import annotations

import json
import math

import numpy as np

import dascore as dc
from dascore.constants import snap_type, windows_type
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io import BinaryReader, BinaryWriter, FiberIO
from dascore.io.utils import slice_dataset
from dascore.utils.misc import optional_import

from .utils import (
    METADATA_KEY,
    _bad,
    _check,
    _UnsupportedError,
    arrow_type,
    batch_values,
    data_batches,
    document_meta,
    patch_document,
    read_document,
)

# The first bytes of an Arrow IPC file, and of an IPC stream's first message.
_FILE_MAGIC = b"ARROW1"
_STREAM_START = b"\xff\xff\xff\xff"
# The largest schema message read; it holds the patch's metadata.
_MAX_METADATA_BYTES = 256 * 1024**2


def _check_head(resource) -> bool:
    """
    Return True for an Arrow IPC file and False for a stream; refuse others.

    Needs no pyarrow, so detection works without it.
    """
    start = resource.tell()
    head = resource.read(8)
    resource.seek(start)
    if head.startswith(_FILE_MAGIC):
        return True
    # A stream's first message states its length; refuse a huge one before
    # Arrow allocates it (any file starting with these 4 bytes gets here).
    size = int.from_bytes(head[4:8], "little", signed=True)
    ok = head.startswith(_STREAM_START) and 0 < size <= _MAX_METADATA_BYTES
    _check(ok, "not an Arrow IPC file or stream")
    return False


def _open(resource):
    """Return a resource's schema and an iterator over its record batches."""
    is_file = _check_head(resource)
    pa = optional_import("pyarrow")
    try:
        if is_file:
            reader = pa.ipc.open_file(resource)
            count = reader.num_record_batches
            return reader.schema, (reader.get_batch(i) for i in range(count))
        reader = pa.ipc.open_stream(resource)
    except pa.ArrowException as exc:
        raise _bad(str(exc)) from exc
    return reader.schema, iter(reader)


def _read_values(batches, doc) -> np.ndarray:
    """Read and check every batch, returning the patch's array."""
    pa = optional_import("pyarrow")
    dtype, shape = np.dtype(doc["dtype"]), tuple(doc["shape"])
    total = math.prod(shape)
    parts, seen = [], 0
    try:
        for batch in batches:
            parts.append(batch_values(batch, dtype))
            seen += len(parts[-1])
            _check(seen <= total, "more values than the shape holds")
    except (pa.ArrowException, OSError) as exc:
        if isinstance(exc, dc.exceptions.InvalidFiberFileError):
            raise
        raise _bad(f"can't read the data: {exc}") from exc
    _check(seen == total, f"expected {total} values, got {seen}; data is incomplete")
    flat = parts[0] if len(parts) == 1 else np.concatenate([*parts, np.empty(0, dtype)])
    return flat.reshape(shape)


class ArrowV1(FiberIO):
    """
    The DASARROW patch format: one patch per Arrow IPC file or stream.

    The patch's values are one Arrow column, ``data``, flattened in C order
    and split into record batches; the dims, shape, coordinates and attrs
    are a JSON document in the schema metadata. Other Arrow tools read the
    values as an ordinary column. Coordinates and attrs read back exactly,
    including the patch's ids. Complex data and object attrs can't be
    stored. Needs ``pyarrow``.

    Examples
    --------
    >>> import dascore as dc
    >>> import tempfile
    >>> from pathlib import Path
    >>>
    >>> patch = dc.get_example_patch()
    >>> path = Path(tempfile.mkdtemp()) / "patch.arrow"
    >>> _ = dc.write(patch, path, "DASARROW")
    >>> assert dc.spool(path)[0] == patch
    """

    name = "DASARROW"
    version = "1"
    preferred_extensions = ("arrow", "arrows")

    def get_version(self, resource: BinaryReader, **kwargs) -> str | None:
        """Return the version if the resource is an Arrow patch."""
        start = resource.tell()
        try:
            _check_head(resource)
            read_document(_open(resource)[0])
        except dc.exceptions.InvalidFiberFileError:
            return None
        except MissingOptionalDependencyError:
            # Claimed, so reading names the missing package.
            return self.version
        finally:
            resource.seek(start)
        return self.version

    def get_metadata(
        self, resource: BinaryReader, *, snap: snap_type = ()
    ) -> list[dc.PatchMeta]:
        """Describe the patch from its schema metadata, reading no data."""
        schema, _ = _open(resource)
        return [document_meta(read_document(schema), snap=snap)]

    def read_array(
        self, resource: BinaryReader, windows: windows_type = (), key: str = ""
    ) -> np.ndarray:
        """Read and check the patch's values, then take the requested window."""
        schema, batches = _open(resource)
        array = _read_values(batches, read_document(schema))
        out = slice_dataset(array, windows)
        # A copy, so a small window doesn't keep the whole patch in memory.
        return out if out.size == array.size else out.copy()

    def write(
        self,
        spool,
        resource: BinaryWriter,
        compression: str | None = None,
        stream: bool = False,
        **kwargs,
    ):
        """
        Write one patch.

        Parameters
        ----------
        spool
            A patch, or a spool holding exactly one.
        resource
            Where to write.
        compression
            ``"zstd"`` or ``"lz4"`` to compress the data; None (the default)
            doesn't, which keeps reads zero-copy and readable by Arrow tools
            without codecs.
        stream
            Write an Arrow IPC stream (for sending) rather than a file.
        """
        pa = optional_import("pyarrow")
        if not isinstance(spool, dc.Patch):
            if len(spool) != 1:
                msg = f"DASARROW holds one patch per file; got {len(spool)}."
                raise _UnsupportedError(msg)
            spool = spool[0]
        doc = patch_document(spool)
        text = json.dumps(doc, ensure_ascii=False, allow_nan=False)
        field = pa.field("data", arrow_type(np.dtype(doc["dtype"])), nullable=False)
        schema = pa.schema([field], metadata={METADATA_KEY: text.encode("utf-8")})
        options = pa.ipc.IpcWriteOptions(compression=compression)
        new = pa.ipc.new_stream if stream else pa.ipc.new_file
        with new(resource, schema, options=options) as writer:
            for batch in data_batches(spool, schema):
                writer.write_batch(batch)
