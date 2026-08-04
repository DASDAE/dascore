"""
Registry for storage codecs.

Codecs are discovered the same way FiberIO formats are: from a Python
entry-point group (``dascore.codec``). External packages can register a new
codec by pointing an entry point at a :class:`~dascore.io.core.BaseCodec`
subclass, e.g. in their ``pyproject.toml``::

    [project.entry-points."dascore.codec"]
    daspack = "dascore_daspack:Daspack"

DASCore's built-in codecs are always available; plugins are merged on top.
"""

from __future__ import annotations

import functools

from dascore.exceptions import InvalidFiberIOError
from dascore.io.core import BaseCodec
from dascore.utils.plugins import get_entry_point_loaders

_CODEC_ENTRY_POINT_GROUP = "dascore.codec"


def _codec_name(codec_cls: type[BaseCodec]) -> str:
    """Return a codec class's registered (discriminator) name."""
    field = codec_cls.model_fields.get("name")
    name = field.default if field is not None else None
    # A required or missing name field leaves no usable discriminator (name is
    # None or the pydantic-undefined sentinel), which would otherwise register
    # the codec under a bogus shared key and silently shadow other codecs.
    if not isinstance(name, str) or not name:
        msg = (
            f"Codec {codec_cls.__name__} must declare a non-empty 'name' field "
            "with a default, e.g. name: Literal['my_codec'] = 'my_codec'."
        )
        raise InvalidFiberIOError(msg)
    return name


@functools.cache
def get_codec_registry() -> dict[str, type[BaseCodec]]:
    """
    Return all known codecs keyed by their public name.

    Built-in codecs are always included; codecs registered by plugins via the
    ``dascore.codec`` entry-point group are merged on top (and may override a
    built-in of the same name).
    """
    # Imported lazily to avoid an import cycle (hdf5 imports from io.core).
    from dascore.io.hdf5 import Gzip

    registry: dict[str, type[BaseCodec]] = {}
    for codec_cls in (Gzip,):
        registry[_codec_name(codec_cls)] = codec_cls
    for loader in get_entry_point_loaders(_CODEC_ENTRY_POINT_GROUP).values():
        codec_cls = loader()
        registry[_codec_name(codec_cls)] = codec_cls
    return registry


def get_codec(name: str) -> type[BaseCodec]:
    """
    Return a registered codec class from its public name.

    Raises
    ------
    ValueError
        If no codec is registered under ``name``.
    """
    registry = get_codec_registry()
    if name not in registry:
        valid = ", ".join(sorted(registry)) or "(none)"
        msg = f"Unknown codec {name!r}. Registered codecs: {valid}."
        raise ValueError(msg)
    return registry[name]
