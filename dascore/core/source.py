"""Private source metadata used by the I/O framework."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatchSource:
    """The resource and logical patch described by a patch's coordinates."""

    path: str = ""
    format: str = ""
    version: str = ""
    key: str = ""
    windows: dict[str, tuple[int, int]] | None = None
