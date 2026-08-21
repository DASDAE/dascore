"""
What a patch is, apart from its values.

A [`PatchProcessor`](`dascore.workflow.processor.PatchProcessor`) is split
in two: a metadata step which decides what the result's coordinates and
attributes are, and a kernel which does the arithmetic. `PatchMeta` is
what the metadata step works on, and the reason the two halves can be
told apart -- the metadata step never sees an array, and the kernel never
sees a coordinate.

That separation is what makes an operation fusible: something which wants
to compile a chain of them can ask each one what it does to the metadata
without touching, or even holding, the data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs
    from dascore.core.coordmanager import CoordManager

# A dataclass rather than a model, and annotations which are never
# evaluated: `dascore.core.coordmanager` is still being imported when this
# module is, so naming `CoordManager` at runtime would raise. It also
# saves revalidating a coord manager on every operation.


@dataclass(frozen=True, slots=True)
class PatchMeta:
    """
    Everything a patch carries except its data.

    Parameters
    ----------
    coords
        The coordinates, which carry the dimensions and the shape too.
    attrs
        The patch's attributes.
    dtype
        What the data are. Held separately because a kernel may promote --
        `normalize` divides, so integers come back as floats -- and the
        metadata cannot always say in advance what it will be.
    backend
        Which array library the data belong to, as
        [`backend_name`](`dascore.utils.array_api.backend_name`) spells
        it. This is how a kernel is chosen without looking at an array.
    """

    coords: CoordManager
    attrs: PatchAttrs
    dtype: Any
    backend: str = "numpy"

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimension names, in order."""
        return self.coords.dims

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape the data have."""
        return self.coords.shape

    @property
    def ndim(self) -> int:
        """How many dimensions the patch has."""
        return len(self.coords.dims)

    def get_axis(self, dim: str) -> int:
        """Return the axis a dimension name refers to."""
        return self.coords.get_axis(dim)

    @classmethod
    def from_patch(cls, patch) -> PatchMeta:
        """Return what a patch is, apart from its values."""
        # Imported here rather than at module scope, for the reason the
        # header gives: this module is imported mid-way through
        # `dascore.core`.
        from dascore.utils.array_api import backend_name  # noqa: PLC0415

        data = patch.data
        return cls(
            coords=patch.coords,
            attrs=patch.attrs,
            dtype=data.dtype,
            backend=backend_name(data),
        )

    def update(self, **kwargs) -> PatchMeta:
        """Return metadata with some of it changed."""
        return replace(self, **kwargs)

    def to_patch(self, data):
        """
        Return the patch this metadata and some data make.

        The coords check the data's shape on the way in, so a kernel
        which returned the wrong shape is refused here rather than
        somewhere later and stranger.
        """
        import dascore as dc  # noqa: PLC0415

        return dc.Patch(data=data, coords=self.coords, attrs=self.attrs)
