"""IO support for MiniSEED files.

DASCore maps MiniSEED source IDs into 2D ``("channel", "time")`` patches.
MiniSEED source identity is preserved as per-channel coordinates instead of
scalar attrs because one DASCore patch can contain many MiniSEED sources.
Compatible MiniSEED sources share one patch only when they also have the same
sample count. Unequal-length sources are returned as separate patches with
stable ``channel`` coordinate values so selections still refer to the same
source across split patches.

The mapping follows the FDSN source identifier convention, where a SEED NSLC
code maps to ``FDSN:<network>_<station>_<location>_<band>_<source>_<subsource>``.
See https://docs.fdsn.org/projects/source-identifiers/en/latest/definition.html.

For DAS MiniSEED, GEOFON recommends representing each fiber sampling point as
a station because each sampling point has its own position; the channel code
``HSF`` is recommended for fiber optic DAS. See
https://geofon.gfz.de/redmine/projects/redmine/wiki/DAS.
"""

from __future__ import annotations

import dascore as dc
from dascore.constants import SpoolType, opt_timeable_types
from dascore.io import FiberIO, ScanPayload
from dascore.utils.io import LocalPath
from dascore.utils.misc import optional_import

from .utils import _detect_format, _get_patches, _scan_patches


class MSeedV2(FiberIO):
    """Support MiniSEED version 2 files."""

    name = "MSEED"
    preferred_extensions = ("mseed", "msd", "miniseed")
    version = "2"

    def get_format(self, path: LocalPath, **kwargs) -> tuple[str, str] | bool:
        """Determine if path is a MiniSEED file."""
        return _detect_format(path)

    def scan(self, path: LocalPath, **kwargs) -> list[ScanPayload]:
        """Scan a MiniSEED file."""
        pymseed = optional_import("pymseed")
        return _scan_patches(path, pymseed)

    def read(
        self,
        path: LocalPath,
        time: tuple[opt_timeable_types, opt_timeable_types] | None = None,
        channel: tuple[int | None, int | None] | None = None,
        source_patch_id=(),
        **kwargs,
    ) -> SpoolType:
        """Read a MiniSEED file."""
        pymseed = optional_import("pymseed")
        patches = _get_patches(
            path,
            pymseed,
            time=time,
            channel=channel,
            source_patch_id=source_patch_id,
        )
        return dc.spool(patches)


class MSeedV3(MSeedV2):
    """Support MiniSEED version 3 files."""

    version = "3"
