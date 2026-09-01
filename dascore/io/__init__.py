"""
Modules for reading and writing fiber data.
"""
from __future__ import annotations

from dascore.io.core import (
    FiberIO,
    ScanPayload,
    make_scan_payload,
    read,
    scan,
    scan_payloads,
    scan_to_df,
    write,
)
from dascore.utils.io import BinaryReader, BinaryWriter
from dascore.utils.hdf5 import (
    H5Reader,
    H5Writer,
)
from dascore.core.annotations import (
    annotation_set_to_csv,
    annotation_set_to_dataframe,
    annotation_set_to_parquet,
    annotation_set_to_vertices,
    save_annotation_set,
)
from dascore.core.inventory import inventory_to_yaml
from dascore.utils.namespace import (
    AnnotationNameSpace,
    InventoryNameSpace,
    PatchNameSpace,
    SpoolNameSpace,
)
from dascore.utils.pd import dataframe_to_patch, patch_to_dataframe
from dascore.utils.io import (
    xarray_to_patch,
    patch_to_xarray,
    patch_to_obspy,
    obspy_to_patch,
    spool_to_xarray,
)


class PatchIO(PatchNameSpace):
    name = "io"

    write = write
    to_dataframe = patch_to_dataframe
    to_xarray = patch_to_xarray
    to_obspy = patch_to_obspy


class SpoolIO(SpoolNameSpace):
    name = "io"

    to_xarray = spool_to_xarray


class InventoryIO(InventoryNameSpace):
    name = "io"

    to_yaml = inventory_to_yaml


class AnnotationIO(AnnotationNameSpace):
    name = "io"

    to_dataframe = annotation_set_to_dataframe
    to_vertices = annotation_set_to_vertices
    to_csv = annotation_set_to_csv
    to_parquet = annotation_set_to_parquet
    save = save_annotation_set
