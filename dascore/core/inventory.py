"""
DASCore's metadata model.

Inventory is intended to be a mixed-record metadata catalog for describing
patches and related acquisition context. Patch data, coordinates, and attrs
remain the primary source of truth, while Inventory records provide richer
context which can be resolved explicitly by acquisition id and time.

  Core Requirements

  - Model “full” metadata separately from Patch and Spool, because those primarily model data, processing interfaces, and partial metadata.
  - Allow data to be managed separate from metadata.
  - Apply beyond one DAS niche: support many geophysical monitoring scenarios.
  - Avoid assuming a single external standard is complete or canonical.
  - Be flexible because the first model will likely need revision.

  Inventory Requirements

  - Inventory should contain all metadata record objects.
  - Every metadata object should have a stable resource_id.
  - Objects should be immutable.
  - Updates should affect exactly one object at a time.
  - Updating should append a new object under the same resource_id, not mutate the old one.
  - Later objects in the Inventory tuple supersede earlier objects once their effective_time applies.
  - Inventory should store history/provenance.
  - Inventory should support restoring or viewing metadata at a specific time.
  - Inventory should index by object type for efficient lookup and iteration.

  Base Metadata Object Requirements
  Each inventory record should share common fields:

  - resource_id
  - creation_time
  - author_id
  - notes
  - schema_version
  - migration support between schema versions

  Patch Relationship Requirements

  - An Acquisition record links patch-like data to an instrument configuration and an optical path.
  - Acquisition-like metadata should include fields such as instrument configuration, optical path, data units, time range, network, station, dims, tag, and history reference.
  - Patch coordinates derived from Inventory should be added explicitly through patch processing methods rather than hidden fallback lookups.

  Design Direction

  - Use Pydantic for schema creation, serialization, and validation.
  - Capture DAS-RCN concepts where useful.
  - Prefer ProdML terminology where it maps well to DFOS concepts.
  - Keep the model plain-textable.
  - Include schema migrations from the start.

  Source: DASDAE presentations das_data_models.qmd (https://github.com/DASDAE/presentations/blob/master/cwp_fall_2023/das_data_models.qmd)


## Regarding bootstrapping existing archives.

 Use a small plain-text inventory YAML file in each directory where a new acquisition context begins.

  Semantics:
  - Applies to all data files in that directory subtree unless overridden by a deeper acquisition file.
  - Captures context raw formats often lack: acquisition id, instrument/config id, optical path, station/network/tag, notes, provenance.
  - Can be partial; DASCore can infer time ranges, dims, dtype, shape, and source info from dc.scan.
  - Patch-local metadata remains primary; acquisition metadata fills gaps.
  - Resource IDs identify stable things; append-only, time-aware history handles changes to the same thing.
  - Starting a new directory plus adding a new acquisition.yaml becomes the practical field workflow when instrument/cable/acquisition context changes.

  In short: directory-local acquisition manifests bootstrap inventories for real field archives without changing raw data files.


## Channel mapping, instrument configuration, and geometry.

 ChannelMap was a mistaken core inventory object. The mapping from sample
 index/channel/locus to optical distance is an instrument configuration concern,
 not intrinsic cable metadata. It depends on interrogator settings and
 acquisition choices such as first channel, number of loci, spatial sampling
 interval, gauge length, lead-in/zero reference, cropping, decimation, and
 vendor conventions.

  Revised semantics:
  - InstrumentConfiguration defines the sampling/grid configuration used to produce patch coordinates from optical distance or locus values.
  - Acquisition links patch data to an InstrumentConfiguration and the observed optical path.
  - OpticalPath describes the ordered physical light path through optical components such as fiber segments, connectors, splices, couplers, turnarounds, or terminators.
  - Optical component records describe physical path components, not sampled channel maps.
  - CoordinateReferenceSystem records describe coordinate systems used by Geometry records.
  - OpticalPath annotations describe named or categorized intervals on the path.
  - Optical path lengths are stored in meters.

  In short: geometry should be represented on optical paths/ranges, while
  channel/locus sampling belongs to the instrument/acquisition configuration.


## Cable and fiber metadata.

 Cable and FiberSegment records include a small set of common telecom and DFOS
 metadata rather than a full standards compliance table. ITU-T G.652 describes
 single-mode optical fibre and cable using geometrical, mechanical, and
 transmission attributes such as fibre dimensions, attenuation, chromatic
 dispersion, and cable-level PMD/attenuation. PRODML represents optical paths
 as ordered components such as fiber segments, connectors, splices, and
 turnarounds, and its worked examples distinguish optical path length from
 facility/cable context.

  Practical inventory split:
  - Cable stores whole-cable identity, construction, jacket/armor, physical constraints, and datasheet/specification references.
  - FiberSegment stores per-fiber identity, subunit/buffer metadata, fiber standard/type, color, and optical calibration metadata such as group refractive index.
  - Tight/loose buffering belongs on FiberSegment because a cable can contain mixed subunit constructions.
  - Values remain SI-only except attenuation_coefficient, which uses the optical-domain convention dB/km.

  Reference sources:
  - ITU-T G.652, Characteristics of a single-mode optical fibre and cable: https://www.itu.int/dms_pubrec/itu-t/rec/g/T-REC-G.652-202408-I!!TOC-HTM-E.htm
  - PRODML Fiber Optical Path: https://docs.energistics.org/PRODML/PRODML_TOPICS/PRO-DTS-000-015-0-C-sv2000.html
  - PRODML optical fiber installation worked example: https://docs.energistics.org/PRODML/PRODML_TOPICS/PRO-DTS-000-028-0-C-sv2000.html


## Patch coordinate integration plan.

 Inventory-derived patch coordinates should be implemented as a patch processing
 method in ``dascore/proc/inventory.py`` and exposed on ``Patch`` with the other
 coordinate-related processing methods. It should not overload ``Patch.get_coord``;
 that method remains a direct accessor for coordinates already owned by the
 patch.

  Proposed API:
  - ``patch.add_inventory_coords(inventory, coords=("x", "y", "z", "label"), acquisition_id=None, on_boundary="raise")``
  - ``inventory`` must be an ``Inventory``.
  - ``coords`` accepts one string or a sequence of strings.
  - ``acquisition_id`` defaults to ``patch.attrs.acquisition_id``.
  - If no acquisition id is available, raise a clear ``ParameterError``.
  - Resolve acquisition context with ``inventory.view(acquisition_id, time=patch_time)`` when an absolute patch time is available; otherwise call ``inventory.view(acquisition_id)``.
  - Return a new patch via ``patch.update_coords`` with derived coordinates attached along the distance dimension.

  Coordinate sources:
  - ``x``, ``y``, and ``z`` are shortcuts derived from ``OpticalPath.geometries``.
  - ``label`` is a shortcut derived from ``OpticalPath.annotations``.
  - Other string coordinate requests may project interval metadata from optical path sub-objects, such as ``fiber_type``, ``coupling_type``, or ``deployment_type``.
  - Simple metadata field names are accepted only when unambiguous across optical path interval tracks.
  - Ambiguous fields should raise and suggest qualified names such as ``optical_component.fiber_type``, ``geometry.geometry_type``, or ``coupling_condition.quality``.

  Geometry semantics:
  - ``Geometry.length`` is the optical distance interval covered by the geometry record.
  - The physical length implied by ``Geometry.coordinates`` may differ from ``Geometry.length``.
  - For ``geometry_type="linear"``, interpolate physical coordinates over normalized optical distance across the geometry interval.
  - For ``geometry_type="unknown"`` or an empty geometry type, requested numeric geometry coordinates should produce NaN over that interval.
  - Do not transform coordinate reference systems in the first implementation; CRS metadata remains informational.

  Boundary behavior:
  - ``on_boundary="raise"`` raises if a distance sample falls exactly on an interior interval boundary.
  - ``on_boundary="warn"`` warns once and assigns boundary samples to the interval on the right.
  - ``on_boundary="ignore"`` silently assigns boundary samples to the interval on the right.
  - The final path endpoint belongs to the final interval and is not considered ambiguous.

  Later implementation tests should cover example inventories, explicit and attr-based acquisition lookup, linear geometry where optical length differs from physical span, interval metadata projection, ambiguous fields, boundary modes, and missing acquisition or optical path errors.



  # Clean Path-Table Inventory Import

  ## Summary

  Replace the current broad dataframe-import attempt with a simpler path-table workflow: users provide separate measurement tables for geometry, fiber, coupling, and labels, and DASCore compiles them into one
  OpticalPath plus supporting inventory records. The API should hide resource_id and path relationship fields from users.

  ## Public API

  Add one focused constructor on Inventory:

  Inventory.from_path_tables(
      *,
      geometry=None,
      fiber=None,
      coupling=None,
      labels=None,
      acquisition=None,
      instrument=None,
      instrument_configuration=None,
      cable=None,
      crs=None,
      optical_path=None,
      validate=True,
  )

  - Dataframe inputs are pandas DataFrames.
  - Metadata inputs may be inventory objects or plain mappings.
  - At least one of geometry, fiber, coupling, or labels is required.
  - This builds one optical path. Multi-path/table-workbook import is deferred.
  - Do not expose table helper classes publicly.

  ## Table Semantics

  - geometry is point/control-point based:
      - required: distance
      - coordinate axes are all non-reserved columns such as x, y, z or latitude, longitude, elevation
      - consecutive rows become linear Geometry intervals
      - Geometry.length = next_distance - distance
      - CoordinateReferenceSystem.axis_order comes from coordinate column order unless supplied in crs
  - fiber, coupling, and labels are interval based:
      - required: start_distance, end_distance
      - interval length is end_distance - start_distance
      - fiber rows become FiberSegment
      - coupling rows become CouplingCondition
      - label rows become OpticalPathAnnotation records
  - Build one synthesized OpticalPath:
      - length is the max endpoint/control-point distance
      - ordered geometry_ids, optical_component_ids, coupling_condition_ids, and annotation_ids are created from sorted table rows
  - Generate deterministic internal resource_ids from the table kind and row order, e.g. geometry:0000, fiber_segment:0000, optical_path:default.

  ## Cleanup / Implementation

  - Keep any future dataframe/table importer narrow and path-table oriented.
  - Implement small private helpers in dascore/utils/inventory.py:
      - validate/coerce dataframe inputs
      - normalize distance intervals
      - build deterministic ids
      - convert geometry control points into Geometry records
      - convert interval tables into record sequences
  - Keep dascore/core/inventory.py focused on model definitions and the thin Inventory.from_path_tables classmethod.
  - Do not add top-level dascore exports.

  ## Test Plan

  - Build an inventory from only geometry and verify add_inventory_coords(..., coords=("latitude", "longitude", "elevation")).
  - Build from geometry + fiber + coupling + labels and verify Inventory.view(acquisition_id) resolves the full path.
  - Verify geometry control-point rows produce correct segment lengths and coordinate interpolation.
  - Verify interval tables reject negative or zero-length intervals where inappropriate.
  - Verify missing required columns raise clear ParameterErrors.
  - Verify generated IDs are deterministic across repeated imports.
  - Verify validate=True checks references and path lengths; validate=False skips final reference/length validation only.

  ## Assumptions

  - This first API builds one optical path, not a whole multi-path workbook.
  - Real-world location is the primary workflow, so geometry is point-based.
  - Fiber, coupling, and labels remain interval-based because that matches field annotations.
  - Users do not provide inventory resource_ids in this workflow.

  To continue this session, run codex resume 019dddc0-0286-7bf3-bbaf-aa983349b84d

"""
# ruff: noqa: E501

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import yaml
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from dascore.constants import path_types
from dascore.exceptions import ParameterError
from dascore.utils.inventory import (
    _IntervalSequence,
    _IntervalTrackName,
    _InventoryRecord,
    _merge_annotations,
    _record_from_mapping,
    _records_from_geometry_table,
    _reverse_annotations,
    _select_annotations,
)
from dascore.utils.misc import sanitize_range_param
from dascore.utils.models import DascoreBaseModel, DateTime64
from dascore.utils.time import to_datetime64

InventoryRecord = _InventoryRecord


class Acquisition(InventoryRecord):
    """Observing context for data produced under coherent metadata."""

    instrument_configuration_id: str = Field(
        default="",
        description="Resource id of the instrument configuration used for acquisition.",
    )
    instrument_configuration: InstrumentConfiguration | None = Field(
        default=None,
        exclude=True,
    )
    optical_path_id: str = Field(
        default="", description="Resource id of the optical path observed."
    )
    optical_path: OpticalPath | None = Field(
        default=None,
        exclude=True,
    )
    data_units: str = Field(
        default="", description="Default units for data produced by this acquisition."
    )
    start_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Start time of the acquisition context.",
    )
    end_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="End time of the acquisition context.",
    )
    network: str = Field(default="", description="Network code for acquired data.")
    station: str = Field(default="", description="Station code for acquired data.")
    dims: str = Field(
        default="", description="Comma-separated default patch dimensions."
    )
    tag: str = Field(default="", description="Default tag for acquired patches.")
    history_id: str = Field(
        default="", description="Resource id of provenance/history metadata."
    )


class Instrument(InventoryRecord):
    """Physical instrument identity."""

    instrument_id: str = Field(
        default="",
        description="Instrument identifier used by acquisition systems or operators.",
    )
    serial_number: str = Field(
        default="", description="Manufacturer serial number for the instrument."
    )
    manufacturer: str = Field(default="", description="Manufacturer of the instrument.")
    model: str = Field(default="", description="Model name of the instrument.")
    instrument_type: str = Field(
        default="",
        description="General instrument category, such as DAS interrogator or point sensor.",
    )


class InstrumentConfiguration(InventoryRecord):
    """Instrument and acquisition settings."""

    instrument_id: str = Field(
        default="", description="Resource id of the configured instrument."
    )
    instrument: Instrument | None = Field(
        default=None,
        exclude=True,
    )
    gauge_length: float | None = Field(
        default=None, description="Gauge length used by the instrument in meters."
    )
    pulse_rate: float | None = Field(
        default=None, description="Pulse repetition rate for the configuration in Hz."
    )
    sample_rate: float | None = Field(
        default=None, description="Temporal sample rate for acquired data in Hz."
    )
    spatial_sampling_interval: float | None = Field(
        default=None,
        description="Spatial sampling interval between channels in meters.",
    )
    first_channel_index: int = Field(
        default=0,
        description="Index of the first channel used to anchor optical distance.",
    )
    first_channel_distance: float = Field(
        default=0.0,
        description="Optical distance in meters corresponding to first_channel_index.",
    )


class _OpticalLengthRecord(InventoryRecord):
    """Base class for optical path records with SI-only lengths."""

    model_config = ConfigDict(extra="forbid")

    length: float | None = Field(
        default=None, description="Length of this optical path interval in meters."
    )


class Cable(InventoryRecord):
    """Physical cable containing one or more fiber segments."""

    cable_id: str = Field(
        default="", description="Operator or vendor identifier for the cable."
    )
    name: str = Field(default="", description="Human-readable cable name.")
    manufacturer: str = Field(default="", description="Manufacturer of the cable.")
    model: str = Field(default="", description="Model name of the cable.")
    serial_number: str = Field(
        default="", description="Manufacturer serial number for the cable."
    )
    cable_type: str = Field(
        default="", description="Cable type or construction, if known."
    )
    specification: str = Field(
        default="",
        description=(
            "Cable or fiber specification, standard, or vendor datasheet "
            "designation, such as ITU-T G.652.D."
        ),
    )
    cable_construction: str = Field(
        default="",
        description=(
            "Whole-cable construction, such as armored, hybrid, tactical, "
            "flat_drop, loose_tube, or unknown."
        ),
    )
    jacket_material: str = Field(
        default="", description="Outer jacket material, if known."
    )
    armor_type: str = Field(
        default="", description="Cable armor type or strength member style."
    )
    outer_diameter: float | None = Field(
        default=None, description="Outer cable diameter in meters."
    )
    minimum_bend_radius: float | None = Field(
        default=None, description="Minimum allowable cable bend radius in meters."
    )
    maximum_tensile_load: float | None = Field(
        default=None, description="Maximum allowable cable tensile load in newtons."
    )
    datasheet_uri: str = Field(
        default="", description="URI or path to the cable datasheet or specification."
    )
    fiber_count: int | None = Field(
        default=None, description="Number of fibers contained in the cable."
    )


class OpticalComponent(_OpticalLengthRecord):
    """Base class for physical optical components in an optical path."""

    length: float | None = Field(
        default=None, description="Length of this optical component in meters."
    )
    name: str = Field(default="", description="Human-readable component name.")


class FiberSegment(OpticalComponent):
    """Physical fiber interval in an optical path."""

    fiber_type: str = Field(
        default="", description="Fiber type, such as single_mode or multi_mode."
    )
    fiber_standard: str = Field(
        default="",
        description=(
            "Fiber standard or grade, such as ITU-T G.652.D or ITU-T G.657.A1."
        ),
    )
    fiber_index: int | None = Field(
        default=None, description="Fiber position or index within the parent cable."
    )
    subunit_id: str = Field(
        default="",
        description=(
            "Identifier for the tube, ribbon, bundle, or other cable subunit "
            "containing this fiber."
        ),
    )
    buffer_type: str = Field(
        default="",
        description=(
            "Per-fiber or subunit buffer construction, such as tight_buffered, "
            "loose_buffered, bare, ribbon, or unknown."
        ),
    )
    buffer_material: str = Field(
        default="", description="Buffer or coating material around this fiber."
    )
    buffer_outer_diameter: float | None = Field(
        default=None, description="Outer diameter of the fiber buffer in meters."
    )
    color: str = Field(
        default="", description="Fiber or buffer color used for field identification."
    )
    group_refractive_index: float | None = Field(
        default=None,
        description=(
            "Group refractive index used to convert optical time-of-flight to "
            "distance."
        ),
    )
    attenuation_coefficient: float | None = Field(
        default=None,
        description="Optical attenuation coefficient in dB/km.",
    )
    container_id: str = Field(
        default="",
        description="Optional parent container or component identifier.",
    )
    container: Cable | None = Field(
        default=None,
        exclude=True,
    )
    fiber_id: str = Field(
        default="", description="Operator or vendor identifier for the fiber."
    )


class Connector(OpticalComponent):
    """Optical connector in an optical path."""

    connector_type: str = Field(default="", description="Connector type.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )
    return_loss: float | None = Field(
        default=None, description="Return loss in dB, if known."
    )


class Splice(OpticalComponent):
    """Optical splice in an optical path."""

    splice_type: str = Field(default="", description="Splice type.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )


class Coupler(OpticalComponent):
    """Optical coupler or splitter in an optical path."""

    coupler_type: str = Field(default="", description="Coupler or splitter type.")
    coupling_ratio: str = Field(default="", description="Coupling ratio, if known.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )


class Turnaround(OpticalComponent):
    """Optical turnaround or loopback component in an optical path."""

    turnaround_type: str = Field(default="", description="Turnaround type.")


class Terminator(OpticalComponent):
    """Optical path terminator."""

    terminator_type: str = Field(default="", description="Terminator type.")
    reflectance: float | None = Field(
        default=None, description="Reflectance in dB, if known."
    )


class CoordinateReferenceSystem(InventoryRecord):
    """Coordinate reference system used by geometry records."""

    crs_type: str = Field(
        default="",
        description="Kind of coordinate system, such as epsg, local, or unknown.",
    )
    name: str = Field(default="", description="Human-readable CRS name.")
    definition: str = Field(
        default="", description="CRS definition, such as an EPSG code or WKT string."
    )
    origin: tuple[float, ...] = Field(
        default=(), description="Origin for local coordinate systems."
    )
    axis_order: tuple[str, ...] = Field(
        default=(), description="Coordinate axis order, such as x, y, z."
    )
    units: str = Field(default="", description="Coordinate units when not implied.")
    vertical_datum: str = Field(
        default="", description="Vertical datum or reference surface, if known."
    )


class Geometry(_OpticalLengthRecord):
    """Geometry for an interval of an optical path."""

    length: float | None = Field(
        default=None, description="Length of the path interval described in meters."
    )
    geometry_type: str = Field(
        default="",
        description="Kind of geometry, such as linear, curve, or unknown.",
    )
    coordinate_reference_system_id: str = Field(
        default="", description="Resource id of the coordinate reference system."
    )
    coordinate_reference_system: CoordinateReferenceSystem | None = Field(
        default=None,
        exclude=True,
    )
    coordinates: tuple[tuple[float, ...], ...] = Field(
        default=(), description="Coordinate points describing the geometry."
    )
    reference: str = Field(
        default="", description="Optional URI, path, or name for external geometry."
    )


class ClampPoint(DascoreBaseModel):
    """Discrete clamp location within a coupling interval."""

    model_config = ConfigDict(
        title="Clamp Point",
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    distance: float = Field(
        default=0.0,
        description=(
            "Distance in meters from the start of the coupling interval to the "
            "clamp point."
        ),
    )
    clamp_type: str = Field(
        default="", description="Clamp type or hardware style at this point."
    )
    attachment: str = Field(
        default="", description="Attachment method used by this clamp point."
    )
    notes: str = Field(default="", description="Free-form notes for this clamp point.")


class CouplingCondition(_OpticalLengthRecord):
    """Acoustic coupling condition for an interval of an optical path."""

    length: float | None = Field(
        default=None, description="Length of the path interval described in meters."
    )
    coupling_type: str = Field(
        default="",
        description=(
            "Kind of acoustic coupling, such as buried, trenched, cemented, "
            "clamped, conduit, suspended, loose, surface_laid, or unknown."
        ),
    )
    medium: str = Field(
        default="", description="Surrounding medium, such as soil, rock, or steel."
    )
    attachment: str = Field(
        default="", description="Attachment method, such as clamped or bonded."
    )
    quality: str = Field(
        default="", description="Qualitative coupling quality, if known."
    )
    deployment_type: str = Field(
        default="",
        description=(
            "Deployment setting, such as borehole, surface, trench, conduit, "
            "structure, or unknown."
        ),
    )
    installation_method: str = Field(
        default="",
        description=(
            "How the cable was installed or coupled, such as cemented, clamped, "
            "buried, pinned, weighted, loose, or unknown."
        ),
    )
    contact_medium: str = Field(
        default="",
        description=(
            "Material directly contacting the cable, such as cement, grout, soil, "
            "sand, casing, tubing, conduit, air, or water."
        ),
    )
    embedment_depth: float | None = Field(
        default=None,
        description="Depth below the local surface or host interface in meters.",
    )
    confining_pressure: float | None = Field(
        default=None,
        description=(
            "Approximate normal pressure coupling the cable to the surrounding "
            "medium in pascals."
        ),
    )
    tension: float | None = Field(
        default=None, description="Approximate axial cable tension in newtons."
    )
    coupling_quality_index: float | None = Field(
        default=None,
        description="Optional normalized or method-specific coupling quality estimate.",
    )
    quality_method: str = Field(
        default="",
        description=(
            "Method used to estimate coupling quality, such as visual inspection, "
            "tap test, coherence, SNR, or unknown."
        ),
    )
    clamp_spacing: float | None = Field(
        default=None, description="Nominal spacing between clamp points in meters."
    )
    clamp_points: tuple[ClampPoint, ...] = Field(
        default=(),
        description=(
            "Discrete clamp locations measured in meters from the start of this "
            "coupling interval."
        ),
    )

    def select(self, *, distance) -> Self:
        """Return this coupling condition selected by local distance."""
        start, stop = OpticalPath._get_distance_limits(distance, self.length)
        clamp_points = tuple(
            point.model_copy(update={"distance": point.distance - start})
            for point in self.clamp_points
            if start <= point.distance < stop
        )
        return self.model_copy(
            update={
                "length": stop - start,
                "clamp_points": clamp_points,
            }
        )


class OpticalPathAnnotation(InventoryRecord):
    """Named or categorized interval on an optical path."""

    start_distance: float = Field(
        default=0.0,
        description="Start distance of the annotation interval in meters.",
    )
    end_distance: float = Field(
        default=0.0,
        description="End distance of the annotation interval in meters.",
    )
    label: str = Field(
        default="", description="Human-readable label for this path annotation."
    )
    category: str = Field(
        default="", description="Optional annotation category or namespace."
    )
    notes: str = Field(
        default="", description="Free-form notes for this optical path annotation."
    )

    @model_validator(mode="after")
    def _validate_interval(self) -> Self:
        """Ensure annotation intervals have positive length."""
        if self.start_distance >= self.end_distance:
            msg = "Annotation start_distance must be less than end_distance."
            raise ValueError(msg)
        return self


class OpticalPath(_OpticalLengthRecord):
    """Continuous optical path described by independent component sequences."""

    length: float | None = Field(
        default=None, description="Declared total optical path length in meters."
    )
    optical_component_ids: tuple[str, ...] = Field(
        default=(), description="Ordered optical component resource ids."
    )
    optical_components: tuple[OpticalComponent, ...] = Field(
        default=(),
        exclude=True,
    )
    geometry_ids: tuple[str, ...] = Field(
        default=(), description="Ordered Geometry resource ids."
    )
    geometries: tuple[Geometry, ...] = Field(
        default=(),
        exclude=True,
    )
    coupling_condition_ids: tuple[str, ...] = Field(
        default=(), description="Ordered CouplingCondition resource ids."
    )
    coupling_conditions: tuple[CouplingCondition, ...] = Field(
        default=(),
        exclude=True,
    )
    annotation_ids: tuple[str, ...] = Field(
        default=(), description="Ordered optical path annotation resource ids."
    )
    annotations: tuple[OpticalPathAnnotation, ...] = Field(
        default=(),
        exclude=True,
    )

    @staticmethod
    def _add_lengths(length_1: float | None, length_2: float | None) -> float | None:
        """Return summed lengths, or None when either length is unknown."""
        if length_1 is None or length_2 is None:
            return None
        return length_1 + length_2

    @property
    def is_empty(self) -> bool:
        """Return True if this optical path has no known interval length."""
        if self.length is not None:
            return self.length == 0
        return not any(
            (
                self.optical_component_ids,
                self.optical_components,
                self.geometry_ids,
                self.geometries,
                self.coupling_condition_ids,
                self.coupling_conditions,
                self.annotation_ids,
                self.annotations,
            )
        )

    @staticmethod
    def _get_distance_limits(distance, length: float | None) -> tuple[float, float]:
        """Validate and normalize a distance selection."""
        start, stop = sanitize_range_param(distance)
        start = 0.0 if start is None else float(start)
        if stop is None:
            if length is None:
                msg = "Open-ended distance selections require optical path length."
                raise ParameterError(msg)
            stop = float(length)
        else:
            stop = float(stop)
        if start > stop:
            msg = "Distance selection start must be less than or equal to stop."
            raise ParameterError(msg)
        if length is not None:
            start = max(start, 0.0)
            stop = min(stop, float(length))
        stop = max(stop, start)
        return start, stop

    @staticmethod
    def _is_full_distance_selection(
        start: float,
        stop: float,
        length: float | None,
    ) -> bool:
        """Return True when the distance selection covers the whole known path."""
        return length is not None and start == 0.0 and stop == float(length)

    def _get_interval_sequence(self, name: _IntervalTrackName) -> _IntervalSequence:
        """Return an interval sequence wrapper for a named path track."""
        match name:
            case "optical_component":
                return _IntervalSequence(
                    name,
                    records=self.optical_components,
                    ids=self.optical_component_ids,
                )
            case "geometry":
                return _IntervalSequence(
                    name,
                    records=self.geometries,
                    ids=self.geometry_ids,
                )
            case "coupling_condition":
                return _IntervalSequence(
                    name,
                    records=self.coupling_conditions,
                    ids=self.coupling_condition_ids,
                )
            case _:
                msg = (
                    "kind must be 'optical_component', 'geometry', "
                    f"or 'coupling_condition', not {name!r}."
                )
                raise ParameterError(msg)

    def _select_interval_sequence(
        self,
        name: _IntervalTrackName,
        start: float,
        stop: float,
        full_selection: bool,
    ) -> tuple[tuple[str, ...], tuple]:
        """Select one named interval sequence."""
        sequence = self._get_interval_sequence(name)
        return sequence.select(start, stop, full_selection=full_selection)

    @staticmethod
    def _get_interval_target(
        component,
        optical_component,
        geometry,
        coupling_condition,
    ) -> tuple[_IntervalTrackName, object]:
        """Normalize path interval target inputs to a single kind and object."""
        targets = {
            "optical_component": optical_component,
            "geometry": geometry,
            "coupling_condition": coupling_condition,
        }
        if component is not None:
            if isinstance(component, OpticalComponent):
                targets["optical_component"] = component
            elif isinstance(component, Geometry):
                targets["geometry"] = component
            elif isinstance(component, CouplingCondition):
                targets["coupling_condition"] = component
            elif isinstance(component, OpticalPathAnnotation):
                msg = "Annotations are not valid get_distance_interval targets."
                raise ParameterError(msg)
            else:
                msg = (
                    "Positional interval targets must be inventory objects. "
                    "Pass string resource ids with optical_component=, geometry=, "
                    "or coupling_condition=."
                )
                raise ParameterError(msg)
        targets = {key: value for key, value in targets.items() if value is not None}
        if len(targets) != 1:
            msg = "exactly one optical path interval target is required."
            raise ParameterError(msg)
        return next(iter(targets.items()))

    def __add__(self, other):
        """Concatenate two optical paths."""
        if not isinstance(other, OpticalPath):
            return NotImplemented
        annotations = _merge_annotations(
            self.annotations,
            other.annotations,
            self.length,
        )
        return self.__class__(
            length=self._add_lengths(self.length, other.length),
            optical_component_ids=(
                *self.optical_component_ids,
                *other.optical_component_ids,
            ),
            optical_components=(
                *self.optical_components,
                *other.optical_components,
            ),
            geometry_ids=(*self.geometry_ids, *other.geometry_ids),
            geometries=(*self.geometries, *other.geometries),
            coupling_condition_ids=(
                *self.coupling_condition_ids,
                *other.coupling_condition_ids,
            ),
            coupling_conditions=(
                *self.coupling_conditions,
                *other.coupling_conditions,
            ),
            annotation_ids=tuple(annotation.resource_id for annotation in annotations),
            annotations=annotations,
        )

    def __radd__(self, other):
        """Support summing optical path sequences."""
        if other == 0:
            return self
        return NotImplemented

    def reverse(self) -> Self:
        """Return a new optical path with all ordered subcomponents reversed."""
        annotations = _reverse_annotations(self.annotations, self.length)
        return self.__class__(
            length=self.length,
            optical_component_ids=tuple(reversed(self.optical_component_ids)),
            optical_components=tuple(reversed(self.optical_components)),
            geometry_ids=tuple(reversed(self.geometry_ids)),
            geometries=tuple(reversed(self.geometries)),
            coupling_condition_ids=tuple(reversed(self.coupling_condition_ids)),
            coupling_conditions=tuple(reversed(self.coupling_conditions)),
            annotation_ids=tuple(annotation.resource_id for annotation in annotations),
            annotations=annotations,
        )

    def select(self, **kwargs) -> Self:
        """Return a new optical path selected by distance."""
        extra = set(kwargs) - {"distance"}
        if extra:
            extra_str = ", ".join(sorted(extra))
            msg = f"Unsupported optical path selection argument(s): {extra_str}."
            raise ParameterError(msg)
        if "distance" not in kwargs:
            return self
        start, stop = self._get_distance_limits(kwargs["distance"], self.length)
        full_selection = self._is_full_distance_selection(start, stop, self.length)
        optical_component_ids, optical_components = self._select_interval_sequence(
            "optical_component",
            start,
            stop,
            full_selection,
        )
        geometry_ids, geometries = self._select_interval_sequence(
            "geometry",
            start,
            stop,
            full_selection,
        )
        coupling_condition_ids, coupling_conditions = self._select_interval_sequence(
            "coupling_condition",
            start,
            stop,
            full_selection,
        )
        annotations = _select_annotations(self.annotations, start, stop)
        return self.__class__(
            length=stop - start,
            optical_component_ids=optical_component_ids,
            optical_components=optical_components,
            geometry_ids=geometry_ids,
            geometries=geometries,
            coupling_condition_ids=coupling_condition_ids,
            coupling_conditions=coupling_conditions,
            annotation_ids=tuple(annotation.resource_id for annotation in annotations),
            annotations=annotations,
        )

    def split_at(self, distance: float) -> tuple[Self, Self]:
        """Split the optical path into two paths at a distance."""
        distance = float(distance)
        return (
            self.select(distance=(0.0, distance)),
            self.select(distance=(distance, None)),
        )

    def split(self, *, distance: float) -> tuple[Self, Self]:
        """Split the optical path by keyword distance."""
        return self.split_at(distance)

    def get_distance_interval(
        self,
        component_or_id,
        *,
        kind: str | None = None,
    ) -> tuple[float, float]:
        """Return the distance interval for a component, geometry, or coupling."""
        if kind is None:
            if isinstance(component_or_id, str):
                msg = (
                    "String resource ids require kind='optical_component', "
                    "'geometry', or 'coupling_condition'."
                )
                raise ParameterError(msg)
            kind, target = self._get_interval_target(component_or_id, None, None, None)
        else:
            target = component_or_id
        if kind == "annotation":
            msg = "Annotations are not valid get_distance_interval targets."
            raise ParameterError(msg)
        sequence = self._get_interval_sequence(kind)
        return sequence.get_interval(target)

    def validate_lengths(self, tolerance: float = 1e-9) -> Self:
        """Validate that populated interval sequences match path length."""
        expected = self.length
        sequences = tuple(
            self._get_interval_sequence(name)
            for name in (
                "optical_component",
                "geometry",
                "coupling_condition",
            )
        )
        lengths = tuple(sequence.length for sequence in sequences)
        if expected is None:
            known_lengths = tuple(length for length in lengths if length is not None)
            expected = known_lengths[0] if known_lengths else None
        errors = tuple(
            error
            for error in (
                sequence.validate_length(expected, tolerance=tolerance)
                for sequence in sequences
            )
            if error
        )
        if errors:
            msg = "Optical path length validation failed:\n" + "\n".join(errors)
            raise ParameterError(msg)
        return self

    def remove_component(
        self,
        component=None,
        *,
        optical_component=None,
        geometry=None,
        coupling_condition=None,
    ) -> Self:
        """Return a new optical path with one component interval removed."""
        target_type, target = self._get_interval_target(
            component,
            optical_component,
            geometry,
            coupling_condition,
        )
        match target_type:
            case "optical_component":
                start, stop = self.get_distance_interval(
                    target,
                    kind="optical_component",
                )
            case "geometry":
                start, stop = self.get_distance_interval(
                    target,
                    kind="geometry",
                )
            case "coupling_condition":
                start, stop = self.get_distance_interval(
                    target,
                    kind="coupling_condition",
                )
        return self.select(distance=(0.0, start)) + self.select(distance=(stop, None))


INVENTORY_RECORD_TYPES: dict[str, type[InventoryRecord]] = {
    cls.__name__: cls
    for cls in (
        Acquisition,
        Cable,
        CoordinateReferenceSystem,
        Connector,
        CouplingCondition,
        Coupler,
        FiberSegment,
        Geometry,
        Instrument,
        InstrumentConfiguration,
        OpticalPathAnnotation,
        OpticalPath,
        Splice,
        Terminator,
        Turnaround,
    )
}

OPTICAL_COMPONENT_TYPES = (
    Connector,
    Coupler,
    FiberSegment,
    Splice,
    Terminator,
    Turnaround,
)


class Inventory(DascoreBaseModel):
    """
    An append-only catalog of DASCore metadata records.

    See the [inventory tutorial](/tutorial/inventory.qmd) for examples of
    enriching patches from inventories and the
    [inventory design note](/notes/inventory_design.qmd) for model rationale.
    """

    model_config = ConfigDict(
        title="Inventory",
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    default_file_name: ClassVar[str] = "dascore-inventory.yaml"
    _relationship_plural_map: ClassVar[dict[str, str]] = {
        "annotation": "annotations",
        "optical_component": "optical_components",
        "geometry": "geometries",
        "coupling_condition": "coupling_conditions",
    }

    dascore_inventory: bool = Field(
        default=True, description="Marker identifying a DASCore inventory manifest."
    )
    schema_version: int = Field(
        default=1,
        description=(
            "Version of the inventory manifest/envelope schema; used for "
            "manifest migration."
        ),
    )
    inventory_id: str = Field(
        default="", description="Optional identifier for this inventory manifest."
    )
    creation_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Time this inventory manifest was created.",
    )
    author_id: str = Field(
        default="", description="Identifier for the manifest author."
    )
    notes: str = Field(default="", description="Free-form notes for the manifest.")
    objects: tuple[InventoryRecord, ...] = Field(
        default_factory=tuple,
        description="Inventory records contained in this manifest.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        """Normalize mappings and default the inventory marker when omitted."""
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        data["dascore_inventory"] = bool(data.get("dascore_inventory", True))
        data["objects"] = tuple(
            obj
            if isinstance(obj, InventoryRecord)
            else _record_from_mapping(obj, INVENTORY_RECORD_TYPES)
            for obj in data.get("objects", ())
        )
        return data

    @model_validator(mode="after")
    def _check_header(self) -> Self:
        """Ensure the manifest identifies itself as a DASCore inventory."""
        if not self.dascore_inventory:
            msg = "Inventory manifest must set dascore_inventory to true."
            raise ValueError(msg)
        return self

    @cached_property
    def contents(self) -> dict[str, tuple[InventoryRecord, ...]]:
        """Return records grouped by stable resource id."""
        grouped = defaultdict(list)
        for obj in self.objects:
            grouped[obj.resource_id].append(obj)
        return {key: tuple(value) for key, value in grouped.items()}

    @cached_property
    def classes(self) -> dict[str, tuple[str, ...]]:
        """Return resource ids grouped by inventory object type."""
        grouped = defaultdict(list)
        for obj in self.objects:
            grouped[obj.type].append(obj.resource_id)
        return {key: tuple(value) for key, value in grouped.items()}

    def get_records(
        self,
        resource_id: str | None = None,
        *,
        record_type: str | type[InventoryRecord] | None = None,
    ) -> tuple[InventoryRecord, ...]:
        """Return records filtered by resource id and/or record type."""
        if isinstance(record_type, str) or record_type is None:
            type_name = record_type
        else:
            type_name = record_type.__name__
        out = self.objects
        if resource_id is not None:
            out = tuple(obj for obj in out if obj.resource_id == resource_id)
        if type_name is not None:
            out = tuple(obj for obj in out if obj.type == type_name)
        return out

    def _current(
        self,
        resource_id: str,
        time=None,
        *,
        record_type: str | type[InventoryRecord] | None = None,
    ) -> InventoryRecord:
        """Return the current or time-effective stored record for one resource id."""
        records = self.get_records(resource_id, record_type=record_type)
        if not records:
            msg = f"No inventory record found for resource_id {resource_id!r}."
            if record_type is not None:
                msg = f"{msg} and record_type {record_type!r}."
            raise KeyError(msg)
        time = to_datetime64(time)
        if not pd.isnull(time):
            records = tuple(obj for obj in records if obj.is_effective_at(time))
            if not records:
                msg = (
                    f"No inventory record found for resource_id {resource_id!r} "
                    f"effective at {time!r}."
                )
                raise KeyError(msg)
        return records[-1]

    def validate_references(self, time=None) -> Self:
        """Validate that inventory id references point to expected record types."""
        time = to_datetime64(time)
        errors: list[str] = []

        def _check(
            owner: InventoryRecord,
            field: str,
            resource_id: str,
            expected_type: type[InventoryRecord] | tuple[type[InventoryRecord], ...],
        ) -> None:
            """Record an error if a referenced resource is missing or mistyped."""
            if not resource_id:
                return
            try:
                record = self._current(resource_id, time=time)
            except KeyError:
                errors.append(
                    f"{owner.type} {owner.resource_id!r} field {field!r} "
                    f"references missing resource_id {resource_id!r}."
                )
                return
            if not isinstance(record, expected_type):
                expected = (
                    expected_type.__name__
                    if isinstance(expected_type, type)
                    else ", ".join(cls.__name__ for cls in expected_type)
                )
                errors.append(
                    f"{owner.type} {owner.resource_id!r} field {field!r} "
                    f"references {record.type} {resource_id!r}; expected {expected}."
                )

        records = self.objects
        if not pd.isnull(time):
            records = tuple(obj for obj in records if obj.is_effective_at(time))
        for obj in records:
            if isinstance(obj, Acquisition):
                _check(
                    obj,
                    "instrument_configuration_id",
                    obj.instrument_configuration_id,
                    InstrumentConfiguration,
                )
                _check(obj, "optical_path_id", obj.optical_path_id, OpticalPath)
            elif isinstance(obj, InstrumentConfiguration):
                _check(obj, "instrument_id", obj.instrument_id, Instrument)
            elif isinstance(obj, FiberSegment):
                _check(obj, "container_id", obj.container_id, Cable)
            elif isinstance(obj, Geometry):
                _check(
                    obj,
                    "coordinate_reference_system_id",
                    obj.coordinate_reference_system_id,
                    CoordinateReferenceSystem,
                )
            elif isinstance(obj, OpticalPath):
                for resource_id in obj.optical_component_ids:
                    _check(
                        obj,
                        "optical_component_ids",
                        resource_id,
                        OPTICAL_COMPONENT_TYPES,
                    )
                for resource_id in obj.geometry_ids:
                    _check(obj, "geometry_ids", resource_id, Geometry)
                for resource_id in obj.coupling_condition_ids:
                    _check(
                        obj,
                        "coupling_condition_ids",
                        resource_id,
                        CouplingCondition,
                    )
                for resource_id in obj.annotation_ids:
                    _check(
                        obj,
                        "annotation_ids",
                        resource_id,
                        OpticalPathAnnotation,
                    )
        if errors:
            joined = "\n".join(errors)
            msg = f"Inventory reference validation failed:\n{joined}"
            raise ValueError(msg)
        return self

    def _get_runtime_relationship_field(self, id_field: str, fields) -> str | None:
        """Return the runtime relationship field paired with an id field."""
        if id_field.endswith("_id"):
            runtime_field = id_field[:-3]
        else:
            stem = id_field[:-4]
            runtime_field = self._relationship_plural_map.get(stem)
            if runtime_field is None:
                msg = f"No relationship plural mapping is defined for {id_field!r}."
                raise ParameterError(msg)
        if runtime_field in fields and fields[runtime_field].exclude:
            return runtime_field
        return None

    def _resolve_one(
        self,
        resource_id: str,
        time,
        recursive: bool,
    ) -> InventoryRecord | None:
        """Return a resolved record, or None if the reference cannot resolve."""
        if not resource_id:
            return None
        try:
            record = self._current(resource_id, time=time)
        except KeyError:
            return None
        if not recursive:
            return record
        return self.view(record, time=time, recursive=recursive)

    def _resolve_many(
        self,
        resource_ids: tuple[str, ...],
        time,
        recursive: bool,
    ) -> tuple[InventoryRecord, ...]:
        """Return resolved records for ids which can be resolved."""
        out = tuple(
            self._resolve_one(
                resource_id,
                time=time,
                recursive=recursive,
            )
            for resource_id in resource_ids
        )
        return tuple(x for x in out if x is not None)

    @staticmethod
    def _with_runtime_fields(record: InventoryRecord, **kwargs) -> InventoryRecord:
        """Return a record copy with resolved runtime fields populated."""
        return record.model_copy(update=kwargs)

    def view(
        self,
        record_or_id: str | InventoryRecord,
        time=None,
        *,
        recursive: bool = True,
        record_type: str | type[InventoryRecord] | None = None,
    ) -> InventoryRecord:
        """Return a time-aware record view with runtime relationships populated."""
        if isinstance(record_or_id, str):
            record = self._current(record_or_id, time=time, record_type=record_type)
        else:
            record = record_or_id
        updates = {}
        fields = record.__class__.model_fields
        for id_field in fields:
            if id_field.endswith("_id"):
                runtime_field = self._get_runtime_relationship_field(id_field, fields)
                if runtime_field:
                    updates[runtime_field] = self._resolve_one(
                        getattr(record, id_field),
                        time=time,
                        recursive=recursive,
                    )
            elif id_field.endswith("_ids"):
                runtime_field = self._get_runtime_relationship_field(id_field, fields)
                if runtime_field:
                    updates[runtime_field] = self._resolve_many(
                        getattr(record, id_field),
                        time=time,
                        recursive=recursive,
                    )
        if not updates:
            return record
        return self._with_runtime_fields(record, **updates)

    def add(self, *records: InventoryRecord | Mapping[str, Any]) -> Self:
        """Return a new inventory with records appended."""
        new_records = tuple(
            record
            if isinstance(record, InventoryRecord)
            else _record_from_mapping(record, INVENTORY_RECORD_TYPES)
            for record in records
        )
        return self.new(objects=(*self.objects, *new_records))

    @classmethod
    def from_geometry_table(
        cls,
        geometry,
        *,
        crs: CoordinateReferenceSystem | Mapping[str, Any] | None = None,
        optical_path: OpticalPath | Mapping[str, Any] | None = None,
        id_prefix: str = "path",
        validate: bool = True,
    ) -> Self:
        """
        Build a geometry-only inventory from optical-distance control points.

        Parameters
        ----------
        geometry
            Pandas dataframe with a required ``distance`` column. All other
            columns are treated as coordinate axes such as ``x``, ``y``, ``z``,
            ``latitude``, ``longitude``, or ``elevation``.
        crs
            Optional coordinate reference system record or mapping. If omitted,
            one is created with ``axis_order`` taken from the coordinate columns.
        optical_path
            Optional optical path record or mapping. Generated geometry ids and
            length are applied to this record.
        id_prefix
            Prefix for deterministic generated resource ids.
        validate
            If True, validate generated references and optical path lengths.

        See Also
        --------
        [Inventory tutorial](/tutorial/inventory.qmd)
            Examples of creating geometry inventories from survey tables.
        """
        records = _records_from_geometry_table(
            geometry,
            crs=crs,
            optical_path=optical_path,
            id_prefix=id_prefix,
            coordinate_reference_system_type=CoordinateReferenceSystem,
            geometry_type=Geometry,
            optical_path_type=OpticalPath,
        )
        out = cls(objects=records)
        if validate:
            out.validate_references()
            out.view(records[-1].resource_id).validate_lengths()
        return out

    def replace_at(
        self,
        old: str | InventoryRecord,
        new: InventoryRecord,
        time,
        *supporting_records: InventoryRecord | Mapping[str, Any],
        notes: str | None = None,
        author_id: str | None = None,
        record_type: str | type[InventoryRecord] | None = None,
        validate: bool = True,
    ) -> Self:
        """Append a new state for an existing inventory record at a time."""
        old_record = (
            self._current(old, record_type=record_type) if isinstance(old, str) else old
        )
        if new.__class__ is not old_record.__class__:
            msg = (
                f"Replacement for {old_record.type} {old_record.resource_id!r} "
                f"must also be {old_record.type}, not {new.type}."
            )
            raise ParameterError(msg)
        updates = {
            "resource_id": old_record.resource_id,
            "effective_time": to_datetime64(time),
        }
        if notes is not None:
            updates["notes"] = notes
        if author_id is not None:
            updates["author_id"] = author_id
        replacement = new.model_copy(update=updates)
        out = self.add(*supporting_records, replacement)
        if validate:
            out.validate_references()
        return out

    def dump_manifest(self) -> dict[str, Any]:
        """Return a plain manifest mapping suitable for YAML serialization."""
        out = self.model_dump(mode="json", exclude_none=True)
        out["objects"] = [
            obj.model_dump(mode="json", exclude_none=True) for obj in self.objects
        ]
        return out

    @classmethod
    def from_yaml(cls, source: path_types | Any) -> Self:
        """Load an inventory from a YAML path or readable stream."""
        if hasattr(source, "read"):
            data = yaml.safe_load(source) or {}
        else:
            with Path(source).open() as fi:
                data = yaml.safe_load(fi) or {}
        return cls.model_validate(data)

    def to_yaml(self, destination: path_types | Any | None = None) -> str | None:
        """Write inventory to YAML, or return a YAML string if no destination."""
        contents = yaml.safe_dump(
            self.dump_manifest(),
            sort_keys=False,
            allow_unicode=True,
        )
        if destination is None:
            return contents
        if hasattr(destination, "write"):
            destination.write(contents)
        else:
            Path(destination).write_text(contents)
        return None
