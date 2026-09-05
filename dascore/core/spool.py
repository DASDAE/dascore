"""Module for spools, containers of patches."""

from __future__ import annotations

import numbers
import os
import warnings
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Self, TypeVar, overload

import numpy as np
import pandas as pd
from pandas.errors import (
    PerformanceWarning,
)
from rich.text import Text

import dascore as dc
from dascore.compat import UPath, is_array
from dascore.constants import (
    PROGRESS_LEVELS,
    WARN_LEVELS,
    ExecutorType,
    PatchType,
    attr_conflict_description,
    dascore_styles,
    enrich_attrs_description,
    enrich_conflict_description,
    enrich_coords_description,
    enrich_on_missing_description,
    namespace_select_type,
    numeric_types,
    path_types,
    progress_description,
    timeable_types,
)
from dascore.core._spool_inventory import (
    NO_EPOCHS,
    UNPLACEABLE,
    UNRESOLVED_WARNING,
    VALID_ON_UNRESOLVED,
    InventoryRef,
    acquisition_conflicts,
    check_stampable,
    combine_inventories,
    drops_samples,
    effective_matches,
    enriched_attr_names,
    get_attr_values,
    glob_filter,
    match_resolved,
    normalize_enrich_kwargs,
    refuse_rows,
    report_unconformed,
    resolution_columns,
    resolve_channel_pieces,
    resolve_contexts,
    resolve_row_epochs,
    resolve_split_pieces,
    stated_channels,
    unsubdividable,
)
from dascore.core.inventory import _SYSTEM_FACT_NAMES, Inventory
from dascore.core.inventory_loader import BLESSED_NAME, carries_inventory
from dascore.exceptions import (
    InvalidInventoryError,
    InvalidSpoolError,
    InvalidSpoolQueryError,
    MissingPatchError,
    ParameterError,
    UnresolvedPatchError,
)
from dascore.units import Quantity
from dascore.utils.chunk_plan import (
    _SOURCE_COLUMNS,
    ChunkPlan,
    _drop_patch_local_empty,
    _ensure_patch_id,
    _resolve_group_attrs,
    _structural,
    build_chunk_plan,
    build_concat_plan,
    build_coverage_frame,
    build_gap_frame,
    build_subdivision_plan,
    subdivision_pieces,
)
from dascore.utils.display import (
    _TIME_TYPES,
    ACQUISITION_ATTR,
    NodeRepr,
    Repr,
    elision_text,
    get_header_text,
    get_nice_text,
    group_names,
    range_texts,
    span_text,
    split_block,
)
from dascore.utils.docs import compose_docstring
from dascore.utils.downloader import resolve_example_uri
from dascore.utils.misc import (
    _spool_map,
    deep_equality_check,
    suppress_warnings,
)
from dascore.utils.namespace import NamespaceOwner
from dascore.utils.patch import (
    get_patch_names,
    stack_patches,
)
from dascore.utils.paths import coerce_to_upath, requires_local_directory
from dascore.utils.pd import (
    drop_selector_names,
    get_dim_names_from_columns,
    present_columns,
    requested_selector_names,
    resolve_selector_namespaces,
    selector_spec_names,
)

if TYPE_CHECKING:
    from dascore.io.index.catalog import PatchCatalog

T = TypeVar("T")


class _InventoryQuery(NamedTuple):
    """How one selection call splits between the index and the inventory."""

    # Selectors naming coordinates the inventory defines along the fiber
    # (a bare `...` entry names one without asking anything of it).
    channels: dict
    # The attr names the attached inventory could state.
    selectable: set
    # The attr and coordinate names the index itself knows.
    known_attrs: set
    known_coords: set
    # Every name the call named, in any form.
    requested: set
    # The `_coords` spec and kwargs with the channel names taken out.
    coords: namespace_select_type
    kwargs: dict


# The types a dimension measured in time is stated in, instants and
# offsets alike. Two such ends are a duration apart, which is a fact
# neither end carries; every other dimension states its own magnitude
# in its own units.
_TIMES = _TIME_TYPES


class Spool(NodeRepr, NamespaceOwner):
    """
    A container of patches: a view over a `PatchCatalog`.

    Constructed from in-memory patches directly (or via
    [`dascore.spool`](`dascore.spool`)), from a directory of files with
    [`Spool.from_directory`](`dascore.core.spool.Spool.from_directory`),
    or from a single file with
    [`Spool.from_file`](`dascore.core.spool.Spool.from_file`).

    Parameters
    ----------
    data
        A patch or sequence of patches this spool should hold, or
        another spool, whose catalog and provenance the new spool shares
        rather than copies; None creates an empty spool.

    Notes
    -----
    The catalog is the spool's entire state: live patches sit in its
    resolver registry, file-backed patches in its index tables, and
    restructured views (chunk/concat) are derived in-memory catalogs
    whose rows are the plan outputs. Selection, ordering, and windowing
    are lazy specs composed on the catalog; one engine serves every
    construction path.
    """

    _rich_style = "bold"
    _namespace_entry_point_group = "dascore.spool_namespace"
    _namespace_attr_errors: ClassVar[dict[str, str]] = {}
    # synthetic catalog identity columns must not join patch kwargs
    # comparisons or chunk merge-compatibility checks
    _drop_columns = (
        "patch",
        "source_path",
        "source_format",
        "source_version",
        "source_patch_key",
    )
    # The catalog backing this spool; every construction path sets one.
    _catalog: PatchCatalog
    # An attached inventory -- itself, or a reference which reads it when
    # something asks -- the enrich kwargs to apply on extraction (None
    # means attached without automatic enrichment), and what to do with a
    # patch the inventory does not describe.
    _inventory = None
    _enrich_kwargs: dict | None = None
    _on_unresolved: WARN_LEVELS = "warn"
    # Whether this spool has already said its inventory covers only part of
    # it; the warning is worth making once, not once per patch.
    _warned_unresolved: bool = False
    # single-file provenance (set by from_file; drives update())
    _file_path = None
    _file_format = None
    _file_version = None

    def __init__(
        self,
        data: PatchType | Sequence[PatchType] | Spool | None = None,
    ):
        from dascore.io.index.catalog import PatchCatalog  # noqa: PLC0415

        if isinstance(data, Spool):
            # copy-construction: share the catalog and provenance. A
            # subclass which never ran this __init__ has no catalog, and
            # copying its state would defer the failure to some later call.
            if not hasattr(data, "_catalog"):
                msg = f"{type(data).__name__} has no catalog; Spool.__init__ never ran."
                raise InvalidSpoolError(msg)
            self.__dict__.update(data.__dict__)
            return
        if data is None:
            patches = ()
        elif isinstance(data, dc.Patch):
            patches = (data,)
        elif isinstance(data, Sequence) and all(isinstance(x, dc.Patch) for x in data):
            patches = data
        else:
            msg = (
                "Spool accepts a Patch, a sequence of patches, or a "
                f"spool; got {type(data)}."
            )
            raise InvalidSpoolError(msg)
        self._catalog = PatchCatalog.from_patches(patches)

    # --- presented relation --------------------------------------------

    @property
    def _df(self) -> pd.DataFrame:
        """The realized flat relation (cached by the catalog)."""
        return self._catalog.to_df()

    def get_contents(self) -> pd.DataFrame:
        """
        Get a dataframe of the spool contents.

        Notes
        -----
        Each call returns a caller-owned dataframe; mutating it never
        changes the spool. Use ``frame.copy(deep=True)`` when an eager
        block copy is needed.

        The columns the index keeps for itself are not part of the
        frame: everything returned is public, so no column name starts
        with an underscore. A few are private only because the chunk
        planner polices public columns; those are presented under their
        public names rather than dropped (`dascore.utils.pd.PRESENTERS`
        lists them), so the frame states each patch's `dtype` and
        `data_size` -- the samples its data array holds. A row which
        does not know its size states none: a merged or subdivided
        chunk output, or a patch a selection trims.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("random_das")
        >>> df = spool.get_contents()
        >>> assert not [x for x in df.columns if str(x).startswith("_")]
        >>> assert df["data_size"].iloc[0] == spool[0].size
        """
        # Shallow: copy-on-write is always on from pandas 3, so the
        # caller's frame detaches from the cached one on its first write.
        return present_columns(self._df.copy(deep=False))

    def __len__(self) -> int:
        """Return len of spool."""
        # counting pushes to SQL (or the cold live registry); the flat
        # relation is never realized just for a length
        return len(self._catalog)

    def _as_selector_array(self, item) -> np.ndarray:
        """Convert an array-like selector to an array which fits the spool."""
        if isinstance(item, pd.Series) and pd.api.types.is_integer_dtype(item):
            # Keep the Series' own integer width: nullable Int64/UInt64 unbox
            # to it (NA raises) and unsigned values can't wrap negative.
            dtype = getattr(item.dtype, "numpy_dtype", item.dtype)
            try:
                array = item.to_numpy(dtype=dtype)
            except ValueError:  # a nullable dtype holding NA has no position
                msg = "An integer selector cannot contain missing values."
                raise ParameterError(msg) from None
        elif isinstance(item, pd.Series) and pd.api.types.is_bool_dtype(item):
            # A mask is applied by position, so it must line up with the frame
            # get_contents returns; one built from a filtered frame would
            # otherwise silently select the wrong patches.
            if not item.index.equals(self._df.index):
                msg = (
                    "The index of a boolean Series selector must match this "
                    "spool's get_contents; build the mask from it, or pass a "
                    "numpy array to select by position."
                )
                raise ParameterError(msg)
            # to_numpy handles nullable boolean dtypes; NA counts as False.
            array = item.to_numpy(dtype=bool, na_value=False)
        else:
            array = np.asarray(item)
            if array.size == 0:  # an empty list selects nothing
                array = array.astype(np.int64)
        if array.ndim != 1:
            msg = f"Spool selectors must be one dimensional, got {array.ndim}D."
            raise ParameterError(msg)
        length = len(self)
        if np.issubdtype(array.dtype, np.bool_):
            if len(array) != length:
                msg = (
                    f"Boolean selector has {len(array)} values but the spool "
                    f"has {length} patches; it must have one per patch."
                )
                raise ParameterError(msg)
        elif np.issubdtype(array.dtype, np.integer):
            # numpy bounds-checks after casting to the platform's index type,
            # which wraps a huge index into range on a 32 bit build.
            if array.size and (array.min() < -length or array.max() >= length):
                msg = f"Integer selector is out of bounds for {length} patches."
                raise IndexError(msg)
        else:
            msg = "Only bool or int dtypes are supported for spool array selection."
            raise ValueError(msg)
        return array

    # An int selects one patch; anything array-like selects a sub-spool.
    @overload
    def __getitem__(self, item: int) -> dc.Patch: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray | pd.Series | list) -> Spool: ...

    def __getitem__(self, item) -> dc.Patch | Spool:
        """
        Return a patch (int index) or a new spool (anything else).

        A slice, an array, a pandas Series, or a list selects patches: with
        booleans, one per patch, True keeps; with integers, by position.
        Boolean masks built from `get_contents` line up with the spool.

        Notes
        -----
        A boolean Series is applied by position, and `get_contents` hands
        out a fresh positional index, so a mask must come from the contents
        of the spool it selects. A mask built from a differently ordered
        view of the same patches has a matching index and selects by its
        own positions rather than raising.
        """
        if isinstance(item, slice):
            # a lazy id-membership window (D2); never realizes the flat
            # relation, and keeps split()/map() parts cheap
            return self._new_from_catalog(self._catalog.window(item))
        if is_array(item) or isinstance(item, pd.Series | list):
            array = self._as_selector_array(item)
            return self._new_from_catalog(self._catalog.restrict(array))
        try:
            return self._maybe_enrich(self._catalog.get_patch(int(item)))
        except MissingPatchError:
            # MissingPatchError subclasses IndexError for backwards
            # compatibility; it must never masquerade as out-of-bounds
            raise
        except IndexError:
            msg = f"index of [{item}] is out of bounds for spool."
            raise IndexError(msg) from None

    def __iter__(self) -> Iterator[dc.Patch]:
        """
        Iterate through the Patches in the spool.

        Notes
        -----
        Iteration may skip patches in certain cases (e.g., when coordinate
        mismatches occur as described in issue #583). Therefore, the number
        of patches yielded during iteration may differ from len(spool).
        """
        # The catalog snapshots the relation once and skips patches which
        # cannot be resolved (see #583).
        for patch in self._catalog:
            yield self._maybe_enrich(patch)

    def __add__(self, other) -> Spool:
        """
        Combine two spools into one containing the patches of both.

        The result is a lazy spool over the union of both spools'
        metadata: file-backed patches stay unloaded, in-memory patches
        are shared (not copied), and the same source appearing in both
        spools keeps a single entry. Selections on the inputs carry over
        by row membership. An attached inventory, and any enrichment set
        up from it, is spool-wide state: the union carries whichever
        operand has one and applies it to every patch it yields, as
        attaching and enriching the union would; two different ones raise.

        Examples
        --------
        >>> import dascore as dc
        >>> sp1 = dc.get_example_spool("random_das")
        >>> sp2 = dc.get_example_spool("diverse_das")
        >>> combined = sp1 + sp2
        >>> assert len(combined) == len(sp1) + len(sp2)
        """
        if not isinstance(other, Spool):
            return NotImplemented
        from dascore.io.index.catalog import PatchCatalog  # noqa: PLC0415

        members = [self._as_union_member(), other._as_union_member()]
        union = PatchCatalog.union(members)
        new = Spool()
        new._catalog = union
        # An attached inventory is part of what a spool yields, so it must
        # survive the union; two different ones have no combined answer.
        new._inventory, enrichment = combine_inventories(self, other)
        if enrichment is not None:
            new._enrich_kwargs, new._on_unresolved = enrichment
        return new

    # --- selection and presentation specs -------------------------------

    def select(
        self,
        *,
        _attrs: namespace_select_type = None,
        _coords: namespace_select_type = None,
        samples: bool = False,
        relative: bool = False,
        **kwargs,
    ) -> Self:
        """
        Sub-select parts of the spool.

        Can be used to specify dimension ranges, or glob matches on
        string attributes, read as SQLite's GLOB reads them. Bare keyword
        names resolve against attributes first, then coordinates; unknown
        names raise.

        Parameters
        ----------
        _attrs
            Attribute selections: a dict of ``name -> selector`` (the
            general form — required when a name cannot be a Python
            keyword) or a name/collection of names tagging bare kwargs
            as attributes (disambiguates names shared with coordinates).
        _coords
            Coordinate selections; same forms as ``_attrs``, validating
            names as coordinates only.
        samples
            If True, selections are coordinate-only and given in sample
            indices; they never exclude patches, but are applied to each
            patch as it loads.
        relative
            If True, coordinate range bounds are relative to each patch:
            positive from its start, negative from its end. Patches without
            the selected coordinate are excluded.
        **kwargs
            Specifies query. Can be of the form {dim_name=(start, stop)}
            or {attr_name=query}.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("diverse_das")
        >>> # subselect data in a particular time range
        >>> time = ('2020-01-03', '2020-01-03T00:00:10')
        >>> time_spool = spool.select(time=time)
        >>> # subselect based on matching tag parameter
        >>> tag_spool = spool.select(tag='some*')
        """
        if self._inventory is None:
            catalog = self._catalog.select(
                _attrs=_attrs,
                _coords=_coords,
                samples=samples,
                relative=relative,
                **kwargs,
            )
            return self._new_from_catalog(catalog)
        query = self._classify_query(_attrs, _coords, kwargs)
        channels = stated_channels(query.channels)
        # Neither keyword has anything to mean about a value the fiber
        # states: it has no sample numbering of its own -- the channels it
        # describes are the patch's -- and no endpoints to be relative to,
        # since what it says varies from one acquisition to the next.
        # Ignoring either quietly would answer a question nobody asked.
        # Judged against the selectors which actually select something: a
        # bare `...` names a fiber coordinate without asking anything of
        # it, so it must not veto a flag the rest of the query needs.
        for flag, label in ((samples, "samples"), (relative, "relative")):
            if not (channels and flag):
                continue
            msg = (
                f"{sorted(channels)} name coordinates the inventory "
                f"defines along the fiber, which {label}=True cannot describe: it "
                "asks about the patch's own axis, and these say what is "
                "attached to each channel of it."
            )
            raise InvalidSpoolQueryError(msg)
        _coords, kwargs = query.coords, query.kwargs
        attr_query: dict = {}
        # A name the attached inventory could contribute is evaluated per
        # row rather than pushed into SQL: the index states it for some
        # rows and the inventory only fills in the others. samples=True
        # selections are coordinate-only, so an attr among them is an
        # error the index states better than this can.
        if not samples and query.requested & query.selectable:
            attrs, coords = resolve_selector_namespaces(
                query.known_attrs | query.selectable,
                query.known_coords,
                _attrs=_attrs,
                _coords=_coords,
                kwargs=kwargs,
            )
            for name in list(attrs):
                if name not in query.selectable:
                    continue
                value = attrs.pop(name)
                # A bare None or ... selects everything, here as everywhere.
                if value is not None and value is not Ellipsis:
                    attr_query[name] = value
            _attrs, _coords, kwargs = attrs, coords, {}
        catalog = self._catalog.select(
            _attrs=_attrs,
            _coords=_coords,
            samples=samples,
            relative=relative,
            **kwargs,
        )
        out = self._new_from_catalog(catalog)
        if attr_query:
            out = out._select_from_inventory(attr_query)
        if channels:
            out = out._select_channels(channels)
        return out

    def unselect(
        self,
        *,
        _attrs: namespace_select_type = None,
        _coords: namespace_select_type = None,
        **kwargs,
    ) -> Self:
        """
        Return the spool without the patches a selection would keep.

        The complement of
        [`select`](`dascore.core.spool.Spool.select`): each keyword means
        exactly what it means there, and the patches it would match are
        the ones removed. This is how a spool says what it does not want —
        one bad tag, an instrument being serviced — without spelling the
        rest of the archive as a selection.

        The patches' own coordinates are not accepted. Selecting on one
        trims each patch to the range rather than choosing between
        patches, so the complement is every patch cut into the pieces
        outside it — one patch becoming two. Select the ranges to keep
        instead, or use [`Patch.unselect`](`dascore.Patch.unselect`) on
        each patch, which can take samples out of its middle.

        The coordinates an attached inventory defines along the fiber are
        different, and are accepted: removing one of those chooses which
        channels a patch holds. A patch may then be cut into the pieces
        the query did not match, so `len` can grow here as well.

        Naming nothing raises, and so does naming only no-op selectors
        (`None`, `...`). `select()` with no selection is the whole spool,
        so its complement is an empty one — but "remove nothing" reads
        just as naturally, and silently emptying a spool is not something
        to guess at.

        Parameters
        ----------
        _attrs
            Attribute selections, in the forms
            [`select`](`dascore.core.spool.Spool.select`) accepts.
        _coords
            The patches' own coordinates only to say they are refused; a
            coordinate an attached inventory defines along the fiber is
            accepted and chooses channels. See above.
        **kwargs
            The selection whose matches are removed.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("diverse_das")
        >>> # everything except the patches tagged 'some_tag'
        >>> rest = spool.unselect(tag='some_tag')
        >>> assert len(rest) + len(spool.select(tag='some_tag')) == len(spool)
        """
        query = self._classify_query(_attrs, _coords, kwargs)
        attrs, coords = resolve_selector_namespaces(
            query.known_attrs | query.selectable,
            query.known_coords,
            _attrs=_attrs,
            _coords=query.coords,
            kwargs=query.kwargs,
        )
        if coords:
            msg = (
                f"{sorted(coords)} name coordinates of the patches "
                "themselves, which unselect cannot take: removing a range "
                "means cutting every patch into the pieces outside it "
                "rather than choosing between patches. Select the ranges "
                "to keep instead, or use Patch.unselect on each patch. The "
                "coordinates an inventory defines along the fiber are "
                "different -- removing one of those chooses channels."
            )
            raise InvalidSpoolQueryError(msg)
        # A no-op selector selects everything, so its complement is an
        # empty spool -- and "remove nothing" reads just as naturally.
        stated = {k: v for k, v in attrs.items() if v is not None and v is not Ellipsis}
        if not stated and not stated_channels(query.channels):
            msg = (
                "unselect needs something to remove; "
                f"{sorted(query.requested) or 'nothing'} names no selection. "
                "Naming nothing could mean the whole spool (the complement "
                "of selecting all of it) or none of it, so it says neither."
            )
            raise ParameterError(msg)
        # The complement is taken against select itself rather than by
        # negating each predicate, so the two can never drift apart.
        if not query.channels:
            removed = self.select(_attrs=stated)._catalog.ordered_ids()
            return self._restrict_to_rows(removed, keep=False)
        # With both, the complement is still one set: a patch keeps every
        # channel unless the attrs matched it, and the channels the fiber
        # query did not match when they did. Complementing the two halves
        # apart would drop a patch the whole selection never held.
        matched = self if not stated else self.select(_attrs=stated)
        return self._select_channels(
            stated_channels(query.channels),
            complement=True,
            applies_to=matched._catalog.ordered_ids(),
        )

    def _classify_query(self, _attrs, _coords, kwargs) -> _InventoryQuery:
        """
        Split what a selection call names between the index and the inventory.

        Whether an inventory has anything to say here is settled without
        reading one: the observing-system facts are the models' own, the
        same for every inventory, and a name the index already carries
        keeps the index's meaning even where an inventory could also
        place it on the fiber. So a query about what the index already
        knows leaves a lazily attached inventory unread, and one naming
        anything else is asking a question only the inventory can answer.
        """
        requested = requested_selector_names(_attrs, _coords, kwargs)
        backend = self._catalog.backend
        known_attrs = set(backend.attr_names())
        known_coords = set(backend.coord_names())
        channels: dict = {}
        selectable: set = set()
        if self._inventory is not None:
            # A name the index already uses for a coordinate keeps its
            # meaning; bare names resolve to attrs first, and an inventory
            # must not quietly move one out of the namespace it has always
            # been in.
            selectable = set(_SYSTEM_FACT_NAMES) - known_coords
            outside = (requested - known_attrs - known_coords - selectable) | (
                selector_spec_names(_coords) - known_coords
            )
            if outside:
                # `selectable` rather than the inventory's own attr names,
                # which are the same set: one spelling of what an inventory
                # could state keeps this from deciding to read on one rule
                # and then reading under another.
                channels = self._channel_selectors(
                    requested,
                    known_attrs | known_coords | selectable,
                    known_coords,
                    _coords,
                    kwargs,
                )
        return _InventoryQuery(
            channels,
            selectable,
            known_attrs,
            known_coords,
            requested,
            drop_selector_names(_coords, channels),
            {k: v for k, v in kwargs.items() if k not in channels},
        )

    def _channel_selectors(
        self, requested, known, known_coords, _coords, kwargs
    ) -> dict:
        """
        Return the selectors naming coordinates the inventory runs along
        the fiber.

        A name which is also an attr resolves to the attr, as bare names
        always do, so only one the caller put in `_coords` is read as the
        coordinate — a label group may share an acquisition field's
        name, and selecting on the field must keep working. A name the
        index already uses for a coordinate keeps that meaning outright:
        `distance` is the patch's own axis whether or not an inventory
        could also place it on the fiber, and an inventory must not
        quietly move a name out of the namespace it has always been in.
        """
        coord_names = self._resolved_inventory().get_names().coords
        coords = set(coord_names) - known_coords
        candidates = (requested - known) | (selector_spec_names(_coords) & coords)
        wanted = candidates & coords
        if not wanted:
            return {}
        # The tag form of `_coords` designates bare kwargs, so the selector
        # is among them; only the mapping form carries one itself.
        mapped = _coords if isinstance(_coords, Mapping) else {}
        stated = {**mapped, **kwargs}
        if named := sorted(wanted - set(stated)):
            msg = (
                f"{named} name coordinates the attached inventory defines "
                "along the fiber. They are selected as ordinary keywords or "
                "through _coords, not through _attrs, since they describe "
                "channels rather than whole patches."
            )
            raise InvalidSpoolQueryError(msg)
        # Sorted so a query built from a set does not order itself by hash:
        # the masks are combined with AND either way, but which selector a
        # message complains about first should not move between runs.
        return {name: stated[name] for name in sorted(wanted)}

    def _select_channels(self, query: dict, *, complement=False, applies_to=None):
        """
        Trim each patch to the channels the inventory says match.

        Where the matching region is disjoint along the fiber — a track
        which passes in and out of the selection, or an uncovered zone in
        the middle of a path — the patch is subdivided so each contiguous
        run becomes its own patch. Selection therefore changes both the
        shape and the number of patches, which is why it builds a plan
        rather than filtering rows.

        Parameters
        ----------
        query
            The channel-level selectors, by inventory name.
        complement
            Keep the channels the query does *not* match. The mask is one
            dimensional, so unlike a patch's rectangle its complement is
            exactly expressible.
        applies_to
            The rows the query judges; any other row keeps every channel.
            `unselect` uses it to leave a patch its attrs never matched
            whole, which is what makes the two halves one complement.
        """
        source_rows, working = self._plan_frames()
        if not len(working):
            return self
        contexts = self._plan_contexts(working)
        if applies_to is not None:
            # A row the attrs did not match is a row the selection never
            # held, so it is left unjudged rather than judged and kept.
            judged = np.isin(working["_patch_id"].to_numpy(), np.asarray(applies_to))
            contexts[~judged] = None
        name, pieces, reasons = resolve_channel_pieces(
            self._resolved_inventory(),
            contexts,
            working,
            query,
            complement=complement,
        )
        refuse_rows(source_rows, reasons, UNPLACEABLE)
        if name is None:
            # No row has a fiber to be judged along, so the query matched
            # nothing: an empty spool, or the whole of it complemented.
            return self if complement else self._restrict_to_rows([])
        bounds = list(zip(working[f"{name}_min"], working[f"{name}_max"], strict=True))
        whole = [
            len(row) == 1 and tuple(row[0]) == pair
            for row, pair in zip(pieces, bounds, strict=True)
        ]
        if all(whole):  # every patch kept entire: nothing to plan
            return self
        if all(keep or not row for keep, row in zip(whole, pieces, strict=True)):
            # Every patch is kept whole or dropped, so this is a filter and
            # the relation it presents need not be rebuilt.
            kept = working["_patch_id"].to_numpy()[[bool(x) for x in pieces]]
            return self._restrict_to_rows(kept)
        return self._subdivided(source_rows, working, pieces, name)

    def _select_from_inventory(self, query: dict) -> Self:
        """
        Keep the rows whose inventory-backed values match.

        Precedence is per row, and is the precedence extraction applies.
        A row which states the name is judged by the index, exactly as it
        would be without an inventory, and only the rows leaving it
        unstated are resolved. A spool whose headers state everything
        therefore touches the inventory only when enrichment will rewrite
        the name, and one which states nothing resolves once per epoch
        rather than per row. A row the inventory has no answer for is not
        selected, as a patch lacking the attr entirely is not. Straddling
        is decided against the row as it now stands, so a range which has
        already trimmed a row inside one epoch leaves it resolvable.

        Pending enrichment changes what a stated row comes out holding,
        so stated rows are resolved too where it would write the name:
        `conflict="keep_last"` makes the inventory's answer the row's
        value, `conflict="drop"` leaves a disagreeing row with none, and
        `on_missing="null"` on named attrs blanks a resolved row the
        inventory cannot answer. A row extraction refuses rather than
        rewrites -- a disagreeing header under `conflict="raise"`, a dated
        row under a pending `time`, a stated key disagreeing with a
        pending `acquisition_key` -- is judged as it stands; the refusal
        is extraction's to make.
        """
        ids = np.asarray(self._catalog.ordered_ids(), dtype=np.int64)
        if not len(ids):
            return self
        backend = self._catalog.backend
        known = set(backend.attr_names())
        contexts = None
        mask = np.ones(len(ids), dtype=bool)
        # How pending enrichment rewrites a stated header, if at all; the
        # `raise` policy refuses rather than rewrites, `keep_first` leaves
        # the header standing, and `on_missing` applies to named attrs only.
        pending = self._enrich_kwargs or {}
        conflict = pending.get("conflict")
        rewritten = frozenset()
        if conflict is not None and conflict not in ("raise", "keep_first"):
            rewritten = enriched_attr_names(pending["attrs"])
        nulls_missing = (
            pending.get("on_missing") == "null" and pending.get("attrs") is not True
        )
        for name, selector in query.items():
            # Which rows state the name is asked of the index rather than
            # read off the relation, so a spool whose headers state it
            # everywhere is realized only when enrichment will rewrite it.
            stated = np.isin(ids, list(backend.attr_stated_ids(name, patch_ids=ids)))
            # A name no patch states is asked about rather than tried: the
            # index rejects it and the inventory answers for every row, and
            # catching that rejection would catch a malformed selector with
            # it. SQL never matches a row which states nothing, so this is
            # the verdict for the stated rows and False everywhere else.
            index_ids = (
                np.asarray(
                    self._catalog.select(_attrs={name: selector}).ordered_ids(),
                    dtype=np.int64,
                )
                if name in known
                else np.empty(0, dtype=np.int64)
            )
            matched = np.isin(ids, index_ids)
            # A stated row is resolved too when enrichment will rewrite it.
            rewriting = name in rewritten
            if not stated.all() or rewriting:
                if contexts is None:
                    contexts = self._row_contexts(ids)
                answers = get_attr_values(self._resolved_inventory(), contexts, name)
                resolved = None
                if rewriting and nulls_missing:
                    resolved = np.array([x is not None for x in contexts], dtype=bool)
                matched = effective_matches(
                    stated,
                    matched,
                    answers,
                    match_resolved(answers, name, selector, backend.attr_units(name)),
                    self._row_headers(ids, name) if rewriting else None,
                    conflict if rewriting else None,
                    resolved,
                )
            mask &= matched
        return self._new_from_catalog(self._catalog.restrict(mask, ids=ids))

    def _row_contexts(self, ids) -> np.ndarray:
        """
        Resolve each presented row to its inventory context, or to None.

        This is where the relation is realized, which is why it is only
        reached for a name some row leaves unstated or pending enrichment
        will rewrite. Rows resolve as extraction will, so pending
        enrichment's own `acquisition_key` and `time` each stand in where
        a row has none of its own.
        """
        df = self._df
        columns = resolution_columns(df, self._enrich_kwargs)
        if columns is None:
            return np.full(len(ids), None, dtype=object)
        return self._aligned(
            ids, df, resolve_contexts(self._resolved_inventory(), *columns)
        )

    def _row_headers(self, ids, name: str) -> np.ndarray:
        """Each presented row's own value of a name, or None where unstated."""
        df = self._df
        values = (
            df[name].to_numpy(dtype=object)
            if name in df.columns
            else np.full(len(df), None, dtype=object)
        )
        return self._aligned(ids, df, values)

    @staticmethod
    def _aligned(ids, df, values) -> np.ndarray:
        """
        Order per-relation-row values by the presented ids.

        Aligned by id rather than by position: the relation is realized
        by a route of its own and need not present every row the id list
        does, and a row it leaves out is one nothing was resolved for.
        """
        by_id = dict(zip(df["_patch_id"].to_numpy(), values, strict=True))
        out = np.full(len(ids), None, dtype=object)
        for position, patch_id in enumerate(ids):
            out[position] = by_id.get(patch_id)
        return out

    def attach_inventory(self, inventory=None) -> Self:
        """
        Attach a DASDAE inventory to this spool.

        The spool carries the reference and nothing else: attaching costs
        no work per patch and adds nothing to the patches it yields. Call
        [`Spool.enrich`](`dascore.core.spool.Spool.enrich`) to copy the
        inventory's metadata onto the patches as they are extracted.

        Attaching replaces whatever the spool carried before, and clears
        enrichment set up from it — swapping the inventory silently under
        a configured enrichment would change every patch's metadata, so
        the new one has to be asked for. `enrich()` resumes with defaults.

        Parameters
        ----------
        inventory
            The inventory to carry: an `Inventory`, or the path of one
            (an authoring directory or a serialized file), which is read
            at the first question rather than now. None means the one
            the spool's own directory carries, read again.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).attach_inventory(inventory)
        >>> assert "gauge_length" not in dict(spool[0].attrs)
        >>> assert spool.enrich()[0].attrs.gauge_length == 10.0

        Notes
        -----
        Attaching promises nothing about the spool matching the
        inventory;
        [`conform_to_inventory`](`dascore.core.spool.Spool.conform_to_inventory`)
        is what makes it so. Once attached, the coordinates the
        inventory defines along the fiber become selectable, and
        [`expand_by`](`dascore.core.spool.Spool.expand_by`) can expand the
        spool by the values of one.

        A spool opened on a directory which carries an inventory under
        the name `.inventory` starts out attached to it, so this is
        needed there only to attach a different one — or, with no
        argument, to read that one again after editing it. An inventory
        is read once and held, since it is an input rather than a cache;
        re-reading it is a thing the program says, not something which
        happens behind it.
        """
        if inventory is None:
            inventory = self._blessed_inventory(demanded=True)
        elif isinstance(inventory, str | os.PathLike):
            path = Path(inventory)
            # Eager, though the read is not: a path which is not there is
            # the caller's own mistake, and saying so later would blame
            # whichever call first happened to ask a question.
            if not path.exists():
                msg = f"No inventory at {path}."
                raise InvalidInventoryError(msg)
            inventory = InventoryRef(path)
        elif not isinstance(inventory, Inventory):
            msg = (
                "attach_inventory needs an Inventory or the path of one, "
                f"got {type(inventory)}."
            )
            raise ParameterError(msg)
        new = self.__class__(self)
        new._inventory = inventory
        new._enrich_kwargs = None
        return new

    def _blessed_inventory(self, demanded: bool = False):
        """
        A reference to the inventory this spool's directory carries.

        Whether one is there is settled now, by a stat; which form it
        takes and whether it can be read wait until something asks. When
        `demanded`, having none is an error rather than an answer, since
        the caller asked for that one in particular.
        """
        path = self.spool_path
        on_directory = path is not None and path.is_dir()
        if on_directory and carries_inventory(path):
            return InventoryRef(path, blessed=True)
        if not demanded:
            return None
        where = (
            f"{path} holds nothing named {BLESSED_NAME}"
            if on_directory
            else "this spool was not opened on a directory which could hold one"
        )
        msg = (
            f"This spool carries no inventory of its own: {where}. Pass the "
            "inventory to attach, or the path of one."
        )
        raise InvalidInventoryError(msg)

    def remove_inventory(self) -> Self:
        """
        Return a spool carrying no inventory.

        Any enrichment set up by [`Spool.enrich`](`dascore.core.spool.Spool.enrich`)
        goes with it, since there is nothing left to enrich from. A spool
        with no inventory is returned unchanged in substance; as everywhere
        else, the original spool is left alone.

        Removal sticks, including on a spool which found its inventory in
        its own directory: the slot is filled when the spool is opened
        and nothing fills it again.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).attach_inventory(inventory).enrich()
        >>> plain = spool.remove_inventory()
        >>> assert "gauge_length" not in dict(plain[0].attrs)
        >>> assert spool[0].attrs.gauge_length == 10.0  # the original stands
        """
        new = self.__class__(self)
        new._inventory = None
        new._enrich_kwargs = None
        return new

    @compose_docstring(
        attrs_desc=enrich_attrs_description,
        coords_desc=enrich_coords_description,
        on_missing_desc=enrich_on_missing_description,
        conflict_desc=enrich_conflict_description,
    )
    def enrich(
        self,
        *,
        on_unresolved: WARN_LEVELS = "warn",
        **kwargs,
    ) -> Self:
        """
        Enrich each patch this spool yields from an inventory.

        The work happens as each patch is extracted, not now, so this is
        cheap on a large spool and costs one
        [`Patch.enrich`](`dascore.proc.inventory.enrich`) per patch which
        actually comes out. Enrichment survives `select`, `sort`, `chunk`
        and friends; `Spool.remove_inventory` undoes it.

        Enriching never removes a patch: one the inventory does not
        describe comes out unchanged rather than missing, so an inventory
        covering part of an archive needs no pruning first. Deciding
        membership is
        [`conform_to_inventory`](`dascore.core.spool.Spool.conform_to_inventory`)'s
        job, and leaving it there is what keeps this lazy — nothing
        resolves until a patch is pulled.

        The inventory is the one
        [`attach_inventory`](`dascore.core.spool.Spool.attach_inventory`)
        put on the spool, which is the only way a spool gets one.

        Parameters
        ----------
        on_unresolved
            What to do with a patch the inventory does not describe — one
            naming no entry, or naming one the inventory does not resolve
            to exactly one of. "warn" (the default) leaves it un-enriched
            and says so, "ignore" leaves it silently, and "raise" fails.
            A patch which *straddles* two epochs is described twice rather
            than not at all, and raises regardless: it needs subdividing.
        **kwargs
            Held and passed to
            [`Patch.enrich`](`dascore.proc.inventory.enrich`) for each
            extracted patch. The names accepted are read from its
            signature, so the two cannot disagree. The names, the policies
            and the shape of an `acquisition_key` are checked now — the
            rest each patch's own enrichment checks as it is extracted.
            Calling `enrich` again replaces these rather than adding to
            them. They are:

        Other Parameters
        ----------------
        {attrs_desc}
        {coords_desc}
        acquisition_key
            The inventory identity to resolve, for patches which do not
            carry one. Given both, each patch and this argument must agree.
        time
            The instant to resolve at, for patches whose time axis is not
            physical. A patch with a real time coordinate resolves at its
            own time and passing this raises.
        {on_missing_desc}
        {conflict_desc}

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).attach_inventory(inventory).enrich()
        >>> assert spool[0].attrs.gauge_length == 10.0
        >>>
        >>> # Or name what is wanted, as with Patch.enrich.
        >>> attached = dc.spool(patch).attach_inventory(inventory)
        >>> spool = attached.enrich(coords=False)
        """
        # Settled now rather than on extraction: a misspelled argument
        # should be an error here, not on some patch pulled much later.
        enrich_kwargs = normalize_enrich_kwargs(kwargs)
        self._check_inventory_policy(on_unresolved, "enrich")
        new = self.__class__(self)
        new._enrich_kwargs = enrich_kwargs
        new._on_unresolved = on_unresolved
        # What this enrichment leaves unresolved is worth saying once more.
        new._warned_unresolved = False
        return new

    def expand_by(
        self,
        name: str,
        *,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
        stamp: bool = True,
    ) -> Self:
        """
        Expand the spool into one patch per value of an inventory coordinate.

        Most often a label group. Every kind of group expands: a
        categorical one by each of its strings, a membership group into
        the channels it includes and those it does not, and a numeric one
        by each distinct measurement. Intervals of one group may overlap,
        but a channel still holds only one of its values, so the outputs
        of one call divide the fiber rather than share it. A patch whose
        channels take several values becomes several patches — this can
        greatly expand the spool.

        Parameters
        ----------
        name
            The inventory-derived coordinate to expand by.
        include, exclude
            Glob patterns matched against each value *written as a
            string*, which is what lets one spelling cover all three
            kinds of group: `"hole_*"` reads a categorical one, `"Tru*"`
            a membership one, and `"1.*"` a numeric one. Selecting on the
            stamp afterwards compares typed values instead, so the two
            are not interchangeable. With `include`, only values matching
            one of them are kept; `exclude` drops the values it matches,
            and wins where both match.
        stamp
            Whether to record the value on each output patch as an attr
            named after the coordinate, so overlapping siblings stay
            distinguishable and later operations can select on it. Pass
            False for a nested expansion, where the second should not
            overwrite the first.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).attach_inventory(inventory)
        >>>
        >>> # The example path annotates two zones along the fiber.
        >>> zones = spool.expand_by("zone")
        >>> assert len(zones) == 2
        >>> assert set(zones.get_contents()["zone"]) == {"north", "south"}
        >>>
        >>> # Which can be narrowed by a glob over the values.
        >>> assert len(spool.expand_by("zone", include="nor*")) == 1
        """
        if self._inventory is None:
            msg = (
                "Spool.expand_by needs an inventory to expand by: the values "
                "it expands into are the ones an inventory states along the "
                "fiber. Attach one with Spool.attach_inventory."
            )
            raise ParameterError(msg)
        # A name the inventory could not contribute has no values to
        # expand into, so it would quietly give an empty spool. Selection refuses
        # a name it does not know, and a misspelling is no more meaningful
        # here than it is there.
        if name not in set(self._resolved_inventory().get_names().coords):
            msg = (
                f"{name!r} is not a coordinate the attached inventory defines "
                "along the fiber, so there is nothing to expand by. "
                "Inventory.get_names().coords lists the names it could."
            )
            raise InvalidSpoolQueryError(msg)
        source_rows, working = self._plan_frames()
        if stamp:
            check_stampable(name, working)
        contexts = self._plan_contexts(working)
        dim, rows, reasons = resolve_split_pieces(
            self._resolved_inventory(),
            contexts,
            working,
            name,
            glob_filter(include, exclude),
        )
        refuse_rows(source_rows, reasons, UNPLACEABLE)
        if dim is None:  # nothing to split: no row has a fiber to split on
            return self._restrict_to_rows([])
        pieces = [[piece for _, piece in row] for row in rows]
        marks = None
        if stamp:
            marks = (name, [value for row in rows for value, _ in row])
        return self._subdivided(source_rows, working, pieces, dim, marks)

    def _plan_contexts(self, working) -> np.ndarray:
        """Resolve each row of a planning frame to its inventory context."""
        columns = resolution_columns(working, self._enrich_kwargs)
        if columns is None:
            return np.full(len(working), None, dtype=object)
        return resolve_contexts(self._resolved_inventory(), *columns)

    def conform_to_inventory(
        self,
        *,
        on_unresolved: WARN_LEVELS = "raise",
    ) -> Self:
        """
        Return a spool the inventory describes exactly, patch for patch.

        The one eager step of the inventory workflow: every row is
        resolved now, patches the inventory does not describe are
        dropped, and a patch whose span crosses a change of optical path
        is subdivided at each such change — so the spool can grow as well
        as shrink. A bound the answers survive unchanged is not a change,
        and does not divide anything. It is metadata work; no patch data
        is read.

        Subdivision is exact. Each piece begins at the first sample at or
        after the change which opens it, so together they hold every
        sample the patch held and hold none of them twice, and `len` and
        `get_contents` describe the pieces rather than the original.

        The inventory is the one
        [`attach_inventory`](`dascore.core.spool.Spool.attach_inventory`)
        put on the spool, which is the only way a spool gets one.

        Parameters
        ----------
        on_unresolved
            What to do with a patch the inventory does not describe — one
            carrying no `acquisition_key` (and given none by a pending
            `enrich`), one carrying a key the
            inventory does not resolve to exactly one entry, one reaching
            outside every matching epoch, or one with no instants to
            resolve at because its time axis is not physical. A patch is
            judged over its whole span, so one described at its start but
            not at its end is undescribed. "raise" (the default)
            fails and names them, "warn" drops them and says so, and
            "ignore" discards them silently, which is what an inventory
            deliberately covering part of an archive wants.

        Raises
        ------
        PatchError
            If a patch spans a change of *acquisition*, or must be
            subdivided but states no time step to find its samples with.
            An acquisition change means the two halves were recorded
            under different configurations, so no subdivision makes it
            one honest patch, and `on_unresolved` does not cover it: the
            inventory describes such a patch twice rather than not at
            all.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> patch, inventory = inventory_patch_pair()
        >>> spool = dc.spool(patch).attach_inventory(inventory)
        >>> assert len(spool.conform_to_inventory()) == 1
        >>>
        >>> # A patch the inventory says nothing about can be dropped.
        >>> other = patch.update_attrs(acquisition_key="DAS.R2D1..OTHER")
        >>> mixed = dc.spool([patch, other]).attach_inventory(inventory)
        >>> assert len(mixed.conform_to_inventory(on_unresolved="ignore")) == 1
        """
        self._check_inventory_policy(on_unresolved, "conform_to_inventory")
        new = self
        source_rows, working = new._plan_frames()
        # The two frames are one relation split by column, so a row of
        # either is the same patch as the row beside it; the messages
        # below name files from one while judging the other.
        assert (source_rows["_patch_id"].to_numpy() == working["_patch_id"]).all()
        columns = resolution_columns(working, new._enrich_kwargs)
        epochs = (
            [NO_EPOCHS] * len(working)
            if columns is None
            else resolve_row_epochs(new._resolved_inventory(), *columns)
        )
        refuse_rows(
            source_rows,
            acquisition_conflicts(epochs),
            "span a change of acquisition, which subdividing cannot "
            "reconcile; select the side you want, or correct the inventory",
        )
        described = np.array([x.described for x in epochs], dtype=bool)
        if not described.all():
            report_unconformed(source_rows[~described], on_unresolved)
        kept = working[described].reset_index(drop=True)
        cuts = [x.cuts for x, keep in zip(epochs, described, strict=True) if keep]
        if not any(cuts):  # nothing to subdivide: a filter is the whole job
            return new._restrict_to_rows(kept["_patch_id"].to_numpy())
        sources = source_rows[described].reset_index(drop=True)
        refuse_rows(
            sources,
            unsubdividable(kept, cuts, "time"),
            "must be subdivided at an epoch boundary but state no time step "
            "to find their samples with",
        )
        return new._subdivided(
            sources, kept, subdivision_pieces(kept, cuts, "time"), "time"
        )

    def _subdivided(self, sources, rows, pieces, name: str, stamp=None) -> Self:
        """
        Return the spool whose patches are the given pieces of these rows.

        The pieces are a plan rather than a rewritten relation, so the
        outputs *are* the contents rows — `len` and `get_contents` stay
        exact — and loading goes through the machinery which already
        trims a member on extraction. `stamp` names an attr to record on
        each output, in the order the pieces were given.

        A plan whose pieces do not cover their rows is marked lossy, so
        that re-planning the same dimension nests rather than collapsing
        onto the sources — which would load back the samples the pieces
        left out.
        """
        from dascore.io.index.planned import derived_catalog  # noqa: PLC0415

        plan = build_subdivision_plan(rows, pieces, name)
        stamped = ()
        if stamp is not None:
            stamp_name, values = stamp
            plan = replace(plan, outputs=plan.outputs.assign(**{stamp_name: values}))
            stamped = (stamp_name,)
        catalog = derived_catalog(
            source_rows=sources,
            plan=plan,
            parent=self._catalog,
            merge_kwargs={},
            mode="chunk",
            origin_path=self.spool_path,
            stamped=stamped,
            lossy=drops_samples(rows, pieces, name),
        )
        return self._new_from_catalog(catalog)

    def _check_inventory_policy(self, on_unresolved, method) -> None:
        """
        Refuse an inventory verb's arguments before it does any work.

        Both verbs read the attached inventory and police the same policy
        vocabulary. Sharing the check keeps the two from drifting into
        saying the same refusal differently.
        """
        if on_unresolved not in VALID_ON_UNRESOLVED:
            msg = (
                f"on_unresolved must be one of {VALID_ON_UNRESOLVED}, "
                f"got {on_unresolved!r}."
            )
            raise ParameterError(msg)
        if self._inventory is None:
            msg = (
                f"Spool.{method} needs an inventory; attach one first with "
                "Spool.attach_inventory."
            )
            raise ParameterError(msg)

    def _restrict_to_rows(self, patch_ids, keep: bool = True) -> Self:
        """
        Return the view holding the named rows, or all but them.

        Presentation order is the catalog's throughout, so this narrows
        which rows a spool holds without saying anything about how they
        come out.
        """
        ids = np.asarray(self._catalog.ordered_ids(), dtype=np.int64)
        named = np.isin(ids, np.asarray(patch_ids, dtype=np.int64))
        mask = named if keep else ~named
        if mask.all():
            return self
        return self._new_from_catalog(self._catalog.restrict(mask, ids=ids))

    def _resolved_inventory(self) -> Inventory:
        """
        The attached inventory itself, read now if it has not been.

        Every question answered *from* an inventory goes through here,
        and nothing else reads one -- comparing two attachments, which is
        the other thing a spool does with them, deliberately does not.
        The cheap `self._inventory is None` says whether one is attached
        at all, which is what lets a spool be opened, counted, ordered,
        chunked, and read without a lazily attached inventory ever being
        touched. Data access is never hostage to a metadata file.
        """
        # Every caller is already behind that cheap check, one way or
        # another: asking an inventory question of a spool carrying none
        # is refused where the question is asked, in its own words.
        attached = self._inventory
        assert attached is not None
        return attached.resolve() if isinstance(attached, InventoryRef) else attached

    def _enrichment(self):
        """Return how this spool enriches, or None if it does not."""
        if self._inventory is None or self._enrich_kwargs is None:
            return None
        return self._enrich_kwargs, self._on_unresolved

    def _maybe_enrich(self, patch):
        """Enrich an extracted patch when enrichment is set up."""
        if (enrichment := self._enrichment()) is None:
            return patch
        kwargs, on_unresolved = enrichment
        try:
            return patch.enrich(self._resolved_inventory(), **kwargs)
        except UnresolvedPatchError:
            # The inventory does not describe this patch. Dropping it is
            # conform_to_inventory's job, so it comes out as it went in.
            if on_unresolved == "raise":
                raise
            if on_unresolved == "warn" and not self._warned_unresolved:
                self._warned_unresolved = True
                # Deduped per spool, which is the granularity that means
                # something. Naming the patch would warn once per distinct
                # key -- thousands of them on exactly the partly-covered
                # archive this default exists to serve -- while leaving a
                # constant message to Python's registry would silence every
                # spool after the first in a session.
                warnings.warn_explicit(
                    UNRESOLVED_WARNING, UserWarning, __file__, 0, registry=None
                )
            return patch

    def sort(self, attribute) -> Self:
        """
        Sort the Spool based on a specific attribute.

        Parameters
        ----------
        attribute
            The attribute or coordinate used for sorting. If a coordinate name
            is used, the sorting will be based on the minimum value.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool()
        >>> # sort spool based on values in time coordinate.
        >>> spool_time_sorted = spool.sort("time")
        >>> # sort spool based on values in tag
        >>> spool_tag_sorted = spool.sort("tag")
        """
        # a lazy ORDER BY spec (D2): no copy, no realization; the
        # ordinal contract supplies the deterministic tiebreak
        return self._new_from_catalog(self._catalog.order_by(attribute))

    def split(
        self,
        size: int | None = None,
        count: int | None = None,
    ) -> Generator[Spool, None, None]:
        """
        Yield sub-patches based on specified parameters.

        Parameters
        ----------
        size
            The number of patches desired in each output spool. The last
            spool may have fewer patches. Must be greater than zero.
        count
            The number of spools to include. If count is greater than
            the length of the spool then the output will be smaller than
            count, with one patch per spool. Must be greater than zero.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("diverse_das")
        >>> # split spool into list of spools each with 3 patches.
        >>> split = spool.split(size=3)
        >>> # split spool into 3 evenly sized (if possible) spools
        >>> split = spool.split(count=3)
        """
        if not ((count is not None) ^ (size is not None)):
            msg = "Spool.split requires either spool_count or spool_size."
            raise ParameterError(msg)
        value = count if count is not None else size
        assert value is not None  # the check above sets exactly one of them
        # A step of zero or less never advances start, so the loop below
        # would yield forever.
        if value <= 0:
            msg = f"Spool.split requires a positive size or count, got {value}."
            raise ParameterError(msg)
        start = 0
        if count is not None:
            step = int(np.ceil(len(self) / value))
        else:
            step = int(np.ceil(value))  # tolerate a non-integral size
        while start < len(self):
            yield self[start : start + step]
            start += step

    @compose_docstring(progress_desc=progress_description)
    def map(
        self,
        func: Callable[..., T],
        *,
        client: ExecutorType | None = None,
        size: int | None = None,
        progress: PROGRESS_LEVELS = "standard",
        **kwargs,
    ) -> list[T]:
        """
        Map a function of all the contents of the spool.

        Parameters
        ----------
        func
            A callable which takes a patch as its first argument.
        client
            A client, or executor, which has a `map` method.
        size
            The number of patches in each spool mapped to a client.
            If not set, defaults to the number of processors on the host.
            Does nothing unless client is defined.
        {progress_desc}
        **kwargs
            kwargs passed to func.

        Notes
        -----
        When a client is specified, the spool is split then passed to the
        client's map method. This is to avoid serializing loaded patches.
        See [`Spool.split`](`dascore.core.spool.Spool.split`) for more
        details about the `size` and `count` parameters.

        Examples
        --------
        import numpy as np
        import dascore as dc

        spool = dc.get_example_spool("random_das")

        # Calculate the std for each channel in 5 second chunks
        results = (
             spool.chunk(time=5)
             .map(lambda x: np.std(x.data, axis=0))
        )
        # stack back into array. dims are (distance, time chunk)
        out = np.stack(results, axis=-1)
        """
        return _spool_map(
            self,
            func,
            client=client,
            size=size,
            progress=progress,
            **kwargs,
        )

    # Bind get_patch names as a spool method.
    get_patch_names = get_patch_names

    # Add method for stacking (adding the data arrays) patches in spool.
    stack = stack_patches

    def _new_from_catalog(self, catalog) -> Self:
        """Create a spool view over a (possibly derived) catalog."""
        new = self.__class__(self)
        new._catalog = catalog
        return new

    def _as_union_member(self):
        """
        Return the catalog which represents this spool in a union.

        A view whose state a table transfer would lose first bakes it
        into a derived catalog (tables only — no patch data is loaded);
        `PatchCatalog.transfer_is_lossy` is what knows which those are.
        """
        catalog = self._catalog
        return self._materialize_lossy() if catalog.transfer_is_lossy() else catalog

    def _materialize_lossy(self):
        """
        Bake residual trims and presentation order into a derived catalog.

        An identity plan over the view's presented rows: one output per
        row (in presentation order, so ordinals record the order spec),
        with trimmed envelopes as the output envelopes and the trims
        themselves re-applied at load through the plan resolver.
        """
        from dascore.io.index.planned import derived_catalog  # noqa: PLC0415

        rows = self._df.reset_index(drop=True)
        working = rows.reset_index(drop=True)
        ids = np.arange(len(working), dtype=np.int64)
        # outputs are not file rows: source bookkeeping stays on the
        # members (where loading needs it), never on the derived rows
        outputs = working.drop(
            columns=["_patch_id", *_SOURCE_COLUMNS], errors="ignore"
        ).assign(output_id=ids)
        members = pd.DataFrame(
            {
                "output_id": ids,
                "_patch_id": working.get("_patch_id", pd.Series(dtype=object)).values,
                "_modified": False,
            }
        )
        plan = ChunkPlan(outputs, members, "", None, {})
        return derived_catalog(
            source_rows=working,
            plan=plan,
            parent=self._catalog,
            merge_kwargs={},
            mode="identity",
            origin_path=self.spool_path,
        )

    # --- restructuring (materializing) operations -----------------------

    def _plan_frames(self, dim: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (source_rows, working) frames for planning along ``dim``.

        Re-planning the *same* dimension collapses (never nests): a
        derived catalog re-plans from its members — the trimmed source
        rows — restricted to the outputs the current view presents.
        Planning a *different* dimension must keep the already-assembled
        boundaries, so it plans over the current output rows themselves
        (loaded through the plan resolver). Patch-local sample and relative
        residuals adjust the working envelopes so plans reflect the
        loading truth.
        """
        from dascore.io.index.planned import (  # noqa: PLC0415
            PlanResolver,
            collapse_working_df,
        )

        resolver = self._catalog.resolver
        same_dim = isinstance(resolver, PlanResolver) and resolver.dim == dim
        base = collapse_working_df(self._catalog) if same_dim else None
        if base is None:
            base = self._catalog.to_df().reset_index(drop=True)
        base = _ensure_patch_id(base)
        working = base.drop(columns=list(self._drop_columns), errors="ignore")
        working = _drop_patch_local_empty(working)
        base = base[base["_patch_id"].isin(working["_patch_id"])]
        return base.reset_index(drop=True), working.reset_index(drop=True)

    def chunk_plan(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float | Quantity | np.timedelta64 = 1.5,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        group: str | Sequence[str] | None = None,
        missing_dim: Literal["raise", "drop"] = "raise",
        **kwargs,
    ):
        """
        Return the plan `chunk` would execute, without touching any data.

        The returned [`ChunkPlan`](`dascore.utils.chunk_plan.ChunkPlan`) is a
        read-only diagnostic: its `outputs` table describes each patch the
        chunked spool would contain (envelopes, step, carried attributes),
        its `members` table shows exactly which slice of which source patch
        feeds each output, and `params` records every resolved parameter
        (including the group attributes and sampling tolerance in effect).
        Accepts the same arguments as
        [`chunk`](`dascore.Spool.chunk`).

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool("random_das")
        >>> plan = spool.chunk_plan(time=3)
        >>> assert len(plan.outputs) == len(spool.chunk(time=3))
        >>> # See which sources contribute to the first output patch.
        >>> members = plan.members
        >>> first = members[members["output_id"] == 0]
        """
        _, working = self._plan_frames(next(iter(kwargs), None))
        return build_chunk_plan(
            working,
            overlap=overlap,
            keep_partial=keep_partial,
            snap_coords=snap_coords,
            tolerance=tolerance,
            conflict=conflict,
            group=group,
            missing_dim=missing_dim,
            **kwargs,
        )

    def _report_relation(self) -> pd.DataFrame:
        """
        Return the relation a gap report reads: the rows the spool presents.

        Unlike [`_plan_frames`](`dascore.core.spool.Spool._plan_frames`)
        a plan-backed spool is never collapsed to its members. Chunk
        collapses because re-planning a dimension replaces the plan; a
        report describes the patches the spool actually holds, so it
        reads the plan's outputs. The catalog has already replayed residual
        selections into those envelopes. No dimension is needed: the whole
        relation is returned and the caller picks its columns.
        """
        base = _ensure_patch_id(self._df.reset_index(drop=True))
        working = base.drop(columns=list(self._drop_columns), errors="ignore")
        return _drop_patch_local_empty(working)

    def get_gaps(
        self,
        dim: str = "time",
        *,
        tolerance: float | Quantity | np.timedelta64 = 1.5,
        group: str | Sequence[str] | None = None,
        missing_dim: Literal["raise", "drop"] = "drop",
    ) -> pd.DataFrame:
        """
        Return a dataframe with one row per gap along a dimension.

        Gaps are found with the rules
        [`chunk`](`dascore.Spool.chunk`) merges by, so every row is
        exactly a boundary that `chunk` would refuse to close.

        Parameters
        ----------
        dim
            The dimension to look for gaps along.
        tolerance
            The maximum number of samples patches can be spaced and still
            count as contiguous, or a quantity or timedelta stating that
            limit in the coordinate's own units (eg `1 * s`). Same
            meaning as chunk's `tolerance`.
        group
            Attributes which separate patches into unrelated groups; a gap
            is never reported between two groups. Defaults to the config
            option `patch_kind_attrs`. Sampling rate and coordinate
            structure split groups too, exactly as they do for `chunk`,
            so one attribute value can span several groups. A value
            nobody recorded is a value of its own, so a patch which never
            stated the attribute is not grouped with one which did.
        missing_dim
            What to do with patches lacking `dim`: "drop" (the default)
            excludes them, "raise" refuses. Chunk defaults to "raise"
            because it must produce those patches; a report need not.

        Notes
        -----
        `{dim}_min` is the last sample before the gap and `{dim}_max` the
        first sample after it, so `gap_size` is their difference — one
        step wider than the missing extent. Subtract the magnitude of
        the returned `{dim}_step` for the extent itself; the step keeps
        the coordinate's sign, which is negative for a descending one.

        `group_id` names the group each gap belongs to, and is the
        column to join against
        [`get_coverage`](`dascore.core.spool.Spool.get_coverage`).

        Overlapping and fully-nested patches never open a gap: each
        boundary is measured against the furthest point reached so far,
        not the previous row.

        A sample-count tolerance scales the step, so patches whose step
        is unknown report no gaps. An absolute tolerance needs no step,
        so it reports their gaps like any other patch's.

        See Also
        --------
        [`Spool.get_coverage`](`dascore.core.spool.Spool.get_coverage`)

        [`get_gap_edges`](`dascore.utils.gaps.get_gap_edges`) finds the
        gaps *inside* one patch's coordinate, which is a different
        question: this method reads the index and never loads data.

        Examples
        --------
        >>> import numpy as np
        >>> from dascore.examples import random_spool
        >>>
        >>> spool = random_spool(time_gap=np.timedelta64(1, "s"), length=3)
        >>> gaps = spool.get_gaps()
        >>> assert len(gaps) == 2
        >>> # A contiguous spool has none.
        >>> assert random_spool().get_gaps().empty
        """
        out = build_gap_frame(
            self._report_relation(),
            dim,
            tolerance=tolerance,
            group=group,
            missing_dim=missing_dim,
        )
        return present_columns(out)

    def get_coverage(
        self,
        dim: str = "time",
        *,
        tolerance: float | Quantity | np.timedelta64 = 1.5,
        group: str | Sequence[str] | None = None,
        missing_dim: Literal["raise", "drop"] = "drop",
    ) -> pd.DataFrame:
        """
        Return a dataframe summarizing how complete the spool is.

        One row per group of related patches — same kind, dims
        signature, coordinate identity, units, and sampling rate,
        exactly as `chunk` groups them before it looks at continuity.
        Each row reports the extent the group spans along `dim` and how
        much of that extent holds data.

        Parameters
        ----------
        dim
            The dimension to measure along.
        tolerance
            The maximum number of samples patches can be spaced and still
            count as contiguous, or a quantity or timedelta stating that
            limit in the coordinate's own units (eg `1 * s`). Same
            meaning as chunk's `tolerance`.
        group
            Attributes which separate patches into unrelated groups.
            Defaults to the config option `patch_kind_attrs`; sampling
            and structural differences split groups too, so one
            attribute value can span several rows. A value nobody
            recorded is a value of its own, so a patch which never
            stated the attribute is not grouped with one which did.
        missing_dim
            What to do with patches lacking `dim`: "drop" (the default)
            excludes them, "raise" refuses.

        Notes
        -----
        `span` is `{dim}_max - {dim}_min`, `gap_total` is the sum of the
        group's gaps as
        [`get_gaps`](`dascore.core.spool.Spool.get_gaps`) reports them,
        `covered` is the rest, and `coverage` is `covered / span` (1.0
        when the span is zero, meaning a single sample). `group_id`
        matches the gap frame's, so the two join on it.

        Coverage is measured between patches, from the envelopes the
        index records; a hole *inside* a patch is not visible here. Nor
        is one in a group whose step is unknown: a sample-count tolerance
        has nothing to scale there, so the group reports no gaps and
        counts as fully covered. An absolute tolerance does measure it.
        Both are what `chunk` would make of the data, so a `coverage` of
        1.0 says "nothing chunk would refuse to merge", not "nothing
        missing".

        See Also
        --------
        [`Spool.get_gaps`](`dascore.core.spool.Spool.get_gaps`)

        [`get_gap_edges`](`dascore.utils.gaps.get_gap_edges`) finds the
        gaps *inside* one patch's coordinate, which is a different
        question: this method reads the index and never loads data.

        Examples
        --------
        >>> import numpy as np
        >>> from dascore.examples import random_spool
        >>>
        >>> spool = random_spool(time_gap=np.timedelta64(1, "s"), length=3)
        >>> coverage = spool.get_coverage()
        >>> assert (coverage["coverage"] < 1).all()
        >>> assert (random_spool().get_coverage()["coverage"] == 1).all()
        """
        out = build_coverage_frame(
            self._report_relation(),
            dim,
            tolerance=tolerance,
            group=group,
            missing_dim=missing_dim,
        )
        return present_columns(out)

    @compose_docstring(conflict_desc=attr_conflict_description)
    def chunk(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float | Quantity | np.timedelta64 = 1.5,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        group: str | Sequence[str] | None = None,
        missing_dim: Literal["raise", "drop"] = "raise",
        **kwargs,
    ) -> Self:
        """
        Chunk the data in the spool along specified dimension.

        Parameters
        ----------
        overlap
            The amount of overlap between each segment, starting with the end of
            first patch. Negative values can be used to create gaps.
        keep_partial
            If True, keep the segments which are smaller than chunk size.
            This often occurs because of data gaps or at end of chunks.
        snap_coords
            If True (default), simplify the coordinates of joined patches to
            an evenly sampled range when doing so moves no coordinate value
            by more than `tolerance` (samples, or the length itself when
            the tolerance states one). Merges whose gaps exceed that keep
            an exact segmented coordinate instead.
        tolerance
            The maximum number of samples a block of data can be spaced (gap)
            and still be considered contiguous. A quantity or timedelta
            states that limit in the coordinate's own units instead (eg
            `tolerance=1 * s`), which also works for patches whose
            sampling interval is unknown. Either way a boundary of one
            sample is contiguous, so a tolerance below one sample never
            splits adjacent patches.
        conflict
            {conflict_desc}
        group
            Attributes which partition patches into separate outputs:
            conflicting values are never an error, the patches simply land
            in different outputs. A missing value (null or "") is a value
            like any other, so patches which never stated an attribute are
            grouped together and apart from those which did. Defaults to
            the config option
            `patch_kind_attrs`; unlike the default, explicitly passed names
            must exist on at least one patch. Dimensions and coordinate
            identities always partition implicitly.
        missing_dim
            What to do when patches lack the chunked dimension: "raise"
            (default) or "drop" (exclude them from the output).
        kwargs
            kwargs are used to specify the dimension along which to chunk, eg:
            `time=10` chunks along the time axis in 10 second increments.
            The value may also be a quantity: one of the coordinate's own
            units (`time=10 * s`) or a data size (`time=25 * megabytes`),
            which chunks so each patch's data array is about that large.
            `overlap` accepts the same forms.

        Examples
        --------
        >>> import dascore as dc
        >>> from dascore.units import s, megabytes
        >>>
        >>> spool = dc.get_example_spool("random_das")
        >>> # get spools with time duration of 10 seconds
        >>> time_chunked = spool.chunk(time=10, overlap=1)
        >>> # the same, with the units stated explicitly
        >>> unit_chunked = spool.chunk(time=10 * s)
        >>> # get patches whose data arrays are at most ~1 MB
        >>> size_chunked = spool.chunk(time=1 * megabytes)
        >>> # merge along time axis
        >>> time_merged = spool.chunk(time=...)

        Notes
        -----
        A data size measures the patch's data array only; coordinates and
        attrs are extra, as are any copies a later processing step makes,
        so the patch as a whole is somewhat larger. The sample count is
        rounded down, so the data never exceeds the requested size, and a
        merge of patches with different dtypes is sized against the dtype
        they upcast to.

        [`Spool.concatenate`](`dascore.Spool.concatenate`) performs a
        similar operation but disregards the coordinate values.

        To inspect what a chunk call will do before running it — which
        output patches it produces and which slice of which source patch
        feeds each one — use
        [`Spool.chunk_plan`](`dascore.core.spool.Spool.chunk_plan`),
        which takes the same arguments and returns the plan without
        touching any data.
        """
        from dascore.io.index.planned import derived_catalog  # noqa: PLC0415

        source_rows, working = self._plan_frames(next(iter(kwargs), None))
        plan = build_chunk_plan(
            working,
            overlap=overlap,
            keep_partial=keep_partial,
            snap_coords=snap_coords,
            tolerance=tolerance,
            conflict=conflict,
            group=group,
            missing_dim=missing_dim,
            **kwargs,
        )
        merge_kwargs = {
            "conflict": conflict,
            "snap_coords": snap_coords,
            # the plan's copy is normalized (eg a dimensionless quantity
            # has become the plain multiple it means)
            "tolerance": plan.params["tolerance"],
        }
        catalog = derived_catalog(
            source_rows=source_rows,
            plan=plan,
            parent=self._catalog,
            merge_kwargs=merge_kwargs,
            mode="chunk",
            origin_path=self.spool_path,
        )
        return self._new_from_catalog(catalog)

    @compose_docstring(conflict_desc=attr_conflict_description)
    def concatenate(
        self,
        check_behavior: WARN_LEVELS | None = None,
        *,
        conflict: Literal["drop", "raise", "keep_first"] = "raise",
        group: str | Sequence[str] | None = None,
        **kwargs,
    ) -> Self:
        """
        Concatenate patches in order along a dimension.

        Patches are partitioned as [`chunk`](`dascore.Spool.chunk`)
        partitions them — by kind (see the
        [patch compatibility note](`docs/notes/patch_compatibility`)),
        dimensions, the identity of every other dimension, and the
        concatenated dimension's units — and each partition's patches are
        then joined by the requested count in the order of the dimension
        (spool order when its step is unknown, or along a new dimension),
        contiguous or not. Patches which cannot be concatenated together
        land in separate outputs; nothing is skipped and planning does not
        raise. A coordinate the index describes only by a summary cannot
        be told from another with the same summary, so such an output is
        settled when it loads: equal values concatenate, and different
        ones raise there rather than being silently mixed.
        Remaining attributes must agree within an output, policed by
        `conflict` as `chunk` polices them. Coordinates are not policed:
        a coordinate riding the concatenated dimension is joined along
        it, every other coordinate must agree, and one which cannot be
        reconciled raises when the output loads rather than being
        dropped from a patch the catalog describes.

        Parameters
        ----------
        check_behavior
            Deprecated and ignored (kept in its old place for callers who
            passed it positionally): patches which cannot be concatenated
            together are placed in separate outputs rather than skipped.
        conflict
            {conflict_desc}
        group
            Attributes which partition patches into separate outputs,
            instead of the config option `patch_kind_attrs`.
        **kwargs
            One keyword naming the dimension and the number of patches per
            output; `None` puts every patch of a partition in one output. A
            dimension no patch has concatenates along a new one.

        Examples
        --------
        >>> import dascore as dc
        >>> spool = dc.get_example_spool()
        >>>
        >>> # Concatenate every patch along time, contiguous or not.
        >>> merged = spool.concatenate(time=None)
        >>> assert len(merged) == 1
        >>>
        >>> # Concatenate copies of a patch along a new dimension; patches
        >>> # sharing a new dimension must agree on the existing ones.
        >>> patch = dc.get_example_patch()
        >>> stacked = dc.spool([patch, patch.new()]).concatenate(wave_rank=None)
        >>> assert stacked[0].shape[stacked[0].get_axis("wave_rank")] == 2

        Notes
        -----
        - [`Spool.chunk`](`dascore.Spool.chunk`) performs a similar
          operation but accounts for coordinate values.
        - [`concatenate_patches`](`dascore.utils.patch.concatenate_patches`)
          concatenates a list of patches directly, relative to the first.
        """
        from dascore.io.index.planned import derived_catalog  # noqa: PLC0415

        if check_behavior is not None:
            msg = (
                "check_behavior is deprecated and ignored by Spool.concatenate: "
                "patches which cannot be concatenated together are placed in "
                "separate outputs. Drop the argument."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        dim = next(iter(kwargs), None)
        source_rows, working = self._plan_frames(dim)
        plan = build_concat_plan(working, conflict=conflict, group=group, **kwargs)
        merge_kwargs = {"conflict": conflict, "count": plan.params["count"]}
        catalog = derived_catalog(
            source_rows=source_rows,
            plan=plan,
            parent=self._catalog,
            merge_kwargs=merge_kwargs,
            mode="concat",
            origin_path=self.spool_path,
            # a plan which drops or picks among conflicting metadata must
            # not be re-planned from its members, which still hold it all
            lossy=conflict != "raise",
        )
        return self._new_from_catalog(catalog)

    # --- construction --------------------------------------------------

    @classmethod
    def from_directory(cls, path, index_path=None) -> Self:
        """
        Create a spool over a directory of fiber files.

        The directory's index (created/updated via ``update()``) backs
        the catalog; ``path`` may also be an existing directory indexer.

        A directory which carries an inventory under the name
        ``.inventory`` — the authoring directory ``.inventory/`` or a
        serialized ``.inventory.yaml``, ``.inventory.yml``, or
        ``.inventory.json`` — hands it to the spool, which reads it at
        the first question only an inventory can answer. See
        [`attach_inventory`](`dascore.core.spool.Spool.attach_inventory`).
        """
        from dascore.io.index.catalog import FileResolver, PatchCatalog  # noqa: PLC0415
        from dascore.io.index.indexer import DBDirectoryIndexer  # noqa: PLC0415

        out = cls()
        if isinstance(path, DBDirectoryIndexer):
            from dascore.io.index.catalog import _DIRECTORY_ORDER  # noqa: PLC0415

            out._catalog = PatchCatalog(
                backend=path._backend,
                resolver=FileResolver(root=path.path),
                syncer=path,
                default_order=_DIRECTORY_ORDER,
            )
        else:
            out._catalog = PatchCatalog.from_directory(path, index_path=index_path)
        # Filling the slot at the moment the spool is opened is what makes
        # `remove_inventory` stick: nothing refills it afterwards, so no
        # sentinel is needed to tell unset from deliberately emptied.
        out._inventory = out._blessed_inventory()
        return out

    @classmethod
    def from_file(
        cls,
        path,
        file_format: str | None = None,
        file_version: str | None = None,
    ) -> Self:
        """
        Create a spool over a single (multi-patch capable) fiber file.

        The file is scanned once; patches load lazily per row.

        Parameters
        ----------
        path
            The file to open.
        file_format
            The format name, sniffed from the file when not given.
        file_version
            The format version, sniffed from the file when not given.

        Notes
        -----
        A format and version given together are believed rather than
        checked against the file, as `dc.read` and `dc.scan` believe them.
        Naming the wrong format therefore yields whatever that reader
        makes of the file, which is usually an empty spool.
        """
        path = path if isinstance(path, UPath) else Path(path)
        if not path.exists() or path.is_dir():
            msg = f"{path} does not exist or is a directory"
            raise FileNotFoundError(msg)
        from dascore.io.index.catalog import PatchCatalog  # noqa: PLC0415

        if file_format and file_version:
            # Both internal callers already know the format, and sniffing
            # it again opens the file a second time; for a remote source
            # that is a round trip. Resolving the reader canonicalizes the
            # names and still rejects a pair no reader claims.
            fiber_io = dc.io.FiberIO.manager.get_fiberio(
                format=file_format, version=file_version
            )
            _format, _version = fiber_io.name, fiber_io.version
        else:
            _format, _version = dc.get_format(path, file_format, file_version)
        out = cls()
        out._catalog = PatchCatalog.from_file(
            path, file_format=_format, file_version=_version
        )
        out._file_path = path
        out._file_format = _format
        out._file_version = _version
        return out

    # --- capabilities --------------------------------------------------

    @property
    def indexer(self):
        """The directory syncer, or None for non-directory spools."""
        return self._catalog.syncer

    @property
    def spool_path(self):
        """The directory or file path this spool derives from, or None."""
        indexer = self.indexer
        if indexer is not None:
            return indexer.path
        if self._file_path is not None:
            return self._file_path
        return getattr(self._catalog.resolver, "origin_path", None)

    @property
    def has_live_patches(self) -> bool:
        """True when any of this spool's patches live in memory."""
        return bool(self._catalog.resolver.live_entries())

    @compose_docstring(progress_desc=progress_description)
    def update(self, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        Updates the contents of the spool, return the updated spool.

        Update is allowed only on a root spool — one no operation has
        been applied to. Directory roots re-index their directory,
        single-file roots rescan the file, and purely in-memory roots
        are trivially current (no-op). Any derived spool (the result of
        select, slicing, sort, chunk, concatenate, or combining spools)
        raises: update the root and re-apply the operations.

        Parameters
        ----------
        {progress_desc}
        """
        from dascore.io.index.catalog import LiveResolver  # noqa: PLC0415

        catalog = self._catalog
        derived_msg = (
            "update() is only allowed on a root spool; this spool is the "
            "result of an operation (select/slice/sort/chunk/combine). "
            "Update the root spool and re-apply the operations, e.g. "
            "root = root.update(); view = root.select(...)."
        )
        if catalog.is_view:
            raise InvalidSpoolError(derived_msg)
        if catalog.syncer is not None:
            catalog.update(progress=progress)
            return self._new_from_catalog(catalog)
        if self._file_path is not None:
            from dascore.io.core import FiberIO  # noqa: PLC0415

            formatter = FiberIO.manager.get_fiberio(
                format=self._file_format, version=self._file_version
            )
            getattr(formatter, "index", lambda _: None)(self._file_path)
            # Sniffed rather than reused: noticing that the file changed is
            # what update is for, and it may now hold another version.
            refreshed = self.from_file(self._file_path)
            # from_file builds a spool from the file alone, but an attached
            # inventory is the caller's state rather than the file's, and
            # re-reading the file is no reason to stop enriching.
            refreshed._inventory = self._inventory
            refreshed._enrich_kwargs = self._enrich_kwargs
            refreshed._on_unresolved = self._on_unresolved
            return refreshed
        if isinstance(catalog.resolver, LiveResolver):
            return self  # in-memory contents are trivially current
        # composite/plan roots are computed spools (unions, chunks)
        raise InvalidSpoolError(derived_msg)

    # --- equality ------------------------------------------------------

    def __eq__(self, other) -> bool:
        """
        Equality check which ignores the state of lazy realization.

        The flat relations are built and compared directly so that
        equality does not depend on whether they were realized yet.
        """
        if self is other:
            return True
        if not isinstance(other, Spool):
            other_dict = getattr(other, "__dict__", {})
            return deep_equality_check(self.__dict__, other_dict)
        # Models compare with ==; deep_equality_check walks their fields,
        # where an unset (NaT) time never equals itself.
        if (self._inventory, self._enrichment()) != (
            other._inventory,
            other._enrichment(),
        ):
            return False
        # views over the same catalog state are equal without realizing
        # the relations (a 200k-row archive must not materialize for ==)
        mine, theirs = self._catalog, other._catalog
        if (
            mine is not None
            and theirs is not None
            and (
                mine is theirs
                or (mine._backend is not None and mine._backend is theirs._backend)
            )
            and mine._queries == theirs._queries
            and mine._residuals == theirs._residuals
            and mine._order == theirs._order
            and mine._ids == theirs._ids
        ):
            return True
        return deep_equality_check(self._eq_state(), other._eq_state())

    def _eq_state(self) -> dict:
        """
        The spool's semantic state, explicitly enumerated for equality.

        Equality is over *effective* rows, never backends or
        representation: same length and order of patch rows, row-wise
        equal semantic columns (source identity like paths and
        live-vs-file backing stripped), with pending residual
        selections folded into the envelopes — a trimmed view equals
        its materialized twin, and spools differing only by a samples
        trim differ in their adjusted envelopes. Whether rows come from
        a live registry, an index file, or a plan is invisible; data
        arrays are never compared (metadata-level, like everything
        here). Because the state is enumerated — never ``__dict__`` —
        new instance attributes cannot silently join equality.
        """
        from dascore.io.index.catalog import _CUT_MASK_SUFFIX  # noqa: PLC0415

        def _private(column, suffix) -> bool:
            # the generated `_<coord><suffix>` column, not an attr which
            # happens to end the same way
            name = str(column)
            return name.startswith("_") and name.endswith(suffix)

        def _strip_identity(df):
            # synthetic per-catalog identities (memory:// paths, ids) and
            # backend provenance (format/version) are not content; equal
            # spools must compare equal without them, and column order
            # (a construction artifact) must not matter. A patch's own
            # lineage ids go too: a plan's outputs state none (a merged
            # patch folds its members' rather than inheriting one), so
            # keeping them would make a view unequal to its own
            # materialization over a column neither describes it by.
            # Coordinate def keys are representation artifacts too: a
            # residual-trimmed
            # view cannot know its trimmed fingerprint without loading,
            # and data values are never compared here anyway.
            drop = [
                "source_path",
                "_patch_id",
                "patch_id",
                "processing_id",
                "source_patch_key",
                "source_format",
                "source_version",
                "_modified",
                "_patch_local_empty",
                # A plan states no size until its outputs are assembled,
                # and equality holds between a view and its
                # materialization: the envelopes already say what the
                # rows describe, so the sample count adds nothing here.
                "_data_size",
                *[c for c in df.columns if _private(c, "_def_key")],
                # a plan's outputs state a placeholder coordinate dtype
                # for the same reason they state no def key
                *[c for c in df.columns if _private(c, "_coord_dtype")],
                # which residual narrowed a row says how this view
                # reached its envelopes, not what the row describes; the
                # trimmed envelopes it explains are compared instead
                *[c for c in df.columns if _private(c, _CUT_MASK_SUFFIX)],
                # whether the index holds every attr a patch defines says
                # what the index can state, not what the patch is
                "_attrs_complete",
                "_attr_dtypes",
            ]
            out = df.drop(columns=drop, errors="ignore")
            return out[sorted(out.columns)]

        rows = self._df
        # Catalog rows already reflect every coordinate residual in call order.
        return {"rows": _strip_identity(rows)}

    def _repr_node(self) -> Repr:
        """
        What the spool is, what it spans, and the tracks it holds.

        Summarizing realizes the index relation, which a spool of many
        patches should not do for a glance; `display_max_patches` says
        how many is too many, and past it the repr states its count and
        path alone.
        """
        count = len(self)
        plural = "" if count == 1 else "es"
        name = f"{self.__class__.__name__} 🧵 ({count:,d} Patch{plural})"
        blocks = [get_header_text(name, style=self._rich_style)]
        if (path := self.spool_path) is not None:
            blocks.append(Text(f"    Path: {path}"))
        # The limit is what it says whatever the spool has been asked
        # before: a repr which summarized only once a frame happened to
        # be built would print two different things for one object.
        limit = dc.get_config().display_max_patches
        if count <= limit:
            # A repr which raises makes an object undebuggable at the
            # one moment someone needs to look at it, so nothing the
            # summary does is allowed to stop the header from printing.
            # Only the summary: counting the spool and asking for its
            # path happen above, and a catalog which cannot answer
            # either of those still raises.
            # Nor is it allowed to warn about how pandas built the
            # frame: a directory index warns once per insert that it is
            # fragmented, a hundred lines of advice about a frame the
            # caller never asked for, from typing a name.
            #
            # Only that category. A realized frame is cached, so the
            # first reader is the only one who ever warns -- suppress
            # anything else here and a repr does not delay it, it eats
            # it, and the `get_contents` after it goes quiet too.
            with suppress(Exception), suppress_warnings(PerformanceWarning):
                blocks.extend(self._summary_blocks())
        else:
            blocks.append(
                Text(
                    f"    Not summarized: {count:,d} patches exceeds "
                    f"display_max_patches={limit:,d}.",
                    style=dascore_styles["keys"],
                )
            )
        header, *rest = blocks
        sections = tuple(split_block(x) for x in rest)
        return Repr(header=header, body=sections)

    def _stated_dims(self, df) -> list[str]:
        """The dimensions this spool's patches actually have."""
        stated = [
            x for x in get_dim_names_from_columns(df) if self._measured(df, x).any()
        ]
        # Time leads where it is one of them: it is the dimension a
        # reader looks for, and the one the tracks are measured along.
        return sorted(stated, key=lambda x: (x != "time", x))

    @staticmethod
    def _measured(df, dim: str):
        """
        Which rows have this dimension as an axis and state both its ends.

        Two things are asked, and the first is asked with the same
        function `get_coverage` asks it with: the relation carries an
        envelope for every coordinate, dimensional or not, and a patch
        may ride a coordinate of this name along a different axis, so
        `dims` is what says whose axis it is, row by row. A track and
        the lane of the same patches have to agree on that, which they
        cannot do from two spellings of one rule.

        Being an axis is not enough on its own: a patch may name one
        whose envelope it never filled in, and an unstated end bounds
        nothing.
        """
        axis = pd.Series(_structural(df, dim), index=df.index)
        return axis & df[f"{dim}_min"].notna() & df[f"{dim}_max"].notna()

    def _comparable(self, df, dim: str) -> bool:
        """Whether this dimension's ends can be measured against each other."""
        measured = df[self._measured(df, dim)]
        # Two kinds do not subtract, and neither do two units: a metre
        # taken from a foot is a number standing for nothing.
        return (
            len(self._value_kinds(measured[f"{dim}_min"])) == 1
            and len(self._dim_units(measured, dim)) <= 1
        )

    @staticmethod
    def _value_kinds(values) -> set[str]:
        """Which kinds of thing an envelope column holds."""
        # One dimension name can be a time on one patch and a number on
        # another. Their envelopes share a column and do not compare, so
        # asking for the extent of the two together raises.
        kinds = set()
        for value in values.dropna():
            # An instant and an offset are both times and still do not
            # compare: a date is not a length, and asking for the
            # smaller of the two raises.
            if isinstance(value, pd.Timestamp | np.datetime64):
                kinds.add("instant")
            elif isinstance(value, pd.Timedelta | np.timedelta64):
                kinds.add("offset")
            elif isinstance(value, numbers.Number):
                kinds.add("number")
            else:
                kinds.add("label")
        return kinds

    def _dims_text(self, df, dims: list[str]) -> Text:
        """The extent this spool covers along each of its dimensions."""
        key_style = dascore_styles["keys"]
        base = Text("➤ ") + Text("Dimensions", style=dascore_styles["dc_blue"])
        base += Text(" (") + Text(", ".join(dims), style="bold") + Text(")")
        width = max(len(x) for x in dims)
        for dim in dims:
            measured = df[self._measured(df, dim)]
            base += Text.assemble("\n    ", Text(f"{dim + ':':<{width + 1}} ", "bold"))
            if len(self._value_kinds(measured[f"{dim}_min"])) > 1:
                # Two kinds do not compare, so there are no two ends to
                # state. Saying so beats the whole summary disappearing.
                base += Text("mixed value kinds", key_style)
                continue
            low, high = measured[f"{dim}_min"].min(), measured[f"{dim}_max"].max()
            units = self._dim_units(measured, dim)
            near, far = range_texts(low, high)
            base += near + Text(" to ", key_style) + far
            if len(units) > 1:
                # Envelopes are stored in the units each patch was read
                # in, so a min and a max from two of them are not two
                # ends of one extent. Say so rather than imply otherwise.
                base += Text(f"  (mixed units: {', '.join(units)})", key_style)
                continue
            # A time states its units in the way it is written; every
            # other dimension is named in what it was measured in.
            if units and not isinstance(low, _TIMES):
                base += Text(" ") + Text(units[0], dascore_styles["units"])
            # How far the two ends lie apart, in what they are measured
            # in: a width said bare beside a max which names its units
            # reads as if it were in others, and the tracks under this
            # line state theirs.
            if (span := span_text(low, high, units[0] if units else None)) is not None:
                base += Text("  ") + span
        return base

    @staticmethod
    def _dim_units(df, dim: str) -> tuple[str, ...]:
        """
        What a dimension is measured in, as its patches state it.

        The units column is private in the relation and renamed only on
        the way out through `present_columns`, so both spellings are
        asked for: a repr reads the frame before that rename.
        """
        stated = df.get(f"{dim}_units", df.get(f"_{dim}_units"))
        # Only a dimension the relation describes is ever asked about,
        # and it describes each of them with a units column, empty or not.
        assert stated is not None, f"the relation states no units for {dim}"
        return tuple(sorted({str(x) for x in stated.dropna().unique() if str(x)}))

    def _span_text(self, low, high, units: tuple[str, ...]) -> Text | None:
        """How wide an extent is, measured in what the dimension is."""
        # A track states its units nowhere else, so the width carries
        # them -- but only where every patch agrees on one, since
        # stating the first of several would label a track in a unit it
        # was not measured in.
        stated = units[0] if len(units) == 1 else None
        return span_text(low, high, stated)

    def _tracks_text(self, df, dim: str) -> Text | None:
        """
        The kinds of patch this spool holds, named as its lanes are named.

        A track is one kind of patch, which is the partition
        [`chunk`](`dascore.Spool.chunk`) starts from. Coverage may cut a
        kind finer still -- two sampling rates are two lanes of one tag
        -- so a lane there is a track here or a part of one, and the two
        are named by one rule rather than given one name.
        """
        # The same resolution chunk does, rather than a second reading
        # of the config which could drift from it. It dedupes, which is
        # what lets the config name an attribute twice: the groupby
        # tolerates that, but reset_index will not put one column back
        # under two labels.
        keys = list(_resolve_group_attrs(None, set(df.columns)))
        if not keys:
            return None
        low, high = f"{dim}_min", f"{dim}_max"
        # A patch which does not have this dimension is not a track along
        # it; get_coverage drops it too rather than report an empty one.
        df = df[self._measured(df, dim)]
        # The dimension came from _stated_dims, which is the set of them
        # some row measures, so something is always left to group.
        assert len(df), f"no patch measures {dim}"
        frame = (
            df.groupby(keys, dropna=False)
            .agg(_n=(low, "size"), _low=(low, "min"), _high=(high, "max"))
            .reset_index()
        )
        # One track is the whole spool, which the dimensions above
        # already state; naming it would only repeat them.
        if len(frame) < 2:
            return None
        units = self._dim_units(df, dim)
        base = Text("➤ ") + Text("Tracks", style=dascore_styles["dc_red"])
        base += Text(f" ({len(frame)} along ") + Text(dim, style="bold") + Text(")")
        limit = dc.get_config().display_max_items
        shown = frame.head(limit)
        names = group_names(
            frame,
            ignore=("_n", "_low", "_high"),
            # The same fallback the plot names a lane by, or a group
            # nothing tells apart is a track called "group 0" and a lane
            # called by its acquisition -- one rule, two names.
            fallback=ACQUISITION_ATTR,
        )[:limit]
        counted = [f"{x:,d} patch{'' if x == 1 else 'es'}" for x in shown["_n"]]
        # The columns are measured over the lines which are drawn: a name
        # elided below should not pad the names above it.
        width = max((len(x) for x in names), default=0)
        held = max((len(x) for x in counted), default=0)
        # iterrows, not itertuples: the aggregate columns are named
        # privately so no attr of that name can collide with them, and
        # itertuples renames a private column to its position.
        for name, count, (_, row) in zip(names, counted, shown.iterrows(), strict=True):
            base += Text.assemble("\n    ", Text(f"{name:<{width}}", "bold"), "  ")
            base += Text(f"{count:>{held}}", dascore_styles["keys"])
            base += Text("  ") + get_nice_text(row["_low"])
            if (span := self._span_text(row["_low"], row["_high"], units)) is not None:
                base += Text(" ") + span
        if (left_out := len(frame) - len(shown)) > 0:
            base += Text("\n    ") + elision_text(left_out)
        base += Text("\n    (coverage: spool.viz.coverage())", dascore_styles["keys"])
        return base

    def _summary_blocks(self) -> list[Text]:
        """The blocks stating what this spool spans and what it holds."""
        df = self._df
        if df is None or not len(df) or "dims" not in df.columns:
            return []
        if not (dims := self._stated_dims(df)):
            return []
        blocks = [self._dims_text(df, dims)]
        # Tracks are measured along one dimension, which has to be one
        # whose ends can be compared.
        measurable = [x for x in dims if self._comparable(df, x)]
        if measurable and (tracks := self._tracks_text(df, measurable[0])) is not None:
            blocks.append(tracks)
        return blocks


# There is one spool class; the old ABC name stays as an alias so
# annotations and isinstance checks written against it keep working.
# `Spool` is the name to use in new code.
BaseSpool = Spool


@singledispatch
def spool(obj: path_types | Spool | Sequence[PatchType], **kwargs) -> Spool:
    """
    Create a spool from a data source.

    This is the main function for loading in DASCore.

    Parameters
    ----------
    obj
        An object from which a spool can be derived.

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> # Get a spool from a single file. An examples:// name refers to a
    >>> # file in DASCore's example data registry; use your own path here.
    >>> file_spool = dc.spool("examples://example_dasdae_event_1.h5")
    >>>
    >>> # get a spool from a directory of files
    >>> directory_path = dc.examples.spool_to_directory(dc.get_example_spool())
    >>> directory_spool = dc.spool(directory_path)
    >>>
    >>> # get a spool from a single patch
    >>> patch = dc.get_example_patch()
    >>> spool = dc.spool(patch)
    """
    msg = f"Could not get spool from: {obj}"
    raise ValueError(msg)


@spool.register(str)
@spool.register(Path)
@spool.register(UPath)
def _spool_from_str(path, **kwargs):
    """Get a spool from a path."""
    path = coerce_to_upath(resolve_example_uri(path))
    # A directory was passed; index it.
    if path.is_dir():
        requires_local_directory(path, label="Directory spool")
        return Spool.from_directory(path, **kwargs)
    # A single file was passed. If the file format supports quick
    # scanning build a lazy file-backed spool, else read it into memory.
    elif path.exists():  # a single file path was passed.
        _format, _version = dc.get_format(path, **kwargs)
        formatter = dc.io.FiberIO.manager.get_fiberio(format=_format, version=_version)
        if formatter.implements_scan:
            return Spool.from_file(path, _format, _version)
        else:
            return Spool(dc.read(path, _format, _version))
    else:
        msg = (
            f"could not get spool from argument: {path}. "
            f"If it is a path, it may not exist."
        )
        raise InvalidSpoolError(msg)


@spool.register(Spool)
def _spool_from_spool(spool, **kwargs):
    """Return a spool from a spool."""
    return spool


@spool.register(list)
@spool.register(tuple)
def _spool_from_patch_list(patch_list, **kwargs):
    """Return a spool from a sequence of patches."""
    return Spool(patch_list)


@spool.register(dc.Patch)
def _spool_from_patch(patch):
    """Get a spool from a single patch."""
    return Spool([patch])
