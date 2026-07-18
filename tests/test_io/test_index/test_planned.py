"""
Tests for derived catalogs (plan-as-catalog) and coverage of their edges.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.io.index.planned import (
    PlanResolver,
    _coord_record_from_row,
    _ns,
    derived_catalog,
)


@pytest.fixture(scope="module")
def patches():
    """Three contiguous example patches."""
    return list(dc.get_example_spool("random_das"))


class TestHelpers:
    """Unit coverage for the conversion helpers."""

    def test_ns_forms(self):
        """All datetime/timedelta forms convert to the same ns."""
        ts = pd.Timestamp("2020-01-01")
        assert _ns(ts) == _ns(ts.to_datetime64()) == ts.value
        td = pd.Timedelta(seconds=1)
        assert _ns(td) == _ns(td.to_timedelta64()) == td.value
        assert _ns(None) is None

    def test_coord_record_numpy_datetimes(self):
        """np.datetime64 envelope values build the same record."""
        lo = np.datetime64("2020-01-01", "ns")
        hi = np.datetime64("2020-01-02", "ns")
        row = {"time_min": lo, "time_max": hi, "time_step": np.timedelta64(1, "s")}
        record = _coord_record_from_row(row, "time")
        assert record.value_kind == "time"
        assert record.min_ns == _ns(lo)

    def test_coord_record_zero_step_length(self):
        """A degenerate step leaves length unknown instead of raising."""
        row = {"time_min": 0.0, "time_max": 1.0, "time_step": 0.0}
        record = _coord_record_from_row(row, "time")
        assert record.length is None

    def test_plan_resolver_requires_output_id(self):
        """member_rows without output_id is a construction error."""
        with pytest.raises(ValueError, match="output_id"):
            PlanResolver(
                token="x",
                dim="time",
                member_rows=pd.DataFrame({"path": []}),
                loader=None,
                merge_kwargs={},
            )

    def test_derived_catalog_adds_patch_ids(self, patches):
        """source_rows without _patch_id get positional ids."""
        from dascore.utils.chunk_plan import ChunkPlan

        spool = dc.spool(patches)
        rows = spool.get_contents().drop(columns=["_patch_id"], errors="ignore")
        rows = rows.reset_index(drop=True)
        members = pd.DataFrame(
            {"output_id": [0], "_patch_id": [0], "_modified": [False]}
        )
        outputs = rows.iloc[:1].assign(output_id=0)
        plan = ChunkPlan(outputs, members, "time", None, {})
        catalog = derived_catalog(
            source_rows=rows,
            plan=plan,
            parent=spool._catalog,
            merge_kwargs={},
            mode="concat",
        )
        assert len(catalog) == 1


class TestDerivedComposition:
    """Operation-order coverage over derived catalogs."""

    def test_collapse_with_value_residual(self, patches):
        """Chunk of a selected chunked spool re-plans from trimmed members."""
        t0 = patches[0].get_coord("time").min()
        t1 = patches[1].get_coord("time").max()
        chunked = dc.spool(patches).chunk(time=2)
        selected = chunked.select(time=(t0, t1))
        merged = selected.chunk(time=None)
        assert len(merged) >= 1
        out = merged[0]
        assert out.get_coord("time").min() >= t0

    def test_sort_by_attr_on_windowed_regex_view(self, patches):
        """Regex selection + window + attr sort compose through SQL."""
        tagged = [
            p.update_attrs(tag=f"t{num}", history=[]) for num, p in enumerate(patches)
        ]
        spool = dc.spool(tagged)
        view = spool.select(tag=re.compile("t[0-9]"))[1:]
        out = view.sort("tag")
        tags = [p.attrs["tag"] for p in out]
        assert tags == sorted(tags)

    def test_attr_membership_array(self, patches):
        """Attr membership with a numpy array of values selects rows."""
        tagged = [
            p.update_attrs(tag=f"t{num}", history=[]) for num, p in enumerate(patches)
        ]
        spool = dc.spool(tagged)
        out = spool.select(tag=np.array(["t0", "t2"]))
        assert len(out) == 2

    def test_sort_by_envelope_column_name(self, patches):
        """Sort accepts the explicit `{dim}_min` column form."""
        spool = dc.spool(list(reversed(patches)))
        out = spool.sort("time_min")
        assert out.get_contents()["time_min"].is_monotonic_increasing

    def test_concatenate_requires_one_kwarg(self, patches):
        """Concatenate validates its dimension keyword."""
        with pytest.raises(ParameterError, match="exactly one dimension"):
            dc.spool(patches).concatenate(time=None, distance=None)

    def test_union_view_of_live_spools_pickles_composite(self, patches):
        """A selected union pickles a membership-restricted composite."""
        import pickle

        t0 = patches[0].get_coord("time")
        combined = dc.spool(patches[:2]) + dc.spool(patches[2:])
        view = combined.select(time=(None, t0.max()))
        assert len(view) == 1
        loaded = pickle.loads(pickle.dumps(view))
        assert len(loaded) == 1
        assert isinstance(loaded[0], dc.Patch)

    def test_missing_live_patch_getitem(self, patches):
        """A missing registry entry surfaces as MissingPatchError, not
        out-of-bounds.
        """
        from dascore.exceptions import MissingPatchError

        spool = dc.spool(patches[:1])
        _ = spool.get_contents()  # realize rows
        spool._catalog.resolver._registry.clear()
        with pytest.raises(MissingPatchError, match="not available"):
            spool[0]

    def test_union_with_third_party_spool(self, patches):
        """The BaseSpool fallback materializes third-party members."""
        from dascore.core.spool import BaseSpool

        class MiniSpool(BaseSpool):
            def __init__(self, inner):
                self._inner = list(inner)

            def __getitem__(self, item):
                return self._inner[item]

            def __iter__(self):
                return iter(self._inner)

            def __len__(self):
                return len(self._inner)

            def chunk(self, **kwargs):
                raise NotImplementedError

            def select(self, **kwargs):
                raise NotImplementedError

            def get_contents(self):
                raise NotImplementedError

        combined = dc.spool(patches[:1]) + MiniSpool(patches[1:])
        assert len(combined) == len(patches)

    def test_samples_negative_index_skips_envelope_adjust(self, patches):
        """Negative samples windows stay candidacy supersets (no crash)."""
        spool = dc.spool(patches[:1]).select(time=(0, -10), samples=True)
        merged = spool.chunk(time=None)
        assert len(merged) == 1

    def test_complete_overlap_merge(self):
        """Two identical-envelope patches merge by keeping the first."""
        patch = dc.get_example_patch()
        twin = patch.new()
        merged = dc.spool([patch, twin]).chunk(time=None)
        assert len(merged) == 1
        assert isinstance(merged[0], dc.Patch)


class TestRemainingEdges:
    """Direct coverage of defensive/rare branches."""

    def test_collapse_with_quantity_residual(self, patches):
        """A quantity-selected chunked view re-chunks without applying
        unit-bearing bounds to envelopes (they stay load residuals).
        """
        from dascore.units import m

        chunked = dc.spool(patches).chunk(time=2)
        selected = chunked.select(_coords={"distance": (0 * m, 10 * m)})
        merged = selected.chunk(time=None)
        assert len(merged) == 1
        coord = merged[0].get_coord("distance")
        assert float(coord.max()) <= 10

    def test_samples_adjust_skips_missing_columns(self):
        """Residuals naming absent envelope columns pass through."""
        from dascore.utils.chunk_plan import samples_adjusted_envelopes

        df = pd.DataFrame({"time_min": [0.0], "time_max": [1.0]})
        residuals = (({"depth": (0, 5)}, True),)
        out = samples_adjusted_envelopes(df, residuals)
        assert out.equals(df)
