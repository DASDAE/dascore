"""Tests for ProdML source patch ids."""

from __future__ import annotations

import pytest

import dascore as dc
from dascore.utils.downloader import fetch


class TestProdMLSourcePatchId:
    """Ensure multi-patch ProdML files support summary-based reload."""

    @pytest.fixture(scope="class")
    def prodml_fbe_path(self):
        """Return a multi-patch ProdML FBE file."""
        return fetch("prodml_fbe_1.h5")

    def test_scan_includes_source_patch_id(self, prodml_fbe_path):
        """Scanned ProdML summaries should include a unique source patch id."""
        summaries = dc.scan(prodml_fbe_path)
        assert len(summaries) > 1
        ids = [summary.source_patch_id for summary in summaries]
        assert all(ids)
        assert len(ids) == len(set(ids))

    def test_read_source_patch_id_selects_single_patch(self, prodml_fbe_path):
        """Reading by source_patch_id should resolve one ProdML patch."""
        target = dc.scan(prodml_fbe_path)[0]
        spool = dc.read(prodml_fbe_path, source_patch_id=target.source_patch_id)
        assert len(spool) == 1
        assert "source_patch_id" not in spool[0].attrs.model_dump()
        assert (
            spool[0].summary.get_coord_summary("time").min
            == target.get_coord_summary("time").min
        )

    def test_read_multiple_source_patch_ids(self, prodml_fbe_path):
        """Reading by multiple source_patch_id values should return each match."""
        summaries = dc.scan(prodml_fbe_path)
        targets = [summaries[0].source_patch_id, summaries[1].source_patch_id]
        spool = dc.read(prodml_fbe_path, source_patch_id=targets)
        assert len(spool) == 2
        assert {patch.summary.get_coord_summary("time").min for patch in spool} == {
            summaries[0].get_coord_summary("time").min,
            summaries[1].get_coord_summary("time").min,
        }
