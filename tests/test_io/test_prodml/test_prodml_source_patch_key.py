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

    def test_scan_includes_source_patch_key(self, prodml_fbe_path):
        """Scanned ProdML summaries should include a unique source patch id."""
        summaries = dc.scan(prodml_fbe_path)
        assert len(summaries) > 1
        ids = [summary.source_patch_key for summary in summaries]
        assert all(ids)
        assert len(ids) == len(set(ids))

    def test_read_source_patch_key_selects_single_patch(self, prodml_fbe_path):
        """Reading by source_patch_key should resolve one ProdML patch."""
        target = dc.scan(prodml_fbe_path)[0]
        spool = dc.read(prodml_fbe_path, source_patch_key=target.source_patch_key)
        assert len(spool) == 1
        assert spool[0].attrs["_source_patch_key"] == target.source_patch_key
        assert (
            spool[0].summary.get_coord_summary("time").min
            == target.get_coord_summary("time").min
        )

    def test_read_multiple_source_patch_keys(self, prodml_fbe_path):
        """Reading by multiple source_patch_key values should return each match."""
        summaries = dc.scan(prodml_fbe_path)
        targets = [summaries[0].source_patch_key, summaries[1].source_patch_key]
        spool = dc.read(prodml_fbe_path, source_patch_key=targets)
        assert len(spool) == 2
        assert {patch.attrs["_source_patch_key"] for patch in spool} == set(targets)
        assert {patch.summary.get_coord_summary("time").min for patch in spool} == {
            summaries[0].get_coord_summary("time").min,
            summaries[1].get_coord_summary("time").min,
        }
