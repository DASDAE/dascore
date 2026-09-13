"""Tests for ProdML source patch ids."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import PatchAttributeError
from dascore.io.core import FiberIO
from dascore.io.prodml.core import ProdMLV2_0
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


class TestReadArrayKeys:
    """The key names which of a file's several nodes is read."""

    @pytest.fixture(scope="class")
    def prodml_fbe_path(self):
        """A ProdML file holding several FBE nodes."""
        return fetch("prodml_fbe_1.h5")

    def test_each_key_reads_its_own_node(self, prodml_fbe_path):
        """Every node is reachable, and none stands in for another."""
        io = ProdMLV2_0()
        payloads = dc.scan(prodml_fbe_path)
        assert len(payloads) > 1
        arrays = {}
        for payload in payloads:
            key = payload.source_patch_key
            out = io.read_array(prodml_fbe_path, {}, source_patch_key=key)
            expected = FiberIO.read_array(io, prodml_fbe_path, {}, source_patch_key=key)
            assert np.array_equal(out, expected, equal_nan=True), key
            arrays[key] = out
        # the nodes share a shape, so only their values tell them apart
        first = next(iter(arrays.values()))
        assert any(
            not np.array_equal(first, x, equal_nan=True) for x in arrays.values()
        )

    def test_unknown_key_raises(self, prodml_fbe_path):
        """A key naming no node is not silently resolved to one."""
        with pytest.raises(PatchAttributeError, match="No patch named"):
            ProdMLV2_0().read_array(prodml_fbe_path, {}, source_patch_key="nope")

    def test_keyless_multi_node_raises(self, prodml_fbe_path):
        """Several nodes and no key cannot be resolved."""
        with pytest.raises(PatchAttributeError, match="source_patch_key"):
            ProdMLV2_0().read_array(prodml_fbe_path, {})


class TestNodeDims:
    """Where a node's stored dimension order comes from."""

    @pytest.fixture
    def fbe_without_dimensions(self, tmp_path):
        """A copy of the FBE file whose nodes state no Dimensions."""
        path = tmp_path / "no_dims.h5"
        shutil.copy(fetch("prodml_fbe_1.h5"), path)
        with h5py.File(path, "a") as h5:
            for group in h5["Acquisition"].values():
                for node in getattr(group, "values", dict)():
                    for name, dataset in getattr(node, "items", dict)():
                        if name.lower().startswith("fbedata["):
                            dataset.attrs.pop("Dimensions", None)
        return path

    def test_missing_dimensions_falls_back(self, fbe_without_dimensions):
        """A node stating nothing takes its parent's order, as scan does."""
        io = ProdMLV2_0()
        payload = dc.scan(fbe_without_dimensions)[0]
        key = payload.source_patch_key
        out = io.read_array(fbe_without_dimensions, {}, source_patch_key=key)
        expected = FiberIO.read_array(
            io, fbe_without_dimensions, {}, source_patch_key=key
        )
        assert out.shape == expected.shape == payload.shape
