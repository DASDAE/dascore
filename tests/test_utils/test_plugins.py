"""Tests for plugin utility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dascore.utils import plugins as plugin_mod


class TestLoadEntryPoint:
    """Tests for loading entry points."""

    def test_missing_entry_point_returns_none(self, monkeypatch):
        """Missing entry points should return None."""
        plugin_mod.get_entry_point_loaders.cache_clear()
        plugin_mod.maybe_load_entry_point.cache_clear()
        monkeypatch.setattr(plugin_mod, "get_entry_point_loaders", lambda _: {})
        assert plugin_mod.maybe_load_entry_point("test.group", "missing") is None

    def test_entry_point_loads_and_caches(self, monkeypatch):
        """Loaded entry points should only invoke the loader once."""
        called = {"count": 0}

        def loader():
            called["count"] += 1
            return "loaded"

        plugin_mod.get_entry_point_loaders.cache_clear()
        plugin_mod.maybe_load_entry_point.cache_clear()
        monkeypatch.setattr(
            plugin_mod, "get_entry_point_loaders", lambda _: {"foo": loader}
        )

        assert plugin_mod.maybe_load_entry_point("test.group", "foo") == "loaded"
        assert plugin_mod.maybe_load_entry_point("test.group", "foo") == "loaded"
        assert called["count"] == 1

    def test_duplicate_entry_points_warn_and_last_one_wins(self, monkeypatch):
        """Duplicate entry-point names should warn and keep the last loader."""

        def first_loader():
            return "first"

        def second_loader():
            return "second"

        entry_point_group = "test.group"
        entry_point_list = [
            SimpleNamespace(name="dup", load=first_loader),
            SimpleNamespace(name="dup", load=second_loader),
            SimpleNamespace(name="unique", load=lambda: "unique"),
        ]

        plugin_mod.get_entry_point_loaders.cache_clear()
        monkeypatch.setattr(
            plugin_mod, "entry_points", lambda *, group: entry_point_list
        )

        with pytest.warns(UserWarning, match="Duplicate entry points found"):
            out = plugin_mod.get_entry_point_loaders(entry_point_group)

        assert out["dup"] is second_loader
        assert out["unique"]() == "unique"
