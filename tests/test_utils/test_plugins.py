"""Tests for plugin utility helpers."""

from __future__ import annotations

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
