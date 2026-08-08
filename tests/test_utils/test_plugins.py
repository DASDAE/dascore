"""Tests for plugin utility helpers."""

from __future__ import annotations

import sys
import threading
from importlib.metadata import EntryPoint
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

    def test_loaders_mapping_is_immutable(self, monkeypatch):
        """The cached loader mapping is shared, so callers cannot mutate it."""
        entry_point_list = [SimpleNamespace(name="unique", load=lambda: "unique")]
        plugin_mod.get_entry_point_loaders.cache_clear()
        monkeypatch.setattr(
            plugin_mod, "entry_points", lambda *, group: entry_point_list
        )

        out = plugin_mod.get_entry_point_loaders("test.group")

        with pytest.raises(TypeError):
            out["another"] = None


class TestConcurrentEntryPointLoading:
    """Tests for concurrent misses on the unsynchronized plugin caches."""

    module_name = "dascore_slow_test_plugin"

    @pytest.fixture
    def execution_log(self, tmp_path):
        """Return the path the test plugin appends to each time it executes."""
        return tmp_path / "executions.txt"

    @pytest.fixture
    def slow_entry_point(self, tmp_path, execution_log, monkeypatch):
        """Return an entry point for a plugin with a deliberately slow import."""
        # The module records executions in a file rather than in its own
        # namespace: a second execution builds a new module object, so a
        # counter kept in module globals could not see the first one.
        source = (
            "import time\n"
            f"open({str(execution_log)!r}, 'a').write('x')\n"
            # Hold the import lock long enough for the other threads to reach
            # it while this one is still running the module body.
            "time.sleep(0.2)\n"
            "class Thing:\n"
            "    pass\n"
        )
        (tmp_path / f"{self.module_name}.py").write_text(source)
        monkeypatch.syspath_prepend(str(tmp_path))
        value = f"{self.module_name}:Thing"
        yield EntryPoint(name="thing", value=value, group="test.group")
        sys.modules.pop(self.module_name, None)

    @pytest.mark.concurrency
    def test_racing_loads_import_the_plugin_once(
        self, slow_entry_point, execution_log, monkeypatch, run_in_threads
    ):
        """Racing callers share one plugin object although the cache has no lock."""
        thread_count = 4
        # Every thread must be inside the loader before any may proceed, so a
        # run whose calls did not really overlap fails instead of quietly
        # asserting something weaker.
        entered = threading.Barrier(thread_count, timeout=30)

        def load():
            entered.wait()
            return slow_entry_point.load()

        plugin_mod.maybe_load_entry_point.cache_clear()
        monkeypatch.setattr(
            plugin_mod, "get_entry_point_loaders", lambda _: {"thing": load}
        )

        results = run_in_threads(
            lambda _: plugin_mod.maybe_load_entry_point("test.group", "thing"),
            count=thread_count,
        )

        # The cache misses on every thread, but the import system runs the
        # module body once, so nothing observes a duplicated or half-built
        # plugin. This is why the caches need no lock of their own.
        assert len({id(x) for x in results}) == 1
        assert execution_log.read_text() == "x"
