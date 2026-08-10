"""Tests for runtime configuration helpers."""

from __future__ import annotations

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

import dascore as dc
from dascore import config as dascore_config
from dascore.config import (
    DascoreConfig,
    config_attr,
    config_context,
    get_config,
    reset_config,
    set_config,
)


def _worker_read_float_precision(patch):
    """Return the runtime float precision a worker task observes."""
    return dc.get_config().display_float_precision


class TestSetConfig:
    """Tests for the permanent, process-wide config base."""

    def setup_method(self):
        """Remember the active config so each test restores it exactly."""
        self._saved = get_config()

    def teardown_method(self):
        """Restore the pre-test config (not hard defaults, which would drop
        session-level overrides such as the temporary index-map dir).
        """
        set_config(self._saved)

    def test_accepts_direct_config_instance(self):
        """Passing a validated config installs it permanently."""
        new = DascoreConfig(debug=True)
        set_config(new)
        assert get_config().debug is True

    def test_persists_until_reset(self):
        """A permanent set stays active (it is not auto-restored)."""
        set_config(display_float_precision=5)
        assert get_config().display_float_precision == 5

    def test_reset_config_returns_to_defaults(self):
        """reset_config restores the process-wide base to defaults."""
        set_config(display_float_precision=5)
        reset_config()
        assert get_config().display_float_precision == 3

    def test_invalid_new_config_raises(self):
        """Arbitrary objects should not be accepted as configs."""
        with pytest.raises(TypeError, match="DascoreConfig"):
            set_config(object())

    def test_new_config_and_kwargs_raise(self):
        """Direct config replacement should not mix with overrides."""
        with pytest.raises(ValueError, match="new_config"):
            set_config(DascoreConfig(), debug=True)

    def test_invalid_patch_history_raises(self):
        """Unsupported patch history policies should be rejected."""
        with pytest.raises(ValueError, match="patch_history"):
            set_config(patch_history="verbose-ish")

    def test_sampling_group_tolerance_must_be_positive(self):
        """Non-positive tolerances are rejected."""
        with pytest.raises(ValueError, match="sampling_group_tolerance"):
            set_config(sampling_group_tolerance=0)

    def test_unknown_field_raises(self):
        """A misspelled override raises rather than being silently ignored."""
        with pytest.raises(ValueError, match="dispplay_float_precision"):
            set_config(dispplay_float_precision=5)

    def test_fork_handler_replaces_held_lock(self):
        """A lock held at fork time is replaced so the child cannot deadlock."""
        old_lock = dascore_config._GLOBAL_CONFIG_LOCK
        with old_lock:
            dascore_config._reinit_config_lock()
            new_lock = dascore_config._GLOBAL_CONFIG_LOCK
        assert new_lock is not old_lock
        assert not new_lock.locked()

    @pytest.mark.concurrency
    def test_visible_from_new_thread(self):
        """A permanent set is visible from threads launched afterward."""
        set_config(display_float_precision=9)
        seen = {}

        def worker():
            seen["value"] = get_config().display_float_precision

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert seen["value"] == 9


class TestConfigContext:
    """Tests for scoped, thread/task-local config overrides."""

    def test_scoped_override_restored_on_exit(self):
        """A scoped override applies inside the block and restores after."""
        previous = get_config()
        with config_context(display_float_precision=6):
            assert get_config().display_float_precision == 6
        assert get_config() == previous

    def test_accepts_direct_config_instance(self):
        """A full config can be installed for the scope of a block."""
        previous = get_config()
        new = DascoreConfig(display_float_precision=6)
        with config_context(new):
            assert get_config().display_float_precision == 6
        assert get_config() == previous

    def test_overrides_stack(self):
        """A nested override builds on the enclosing one and reverts alone."""
        with config_context(display_float_precision=6):
            with config_context(display_array_threshold=7):
                config = get_config()
                assert config.display_array_threshold == 7
                assert config.display_float_precision == 6
            # The inner override reverts; the outer one remains.
            assert get_config().display_array_threshold == 100
            assert get_config().display_float_precision == 6

    def test_remote_cache_controls_can_be_overridden(self):
        """Remote cache policy config round-trips through config_context."""
        previous = get_config()
        with config_context(
            allow_remote_cache=False,
            allow_remote_cache_for_metadata=True,
            warn_on_remote_cache=False,
            allow_dasdae_format_unpickle=True,
        ):
            config = get_config()
            assert config.allow_remote_cache is False
            assert config.allow_remote_cache_for_metadata is True
            assert config.warn_on_remote_cache is False
            assert config.allow_dasdae_format_unpickle is True
        assert get_config() == previous

    def test_groupby_attrs_coerced_to_tuple(self):
        """List inputs coerce to the immutable tuple form."""
        with config_context(groupby_attrs=["tag"]):
            assert get_config().groupby_attrs == ("tag",)

    def test_invalid_new_config_raises(self):
        """Validation is shared with set_config."""
        with pytest.raises(TypeError, match="DascoreConfig"):
            with config_context(object()):
                pass

    def test_config_attr_reflects_runtime_config(self):
        """Config descriptors resolve against the active runtime config."""

        class _UsesConfig:
            value = config_attr("display_float_precision")

        assert _UsesConfig().value == get_config().display_float_precision
        with config_context(display_float_precision=7):
            assert _UsesConfig().value == 7

    @pytest.mark.concurrency
    def test_concurrent_overrides_are_isolated(self):
        """Two threads with different scoped overrides never stomp each other."""
        barrier = threading.Barrier(2)
        results = {}

        def worker(name, value):
            with config_context(display_float_precision=value):
                # Force overlap: both threads sit inside their context at once.
                barrier.wait()
                results[name] = get_config().display_float_precision

        threads = [
            threading.Thread(target=worker, args=("a", 1)),
            threading.Thread(target=worker, args=("b", 2)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == {"a": 1, "b": 2}


class TestConfigDefaults:
    """Sanity checks on default field values."""

    def test_groupby_attrs_default(self):
        """The default group attrs are the conventional identity set."""
        expected = (
            "data_source_id",
            "data_type",
            "data_category",
            "tag",
            "network",
            "station",
        )
        assert get_config().groupby_attrs == expected

    def test_sampling_group_tolerance_default(self):
        """The default sampling group tolerance is 5%."""
        assert get_config().sampling_group_tolerance == 0.05


class TestMapConfigBinding:
    """Spool.map applies the config active when map() was called."""

    @pytest.mark.concurrency
    def test_thread_pool_sees_call_time_config(self):
        """A thread-pool map sees the caller's scoped override."""
        spool = dc.get_example_spool()
        with config_context(display_float_precision=8):
            with ThreadPoolExecutor(2) as executor:
                out = spool.map(_worker_read_float_precision, client=executor)
        assert set(out) == {8}

    @pytest.mark.concurrency
    def test_process_pool_sees_call_time_config(self):
        """A process-pool map sees the caller's scoped override too."""
        spool = dc.get_example_spool()
        with config_context(display_float_precision=8):
            with ProcessPoolExecutor(2) as executor:
                out = spool.map(_worker_read_float_precision, client=executor)
        assert set(out) == {8}
