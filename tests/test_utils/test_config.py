"""Tests for runtime configuration helpers."""

from __future__ import annotations

import pytest

from dascore.config import (
    DascoreConfig,
    config_attr,
    get_config,
    reset_config,
    set_config,
)


class TestSetConfig:
    """Tests for updating runtime configuration."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_config()

    def test_accepts_direct_config_instance(self):
        """Passing a validated config should install it."""
        previous = get_config()
        new = DascoreConfig(debug=True)
        with set_config(new):
            assert get_config().debug is True
        assert get_config() == previous

    def test_invalid_new_config_raises(self):
        """Arbitrary objects should not be accepted as configs."""
        with pytest.raises(TypeError, match="DascoreConfig"):
            set_config(object())

    def test_new_config_and_kwargs_raise(self):
        """Direct config replacement should not mix with overrides."""
        with pytest.raises(ValueError, match="new_config"):
            set_config(DascoreConfig(), debug=True)

    def test_remote_cache_controls_can_be_overridden(self):
        """Remote cache policy config should round-trip through set_config."""
        previous = get_config()
        with set_config(
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

    def test_config_attr_reflects_runtime_config(self):
        """Config descriptors should resolve against the active runtime config."""

        class _UsesConfig:
            value = config_attr("display_float_precision")

        assert _UsesConfig().value == get_config().display_float_precision
        with set_config(display_float_precision=7):
            assert _UsesConfig().value == 7
