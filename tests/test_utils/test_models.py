"""Tests for DASCore models and related functionality."""

from __future__ import annotations

import numpy as np

from dascore.utils.models import DascoreBaseModel, sensible_model_equals


class _TestModel(DascoreBaseModel):
    array: np.ndarray | None = None
    _private: int = 0
    some_str: str = "10"


class TestModelEquals:
    """Tests for seeing if models/dicts are equal."""

    def test_empty(self):
        """Empty dicts should be equal."""
        assert sensible_model_equals({}, {})

    def test_arrays_not_equal(self):
        """Ensure when arrays aren't equal models arent."""
        mod1 = _TestModel(array=np.arange(10))
        mod2 = _TestModel(array=np.arange(10) + 10)
        assert not sensible_model_equals(mod1, mod2)

    def test_private(self):
        """When private attrs arent equal the models should still be."""
        mod1 = _TestModel(_private=1)
        mod2 = _TestModel(_private=2)
        assert sensible_model_equals(mod1, mod2)

    def test_new(self):
        """Ensure a new model can b e created."""
        mod = _TestModel(some_str="test")
        new = mod.new(some_str="bob")
        assert new.some_str == "bob"
        assert new is not mod
