"""Tests for DASCore models and related functionality."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from pydantic import Field

from dascore.utils.models import (
    DascoreBaseModel,
    DateTime64,
    FrozenDictType,
    TimeDelta64,
    sensible_model_equals,
)


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
        """Ensure when arrays aren't equal models aren't."""
        mod1 = _TestModel(array=np.arange(10))
        mod2 = _TestModel(array=np.arange(10) + 10)
        assert not sensible_model_equals(mod1, mod2)

    def test_private(self):
        """When private attrs aren't equal the models should still be."""
        mod1 = _TestModel(_private=1)
        mod2 = _TestModel(_private=2)
        assert sensible_model_equals(mod1, mod2)

    def test_private_disjoint(self):
        """Private attrs not shared should not affect equality."""
        mod1 = _TestModel(_private_1=1)
        mod2 = _TestModel(_private_2=2)
        assert sensible_model_equals(mod1, mod2)

    def test_new(self):
        """Ensure a new model can b e created."""
        mod = _TestModel(some_str="test")
        new = mod.new(some_str="bob")
        assert new.some_str == "bob"
        assert new is not mod


class _NullModel(DascoreBaseModel):
    """A model holding every flavor of null a field can carry."""

    time: DateTime64 = np.datetime64("NaT", "ns")
    duration: TimeDelta64 = np.timedelta64("NaT", "ns")
    number: float = np.nan
    mapping: FrozenDictType[str, float] = Field(default_factory=dict)


class TestModelHash:
    """Hashing agrees with equality, so models work as dict keys."""

    def test_unset_models_hash_equally(self):
        """Nulls compare equal to nulls, so they must hash alike."""
        # Each null is a fresh object, and nan and NaT both hash by identity.
        first, second = _NullModel(), _NullModel()
        assert first == second
        assert hash(first) == hash(second)
        assert {first: "found"}[second] == "found"

    def test_null_from_any_input_hashes_alike(self):
        """None and the string 'NaT' reach the same null."""
        assert hash(_NullModel(time=None)) == hash(_NullModel(time="NaT"))

    def test_nulls_inside_a_mapping_hash_alike(self):
        """A null nested in a frozen mapping is normalized too."""
        first = _NullModel(mapping={"gain": float("nan")})
        second = _NullModel(mapping={"gain": float("nan")})
        assert first == second
        assert hash(first) == hash(second)

    def test_mapping_order_does_not_change_the_hash(self):
        """Mappings compare without regard to order, so they hash that way."""
        first = _NullModel(mapping={"a": 1.0, "b": 2.0})
        second = _NullModel(mapping={"b": 2.0, "a": 1.0})
        assert first == second
        assert hash(first) == hash(second)

    def test_hash_survives_a_pickle_round_trip(self):
        """Unpickling skips validation, so the hash cannot depend on it."""
        model = _NullModel()
        loaded = pickle.loads(pickle.dumps(model))
        assert loaded == model
        assert hash(loaded) == hash(model)

    def test_differing_models_hash_apart(self):
        """A real value is not confused with the null it replaces."""
        assert hash(_NullModel(number=1.0)) != hash(_NullModel())

    def test_declaration_order_does_not_change_the_hash(self):
        """Equality compares field names, so declaration order cannot matter."""

        class Ab(DascoreBaseModel):
            a: int = 1
            b: int = 2

        class Ba(DascoreBaseModel):
            b: int = 2
            a: int = 1

        first, second = Ab(), Ba()
        assert first == second
        assert hash(first) == hash(second)

    def test_unhashable_field_still_refuses(self):
        """A model holding an array is unhashable, and says so."""
        with pytest.raises(TypeError, match="unhashable"):
            hash(_TestModel(array=np.arange(10)))
