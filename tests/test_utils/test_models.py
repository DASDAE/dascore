"""Tests for DASCore models and related functionality."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from pydantic import Field, ValidationError

from dascore.units import Quantity
from dascore.utils.models import (
    DascoreBaseModel,
    DateTime64,
    FrozenDictType,
    TimeDelta64,
    UnitQuantity,
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


class _UnitModel(DascoreBaseModel):
    """A model carrying units."""

    units: UnitQuantity | None = None


class TestUnitQuantity:
    """Units name a single unit, so they stay scalar."""

    @pytest.mark.parametrize(
        "spec", ["m/s", "1/s", "Hz", "2*degC", "2.5*degC", "10 m/s", 1.0, None, ""]
    )
    def test_real_spellings_accepted(self, spec):
        """Every real unit spelling has a scalar magnitude, so it hashes."""
        assert isinstance(hash(_UnitModel(units=spec)), int)

    def test_sequence_refused(self):
        """A sequence makes pint build a mutable, unhashable magnitude."""
        with pytest.raises(ValidationError, match="single unit"):
            _UnitModel(units=(1.0, 2.0))

    @pytest.mark.parametrize("magnitude", [np.array([1.0, 2.0]), np.array(1.0)])
    def test_array_magnitude_refused(self, magnitude):
        """A quantity built by hand is held to the same rule."""
        # A zero dimensional array is as writable as any other, so measuring
        # how many dimensions it has would let this one through.
        with pytest.raises(ValidationError, match="single unit"):
            _UnitModel(units=Quantity(magnitude, "m"))

    def test_serialized_units_validate_again(self):
        """What the serializer writes has to survive being read back."""
        model = _UnitModel(units="m/s")
        assert _UnitModel(**model.model_dump()) == model


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

        class Forward(DascoreBaseModel):
            a: int = 1
            b: int = 2

        class Reversed(DascoreBaseModel):
            b: int = 2
            a: int = 1

        first, second = Forward(), Reversed()
        assert first == second
        assert hash(first) == hash(second)

    def test_unhashable_field_still_refuses(self):
        """A model holding an array is unhashable, and says so."""
        with pytest.raises(TypeError, match="unhashable"):
            hash(_TestModel(array=np.arange(10)))
