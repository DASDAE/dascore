"""Tests for DASCore models and related functionality."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
from pydantic import Field, ValidationError

from dascore.core.attrs import PatchAttrs
from dascore.core.inventory import Cable
from dascore.exceptions import InvalidModelTagError
from dascore.io.core import FiberIO
from dascore.io.sintela.core import SintelaPatchAttrs
from dascore.models import (
    DascoreBaseModel,
    DateTime64,
    FrozenDictType,
    TimeDelta64,
    UnitQuantity,
    registry,
    sensible_model_equals,
)
from dascore.units import Quantity


class _TestModel(DascoreBaseModel):
    array: np.ndarray | None = None
    _private: int = 0
    some_str: str = "10"


class _Inner(DascoreBaseModel):
    """A nested model, which a document must also name."""

    value: int = 1


class _Outer(DascoreBaseModel):
    """A model holding another."""

    inner: _Inner = Field(default_factory=_Inner)


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


def _dascore_patch_attrs_classes():
    """Every PatchAttrs class DASCore itself declares, base included."""
    # The subclasses only exist once their io modules are imported, and other
    # test modules register their own subclasses globally on import, so the
    # walk both forces the load and keeps to DASCore's own classes.
    FiberIO.manager.load_plugins()
    found: dict[str, type[PatchAttrs]] = {}
    stack = [PatchAttrs]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__.startswith("dascore."):
            found[cls.__name__] = cls
    return [found[name] for name in sorted(found)]


# A required field has no default to round trip, so the walk states one. A
# new required field fails the assert below rather than dropping out of it.
_REQUIRED_ATTR_VALUES = {"gauge_length": 10.0}


def _minimal_attrs(cls):
    """Build the emptiest legal instance of a PatchAttrs class."""
    required = {name for name, f in cls.model_fields.items() if f.is_required()}
    assert not (unknown := required - set(_REQUIRED_ATTR_VALUES)), (
        f"{cls.__name__} requires {sorted(unknown)}, which "
        "_REQUIRED_ATTR_VALUES does not state a value for."
    )
    return cls(**{name: _REQUIRED_ATTR_VALUES[name] for name in required})


@pytest.mark.parametrize(
    "attrs_class",
    _dascore_patch_attrs_classes(),
    ids=lambda cls: cls.__name__,
)
class TestPatchAttrsSerialization:
    """
    Every PatchAttrs class must survive a text round trip.

    Parametrized over the class walk rather than a list, so a format added
    later is covered without touching this file.
    """

    def test_json_round_trip(self, attrs_class):
        """A defaulted instance reconstructs from its own json."""
        attrs = _minimal_attrs(attrs_class)
        out = attrs_class.model_validate_json(attrs.model_dump_json())
        assert out == attrs

    def test_no_field_defaults_to_a_non_finite_number(self, attrs_class):
        """
        Nan and inf have no json spelling, so a float defaulting to one
        writes null and then refuses to read it back. Optional numbers are
        spelled `FiniteFloat | None`.
        """
        for name, field in attrs_class.model_fields.items():
            default = field.get_default(call_default_factory=True)
            if isinstance(default, float):
                assert np.isfinite(default), (
                    f"{attrs_class.__name__}.{name} defaults to {default}."
                )


@pytest.fixture
def clean_registry():
    """Undo whatever a test registers, so the real registry is untouched."""
    before = registry.registered_models()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(before)


def _model_in(module: str, name: str = "Square", base=DascoreBaseModel):
    """Declare a model as though it lived in another package."""
    return type(name, (base,), {"__module__": module, "__qualname__": name})


class TestModelTagRegistry:
    """A tag names one class, and the registry is what resolves it."""

    def test_dascore_models_register_bare(self):
        """A bare name means dascore, which keeps files hand-authorable."""
        assert registry.registered_models()["PatchAttrs"] is PatchAttrs
        assert registry.get_model_tag(PatchAttrs) == "PatchAttrs"

    def test_out_of_tree_models_are_namespaced(self, clean_registry):
        """A plugin's namespace is derived, so it needs no ceremony."""
        cls = _model_in("myplugin.shapes")
        assert registry.get_model_tag(cls) == "myplugin:Square"
        assert registry.registered_models()["myplugin:Square"] is cls

    def test_colliding_dascore_names_raise(self, clean_registry):
        """Two of our own classes may not claim one tag; this is the pin."""
        first = _model_in("dascore.somewhere", "Doubled")
        with pytest.raises(InvalidModelTagError, match="claim the tag"):
            _model_in("dascore.elsewhere", "Doubled")
        # The collision does not replace what was there before it.
        assert registry.registered_models()["Doubled"] is first

    def test_colliding_plugin_names_warn(self, clean_registry):
        """A user cannot rename another package's class, so this only warns."""
        _model_in("myplugin.a", "Doubled")
        with pytest.warns(UserWarning, match="claim the tag"):
            second = _model_in("myplugin.b", "Doubled")
        assert registry.registered_models()["myplugin:Doubled"] is second

    def test_a_class_declared_in_a_function_is_not_registered(self, clean_registry):
        """Nothing can resolve a name which exists only while a call runs."""

        class Local(DascoreBaseModel):
            """A model which cannot be addressed from a document."""

        assert "Local" not in registry.registered_models()

    def test_a_reimported_module_replaces_its_own_entry(self, clean_registry):
        """Re-importing a module is not two classes claiming one tag."""
        _model_in("myplugin.shapes")
        second = _model_in("myplugin.shapes")
        assert registry.registered_models()["myplugin:Square"] is second

    @pytest.mark.parametrize(
        "tag", ["Cable", "myplugin:Square", "my_plugin.sub:Square", "Cable-0.0.1"]
    )
    def test_legal_tags(self, tag):
        """The grammar takes a name, a namespace and room for a version."""
        assert registry.TAG_PATTERN.match(tag)

    @pytest.mark.parametrize(
        "tag", ["", "9Cable", "Cable-1", ":Cable", "dascore.core.inventory.Cable", 3]
    )
    def test_illegal_tags_are_refused(self, tag):
        """A tag which is not a tag is a malformed document, not an unknown one."""
        with pytest.raises(InvalidModelTagError, match="is not a legal"):
            registry.resolve_model_tag(tag)

    def test_an_unknown_tag_falls_back_with_a_warning(self):
        """A document from an uninstalled package still reads as its base."""
        data = {registry.TAG_FIELD: "absent:Whatever"}
        with pytest.warns(UserWarning, match="Nothing registers"):
            out = registry.resolve_tagged_model(data, default=PatchAttrs)
        assert out is PatchAttrs

    def test_an_unknown_tag_without_a_default_raises(self):
        """A standalone document has no other class to fall back on."""
        data = {registry.TAG_FIELD: "absent:Whatever"}
        with pytest.raises(InvalidModelTagError, match="Nothing registers"):
            registry.resolve_tagged_model(data)

    def test_an_untagged_document_without_a_default_raises(self):
        """Nothing but the document says what a standalone document holds."""
        with pytest.raises(InvalidModelTagError, match="declares no"):
            registry.resolve_tagged_model({"a": 1})

    def test_an_untagged_document_takes_the_default(self):
        """A caller which names the class does not need the document to."""
        assert registry.resolve_tagged_model({}, default=PatchAttrs) is PatchAttrs

    def test_a_model_in_an_unimported_module_is_found(self):
        """
        A format's models only exist once its module is imported.

        Run in a fresh interpreter because the io modules are imported by
        the time any other test runs, which is exactly what hides this.
        """
        code = (
            "import dascore.models.registry as r\n"
            "assert 'ODH4PatchAttrs' not in r.registered_models(), 'already there'\n"
            "assert r.resolve_model_tag('ODH4PatchAttrs') is not None, 'not found'\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr


class TestObjectTypeSerialization:
    """Every model names its class in a text document, and nowhere else."""

    def test_json_states_the_class(self):
        """A document says what it holds."""
        dumped = json.loads(PatchAttrs(tag="a").model_dump_json())
        assert dumped[registry.TAG_FIELD] == "PatchAttrs"

    def test_a_python_dump_is_untagged(self):
        """
        The tag is not a field, and python dumps are not documents.

        Equality compares them, `new` reconstructs from them and the spool
        index ingests them; a key which is not a field belongs in none.
        """
        assert registry.TAG_FIELD not in PatchAttrs().model_dump()
        assert registry.TAG_FIELD not in PatchAttrs().new(tag="a").model_dump()

    def test_equality_is_unaffected(self, clean_registry):
        """Two classes' instances are still told apart by their fields."""
        assert PatchAttrs(tag="a") == PatchAttrs(tag="a")
        assert PatchAttrs(tag="a") != PatchAttrs(tag="b")

    def test_nested_models_state_their_class(self):
        """Universal, so a nested object can be read on its own later."""
        dumped = json.loads(_Outer(inner=_Inner()).model_dump_json())
        assert dumped[registry.TAG_FIELD] == "tests:_Outer"
        assert dumped["inner"][registry.TAG_FIELD] == "tests:_Inner"

    def test_a_subclass_reads_back_through_its_base(self):
        """The document holds everything the base declares, so this is fine."""
        attrs = SintelaPatchAttrs(gauge_length=10.0)
        out = PatchAttrs.model_validate_json(attrs.model_dump_json())
        assert out.gauge_length == 10.0

    def test_a_foreign_tag_is_refused(self):
        """A document which names another class is misfiled, not reinterpreted."""
        data = {registry.TAG_FIELD: "Cable"}
        with pytest.raises(ValidationError, match="cannot be read as"):
            PatchAttrs(**data)

    def test_an_unknown_tag_is_accepted(self):
        """The caller named the class, so there is nothing to disagree with."""
        attrs = PatchAttrs(**{registry.TAG_FIELD: "absent:Whatever"})
        assert isinstance(attrs, PatchAttrs)

    def test_the_tag_does_not_become_an_extra_field(self):
        """PatchAttrs keeps extras, and the tag is not one of them."""
        attrs = PatchAttrs(**{registry.TAG_FIELD: "PatchAttrs"})
        assert not hasattr(attrs, registry.TAG_FIELD)

    def test_a_union_member_states_it_once(self):
        """
        The five resource models declare the tag as a real field.

        Pydantic must pick a class before an object exists, so their tag
        cannot be a serializer concern; the base class leaves them alone
        rather than writing a second copy of what they already wrote.
        """
        text = Cable(resource_id="c1").model_dump_json()
        assert json.loads(text)[registry.TAG_FIELD] == "Cable"
        assert text.count(f'"{registry.TAG_FIELD}"') == 1
