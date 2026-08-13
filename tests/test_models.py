"""Tests for DASCore models and related functionality."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pydantic
import pytest
from pydantic import Field, ValidationError

import dascore.models as dc_models
from dascore.core.attrs import PatchAttrs
from dascore.core.inventory import Cable, Inventory
from dascore.exceptions import InvalidModelTagError
from dascore.io.core import FiberIO
from dascore.io.sintela.core import SintelaPatchAttrs
from dascore.models import (
    DascoreBaseModel,
    DateTime64,
    FrozenDictType,
    OptionalFiniteFloat,
    TimeDelta64,
    UnitQuantity,
    registry,
    sensible_model_equals,
)
from dascore.units import Quantity
from dascore.utils import models as models_shim


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


# A floor, not a count: raise it when a format adds a class. It exists
# because FiberIO swallows a plugin's import failure as a warning, which
# would otherwise drop that format's class out of the walk below and leave
# this file green while testing one format fewer.
_ATTRS_CLASS_FLOOR = 18


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
    assert len(found) >= _ATTRS_CLASS_FLOOR, (
        f"only {len(found)} PatchAttrs classes were found; a format's plugin "
        "probably failed to import, which would silently go untested here."
    )
    return [found[name] for name in sorted(found)]


# A required field has no default to round trip, so the walk states one. A
# new required field fails the assert below rather than dropping out of it.
_REQUIRED_ATTR_VALUES = {"gauge_length": 10.0}


def _optional_float_names(cls):
    """Names of the fields on a class which hold an optional number."""
    annotation = OptionalFiniteFloat.__origin__
    return [n for n, f in cls.model_fields.items() if f.annotation == annotation]


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
        # Compared as text, not with ==, which counts any null equal to any
        # other and so cannot see a value degrade to None on the way through.
        assert out.model_dump_json() == attrs.model_dump_json()

    def test_no_field_holds_a_number_json_cannot_spell(self, attrs_class):
        """
        No field holds a number json cannot spell.

        NaN and inf write as `null` and then refuse to read back, so an
        optional number is spelled `OptionalFiniteFloat`. Checked on the
        built instance rather than the raw default, since `np.float32("nan")`
        is not a `float` and would slip past that.
        """
        instance = _minimal_attrs(attrs_class)
        for name in attrs_class.model_fields:
            value = getattr(instance, name, None)
            if isinstance(value, float | np.floating):
                assert np.isfinite(value), (
                    f"{attrs_class.__name__}.{name} holds {value}."
                )

    def test_a_non_finite_value_from_a_file_reads_as_absent(self, attrs_class):
        """
        A format which spells "unknown" as nan is read, not refused.

        Readers hand vendor header floats straight to these classes, and a
        scan swallows a ValidationError as "failed to scan", so refusing one
        would silently drop the file from a spool rather than report it.
        """
        for name in _optional_float_names(attrs_class):
            built = _minimal_attrs(attrs_class).new(**{name: np.float32("nan")})
            assert getattr(built, name) is None


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


class TestOptionalFiniteFloat:
    """How a number which may be absent is spelled."""

    class _Model(DascoreBaseModel):
        value: OptionalFiniteFloat = None

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.float32("nan")])
    def test_a_non_finite_number_is_absent(self, value):
        """A file which spells "unknown" as nan is read, not refused."""
        assert self._Model(value=value).value is None

    def test_a_number_survives(self):
        """Everything else is the number it was."""
        assert self._Model(value=1.5).value == 1.5

    def test_a_non_number_is_left_to_the_field(self):
        """
        Only numbers can be finite, so anything else passes through here.

        A string is what a yaml or json document holds, and pydantic's own
        coercion is what should read it -- or refuse it, in its own words.
        """
        assert self._Model(value="1.5").value == 1.5
        with pytest.raises(ValidationError):
            self._Model(value="not a number")


def test_optional_numbers_are_declared_across_the_formats():
    """
    The nan-to-None loop above tests nothing on a class with no such field.

    So the fields it walks are counted here: were the annotation renamed or
    the migration reverted, every one of those loops would quietly empty.
    """
    total = sum(
        len(_optional_float_names(cls)) for cls in _dascore_patch_attrs_classes()
    )
    assert total >= 25


class TestDeprecatedModelPath:
    """`dascore.utils.models` keeps working for readers which import it."""

    def test_names_resolve_to_their_new_home(self):
        """The same objects, not copies of them."""
        assert models_shim.DascoreBaseModel is DascoreBaseModel
        assert models_shim.DateTime64 is DateTime64

    def test_it_re_exports_everything_the_package_does(self):
        """
        Re-exported wholesale, so a name added later cannot go missing.

        Which is the point of the star import: a second hand-maintained
        list would drift from the first without anything noticing.
        """
        assert set(dc_models.__all__) <= set(models_shim.__all__)
        for name in models_shim.__all__:
            assert hasattr(models_shim, name), name

    def test_it_still_carries_pydantic_base_model(self):
        """It re-exported this, so something out of tree may import it."""
        assert models_shim.BaseModel is pydantic.BaseModel


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

    def test_colliding_plugin_names_stop_resolving(self, clean_registry):
        """
        A user cannot rename another package's class, so importing both works.

        What the tag may not do is quietly pick one: a document written by
        the first would then be read as the second.
        """
        _model_in("myplugin.a", "Doubled")
        with pytest.warns(UserWarning, match="claim the tag"):
            _model_in("myplugin.b", "Doubled")
        assert "myplugin:Doubled" not in registry.registered_models()
        with pytest.raises(InvalidModelTagError, match="names two classes"):
            registry.resolve_model_tag("myplugin:Doubled")

    def test_an_uppercase_package_is_a_legal_namespace(self, clean_registry):
        """
        A package name may start with a capital, and many do (PIL, Terra15).

        The tag DASCore writes must be one it can read: a namespace the
        grammar refused would make its own files unreadable.
        """
        cls = _model_in("Terra15.attrs", "Terra15Attrs")
        tag = registry.get_model_tag(cls)
        assert tag == "Terra15:Terra15Attrs"
        assert registry.resolve_model_tag(tag) is cls

    def test_a_class_which_cannot_be_named_is_not_registered(self, clean_registry):
        """
        A parametrized generic is spelled `G[int]`, which no document states.

        Writing that tag anyway would produce a file DASCore refuses to
        read, so such a class is simply not named.
        """
        cls = _model_in("myplugin.generics", "Square[int]")
        assert registry.get_model_tag(cls) is None
        assert "myplugin:Square[int]" not in registry.registered_models()

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
        # Through the function rather than the pattern: what matters is that
        # a legal tag is looked up rather than refused outright.
        registry.resolve_model_tag(tag)

    @pytest.mark.parametrize(
        "tag", ["", "9Cable", "Cable-1", ":Cable", "dascore.core.inventory.Cable", 3]
    )
    def test_illegal_tags_are_refused(self, tag):
        """A tag which is not a tag is a malformed document, not an unknown one."""
        with pytest.raises(InvalidModelTagError, match="is not a legal"):
            registry.resolve_model_tag(tag)

    def test_an_unknown_tag_falls_back_with_a_warning(self):
        """A document from an uninstalled package still reads as its base."""
        with pytest.warns(UserWarning, match="Nothing registers"):
            out = registry.resolve_tagged_model("absent:Whatever", default=PatchAttrs)
        assert out is PatchAttrs

    def test_a_resolved_tag_must_be_the_kind_asked_for(self):
        """A file naming a class of the wrong kind is refused, not built."""
        with pytest.raises(InvalidModelTagError, match="not a PatchAttrs"):
            registry.resolve_tagged_model("Cable", default=PatchAttrs)

    def test_an_unknown_tag_without_a_default_raises(self):
        """A standalone document has no other class to fall back on."""
        with pytest.raises(InvalidModelTagError, match="Nothing registers"):
            registry.resolve_tagged_model("absent:Whatever")

    def test_an_untagged_document_without_a_default_raises(self):
        """Nothing but the document says what a standalone document holds."""
        with pytest.raises(InvalidModelTagError, match="declares no"):
            registry.resolve_tagged_model(None)

    def test_an_untagged_document_takes_the_default(self):
        """A caller which names the class does not need the document to."""
        assert registry.resolve_tagged_model(None, default=PatchAttrs) is PatchAttrs

    def test_a_broken_plugin_does_not_break_resolution(self, monkeypatch):
        """A plugin which cannot be imported has no models to find."""

        def _boom():
            raise ImportError("this plugin is a stale entry point")

        monkeypatch.setattr(registry, "_plugins_swept", False)
        monkeypatch.setattr(
            registry, "get_entry_point_loaders", lambda group: {"broken": _boom}
        )
        assert registry.resolve_model_tag("absent:Whatever") is None

    @pytest.mark.concurrency
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

    def test_include_and_exclude_still_filter_fields(self):
        """
        A model serializer must not cost the caller `include`/`exclude`.

        Declaring it with `when_used="json"` reads better and silently
        breaks them: pydantic skips the wrapper in python mode, and the
        field filtering goes with it. `Patch.equals` dumps with `include`.
        """
        attrs = PatchAttrs(tag="a")
        assert set(attrs.model_dump(include={"tag"})) == {"tag"}
        assert "tag" not in attrs.model_dump(exclude={"tag"})

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

    @pytest.mark.parametrize("value", ["my_sensor", "a b c", 3])
    def test_an_attr_which_names_no_class_is_kept(self, value):
        """
        A reader's own metadata spelled like the tag is data, not a tag.

        PatchAttrs keeps extra fields, so consuming a value which names no
        class would lose it silently; a tag DASCore wrote always resolves.
        """
        attrs = PatchAttrs(**{registry.TAG_FIELD: value})
        assert attrs.model_dump()[registry.TAG_FIELD] == value

    def test_the_tag_does_not_become_an_extra_field(self):
        """PatchAttrs keeps extras, and the tag is not one of them."""
        attrs = PatchAttrs(**{registry.TAG_FIELD: "PatchAttrs"})
        assert not hasattr(attrs, registry.TAG_FIELD)

    def test_a_union_member_writes_its_own_tag(self, clean_registry):
        """
        The nine union members declare the tag as a real field.

        Pydantic must pick a class before an object exists, so their tag
        cannot be a serializer concern. The base class must leave the value
        alone rather than overwrite it with the registry's name for the
        class: the two differ for a subclass declared out of tree, and the
        closed Literal would refuse to read the registry's spelling back.
        """
        dumped = json.loads(Cable(resource_id="c1").model_dump_json())
        assert dumped[registry.TAG_FIELD] == "Cable"
        # A plugin's subclass of a union member: tag and Literal disagree.
        plugin_cable = type("PluginCable", (Cable,), {"__module__": "myplugin.cables"})
        assert registry.get_model_tag(plugin_cable) == "myplugin:PluginCable"
        sub_dump = json.loads(plugin_cable(resource_id="c1").model_dump_json())
        assert sub_dump[registry.TAG_FIELD] == "Cable"
        # Which is what keeps the document readable at all.
        assert isinstance(Cable(**sub_dump), Cable)

    def test_a_union_member_states_its_tag_under_exclude_defaults(self):
        """
        The tag is defaulted, so a dump flag would otherwise drop it.

        The serializer puts it back: a document which cannot say which union
        member it holds cannot be read at all.
        """
        inventory = Inventory(resources={"c1": Cable(resource_id="c1")})
        dumped = inventory.model_dump(mode="json", exclude_defaults=True)
        assert dumped["resources"]["c1"][registry.TAG_FIELD] == "Cable"
        assert Inventory(**dumped) == inventory
