"""Tests for namespace utilities."""

from __future__ import annotations

import warnings
from typing import ClassVar

import pandas as pd
import pytest

import dascore.utils.namespace as ns_module
from dascore.exceptions import DASCorePluginError
from dascore.utils.namespace import (
    NamespaceOwner,
    _load_plugin_registry,
    _MethodNameSpace,
)


class ParentClass(NamespaceOwner):
    """A test parent class."""

    _namespace_entry_point_group = "dascore.ParentClass"
    _namespace_attr_errors: ClassVar[dict[str, str]] = {
        "test_error": "test_error emitted"
    }

    def unaltered_function(self, value):
        """A simple function with no changes."""
        return value


class ParentClassNamespace(_MethodNameSpace):
    """A test child class."""

    entry_point_group = "dascore.ParentClass"


class Namespace1(ParentClassNamespace):
    """Method namespace subclass."""

    name = "bob"

    def return_self(self):
        """First func."""
        return self


class Namespace2(ParentClassNamespace):
    """Method namespace subclass."""

    name = "bill"

    def func1(self):
        """First func."""
        return self.name


class TestNamespace:
    """Tests for namespace mechanism."""

    def test_discoverable(self):
        """The Parent class should have the namespaces available."""
        inst = ParentClass()
        bob = inst.bob
        assert isinstance(bob, ParentClassNamespace)

    def test_discoverable_is_cached(self):
        """The same namespace instance should be returned on repeated access."""
        inst = ParentClass()
        assert inst.bob is inst.bob

    def test_parent_type_bound(self):
        """Ensure the parent type is bound the instances."""
        inst = ParentClass()
        out = inst.bob.return_self()
        assert out is inst

    def test_registered_namespaces(self):
        """Ensure class-level namespace discovery uses the entry point group."""
        out = ParentClass.get_registered_namespaces()
        assert set(out) >= {"bill", "bob"}
        assert out["bob"] is Namespace1
        assert out["bill"] is Namespace2

    def test_registered_namespaces_are_immutable(self):
        """Ensure registry discovery returns an immutable snapshot."""
        out = ParentClass.get_registered_namespaces()
        with pytest.raises(TypeError):
            out["bob"] = Namespace2

    def test_unregistered_namespaces(self):
        """Ensure a namespace not registered fails with the default message."""
        inst = ParentClass()
        msg = "ParentClass has no attribute 'not_an_attr'"
        with pytest.raises(AttributeError, match=msg):
            inst.not_an_attr

    def test_public_method_added_after_class_creation_is_wrapped(self):
        """Ensure public callables set later are rebound to the host object."""
        dynamic_namespace = type("DynamicNamespace", (ParentClassNamespace,), {})
        dynamic_namespace.new_method = lambda self: self
        inst = ParentClass()
        namespace = dynamic_namespace(inst)
        assert namespace.new_method() is inst

    def test_private_method_added_after_class_creation_is_not_wrapped(self):
        """Ensure private callables set later stay bound to the namespace."""
        dynamic_namespace = type("DynamicNamespacePrivate", (ParentClassNamespace,), {})
        dynamic_namespace._private_method = lambda self: self
        inst = ParentClass()
        namespace = dynamic_namespace(inst)
        assert namespace._private_method() is namespace

    def test_registry_is_isolated_by_entry_point_group(self):
        """Ensure namespaces for one group do not appear in another."""

        class OtherParent(NamespaceOwner):
            _namespace_entry_point_group = "dascore.other_parent"

        class OtherParentNamespace(_MethodNameSpace):
            entry_point_group = "dascore.other_parent"

        class OtherNamespace(OtherParentNamespace):
            name = "other"

        out = OtherParent.get_registered_namespaces()
        assert out["other"] is OtherNamespace
        assert "other" not in ParentClass.get_registered_namespaces()

    def test_unknown_group_does_not_create_registry_bucket(self):
        """Ensure looking up an unknown group does not mutate the registry."""

        class UnknownParent(NamespaceOwner):
            _namespace_entry_point_group = "dascore.unknown_parent"

        assert "dascore.unknown_parent" not in _MethodNameSpace._registry
        out = UnknownParent.get_registered_namespaces()
        assert not out
        assert "dascore.unknown_parent" not in _MethodNameSpace._registry

    def test_namespace_collision_warns_and_last_wins(self):
        """Ensure duplicate names warn and the later namespace replaces the first."""

        class CollisionBase(_MethodNameSpace):
            entry_point_group = "dascore.collision_test"

        class FirstNamespace(CollisionBase):
            name = "dup"

        with pytest.warns(UserWarning, match="Namespace collision"):

            class SecondNamespace(CollisionBase):
                name = "dup"

        out = CollisionBase._registry["dascore.collision_test"]["dup"]
        assert out is SecondNamespace
        assert out is not FirstNamespace

    def test_distinct_namespace_names_do_not_warn(self):
        """Ensure unique names within a group register without warnings."""

        class DistinctBase(_MethodNameSpace):
            entry_point_group = "dascore.distinct_test"

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")

            class FirstNamespace(DistinctBase):
                name = "first"

            class SecondNamespace(DistinctBase):
                name = "second"

        assert not record
        out = DistinctBase._registry["dascore.distinct_test"]
        assert out["first"] is FirstNamespace
        assert out["second"] is SecondNamespace

    def test_init_subclass_is_cooperative(self):
        """Ensure parent __init_subclass__ hooks still run."""

        class HookMixin:
            hook_called = False

            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.hook_called = True

        class CooperativeBase(HookMixin, _MethodNameSpace):
            entry_point_group = "dascore.cooperative_test"

        class CooperativeNamespace(CooperativeBase):
            name = "cooperative"

        assert CooperativeNamespace.hook_called
        out = CooperativeBase._registry["dascore.cooperative_test"]
        assert out["cooperative"] is CooperativeNamespace

    def test_custom_attr_error_message(self):
        """Ensure _namespace_attr_errors messages are raised verbatim."""
        inst = ParentClass()
        with pytest.raises(AttributeError, match="test_error emitted"):
            inst.test_error


@pytest.fixture(autouse=False)
def _clear_registry_cache():
    """Clear the _load_plugin_registry LRU cache before and after each test."""
    _load_plugin_registry.cache_clear()
    yield
    _load_plugin_registry.cache_clear()


class TestPluginRegistry:
    """Tests for the plugin registry CSV lookup."""

    def test_none_group_returns_empty(self, _clear_registry_cache):
        """_load_plugin_registry returns empty dict when group is None."""
        result = _load_plugin_registry(None)
        assert result == {}

    def test_nonexistent_csv_returns_empty(
        self, _clear_registry_cache, monkeypatch, tmp_path
    ):
        """_load_plugin_registry returns empty dict when CSV does not exist."""
        monkeypatch.setattr(ns_module, "_PLUGIN_REGISTRY_DIR", tmp_path)
        result = _load_plugin_registry("dascore.nofile_namespace")
        assert result == {}

    def test_valid_csv_returns_mapping(
        self, _clear_registry_cache, monkeypatch, tmp_path
    ):
        """_load_plugin_registry parses CSV into a namespace → (pkg, url) mapping."""
        csv_path = tmp_path / "myplugin.csv"
        df = pd.DataFrame(
            {
                "package_name": ["mypkg"],
                "package_url": ["https://example.com/mypkg"],
                "namespace": ["myns"],
            }
        )
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(ns_module, "_PLUGIN_REGISTRY_DIR", tmp_path)
        result = _load_plugin_registry("dascore.myplugin_namespace")
        assert result == {"myns": ("mypkg", "https://example.com/mypkg")}

    def test_multiple_rows_all_returned(
        self, _clear_registry_cache, monkeypatch, tmp_path
    ):
        """_load_plugin_registry returns all rows when CSV has multiple entries."""
        csv_path = tmp_path / "multi.csv"
        df = pd.DataFrame(
            {
                "package_name": ["pkgA", "pkgB"],
                "package_url": ["https://a.com", "https://b.com"],
                "namespace": ["nsA", "nsB"],
            }
        )
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(ns_module, "_PLUGIN_REGISTRY_DIR", tmp_path)
        result = _load_plugin_registry("dascore.multi_namespace")
        assert result == {
            "nsA": ("pkgA", "https://a.com"),
            "nsB": ("pkgB", "https://b.com"),
        }

    def test_getattr_plugin_hit_raises_helpful_error(
        self, _clear_registry_cache, monkeypatch, tmp_path
    ):
        """__getattr__ raises AttributeError with package info for known plugins."""
        csv_path = tmp_path / "ParentClass.csv"
        df = pd.DataFrame(
            {
                "package_name": ["coolpkg"],
                "package_url": ["https://example.com/coolpkg"],
                "namespace": ["cool_ns"],
            }
        )
        df.to_csv(csv_path, index=False)
        monkeypatch.setattr(ns_module, "_PLUGIN_REGISTRY_DIR", tmp_path)
        inst = ParentClass()
        msg = (
            "ParentClass has a registered namespace of 'cool_ns' "
            "provided by 'coolpkg' but it is not installed. "
            "Install it from: https://example.com/coolpkg"
        )
        with pytest.raises(DASCorePluginError, match="coolpkg"):
            inst.cool_ns
        # Verify the full message structure.
        with pytest.raises(DASCorePluginError) as exc_info:
            inst.cool_ns
        assert str(exc_info.value) == msg

    def test_getattr_unknown_attr_raises_default_error(
        self, _clear_registry_cache, monkeypatch, tmp_path
    ):
        """__getattr__ raises default AttributeError when attr not in CSV."""
        monkeypatch.setattr(ns_module, "_PLUGIN_REGISTRY_DIR", tmp_path)
        inst = ParentClass()
        with pytest.raises(
            AttributeError, match="ParentClass has no attribute 'totally_unknown'"
        ):
            inst.totally_unknown
