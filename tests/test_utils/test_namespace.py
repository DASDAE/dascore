"""Tests for namespace utilities."""

from __future__ import annotations

import warnings
from typing import ClassVar

import pytest

from dascore.utils.namespace import (
    NamespaceOwner,
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

    def func1(self, expected_type):
        """First func."""
        return self.name


class TestNamespace:
    """Tests for namespace mechanism."""

    def test_discoverable(self):
        """The Parent class should have the namespaces available."""
        inst = ParentClass()
        bob = getattr(inst, "bob")
        assert isinstance(bob, ParentClassNamespace)

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
