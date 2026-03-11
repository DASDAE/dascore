"""Tests for namespace utilities."""

from __future__ import annotations

import pytest

from dascore.core.patch import Patch
from dascore.core.spool import BaseSpool
from dascore.exceptions import ParameterError
from dascore.utils import namespace as namespace_mod
from dascore.utils import plugins as plugin_mod
from dascore.utils.namespace import MethodNameSpace, PatchNameSpace, SpoolNameSpace


class ParentClass:
    """A test parent class."""

    @property
    def namespace(self):
        """Your run-o-the-mill namespace."""
        return MNS(self)


class MNS(MethodNameSpace):
    """Method namespace subclass."""

    def func1(self, expected_type):
        """First func."""
        return isinstance(self, expected_type)


class TestNamespaceClass:
    """Tests for namespace class."""

    def test_parent_self_passed_to_namespace(self):
        """Ensure the parent of namespace gets passed to self."""
        pc = ParentClass()
        assert pc.namespace.func1(ParentClass)

    def test_assign_adhoc_method(self):
        """Ensure methods added after class definition still work."""

        def new_method(self, expected_type):
            return isinstance(self, expected_type)

        MNS.new_method = new_method
        pc = ParentClass()
        assert pc.namespace.new_method(ParentClass)


class TestDescriptorBehavior:
    """Tests for descriptor edge cases."""

    class DescriptorNS(PatchNameSpace):
        """A namespace used to test class-level descriptor access."""

        name = "descriptor_patch"

        def get_dims(self):
            """Return patch dims."""
            return self.dims

    def test_registered_namespace_returns_class_on_class_access(
        self, random_patch, monkeypatch
    ):
        """Accessing a namespace on the class should return the namespace class."""
        monkeypatch.setattr(
            Patch,
            "_method_namespace_registry",
            dict(Patch.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(Patch, "descriptor_patch", raising=False)
        assert random_patch.descriptor_patch.get_dims() == random_patch.dims
        assert Patch.descriptor_patch is self.DescriptorNS


class TestNamespaceRegistration:
    """Tests for registering method namespaces."""

    class PatchNS(PatchNameSpace):
        """A test namespace for Patch."""

        name = "plugin_test_patch"

        def get_dims(self):
            """Return patch dims."""
            return self.dims

    class SpoolNS(SpoolNameSpace):
        """A test namespace for Spool."""

        name = "plugin_test_spool"

        def get_patch_count(self):
            """Return patch count."""
            return len(self)

    def test_patch_namespace_subclass_binds_on_access(self, random_patch, monkeypatch):
        """Imported patch namespaces should bind on first access."""
        monkeypatch.setattr(
            Patch,
            "_method_namespace_registry",
            dict(Patch.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(Patch, "plugin_test_patch", raising=False)
        assert random_patch.plugin_test_patch.get_dims() == random_patch.dims
        assert Patch.get_registered_namespaces()["plugin_test_patch"] is self.PatchNS

    def test_spool_namespace_subclass_binds_on_access(self, random_spool, monkeypatch):
        """Imported spool namespaces should bind on first access."""
        monkeypatch.setattr(
            BaseSpool,
            "_method_namespace_registry",
            dict(BaseSpool.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(BaseSpool, "plugin_test_spool", raising=False)
        assert random_spool.plugin_test_spool.get_patch_count() == len(random_spool)
        assert (
            BaseSpool.get_registered_namespaces()["plugin_test_spool"] is self.SpoolNS
        )

    def test_invalid_namespace_binding_raises(self):
        """Non-namespace classes should be rejected."""
        with pytest.raises(ParameterError, match="PatchNameSpace"):
            namespace_mod._bind_method_namespace(Patch, "invalid_plugin_ns", object)

    def test_wrong_namespace_type_raises(self):
        """Patch and spool namespaces should not be interchangeable."""
        with pytest.raises(ParameterError, match="PatchNameSpace"):
            namespace_mod._bind_method_namespace(Patch, "wrong_patch_ns", self.SpoolNS)
        with pytest.raises(ParameterError, match="SpoolNameSpace"):
            namespace_mod._bind_method_namespace(
                BaseSpool, "wrong_spool_ns", self.PatchNS
            )

    def test_invalid_namespace_name_raises(self):
        """Namespace subclass names must be valid identifiers."""
        with pytest.raises(ParameterError, match="not a valid namespace name"):

            class BadPatchNS(PatchNameSpace):
                name = "not-valid"

    def test_duplicate_namespace_name_raises(self):
        """Namespace subclass names should be unique within a base type."""
        with pytest.raises(ParameterError, match="Patch.duplicate_patch_ns"):

            class DuplicatePatchNS1(PatchNameSpace):
                name = "duplicate_patch_ns"

            class DuplicatePatchNS2(PatchNameSpace):
                name = "duplicate_patch_ns"

    def test_existing_attribute_conflict_raises(self):
        """Existing attributes should not be overwritten by default."""
        namespace_mod.load_namespace(Patch, "io")
        with pytest.raises(ParameterError, match="already exists"):
            namespace_mod._bind_method_namespace(Patch, "io", self.PatchNS)

    def test_invalid_bound_namespace_name_raises(self):
        """Low-level binding should reject invalid attribute names."""
        with pytest.raises(ParameterError, match="not a valid namespace name"):
            namespace_mod._bind_method_namespace(Patch, "not-valid", self.PatchNS)

    def test_patch_builtin_namespaces_are_registered(self, random_patch):
        """Built-in patch namespaces should share the registry path."""
        _ = random_patch.viz
        _ = random_patch.io
        registered = Patch.get_registered_namespaces()
        assert registered["viz"] is Patch.viz
        assert registered["io"] is Patch.io
        assert random_patch.viz.__class__ is Patch.viz
        assert random_patch.io.__class__ is Patch.io


class TestLazyNamespaceLoading:
    """Tests for lazy loading namespaces from entry points."""

    class LazyPatchNS(PatchNameSpace):
        """A lazily loaded patch namespace."""

        def get_shape(self):
            """Return patch shape."""
            return self.shape

    class LazySpoolNS(SpoolNameSpace):
        """A lazily loaded spool namespace."""

        def get_patch_count(self):
            """Return patch count."""
            return len(self)

    def test_patch_namespace_loaded_on_missing_attr(self, random_patch, monkeypatch):
        """Patch should lazy load matching namespace entry points."""
        called = {"count": 0}

        def loader():
            called["count"] += 1
            return self.LazyPatchNS

        plugin_mod.get_entry_point_loaders.cache_clear()
        plugin_mod.load_entry_point.cache_clear()
        monkeypatch.setattr(
            namespace_mod,
            "get_entry_point_loaders",
            lambda group: {"lazy_patch": loader},
        )
        monkeypatch.setattr(
            namespace_mod, "load_entry_point", lambda group, name: loader()
        )
        monkeypatch.setattr(
            Patch,
            "_method_namespace_registry",
            dict(Patch.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(Patch, "lazy_patch", raising=False)

        assert called["count"] == 0
        assert random_patch.lazy_patch.get_shape() == random_patch.shape
        assert called["count"] == 1

    def test_spool_namespace_loaded_on_missing_attr(self, random_spool, monkeypatch):
        """Spool should lazy load matching namespace entry points."""
        called = {"count": 0}

        def loader():
            called["count"] += 1
            return self.LazySpoolNS

        plugin_mod.get_entry_point_loaders.cache_clear()
        plugin_mod.load_entry_point.cache_clear()
        monkeypatch.setattr(
            namespace_mod,
            "get_entry_point_loaders",
            lambda group: {"lazy_spool": loader},
        )
        monkeypatch.setattr(
            namespace_mod, "load_entry_point", lambda group, name: loader()
        )
        monkeypatch.setattr(
            BaseSpool,
            "_method_namespace_registry",
            dict(BaseSpool.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(BaseSpool, "lazy_spool", raising=False)

        assert called["count"] == 0
        assert random_spool.lazy_spool.get_patch_count() == len(random_spool)
        assert called["count"] == 1

    def test_missing_patch_namespace_raises_attribute_error(
        self, random_patch, monkeypatch
    ):
        """Missing patch namespaces should still raise AttributeError."""
        plugin_mod.get_entry_point_loaders.cache_clear()
        plugin_mod.load_entry_point.cache_clear()
        monkeypatch.setattr(namespace_mod, "get_entry_point_loaders", lambda _: {})
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = random_patch.not_a_namespace

    def test_missing_spool_viz_preserves_message(self, random_spool, monkeypatch):
        """Missing spool viz should preserve the existing guidance."""
        plugin_mod.get_entry_point_loaders.cache_clear()
        plugin_mod.load_entry_point.cache_clear()
        monkeypatch.setattr(namespace_mod, "get_entry_point_loaders", lambda _: {})
        with pytest.raises(AttributeError, match="has no 'viz' namespace"):
            _ = random_spool.viz

    def test_registered_plugin_uses_registry_without_reloading(self, monkeypatch):
        """Already registered plugins should resolve from the registry."""
        monkeypatch.setattr(
            Patch,
            "_method_namespace_registry",
            {"lazy_patch_loaded": self.LazyPatchNS},
            raising=False,
        )
        plugin_mod.get_entry_point_loaders.cache_clear()
        plugin_mod.load_entry_point.cache_clear()
        monkeypatch.setattr(namespace_mod, "get_entry_point_loaders", lambda _: {})
        assert namespace_mod.load_namespace(Patch, "lazy_patch_loaded")


class TestNamespaceHelpers:
    """Tests for helper functions in namespace utilities."""

    def test_owner_cls_without_namespace_config_returns_self(self):
        """Classes without namespace config should resolve to themselves."""

        class NoNamespaceOwner:
            pass

        assert (
            namespace_mod._get_namespace_owner_cls(NoNamespaceOwner) is NoNamespaceOwner
        )

    def test_load_namespace_without_group_returns_false(self):
        """Classes without namespace config should not attempt plugin loads."""

        class NoNamespaceOwner:
            pass

        assert not namespace_mod.load_namespace(NoNamespaceOwner, "missing")
