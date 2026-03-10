"""Tests for namespace utilities."""

from __future__ import annotations

import pytest

from dascore.core.patch import Patch
from dascore.core.spool import BaseSpool
from dascore.exceptions import ParameterError
from dascore.utils.namespace import MethodNameSpace


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

    class DescriptorNS(MethodNameSpace):
        """A namespace used to test class-level descriptor access."""

        def get_dims(self):
            """Return patch dims."""
            return self.dims

    def test_registered_namespace_returns_class_on_class_access(self, monkeypatch):
        """Accessing a namespace on the class should return the namespace class."""
        monkeypatch.setattr(
            Patch,
            "_method_namespace_registry",
            dict(Patch.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(Patch, "descriptor_patch", raising=False)
        Patch.register_namespace("descriptor_patch", self.DescriptorNS)
        assert Patch.descriptor_patch is self.DescriptorNS


class TestNamespaceRegistration:
    """Tests for registering method namespaces."""

    class PatchNS(MethodNameSpace):
        """A test namespace for Patch."""

        def get_dims(self):
            """Return patch dims."""
            return self.dims

    class SpoolNS(MethodNameSpace):
        """A test namespace for Spool."""

        def get_patch_count(self):
            """Return patch count."""
            return len(self)

    def test_patch_can_register_namespace(self, random_patch, monkeypatch):
        """Registered namespaces should bind to Patch instances."""
        monkeypatch.setattr(
            Patch,
            "_method_namespace_registry",
            dict(Patch.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(Patch, "plugin_test_patch", raising=False)
        Patch.register_namespace("plugin_test_patch", self.PatchNS)
        assert random_patch.plugin_test_patch.get_dims() == random_patch.dims
        assert Patch.get_registered_namespaces()["plugin_test_patch"] is self.PatchNS

    def test_spool_can_register_namespace(self, random_spool, monkeypatch):
        """Registered namespaces should bind to Spool instances."""
        monkeypatch.setattr(
            BaseSpool,
            "_method_namespace_registry",
            dict(BaseSpool.get_registered_namespaces()),
            raising=False,
        )
        monkeypatch.delattr(BaseSpool, "plugin_test_spool", raising=False)
        BaseSpool.register_namespace("plugin_test_spool", self.SpoolNS)
        assert random_spool.plugin_test_spool.get_patch_count() == len(random_spool)
        assert (
            BaseSpool.get_registered_namespaces()["plugin_test_spool"]
            is self.SpoolNS
        )

    def test_invalid_namespace_registration_raises(self):
        """Non-namespace classes should be rejected."""
        with pytest.raises(ParameterError, match="MethodNameSpace"):
            Patch.register_namespace("invalid_plugin_ns", object)

    def test_invalid_namespace_name_raises(self):
        """Namespace names must be valid identifiers."""
        with pytest.raises(ParameterError, match="not a valid namespace name"):
            Patch.register_namespace("not-valid", self.PatchNS)

    def test_existing_attribute_conflict_raises(self):
        """Existing attributes should not be overwritten by default."""
        with pytest.raises(ParameterError, match="already exists"):
            Patch.register_namespace("io", self.PatchNS)


class TestLazyNamespaceLoading:
    """Tests for lazy loading namespaces from entry points."""

    class LazyPatchNS(MethodNameSpace):
        """A lazily loaded patch namespace."""

        def get_shape(self):
            """Return patch shape."""
            return self.shape

    class LazySpoolNS(MethodNameSpace):
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

        monkeypatch.setattr(Patch._namespace_manager, "_eps", {"lazy_patch": loader})
        monkeypatch.setattr(Patch._namespace_manager, "_loaded_names", set())
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

        monkeypatch.setattr(
            BaseSpool._namespace_manager, "_eps", {"lazy_spool": loader}
        )
        monkeypatch.setattr(BaseSpool._namespace_manager, "_loaded_names", set())
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
        monkeypatch.setattr(Patch._namespace_manager, "_eps", {})
        monkeypatch.setattr(Patch._namespace_manager, "_loaded_names", set())
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = random_patch.not_a_namespace

    def test_missing_spool_viz_preserves_message(self, random_spool, monkeypatch):
        """Missing spool viz should preserve the existing guidance."""
        monkeypatch.setattr(BaseSpool._namespace_manager, "_eps", {})
        monkeypatch.setattr(BaseSpool._namespace_manager, "_loaded_names", set())
        with pytest.raises(AttributeError, match="has no 'viz' namespace"):
            _ = random_spool.viz

    def test_loaded_plugin_uses_registry_without_reloading(self, monkeypatch):
        """Already loaded plugins should resolve from the registry."""
        monkeypatch.setattr(
            Patch,
            "_method_namespace_registry",
            {"lazy_patch_loaded": self.LazyPatchNS},
            raising=False,
        )
        monkeypatch.setattr(Patch._namespace_manager, "_loaded_names", {"lazy_patch_loaded"})
        monkeypatch.setattr(Patch._namespace_manager, "_eps", {})
        assert Patch._namespace_manager.load_plugin("lazy_patch_loaded")
