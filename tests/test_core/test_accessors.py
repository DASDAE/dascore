"""
Tests for DASCore Accessors.
"""

from __future__ import annotations

import pytest

import dascore as dc
import dascore.core.accessor as accessor_mod
import dascore.core.patch as patch_mod
import dascore.core.spool as spool_mod
from dascore.exceptions import AccessorRegistrationError


class TestAccessorDescriptor:
    """Unit tests for the _Accessor descriptor."""

    def test_class_access_returns_accessor_class(self):
        """Access via class should return accessor class itself."""

        class _Acc:
            pass

        desc = accessor_mod._Accessor("ns", _Acc)
        assert desc.__get__(None, object) is _Acc

    def test_instance_access_instantiates_accessor(self):
        """Access via instance should instantiate accessor with host object."""

        class _Acc:
            def __init__(self, obj):
                self.obj = obj

        host = object()
        desc = accessor_mod._Accessor("ns", _Acc)
        out = desc.__get__(host, object)
        assert isinstance(out, _Acc)
        assert out.obj is host

    def test_instance_access_wraps_attribute_error(self):
        """AttributeError from accessor init should be wrapped as RuntimeError."""

        class _Acc:
            def __init__(self, _):
                raise AttributeError("bad host")

        desc = accessor_mod._Accessor("ns", _Acc)
        with pytest.raises(RuntimeError, match="Accessor 'ns' is not valid"):
            _ = desc.__get__(object(), object)

    def test_attribute_error_cause_is_preserved(self):
        """RuntimeError should chain from the original AttributeError."""
        original = AttributeError("original message")

        class _Acc:
            def __init__(self, _):
                raise original

        desc = accessor_mod._Accessor("ns", _Acc)
        with pytest.raises(RuntimeError) as exc_info:
            desc.__get__(object(), object)
        assert exc_info.value.__cause__ is original


class TestRegisterAccessor:
    """Tests for _register_accessor."""

    def test_invalid_name_raises(self):
        """Non-identifier names should raise."""

        class _Host:
            pass

        with pytest.raises(AccessorRegistrationError, match="valid Python identifier"):
            accessor_mod._register_accessor("not-valid", _Host)

    def test_private_name_raises(self):
        """Names starting with underscore should raise."""

        class _Host:
            pass

        with pytest.raises(AccessorRegistrationError, match="must not start"):
            accessor_mod._register_accessor("_private", _Host)

    def test_conflict_with_existing_attribute_raises(self):
        """Registration should fail if attribute already exists on host."""

        class _Host:
            existing = 1

        dec = accessor_mod._register_accessor("existing", _Host)
        with pytest.raises(AccessorRegistrationError, match="already an attribute"):
            dec(type("Acc", (), {}))

    def test_conflict_with_inherited_attribute_raises(self):
        """Registration should fail for inherited attrs too."""

        class _Parent:
            inherited = 1

        class _Host(_Parent):
            pass

        dec = accessor_mod._register_accessor("inherited", _Host)
        with pytest.raises(AccessorRegistrationError, match="already an attribute"):
            dec(type("Acc", (), {}))

    def test_register_and_access(self):
        """Registering should attach descriptor and allow accessor calls."""

        class _Host:
            def __init__(self, value):
                self.value = value

        dec = accessor_mod._register_accessor("ns", _Host)

        @dec
        class _Acc:
            def __init__(self, obj):
                self.obj = obj

            def get(self):
                return self.obj.value

        host = _Host(10)
        assert host.ns.get() == 10
        assert "ns" in _Host._accessors
        assert _Host.ns is _Acc

    def test_register_returns_accessor_class(self):
        """Decorator should return the unmodified accessor class."""

        class _Host:
            pass

        class _Acc:
            def __init__(self, obj):
                self.obj = obj

        dec = accessor_mod._register_accessor("ret", _Host)
        result = dec(_Acc)
        assert result is _Acc

    def test_idempotent_same_class(self):
        """Registering same class twice under same name should no-op."""

        class _Host:
            pass

        dec = accessor_mod._register_accessor("ns", _Host)

        class _Acc:
            def __init__(self, obj):
                self.obj = obj

        out1 = dec(_Acc)
        out2 = dec(_Acc)
        assert out1 is _Acc
        assert out2 is _Acc
        assert isinstance(_Host.__dict__["ns"], accessor_mod._Accessor)

    def test_second_accessor_reuses_accessors_set(self):
        """Registering a second name on the same class should extend _accessors."""

        class _Host:
            pass

        def _make_acc():
            return type("_A", (), {"__init__": lambda self, o: None})

        accessor_mod._register_accessor("alpha", _Host)(_make_acc())
        accessor_mod._register_accessor("beta", _Host)(_make_acc())
        assert {"alpha", "beta"} <= _Host._accessors


class TestPublicRegisterFunctions:
    """Tests for public registration helpers."""

    def test_register_patch_accessor(self, monkeypatch):
        """register_patch_accessor should register on Patch class."""

        class _Patch:
            pass

        monkeypatch.setattr(patch_mod, "Patch", _Patch)
        dec = accessor_mod.register_patch_accessor("qa")

        @dec
        class _Acc:
            def __init__(self, patch):
                self.patch = patch

        out = _Patch().qa
        assert isinstance(out, _Acc)
        assert isinstance(_Patch.__dict__["qa"], accessor_mod._Accessor)

    def test_register_spool_accessor(self, monkeypatch):
        """register_spool_accessor should register on BaseSpool class."""

        class _Spool:
            pass

        monkeypatch.setattr(spool_mod, "BaseSpool", _Spool)
        dec = accessor_mod.register_spool_accessor("event")

        @dec
        class _Acc:
            def __init__(self, spool):
                self.spool = spool

        out = _Spool().event
        assert isinstance(out, _Acc)
        assert isinstance(_Spool.__dict__["event"], accessor_mod._Accessor)


class TestBuiltinAccessors:
    """Integration tests confirming the built-in viz and io accessors are wired up."""

    def test_viz_accessor_returns_correct_type(self):
        """patch.viz should return a VizPatchAccessor instance."""
        from dascore.viz import VizPatchAccessor

        assert isinstance(dc.get_example_patch().viz, VizPatchAccessor)

    def test_io_accessor_returns_correct_type(self):
        """patch.io should return a PatchIO instance."""
        from dascore.io import PatchIO

        assert isinstance(dc.get_example_patch().io, PatchIO)

    def test_viz_class_access_returns_class(self):
        """Patch.viz (class-level) should return VizPatchAccessor class."""
        from dascore.viz import VizPatchAccessor

        assert dc.Patch.viz is VizPatchAccessor

    def test_io_class_access_returns_class(self):
        """Patch.io (class-level) should return PatchIO class."""
        from dascore.io import PatchIO

        assert dc.Patch.io is PatchIO

    def test_builtin_accessors_in_accessors_set(self):
        """Patch._accessors should contain viz and io."""
        assert "viz" in dc.Patch._accessors
        assert "io" in dc.Patch._accessors

    def test_builtin_accessors_in_dir(self):
        """Viz and io should appear in dir(patch)."""
        names = dir(dc.get_example_patch())
        assert "viz" in names
        assert "io" in names


class TestAttachToPatchAccessor:
    """Tests for dynamic method attachment helper."""

    def test_attach_creates_namespace_and_method(self, monkeypatch):
        """Attach should create accessor namespace when missing."""

        class _Patch:
            def __init__(self, value=1):
                self.value = value

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        def scale(patch, factor):
            return patch.value * factor

        accessor_mod._attach_to_patch_accessor("mathx", scale)
        out = _Patch(2).mathx.scale(3)
        assert out == 6
        assert "mathx" in _Patch._accessors

    def test_attach_uses_existing_accessor_namespace(self, monkeypatch):
        """Attach should add methods to an existing accessor class."""

        class _Patch:
            def __init__(self, value=1):
                self.value = value

        class _Existing:
            def __init__(self, patch):
                self._patch = patch

            def base(self):
                return self._patch.value

        monkeypatch.setattr(patch_mod, "Patch", _Patch)
        accessor_mod._register_accessor("calc", _Patch)(_Existing)

        def plus(patch, value):
            return patch.value + value

        accessor_mod._attach_to_patch_accessor("calc", plus)
        acc = _Patch(4).calc
        assert isinstance(acc, _Existing)
        assert acc.base() == 4
        assert acc.plus(6) == 10

    def test_attach_conflict_with_non_accessor_attribute_raises(self, monkeypatch):
        """Attach should fail when namespace is a non-accessor attribute."""

        class _Patch:
            bad_ns = 1

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        def noop(patch):
            return patch

        with pytest.raises(AccessorRegistrationError, match="non-accessor attribute"):
            accessor_mod._attach_to_patch_accessor("bad_ns", noop)

    def test_attach_preserves_function_metadata(self, monkeypatch):
        """functools.wraps should carry __name__ and __doc__ onto the method."""

        class _Patch:
            def __init__(self, value=1):
                self.value = value

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        def my_named_func(patch):
            """My docstring."""
            return patch.value

        accessor_mod._attach_to_patch_accessor("meta_ns", my_named_func)
        method = _Patch(1).meta_ns.my_named_func
        assert method.__name__ == "my_named_func"
        assert method.__doc__ == "My docstring."


class TestPatchFunctionNamespace:
    """Tests for the namespace parameter of patch_function."""

    def test_namespace_attaches_method_to_accessor(self, monkeypatch):
        """Decorated function should appear on the named accessor namespace."""

        class _Patch:
            def __init__(self, value=0):
                self.value = value

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        @dc.patch_function(namespace="metrics", history=None)
        def triple(patch, factor=3):
            return patch.value * factor

        assert _Patch(4).metrics.triple() == 12
        assert _Patch(4).metrics.triple(factor=2) == 8

    def test_namespace_function_still_callable_standalone(self, monkeypatch):
        """Function decorated with namespace should still work as a regular call."""

        class _Patch:
            def __init__(self, value=0):
                self.value = value

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        @dc.patch_function(namespace="stats", history=None)
        def double(patch):
            return patch.value * 2

        assert double(_Patch(5)) == 10

    def test_namespace_second_function_goes_to_same_accessor(self, monkeypatch):
        """Two functions with the same namespace share one accessor class."""

        class _Patch:
            def __init__(self, value=0):
                self.value = value

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        @dc.patch_function(namespace="ops", history=None)
        def add_one(patch):
            return patch.value + 1

        @dc.patch_function(namespace="ops", history=None)
        def add_two(patch):
            return patch.value + 2

        host = _Patch(10)
        assert host.ops.add_one() == 11
        assert host.ops.add_two() == 12
        assert type(host.ops) is type(host.ops)  # noqa same class

    def test_namespace_conflict_with_existing_attribute_raises(self, monkeypatch):
        """Namespace pointing to a non-accessor attribute should raise."""

        class _Patch:
            blocked = 99

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        with pytest.raises(AccessorRegistrationError, match="non-accessor attribute"):

            @dc.patch_function(namespace="blocked", history=None)
            def fn(patch):
                return patch

    def test_namespace_method_name_preserved(self, monkeypatch):
        """Method name on the accessor should match the function name."""

        class _Patch:
            def __init__(self, value=0):
                self.value = value

        monkeypatch.setattr(patch_mod, "Patch", _Patch)

        @dc.patch_function(namespace="named_ns", history=None)
        def my_special_func(patch):
            """Special docstring."""
            return patch.value

        method = _Patch(0).named_ns.my_special_func
        assert method.__name__ == "my_special_func"
        assert method.__doc__ == "Special docstring."
