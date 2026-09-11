"""Tests for naming, resolving and fingerprinting patch functions."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import Field

import dascore as dc
import dascore.utils.patch_registry as registry_module
from dascore.exceptions import ParameterError
from dascore.units import get_quantity
from dascore.utils.patch_registry import (
    _as_call,
    _bind,
    fingerprint_call,
    patch_function_tag,
    register_patch_function,
    resolve_patch_function,
)


@dc.patch_function()
def registry_test_scaled(patch, factor=2.0):
    """Scale a patch, for the tests which need a function of our own."""
    return patch.new(data=patch.data * factor)


@dc.patch_function()
def registry_test_field_default(patch, value=Field(default=7)):
    """Take a parameter whose default is spelled as a pydantic Field."""
    return patch.update_attrs(seen=value)


@dc.patch_function()
def registry_test_leading_then_group(patch, first, *rest, flag=False):
    """Take one named argument, then a group of them."""
    return patch.update_attrs(seen=f"{first}{rest}{flag}")


@dc.patch_function(version="1.0")
def registry_test_versioned(patch, factor=1):
    """Do nothing, at the version it is declared."""
    return patch


@pytest.fixture()
def own_registry(monkeypatch):
    """Work against a copy of the registry, so a tag left behind changes nothing."""
    monkeypatch.setattr(
        registry_module, "_REGISTERED", dict(registry_module._REGISTERED)
    )
    monkeypatch.setattr(registry_module, "_AMBIGUOUS", {})


class TestTags:
    """A tag names a function: bare for DASCore, namespaced otherwise."""

    def test_a_dascore_function_is_bare(self):
        """DASCore's own are unqualified."""
        assert patch_function_tag(dc.proc.normalize) == "normalize"

    def test_our_own_is_namespaced_by_its_package(self):
        """A function defined here is `tests:name`."""
        assert patch_function_tag(registry_test_scaled) == (
            "tests:registry_test_scaled"
        )
        assert resolve_patch_function("tests:registry_test_scaled") is (
            registry_test_scaled
        )

    def test_a_function_with_no_name(self):
        """A callable object has no `__name__`, so it takes no tag."""

        class Callable:
            """A callable object."""

            __qualname__ = "Callable"

            def __call__(self, patch):
                """Do nothing."""
                return patch

        assert patch_function_tag(Callable()) is None

    def test_a_plugin_never_resolves_to_dascores(self, own_registry):
        """A package's `normalize` is its own and cannot shadow DASCore's."""

        def normalize(patch, factor=1):
            """Share a name with a core patch function."""
            return patch

        normalize.__module__ = "myplugin.filters"
        normalize.__qualname__ = "normalize"
        tagged = dc.patch_function()(normalize)
        assert patch_function_tag(tagged) == "myplugin:normalize"
        assert resolve_patch_function("normalize") is dc.proc.normalize
        assert resolve_patch_function("myplugin:normalize") is tagged

    @pytest.mark.concurrency
    def test_a_deferred_function_resolves_in_a_fresh_process(self):
        """Nothing has imported `dascore.viz`; the sweep is what finds it."""
        code = (
            "import sys, dascore; "
            "assert 'dascore.viz' not in sys.modules; "
            "from dascore.utils.patch_registry import resolve_patch_function; "
            "assert resolve_patch_function('waterfall').__name__ == 'waterfall'; "
            "assert 'dascore.viz' in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code], check=True, timeout=120, capture_output=True
        )


class TestTheRegistry:
    """Two functions may not claim one tag."""

    @pytest.fixture(autouse=True)
    def _own(self, own_registry):
        """Every test here works against a copy."""

    def test_a_tag_the_sweep_finds(self, monkeypatch):
        """A tag missing until the install has been swept still resolves."""

        def late(patch):
            """Arrive only once the sweep has run."""
            return patch

        monkeypatch.setattr(registry_module, "_swept", False)
        monkeypatch.setattr(
            registry_module,
            "_sweep_patch_functions",
            lambda: registry_module._REGISTERED.update({"swept:late": late}),
        )
        assert resolve_patch_function("swept:late") is late

    def test_two_dascore_functions_may_not_share_a_tag(self):
        """DASCore's own tags are its own to keep unique."""

        def normalize(patch):
            """Claim a tag DASCore already uses."""
            return patch

        normalize.__module__ = "dascore.somewhere"
        normalize.__qualname__ = "normalize"
        with pytest.raises(ParameterError, match="claim the tag"):
            register_patch_function(normalize)

    def test_two_plugins_may_not_resolve_one_tag(self):
        """Both import, but the tag no longer resolves to either."""

        def denoise(patch):
            """Claim a tag another package also claims."""
            return patch

        denoise.__module__ = "pkga.filters"
        denoise.__qualname__ = "denoise"
        assert register_patch_function(denoise) == "pkga:denoise"

        def other(patch):
            """The other function of the same name."""
            return patch

        other.__module__ = "pkga.elsewhere"
        other.__name__ = other.__qualname__ = "denoise"
        with pytest.warns(UserWarning, match="claim the tag"):
            register_patch_function(other)
        with pytest.raises(ParameterError, match="names two functions"):
            resolve_patch_function("pkga:denoise")

    def test_a_module_re_imported_keeps_its_entry(self):
        """The same function registered twice is not a collision."""
        tag = register_patch_function(registry_test_scaled)
        assert register_patch_function(registry_test_scaled) == tag

    @pytest.mark.parametrize(
        ("name", "match"),
        [
            ("not_a_patch_function", "DASCore defines none"),
            ("get_axis", "DASCore defines none"),
            ("nosuchpkg:denoise", "install nosuchpkg"),
            ("__main__:my_filter", "script or a notebook"),
        ],
    )
    def test_an_unknown_tag(self, name, match):
        """A tag nothing registers says what is missing."""
        with pytest.raises(ParameterError, match=match):
            resolve_patch_function(name)

    def test_an_unknown_tag_names_its_module(self):
        """Where the function was defined, when that is known."""
        with pytest.raises(ParameterError, match=r"defined in nosuchpkg\.filters"):
            resolve_patch_function("nosuchpkg:denoise", "nosuchpkg.filters")


class TestBinding:
    """Two spellings of one call are one mapping."""

    def test_positional_and_keyword(self):
        """Which is what binding against the signature is for."""
        expected = {"dim": "time", "norm": "l2", "window": None, "samples": False}
        assert _bind(dc.proc.normalize, ("time",), {}) == expected
        assert _bind(dc.proc.normalize, (), {"dim": "time"}) == expected

    def test_a_star_args_group(self):
        """A `*args` group is one entry holding the tuple it is."""
        assert _bind(dc.proc.transpose, ("time", "distance"), {}) == {
            "dims": ("time", "distance")
        }

    def test_an_extra_is_kept_by_name(self):
        """A dimension the signature does not name is bound under its own."""
        bound = _bind(dc.proc.pass_filter, (), {"time": (10, 100)})
        assert bound["time"] == (10, 100)
        assert bound["corners"] == 4

    def test_a_field_default(self):
        """A default spelled as a pydantic Field is the value it holds."""
        assert _bind(registry_test_field_default, (), {}) == {"value": 7}

    def test_an_extra_which_collides_with_a_parameter(self):
        """`append_dims(patch, *empty_dims, **kwargs)` given `empty_dims=3`."""
        with pytest.raises(ParameterError, match="both as a parameter"):
            _bind(dc.proc.append_dims, ("a",), {"empty_dims": 3})

    def test_an_argument_the_signature_rejects(self):
        """An argument the function does not take is refused."""
        with pytest.raises(ParameterError, match="cannot be called that way"):
            _bind(dc.proc.normalize, (), {"dim": "time", "not_a_parameter": 1})

    def test_a_positional_before_a_star_args_group(self, random_patch):
        """The leading argument goes back to being positional to run."""
        func = registry_test_leading_then_group
        bound = _bind(func, (1, 2, 3), {"flag": True})
        assert bound == {"first": 1, "rest": (2, 3), "flag": True}
        args, kwargs = _as_call(func, bound)
        assert (args, kwargs) == ((1, 2, 3), {"flag": True})
        with pytest.raises(ParameterError, match="cannot be called that way"):
            _as_call(func, {"rest": (2, 3)})


class TestFingerprintCall:
    """The digest a call carries."""

    def test_a_call_is_not_fingerprinted_until_something_reads_it(self, random_patch):
        """An argument the serializer cannot encode never fails the call."""
        deep = {}
        deep["self"] = deep

        @dc.patch_function()
        def takes_anything(patch, thing=None):
            """Accept whatever it is given."""
            return patch

        assert takes_anything(random_patch, thing=deep) is not None

    def test_the_version_is_part_of_it(self, monkeypatch):
        """An operation at a new version is a new operation."""
        before = fingerprint_call(registry_test_versioned, (), {})
        monkeypatch.setattr(registry_test_versioned, "__version__", "2.0")
        assert fingerprint_call(registry_test_versioned, (), {}) != before

    def test_the_name_is_part_of_it(self):
        """Two operations given the same arguments are still two."""
        assert fingerprint_call(
            dc.proc.demean, (), {"dim": "time"}
        ) != fingerprint_call(dc.proc.demedian, (), {"dim": "time"})

    def test_equal_quantities_are_two_calls(self):
        """Pint hashes 1 m and 100 cm alike; the cache must not merge them."""
        one_meter, hundred_cm = get_quantity("1 m"), get_quantity("100 cm")
        assert hash(one_meter) == hash(hundred_cm)
        first = fingerprint_call(dc.proc.select, (), {"distance": one_meter})
        second = fingerprint_call(dc.proc.select, (), {"distance": hundred_cm})
        assert first != second
        assert fingerprint_call(dc.proc.select, (), {"distance": one_meter}) == first

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "expected"),
        [
            ("abs", (), {}, "19d28ce8e2762604"),
            ("pass_filter", (), {"time": (10, 100)}, "760edca6e6e15fc1"),
            ("transpose", ("time", "distance"), {}, "97c847a8ef484a40"),
        ],
    )
    def test_it_is_stable(self, func, args, kwargs, expected):
        """A fingerprint written down last week names the same call today."""
        assert fingerprint_call(getattr(dc.proc, func), args, kwargs) == expected
