"""
Tests that pin DASCore's public API to its checked-in snapshot.

A failure here means a public name was added, removed, or called
differently. If that was intended, run `python scripts/public_api.py
freeze` and justify each change in the pull request.
"""

from __future__ import annotations

import enum
import importlib.util
import inspect
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

import dascore as dc

_SCRIPTS = Path(__file__).absolute().parents[1] / "scripts"


def _load_script(name):
    """Import a script by path; scripts/ is not a package."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # public_api imports _api_urls by name, as it would run from scripts/.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_script("_api_urls")
public_api = _load_script("public_api")


class _Color(enum.Enum):
    """An enum used as a default."""

    RED = 1


class _Holder:
    """A class with one member of each kind the snapshot tells apart."""

    LIMIT = 3

    @property
    def size(self):
        """A property."""
        return 1

    @classmethod
    def build(cls, value=1):
        """A classmethod."""
        return cls()

    def run(self, a, *, b=None):
        """A method."""

    ufunc = np.add


class _NoSignature:
    """A callable whose signature cannot be read."""

    def __call__(self):
        """Do nothing."""

    @property
    def __signature__(self):
        raise ValueError("no signature")


class TestSnapshot:
    """The live API matches the baseline."""

    def test_matches_baseline(self):
        """Every public name and signature is the one frozen."""
        problem = public_api.check()
        assert problem is None, problem

    def test_all_is_sorted_and_unique(self):
        """`__all__` is kept sorted, so a diff shows only what changed."""
        assert list(dc.__all__) == sorted(set(dc.__all__))

    def test_no_shadowing_names(self):
        """A star import must not replace builtins or stdlib modules."""
        assert not {"print", "std_warnings", "io", "warnings"} & set(dc.__all__)
        assert not hasattr(dc, "print")
        assert not hasattr(dc, "std_warnings")

    def test_annotations_is_the_loader(self):
        """`dc.annotations` is the function, not the `__future__` feature."""
        assert inspect.isfunction(dc.annotations)
        assert dc.annotations.__module__ == "dascore.core.annotation_loader"


class TestDescribe:
    """Objects are rendered the same on every platform."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "None"),
            (1.5, "1.5"),
            ("a", "'a'"),
            ((1,), "(1,)"),
            ((1, "b"), "(1, 'b')"),
            ([1], "[1]"),
            (frozenset({"b", "a"}), "{'a', 'b'}"),
            ({"a": 1}, "{'a': 1}"),
            (_Color.RED, "_Color.RED"),
            (np.float64(1.0), "float64(1.0)"),
            (np.datetime64("NaT", "ns"), "datetime64(NaT)"),
            (PurePosixPath("/home/x"), "<Path>"),
            (int, "int"),
            (len, "len"),
            (object(), "<object>"),
        ],
    )
    def test_default_text(self, value, expected):
        """Defaults render by value or by name, never with an address."""
        assert public_api._default_text(value) == expected

    def test_signature_drops_annotations(self):
        """Annotations are left out; names, kinds and defaults stay."""

        def func(a: int, /, b: str = "x", *args, c: object = None, **kwargs) -> int:
            return 1

        text = public_api.render_signature(inspect.signature(func))
        assert text == "(a, /, b='x', *args, c=None, **kwargs)"

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (dc, "module"),
            (int, "class"),
            (3, "constant"),
            (_NoSignature(), "callable"),
        ],
    )
    def test_describe(self, obj, expected):
        """Objects without a signature are named by kind."""
        assert public_api.describe(obj) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("size", "attribute"),
            ("build", "(value=1)"),
            ("run", "(self, a, *, b=None)"),
            ("LIMIT", "constant"),
            ("ufunc", "<ufunc>"),
        ],
    )
    def test_describe_member(self, name, expected):
        """Class members are described without running descriptors."""
        assert public_api.describe_member(_Holder, name) == expected

    def test_module_names_without_all(self):
        """A module without `__all__` lists only what it defines."""
        names = public_api.module_names(dc.utils.time)
        assert "to_datetime64" in names
        assert "np" not in names
        assert all(not x.startswith("_") for x in names)

    def test_module_names_with_all(self):
        """A module with `__all__` lists exactly that."""
        assert public_api.module_names(dc.xarray) == sorted(dc.xarray.__all__)


class TestNamespaces:
    """Only DASCore's own namespaces are part of its API."""

    def test_foreign_namespace_skipped(self, monkeypatch):
        """A namespace a plugin or doc example registers is not pinned."""

        class Foreign:
            """Stands in for a namespace registered outside DASCore."""

            def go(self):
                """A method."""

        registered = dict(dc.Patch.get_registered_namespaces())
        monkeypatch.setattr(
            dc.Patch,
            "get_registered_namespaces",
            lambda: {**registered, "foreign": Foreign},
        )
        keys = dict(public_api._namespace_entries(dc.Patch, "dascore.Patch"))
        assert "dascore.Patch.io" in keys
        assert not any(x.startswith("dascore.Patch.foreign") for x in keys)


class TestBaselineFile:
    """Freezing, loading and comparing baselines."""

    def test_compare(self):
        """Added, removed and changed names are reported apart."""
        baseline = {"a": "(x)", "b": "class", "c": "(y)"}
        current = {"a": "(x)", "c": "(y, z=1)", "d": "constant"}
        difference = public_api.compare(current, baseline)
        assert difference == {
            "added": {"d": "constant"},
            "removed": ["b"],
            "changed": {"c": ("(y)", "(y, z=1)")},
        }

    def test_failure_message(self, tmp_path):
        """The failure shows every entry and says how to accept it."""
        path = tmp_path / "public_api.tsv"
        current = public_api.current_api()
        stale = dict(current)
        stale.pop("dascore.read")
        stale["dascore.gone"] = "(x)"
        stale["dascore.spool"] = "(changed)"
        public_api.write_baseline(stale, path)
        message = public_api.check(path)
        assert "1 added, 1 removed, 1 changed" in message
        assert f"added: dascore.read\t{current['dascore.read']}" in message
        assert "removed: dascore.gone" in message
        assert "changed: dascore.spool" in message
        assert "python scripts/public_api.py freeze" in message
        assert "justify each change" in message

    def test_main(self, tmp_path):
        """`freeze` then `check` passes; a stale baseline fails `check`."""
        path = tmp_path / "public_api.tsv"
        assert public_api.main(["freeze"], path) == 0
        assert path.read_text().startswith(public_api._HEADER)
        assert public_api.main(["check"], path) == 0
        public_api.write_baseline({"dascore.gone": "(x)"}, path)
        assert public_api.main(["check"], path) == 1
