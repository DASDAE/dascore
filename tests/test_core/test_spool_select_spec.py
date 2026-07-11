"""
Selector-spec behavior of Spool.select (all spool types).

These encode the hard-break semantics adopted for 0.2: unknown names
raise (#435), samples selections are patch-local (#447), relative ranges
work at spool level (#362), and _attrs/_coords disambiguate explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidSpoolQueryError


@pytest.fixture(scope="module", params=("memory", "directory"))
def spool(request, tmp_path_factory):
    """The same patches served by each spool type."""
    base = dc.get_example_spool("random_das")
    if request.param == "memory":
        return dc.spool(list(base))
    path = dc.examples.spool_to_directory(
        base, path=tmp_path_factory.mktemp("select_spec")
    )
    return dc.spool(path).update(progress=None)


class TestUnknownNames:
    """Unknown selector names raise eagerly (#435)."""

    def test_unknown_kwarg_raises(self, spool):
        """A name that is neither attr nor coord errors."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            spool.select(bad_dimension=(1, 2))

    def test_unknown_attr_namespace_raises(self, spool):
        """_attrs validates against attributes only."""
        with pytest.raises(InvalidSpoolQueryError, match="not an attribute"):
            spool.select(_attrs={"time": (None, None)})

    def test_unknown_coord_namespace_raises(self, spool):
        """_coords validates against coordinates only."""
        with pytest.raises(InvalidSpoolQueryError, match="not a coordinate"):
            spool.select(_coords={"tag": "random"})

    def test_double_specification_raises(self, spool):
        """A name can't be bare and namespaced at once."""
        with pytest.raises(InvalidSpoolQueryError, match="both"):
            spool.select(tag="random", _attrs={"tag": "random"})


class TestNamespaces:
    """Explicit namespaces select as their bare equivalents."""

    def test_attr_namespace(self, spool):
        """_attrs behaves like the bare attr kwarg."""
        assert len(spool.select(_attrs={"tag": "random"})) == len(spool)

    def test_coord_namespace(self, spool):
        """_coords behaves like the bare coord kwarg."""
        df = spool.get_contents()
        t0 = df["time_min"].min()
        out = spool.select(_coords={"time": (t0, t0 + np.timedelta64(2, "s"))})
        assert len(out) == 1


class TestCatalogPushdown:
    """Public spool selection composes a lazy SQLite query."""

    def test_coord_predicate_reaches_backend(self, spool, monkeypatch):
        """Selection does not query all rows before applying its predicate."""
        catalog = spool._catalog or spool._get_catalog()
        backend = catalog.backend
        calls = []
        original = backend.query

        def wrapped(query=None):
            calls.append(query)
            return original(query)

        monkeypatch.setattr(backend, "query", wrapped)
        selected = spool.select(time=("2020-01-03", "2020-01-04"))
        assert calls == []
        assert len(selected)
        queries = calls[0]
        assert isinstance(queries, list)
        assert queries[0].coords["time"] == ("2020-01-03", "2020-01-04")


class TestSamples:
    """samples=True never excludes patches; trims on load (#447)."""

    def test_length_preserved(self, spool):
        """The spool keeps every patch."""
        out = spool.select(distance=(0, 10), samples=True)
        assert len(out) == len(spool)

    def test_patch_trimmed_on_load(self, spool):
        """Loaded patches carry the sample trim."""
        out = spool.select(distance=(0, 10), samples=True)
        patch = out[0]
        assert len(patch.get_coord("distance")) == 10

    def test_survives_chunk(self, spool):
        """Post-selects propagate through derived spools."""
        out = spool.select(distance=(0, 10), samples=True).chunk(time=None)
        patch = out[0]
        assert len(patch.get_coord("distance")) == 10

    def test_non_coord_raises(self, spool):
        """Samples selections must name coordinates."""
        with pytest.raises(InvalidSpoolQueryError, match="coordinate-only"):
            spool.select(tag="random", samples=True)


class TestRelative:
    """relative=True resolves against the spool envelope (#362)."""

    def test_trims_both_ends(self, spool):
        """One second off each end of the spool."""
        df = spool.get_contents()
        gmin = df["time_min"].min()
        gmax = df["time_max"].max()
        out = spool.select(time=(1, -1), relative=True)
        merged = out.chunk(time=None)[0]
        time = merged.get_coord("time")
        assert time.min() >= np.datetime64(gmin) + np.timedelta64(1, "s")
        assert time.max() <= np.datetime64(gmax) - np.timedelta64(1, "s")

    def test_requires_range(self, spool):
        """Scalars are rejected with a clear message."""
        with pytest.raises(InvalidSpoolQueryError, match="requires"):
            spool.select(time=5, relative=True)

    def test_namespaced_coord_with_attr(self, spool):
        """Only the coordinate range is converted to relative offsets."""
        out = spool.select(_coords={"time": (1, -1)}, tag="random", relative=True)
        assert len(out)
        assert set(out.get_contents()["tag"]) == {"random"}


class TestExistingBehaviorKept:
    """The conventional selections still work."""

    def test_attr_glob(self, spool):
        """Unix-style attr matching."""
        assert len(spool.select(tag="rand*")) == len(spool)

    def test_time_range_narrows(self, spool):
        """Plain time range selection."""
        df = spool.get_contents()
        t0 = df["time_min"].min()
        out = spool.select(time=(t0, t0 + np.timedelta64(2, "s")))
        assert len(out) == 1
