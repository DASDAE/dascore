"""
Randomized heterogeneity stress test for the spool index.

Generates hundreds of patch summaries with randomized dimension names,
coord dtypes/units, attr names/kinds (including hostile names that
collide after sanitization), and verifies ingest, counts, and the
no-false-negative query contract on SQLite.
"""

from __future__ import annotations

import numpy as np
import pytest

from dascore.core.summary import PatchSummary
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import summaries_to_records
from dascore.io.index.query import Query
from dascore.units import get_quantity

N_PATCHES = 300

_DIM_POOL = (
    "time",
    "distance",
    "lag_time",
    "frequency",
    "velocity",
    "depth",
    "channel_number",
    "offset",
    "azimuth",
    "Gauge Length (m)",  # hostile coord name
)
_NUM_UNITS = (None, "m", "ft", "Hz", "1/m", "km")
_ATTR_NAMES = (
    "station",
    "tag",
    "shot_number",
    "Shot Number",  # sanitizes identically to shot_number
    "GaugeLength",
    "über_attr",
    "9lives",
    "data quality!",
    "processing_level",
)


def _random_coord(rng, name):
    """Build one random coord summary dict."""
    kind = rng.choice(["datetime", "timedelta", "float", "int", "str"])
    length = int(rng.integers(2, 5000))
    if kind == "datetime":
        start = np.datetime64("2024-01-01", "ns") + np.timedelta64(
            int(rng.integers(0, 10**7)), "s"
        )
        span = np.timedelta64(int(rng.integers(1, 10**5)), "s")
        return {
            "dtype": "datetime64",
            "min": start,
            "max": start + span,
            "step": span / max(length - 1, 1) if rng.random() > 0.3 else None,
            "dims": (name,),
            "len": length,
        }
    if kind == "timedelta":
        lo = np.timedelta64(int(rng.integers(-(10**6), 0)), "ms")
        hi = np.timedelta64(int(rng.integers(1, 10**6)), "ms")
        return {
            "dtype": "timedelta64",
            "min": lo,
            "max": hi,
            "step": None,
            "dims": (name,),
            "len": length,
        }
    if kind == "str":
        lo, hi = sorted(
            [f"A{rng.integers(0, 100):03d}", f"Z{rng.integers(0, 100):03d}"]
        )
        return {"dtype": "str", "min": lo, "max": hi, "dims": (name,), "len": length}
    lo = float(rng.uniform(-1000, 1000))
    hi = lo + float(rng.uniform(0.001, 5000))
    units = rng.choice(_NUM_UNITS)
    out = {
        "dtype": "float64" if kind == "float" else "int64",
        "min": lo if kind == "float" else int(lo),
        "max": hi if kind == "float" else int(hi) + 1,
        "step": None,
        "dims": (name,),
        "len": length,
    }
    if units is not None:
        out["units"] = units
    return out


def _random_attrs(rng) -> dict:
    """Build a random attr dict with mixed kinds and hostile names."""
    out = {}
    for name in _ATTR_NAMES:
        roll = rng.random()
        if roll < 0.35:
            continue  # attr missing on this patch
        if name in ("station", "tag"):
            # typed str fields on PatchAttrs; only extra attrs vary kind
            out[name] = f"v{rng.integers(0, 50)}"
            continue
        kind = rng.choice(["str", "int", "float", "bool", "time", "quantity"])
        if kind == "str":
            out[name] = f"v{rng.integers(0, 50)}"
        elif kind == "int":
            out[name] = int(rng.integers(-1000, 1000))
        elif kind == "float":
            out[name] = float(rng.uniform(-1000, 1000))
        elif kind == "bool":
            out[name] = bool(rng.random() > 0.5)
        elif kind == "time":
            out[name] = np.datetime64("2024-01-01") + np.timedelta64(
                int(rng.integers(0, 10**6)), "s"
            )
        else:
            # "s" is dimensionally incompatible with "m"/"ft": ingest keeps
            # the first-seen dimension and skips the rest with a warning.
            out[name] = float(rng.uniform(0, 100)) * get_quantity(
                str(rng.choice(["m", "ft", "s"]))
            )
    return out


def make_random_summaries(n: int, seed: int = 0) -> list[PatchSummary]:
    """Generate n randomized heterogeneous patch summaries."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_dims = int(rng.integers(1, 4))
        dims = tuple(rng.choice(_DIM_POOL, size=n_dims, replace=False))
        coords = {name: _random_coord(rng, name) for name in dims}
        shape = tuple(int(coords[d]["len"]) for d in dims)
        out.append(
            PatchSummary(
                attrs=_random_attrs(rng),
                coords=coords,
                dims=dims,
                shape=shape,
                dtype="float32",
                source_path=f"stress/file_{i:05d}.h5",
                source_format="DASDAE",
                source_version="1",
            )
        )
    return out


@pytest.fixture(scope="module")
def summaries():
    """The randomized summary population (deterministic seed)."""
    return make_random_summaries(N_PATCHES, seed=42)


@pytest.fixture()
def backend(tmp_path_factory, summaries):
    """A SQLite backend ingesting the random population."""
    path = tmp_path_factory.mktemp("stress") / "index.sqlite3"
    back = get_backend(path)
    back.write_sources(summaries_to_records(summaries))
    yield back
    back.close()


class TestStressIngest:
    """Every random population ingests completely."""

    def test_all_patches_indexed(self, backend):
        """One flat row per summary."""
        assert len(backend.query()) == N_PATCHES

    def test_all_coords_present(self, backend, summaries):
        """Every generated coord name is known to the index."""
        expected = {name for s in summaries for name in s.coords}
        assert expected <= backend.coord_names()

    def test_sanitize_collision_attrs_distinct(self, backend, summaries):
        """Attrs whose names sanitize identically stay distinct."""
        assert "shot_number" in backend.attr_names()
        assert "Shot Number" in backend.attr_names()
        df = backend.query()
        assert "shot_number" in df.columns and "Shot Number" in df.columns
        # every generated attr that carried a value is indexed
        expected = {
            name
            for s in summaries
            for name, value in s.attrs.model_dump().items()
            if name in _ATTR_NAMES and value not in (None, "")
        }
        assert expected <= backend.attr_names()


class TestStressNoFalseNegatives:
    """Random range queries never miss a matching patch."""

    def test_random_numeric_coord_ranges(self, backend, summaries):
        """Numeric coord queries: reference computed from raw summaries."""
        rng = np.random.default_rng(1)
        num_coords = ["distance", "frequency", "velocity", "depth", "offset"]
        for _ in range(20):
            name = str(rng.choice(num_coords))
            lo = float(rng.uniform(-500, 500))
            hi = lo + float(rng.uniform(1, 2000))
            got = set(backend.query(Query(coords={name: (lo, hi)}))["source_path"])
            for summary in summaries:
                csum = summary.coords.get(name)
                if csum is None:
                    continue
                # numeric queries must NOT match datetime/timedelta coords
                # (kind dispatch) — and numpy's type hierarchy makes
                # timedelta64 a signedinteger subtype, so check dtype.kind.
                if np.dtype(csum.dtype).kind not in "iuf":
                    continue
                # A bare range means each coordinate's native units, so the
                # oracle compares raw stored values — no unit conversion.
                if float(csum.min) <= hi and float(csum.max) >= lo:
                    assert str(summary.source_path).replace("\\", "/") in got, (
                        name,
                        lo,
                        hi,
                    )

    def test_random_time_ranges(self, backend, summaries):
        """Absolute time queries against datetime coords."""
        rng = np.random.default_rng(2)
        base = np.datetime64("2024-01-01", "ns")
        for _ in range(20):
            lo = base + np.timedelta64(int(rng.integers(0, 10**7)), "s")
            hi = lo + np.timedelta64(int(rng.integers(1, 10**6)), "s")
            got = set(backend.query(Query(coords={"time": (lo, hi)}))["source_path"])
            for summary in summaries:
                csum = summary.coords.get("time")
                if csum is None or "datetime" not in str(csum.dtype):
                    continue
                if csum.min <= hi and csum.max >= lo:
                    assert str(summary.source_path).replace("\\", "/") in got

    def test_attr_equality_roundtrip(self, backend, summaries):
        """Str attr equality returns every patch carrying that value."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            summary = summaries[int(rng.integers(0, len(summaries)))]
            attrs = {
                k: v
                for k, v in summary.attrs.model_dump().items()
                if isinstance(v, str) and v and k in set(_ATTR_NAMES)
            }
            if not attrs:
                continue
            name, value = next(iter(attrs.items()))
            got = set(backend.query(Query(attrs={name: value}))["source_path"])
            assert str(summary.source_path).replace("\\", "/") in got
