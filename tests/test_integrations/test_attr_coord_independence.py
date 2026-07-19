"""
Regression tests for full attr/coord independence.

Attrs whose names look like flat coordinate metadata (``channel_step``,
``pulse_len``, ...) are ordinary attrs everywhere: model construction,
``update_attrs``, DASDAE round-trips, and the spool index. Coordinates are
never inferred from attr names. The one place the two worlds meet is the
flat ``get_contents()`` frame, where a genuine collision with a coordinate
envelope column warns and the coordinate wins.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.core.coordmanager import get_coord_manager

# The suffix-capture family from the original bug report: each of these was
# previously reinterpreted as coordinate metadata for a phantom coordinate.
COORD_SHAPED_NAMES = (
    "channel_step",
    "pulse_len",
    "sensor_fingerprint",
    "gain_max",
    "noise_min",
    "probe_dims",
    "x_units",
    "y_dtype",
    "gauge_length",
)


@pytest.fixture()
def patch_with_coord_shaped_attrs(random_patch):
    """A patch whose attrs include every coord-shaped name."""
    return random_patch.update_attrs(**{x: 1 for x in COORD_SHAPED_NAMES})


class TestCoordShapedNamesAreAttrs:
    """No attr name manufactures a phantom coordinate on any entry point."""

    @pytest.mark.parametrize("name", COORD_SHAPED_NAMES)
    def test_patch_attrs_construction(self, name):
        """Cause 1: PatchAttrs(...) keeps the attr, invents no coords."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            attrs = dc.PatchAttrs(**{name: 1})
        assert attrs[name] == 1

    @pytest.mark.parametrize("name", COORD_SHAPED_NAMES)
    def test_attrs_update(self, name, random_patch):
        """Cause 3: PatchAttrs.update accepts the name like any other."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            attrs = random_patch.attrs.update(**{name: 1})
        assert attrs[name] == 1

    def test_patch_update_attrs(self, patch_with_coord_shaped_attrs, random_patch):
        """Cause 3: update_attrs sets attrs and never touches coords."""
        patch = patch_with_coord_shaped_attrs
        for name in COORD_SHAPED_NAMES:
            assert patch.attrs[name] == 1
        assert patch.coords == random_patch.coords

    def test_subclass_field(self, random_patch):
        """Cause 3: an explicit subclass field (the Sintela pattern) works."""

        class _VendorAttrs(dc.PatchAttrs):
            channel_step: int | None = None

        attrs = _VendorAttrs(channel_step=3)
        patch = random_patch.update_attrs(**attrs.model_dump(exclude_unset=True))
        assert patch.attrs["channel_step"] == 3

    def test_real_coords_unaffected(self, random_patch):
        """Cause 2: supplying real coords plus a coord-shaped attr is safe."""
        patch = random_patch.update_attrs(channel_step=3)
        assert patch.coords.coord_map.keys() == random_patch.coords.coord_map.keys()
        assert patch.attrs["channel_step"] == 3


class TestCoordsNeverBuiltFromAttrs:
    """Cause 4: the get_coord_manager attrs branch is gone."""

    def test_attrs_param_removed(self):
        """The old crash repro now fails loudly at the signature."""
        attrs = {"time_min": 0, "time_max": 10, "time_step": 1, "channel_step": 3}
        with pytest.raises(TypeError, match="attrs"):
            get_coord_manager(None, dims=("time",), attrs=attrs)


class TestDasdaeRoundTrip:
    """Cause 7a: coord-shaped attrs survive DASDAE write/read exactly."""

    @pytest.fixture()
    def round_tripped(self, patch_with_coord_shaped_attrs, tmp_path_factory):
        """Write and read back a patch with coord-shaped attrs."""
        path = tmp_path_factory.mktemp("dasdae_attrs") / "patch.h5"
        patch_with_coord_shaped_attrs.io.write(path, "dasdae")
        return dc.spool(path)[0], dc.scan(path)[0]

    def test_read_keeps_attrs(self, round_tripped, patch_with_coord_shaped_attrs):
        """Attrs and coords round-trip unchanged."""
        patch, _ = round_tripped
        for name in COORD_SHAPED_NAMES:
            assert patch.attrs[name] == 1
        assert patch.coords == patch_with_coord_shaped_attrs.coords

    def test_scan_keeps_attrs(self, round_tripped):
        """Scan summaries carry the attrs too."""
        _, summary = round_tripped
        for name in COORD_SHAPED_NAMES:
            assert summary.attrs[name] == 1

    def test_shadowing_attr_round_trips(self, random_patch, tmp_path_factory):
        """An attr shadowing the patch's own coord envelope still survives."""
        patch = random_patch.rename_coords(distance="channel")
        patch = patch.update_attrs(channel_step=999)
        path = tmp_path_factory.mktemp("dasdae_shadow") / "patch.h5"
        patch.io.write(path, "dasdae")
        read_back = dc.spool(path)[0]
        assert read_back.attrs["channel_step"] == 999
        assert read_back.coords == patch.coords


class TestIndexKeepsCoordShapedAttrs:
    """Cause 5: ingest indexes every attr; no warn-and-drop."""

    @pytest.fixture()
    def spool_with_attrs(self, patch_with_coord_shaped_attrs):
        """An in-memory spool holding the attr-decorated patch."""
        return dc.spool([patch_with_coord_shaped_attrs])

    def test_no_warning_on_ingest(self, patch_with_coord_shaped_attrs):
        """Building and materializing the spool emits no warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spool = dc.spool([patch_with_coord_shaped_attrs])
            df = spool.get_contents()
        assert df["channel_step"].iloc[0] == 1

    def test_attrs_queryable(self, spool_with_attrs):
        """The attrs work as filters through the _attrs namespace."""
        assert len(spool_with_attrs.select(_attrs={"channel_step": 1})) == 1
        assert len(spool_with_attrs.select(_attrs={"channel_step": 2})) == 0


class TestGetContentsClobberWarning:
    """Cause 6: the one surviving check."""

    @pytest.fixture()
    def shadowing_patch(self, random_patch):
        """A patch with an attr equal to one of its own envelope columns."""
        patch = random_patch.rename_coords(distance="channel")
        return patch.update_attrs(channel_step=999)

    def test_warns_and_coord_wins(self, shadowing_patch):
        """Genuine collision warns; the coord envelope owns the column."""
        name = "channel_step"
        spool = dc.spool([shadowing_patch])
        with pytest.warns(UserWarning, match="collide with coordinate envelope"):
            df = spool.get_contents()
        # the flat column holds the coord step, not the attr value
        step = shadowing_patch.get_coord("channel").step
        assert df[name].iloc[0] == step
        # the attr remains queryable through the explicit namespace
        assert len(spool.select(_attrs={name: 999})) == 1

    def test_cross_patch_collision(self, random_patch):
        """An attr can collide with another patch's coord envelope."""
        renamed = random_patch.rename_coords(distance="channel")
        with_attr = random_patch.update_attrs(channel_step=3)
        # the warning fires on the first flat materialization, which a
        # multi-patch spool performs during construction
        with pytest.warns(UserWarning, match="collide with coordinate envelope"):
            spool = dc.spool([renamed, with_attr])
            df = spool.get_contents()
        assert "channel_step" in df.columns  # the envelope column survives

    def test_no_collision_no_warning(self, patch_with_coord_shaped_attrs):
        """Without a matching coord there is nothing to warn about."""
        spool = dc.spool([patch_with_coord_shaped_attrs])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = spool.get_contents()
        assert df["channel_step"].iloc[0] == 1

    def test_regex_query_reads_attr_under_collision(self, random_patch):
        """A regex _attrs query evaluates the attr, not the coord column."""
        import re

        # one patch carrying both the coord and a same-named string attr:
        # the flat column is the (numeric) envelope even in the filtered
        # frame, so the residual must fall back to the real attr values
        patch = random_patch.rename_coords(distance="channel")
        patch = patch.update_attrs(channel_min="vendor-a7")
        spool = dc.spool([patch])
        with pytest.warns(UserWarning, match="collide with coordinate envelope"):
            df = spool.get_contents()
        assert "vendor-a7" not in set(df["channel_min"].astype(str))
        hits = spool.select(_attrs={"channel_min": re.compile("vendor-.*")})
        assert len(hits) == 1
        misses = spool.select(_attrs={"channel_min": re.compile("nope")})
        assert len(misses) == 0


class TestFixedEnvelopeColumnsReserved:
    """time/distance envelopes are patches-table columns, reserved at ingest."""

    def test_fixed_envelope_attr_warns_and_skips(self, random_patch):
        """An attr named like a fixed envelope column warns and is skipped."""
        patch = random_patch.update_attrs(time_step=123.0)
        with pytest.warns(UserWarning, match="reserved attr name"):
            spool = dc.spool([patch])
            df = spool.get_contents()
        # the column keeps structural values; the attr stays on the patch
        assert df["time_step"].iloc[0] != 123.0
        assert spool[0].attrs["time_step"] == 123.0


class TestDerivedCatalogKeepsAttrs:
    """Chunk/concat derived records must preserve coord-shaped attrs."""

    def test_chunk_preserves_coord_shaped_attr(self, random_patch):
        """A channel_step attr (no channel coord) survives spool.chunk."""
        patch = random_patch.update_attrs(channel_step=3)
        chunked = dc.spool([patch]).chunk(time=1)
        df = chunked.get_contents()
        assert "channel_step" in df.columns
        assert (df["channel_step"] == 3).all()
        assert len(chunked.select(_attrs={"channel_step": 3})) == len(chunked)


class TestCompatStaysInDasdae:
    """The legacy-metadata compat layer must not bleed out of dasdae."""

    def test_import_graph(self):
        """Nothing outside dascore.io.dasdae imports the _compat module."""
        import re
        from pathlib import Path

        root = Path(dc.__file__).parent
        dasdae_dir = root / "io" / "dasdae"
        pattern = re.compile(r"\b_compat\b")
        outside = [
            str(path.relative_to(root))
            for path in root.rglob("*.py")
            if dasdae_dir not in path.parents
            and pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert not outside, f"_compat referenced outside dasdae: {outside}"


class TestLegacyDasdaeFile:
    """Cause 7b: legacy fixture files still parse correctly."""

    def test_legacy_fixture_reads(self):
        """The shipped legacy file loads with coords intact, attrs pure."""
        pytest.importorskip("pooch")
        from dascore.utils.downloader import fetch

        path = fetch("example_dasdae_event_1.h5")
        patch = dc.spool(path)[0]
        # coord metadata must have landed on coords, not attrs
        attr_names = set(patch.attrs.model_dump())
        for coord in patch.coords.coord_map:
            for field_ in ("min", "max", "step"):
                assert f"{coord}_{field_}" not in attr_names
        assert np.issubdtype(patch.coords.get_array("time").dtype, np.datetime64)
