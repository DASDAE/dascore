"""Tests for DASDAE format."""

from __future__ import annotations

import pickle
import shutil
from typing import ClassVar

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.config import config_context
from dascore.core.coords import CoordString
from dascore.exceptions import (
    InvalidFiberFileError,
    MissingPatchError,
    ParameterError,
    PatchAttributeError,
)
from dascore.io.core import FiberIO
from dascore.io.dasdae import utils as dasdae_utils
from dascore.io.dasdae._compat import translate_legacy_attrs
from dascore.io.dasdae.core import DASDAEV1
from dascore.io.dasdae.utils import (
    _ATTRS_CLASS_KEY,
    _SEPARATE_ATTRS_KEY,
    _decode_attr_value,
    _decode_legacy_attr_value,
    _encode_attr_value,
    _get_attrs,
    _get_contents_from_patch_groups_generic,
    _get_coords,
    _get_file_version,
    _get_scan_payload_from_group,
    _save_array,
    _save_patch,
)
from dascore.io.odh4.core import ODH4PatchAttrs
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func
from dascore.utils.time import to_datetime64

# a list of fixture names for written DASDAE files
WRITTEN_FILES = []


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_random(random_patch, tmp_path_factory):
    """Write the example patch to disk."""
    path = tmp_path_factory.mktemp("dasdae_file") / "test.hdf5"
    dc.write(random_patch, path, "dasdae", file_version="1")
    return path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_empty(tmp_path_factory):
    """Write an empty patch to the dascore format."""
    path = tmp_path_factory.mktemp("empty_patch") / "empty.hdf5"
    patch = dc.Patch()
    dc.write(patch, path, "DASDAE", file_version="1")
    return path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_correlate(tmp_path_factory, random_patch):
    """Write a correlate patch to the dascore format."""
    path = tmp_path_factory.mktemp("correlate_patch") / "correlate.hdf5"
    padded_pa = random_patch.pad(time="correlate")
    dft_pa = padded_pa.dft("time", real=True)
    cc_pa = dft_pa.correlate(distance=[0, 1, 2], samples=True)
    dc.write(cc_pa, path, "DASDAE", file_version="1")
    return path


class TestWriteDASDAE:
    """Ensure the format can be written."""

    def test_append(self, written_dascore_v1_random, tmp_path_factory, random_patch):
        """Ensure files can be appended to unindexed dasdae file."""
        # make a copy of the dasdae file.
        new_path = tmp_path_factory.mktemp("dasdae_append") / "tmp.h5"
        shutil.copy(written_dascore_v1_random, new_path)
        # ensure the patch exists in the copied spool.
        df_pre = dc.spool(new_path).get_contents()
        assert len(df_pre) == 1
        # append patch to dasdae file
        new_patch = random_patch.update_coords(time_min="1990-01-01")
        dc.write(new_patch, new_path, "DASDAE")
        # ensure the file has grown in contents
        df = dc.spool(new_path).get_contents()
        assert len(df) == len(df_pre) + 1
        assert (df["time_min"] == to_datetime64("1990-01-01")).any()

    def test_write_again(self, written_dascore_v1_random, random_patch):
        """Ensure a patch can be written again to file (should overwrite old)."""
        random_patch.io.write(written_dascore_v1_random, "dasdae")
        read_patch = dc.spool(written_dascore_v1_random)[0]
        assert random_patch == read_patch

    def test_write_cc_patch(self, written_dascore_correlate):
        """Ensure cross correlated patches can be written and read."""
        sp_cc = dc.spool(written_dascore_correlate)
        assert isinstance(sp_cc[0], dc.Patch)


class TestReadDASDAE:
    """Test for reading a dasdae format."""

    def test_round_trip_empty_patch(self, written_dascore_v1_empty):
        """Ensure an empty patch can be deserialized."""
        spool = dc.read(written_dascore_v1_empty)
        assert len(spool) == 1
        spool[0].equals(dc.Patch())

    def test_append_to_legacy_file_keeps_new_attrs(
        self, random_patch, tmp_path_factory
    ):
        """New groups appended to a legacy file must not be legacy-stripped."""
        path = tmp_path_factory.mktemp("dasdae_legacy_append") / "mixed.h5"
        old_patch = random_patch.update_attrs(tag="old")
        old_patch.io.write(path, "dasdae")
        # simulate a legacy file: strip the root and group markers
        with h5py.File(path, "a") as h5:
            del h5.attrs[_SEPARATE_ATTRS_KEY]
            for group in h5["waveforms"].values():
                del group.attrs[_SEPARATE_ATTRS_KEY]
        # append a new patch carrying an attr shadowing its own coord envelope
        dim = random_patch.dims[0]
        new_patch = random_patch.update_attrs(tag="new", **{f"{dim}_step": 999})
        new_patch.io.write(path, "dasdae")
        with h5py.File(path, "r") as h5:
            assert not h5.attrs.get(_SEPARATE_ATTRS_KEY, False)  # file stays legacy
        patches = {p.attrs["tag"]: p for p in dc.read(path, file_format="DASDAE")}
        assert set(patches) == {"old", "new"}
        # the appended group round-trips exactly; the legacy one still reads
        assert patches["new"].attrs[f"{dim}_step"] == 999
        assert patches["old"].coords == old_patch.coords

    def test_datetimes(self, tmp_path_factory, random_patch):
        """Ensure the datetimes in the attrs come back as datetimes."""
        # create a patch with a custom dt attribute.
        path = tmp_path_factory.mktemp("dasdae_dt_saves") / "rt.h5"
        dt = np.datetime64("2010-09-12")
        patch = random_patch.update_attrs(custom_dt=dt)
        patch.io.write(path, "dasdae")
        patch_2 = dc.read(path)[0]
        # make sure custom tag with dt comes back from read.
        assert patch_2.attrs["custom_dt"] == dt
        # test coords are still dt64
        array = patch_2.coords.get_array("time")
        assert np.issubdtype(array.dtype, np.datetime64)
        # test attrs
        time_summary = patch_2.summary.get_coord_summary("time")
        for value in (time_summary.min, time_summary.max):
            assert isinstance(value, np.datetime64)

    def test_read_file_no_wavegroup(self, generic_hdf5):
        """Ensure an h5 with no wavegroup returns empty patch."""
        parser = DASDAEV1()
        spool = parser.read(generic_hdf5)
        assert not len(spool)

    def test_read_source_patch_key(self, tmp_path):
        """Reading with a source patch id should only load one patch."""
        path = tmp_path / "multi_patch.h5"
        spool = dc.examples.get_example_spool("random_das", length=2)
        dc.write(spool, path, "DASDAE", file_version="1")
        scanned = dc.scan(path)
        target = scanned[1].source_patch_key
        out = dc.read(path, source_patch_key=target)
        assert len(out) == 1
        assert out[0].attrs["_source_patch_key"] == target
        assert out[0].summary.source_patch_key == out[0].attrs["_source_patch_key"]
        assert (
            out[0].summary.get_coord_summary("time").min
            == scanned[1].get_coord_summary("time").min
        )

    def test_read_multiple_source_patch_keys(self, tmp_path):
        """Reading with multiple source patch ids should return each match."""
        path = tmp_path / "multi_patch.h5"
        spool = dc.examples.get_example_spool("random_das", length=3)
        dc.write(spool, path, "DASDAE", file_version="1")
        scanned = dc.scan(path)
        targets = [scanned[0].source_patch_key, scanned[2].source_patch_key]
        out = dc.read(path, source_patch_key=targets)
        assert len(out) == 2
        assert {patch.attrs["_source_patch_key"] for patch in out} == set(targets)
        assert {patch.summary.get_coord_summary("time").min for patch in out} == {
            scanned[0].get_coord_summary("time").min,
            scanned[2].get_coord_summary("time").min,
        }

    def test_read_ignores_multi_dim_coord_filters(
        self, tmp_path, multi_dim_coords_patch
    ):
        """Multi-dimensional coord kwargs should bypass coord selection safely."""
        path = tmp_path / "multi_dim_filter.h5"
        multi_dim_coords_patch.io.write(path, "dasdae")

        out = dc.read(path, quality=(0, 1))[0]

        assert out == multi_dim_coords_patch

    def test_file_spool_loads_distinct_attrs(self, tmp_path, random_patch):
        """Lazy loading should materialize the patch for each DASDAE row."""
        path = tmp_path / "multi_patch.h5"
        patches = [
            random_patch.update_attrs(tag="S100", label="L100"),
            random_patch.update_attrs(tag="S120", label="L120"),
        ]

        dc.write(dc.spool(patches), path, "DASDAE")
        spool = dc.spool(path)

        assert spool.get_contents()["tag"].to_list() == ["S100", "S120"]
        assert [x.attrs.tag for x in spool] == ["S100", "S120"]
        assert [x.attrs.label for x in spool] == ["L100", "L120"]

    def test_read_filters_patch_attrs_before_loading(self, tmp_path, random_patch):
        """DASDAE attr filters should skip non-matching patch groups."""
        path = tmp_path / "multi_patch.h5"
        patches = [
            random_patch.update_attrs(tag="S100"),
            random_patch.update_attrs(tag="S120"),
        ]
        dc.write(dc.spool(patches), path, "DASDAE")

        out = dc.read(path, tag="S120")

        assert len(out) == 1
        assert out[0].attrs.tag == "S120"


@pytest.fixture(scope="class")
def multi_patch_path(tmp_path_factory):
    """A DASDAE file holding three patches with distinct data."""
    path = tmp_path_factory.mktemp("read_array") / "multi.h5"
    spool = dc.examples.get_example_spool("random_das", length=3)
    patches = [p.update(data=p.data + i * 100) for i, p in enumerate(spool)]
    dc.write(dc.spool(patches), path, "DASDAE", file_version="1")
    return path


class TestReadArray:
    """Tests for the data-only hyperslab read."""

    def test_matches_default(self, written_dascore_v1_random, random_patch):
        """The override returns exactly what the read-and-trim default does."""
        io = DASDAEV1()
        windows = {"time": (3, 11), "distance": (2, 5)}
        out = io.read_array(written_dascore_v1_random, windows)
        expected = FiberIO.read_array(io, written_dascore_v1_random, windows)
        assert np.array_equal(out, expected)
        assert out.dtype == random_patch.data.dtype
        assert np.array_equal(out, random_patch.data[2:5, 3:11])

    def test_whole_array(self, written_dascore_v1_random, random_patch):
        """No windows returns the whole array."""
        out = DASDAEV1().read_array(written_dascore_v1_random, {})
        assert np.array_equal(out, random_patch.data)

    def test_keyed_patch(self, multi_patch_path):
        """A source patch key picks that patch's data."""
        io = DASDAEV1()
        payloads = dc.scan(multi_patch_path)
        assert len(payloads) == 3
        for payload in payloads:
            key = payload.source_patch_key
            patch = dc.read(multi_patch_path, source_patch_key=key)[0]
            out = io.read_array(
                multi_patch_path, {"time": (1, 4)}, source_patch_key=key
            )
            assert np.array_equal(out, patch.data[:, 1:4])

    def test_null_key_resolves_single_patch(self, written_dascore_v1_random):
        """A NaN key (an index row with no stored key) means the lone patch."""
        out = DASDAEV1().read_array(
            written_dascore_v1_random, {}, source_patch_key=np.nan
        )
        expected = DASDAEV1().read_array(written_dascore_v1_random, {})
        assert np.array_equal(out, expected)

    def test_strided_window_refused(self, written_dascore_v1_random):
        """A window is a (start, stop) pair; a stride is not part of the contract."""
        with pytest.raises(ParameterError, match="pair"):
            DASDAEV1().read_array(written_dascore_v1_random, {"time": (2, 6, 2)})

    def test_legacy_file(self):
        """A file written before attrs and coords were separated slices the same."""
        io = DASDAEV1()
        path = fetch("UoU_lf_urban.hdf5")
        windows = {"time": (4, 9)}
        out = io.read_array(path, windows)
        assert np.array_equal(out, FiberIO.read_array(io, path, windows))

    def test_open_and_float_bounds(self, written_dascore_v1_random, random_patch):
        """None or ... leaves an end open, as the default does; floats are refused."""
        io = DASDAEV1()
        out = io.read_array(written_dascore_v1_random, {"time": (..., 6)})
        assert np.array_equal(out, random_patch.data[:, :6])
        out = io.read_array(written_dascore_v1_random, {"time": (3, None)})
        assert np.array_equal(out, random_patch.data[:, 3:])
        with pytest.raises(ParameterError, match="integers"):
            io.read_array(written_dascore_v1_random, {"time": (2.0, 6.0)})

    def test_path_like_key_raises(self, multi_patch_path):
        """A key h5py would read as a path names no patch."""
        for key in ("/waveforms", ".", "..", "data"):
            with pytest.raises(PatchAttributeError, match="No patch named"):
                DASDAEV1().read_array(multi_patch_path, {}, source_patch_key=key)

    def test_keyless_multi_patch_raises(self, multi_patch_path):
        """Several patches and no key cannot be resolved."""
        with pytest.raises(PatchAttributeError, match="source_patch_key"):
            DASDAEV1().read_array(multi_patch_path, {})

    def test_unknown_key_raises(self, multi_patch_path):
        """A key naming no patch group raises."""
        with pytest.raises(PatchAttributeError, match="No patch named"):
            DASDAEV1().read_array(multi_patch_path, {}, source_patch_key="nope")

    def test_no_patches_raises(self, generic_hdf5):
        """A file without a waveform group has nothing to read."""
        with pytest.raises(MissingPatchError, match="No patches"):
            DASDAEV1().read_array(generic_hdf5, {})

    def test_unknown_dimension_raises(self, written_dascore_v1_random):
        """A window on a dimension the patch lacks raises."""
        with pytest.raises(ParameterError, match="not among patch dims"):
            DASDAEV1().read_array(written_dascore_v1_random, {"bob": (0, 1)})

    def test_load_filters_refused(self, written_dascore_v1_random):
        """Value filters are not part of the window contract."""
        with pytest.raises(ParameterError, match="takes only windows"):
            DASDAEV1().read_array(written_dascore_v1_random, {}, time_min=1)

    def test_reads_only_the_window(self, written_dascore_v1_random, monkeypatch):
        """The dataset is sliced in the file, not read whole then trimmed."""
        seen = []
        original = h5py.Dataset.__getitem__

        def spy(self, index):
            seen.append(index)
            return original(self, index)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
        DASDAEV1().read_array(written_dascore_v1_random, {"time": (2, 6)})
        assert seen == [(slice(None), slice(2, 6))]


class TestScanDASDAE:
    """Tests for scanning the dasdae format."""

    def test_scan_returns_info(self, written_dascore_v1_random, random_patch):
        """Ensure scanning returns expected values."""
        info1 = dc.scan(written_dascore_v1_random)[0].attrs.model_dump()
        info2 = random_patch.attrs.model_dump()
        # History is excluded because writing is not an operation; the
        # ids are not, because a file carries them and that is the point.
        common_keys = set(info1) & set(info2) - {"history"}
        assert info1["patch_id"] == info2["patch_id"]
        for key in common_keys:
            assert info1[key] == info2[key]

    def test_get_patch_summary_has_file_metadata(self, random_spool):
        """The summary helper should stamp DASDAE metadata on each row."""
        out = DASDAEV1()._get_patch_summary(random_spool)
        assert set(out["source_format"]) == {"DASDAE"}
        assert set(out["source_version"]) == {"1"}
        assert out["source_patch_key"].notnull().all()


class TestAttrsClassRoundTrip:
    """A file names the attrs class it holds, so a custom one comes back."""

    @pytest.fixture(scope="class")
    def odh4_patch(self, random_patch):
        """A patch carrying a format's own attrs class."""
        attrs = ODH4PatchAttrs(**dict(random_patch.attrs), gauge_length=10.0)
        return random_patch.update(attrs=attrs)

    @pytest.fixture(scope="class")
    def odh4_path(self, odh4_patch, tmp_path_factory):
        """The patch above, written to a DASDAE file."""
        path = tmp_path_factory.mktemp("attrs_class") / "odh4.h5"
        odh4_patch.io.write(path, "dasdae")
        return path

    def test_read_keeps_the_class(self, odh4_path):
        """Reading rebuilds the class which was written."""
        attrs = dc.read(odh4_path)[0].attrs
        assert isinstance(attrs, ODH4PatchAttrs)
        assert attrs.gauge_length == 10.0

    def test_scan_keeps_the_class(self, odh4_path):
        """Scanning takes the same path, so it says the same thing."""
        assert isinstance(dc.scan(odh4_path)[0].attrs, ODH4PatchAttrs)

    def test_a_file_naming_no_class_reads_as_patch_attrs(self, odh4_path, tmp_path):
        """A file written before the class was recorded still reads."""
        path = tmp_path / "unnamed.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            for group in h5["waveforms"].values():
                del group.attrs[_ATTRS_CLASS_KEY]
        attrs = dc.read(path)[0].attrs
        assert type(attrs) is dc.PatchAttrs
        # The values were always kept; it is the class which was lost.
        assert attrs.gauge_length == 10.0

    def test_an_unresolvable_class_warns_and_falls_back(self, odh4_path, tmp_path):
        """An archive written by a package we lack is still readable."""
        path = tmp_path / "foreign.h5"
        shutil.copy(odh4_path, path)
        with h5py.File(path, "a") as h5:
            for group in h5["waveforms"].values():
                group.attrs[_ATTRS_CLASS_KEY] = "absent:ForeignAttrs"
        with pytest.warns(UserWarning, match="Nothing registers"):
            attrs = dc.read(path)[0].attrs
        assert type(attrs) is dc.PatchAttrs
        assert attrs.gauge_length == 10.0


class TestLegacyFixtureCompatibility:
    """Tests for the retained legacy DASDAE fixture compatibility helpers."""

    def test_translate_legacy_attrs_coord_manager_like_coords(self):
        """Legacy coord managers should still flatten via to_summary_dict."""

        class CoordManagerLike:
            def to_summary_dict(self):
                return {"time": {"units": "s", "step": 1}}

        out = translate_legacy_attrs({"coords": CoordManagerLike()})
        assert out["time_units"] == "s"
        assert out["time_step"] == 1

    def test_translate_legacy_attrs_summary_like_coord(self):
        """Legacy coord summaries should normalize via to_summary/model_dump."""

        class SummaryLike:
            def to_summary(self):
                return dc.core.CoordSummary(min=0, max=1, step=2, units="m")

        out = translate_legacy_attrs(
            {
                "coords": {"distance": SummaryLike(), "time": object()},
                "dims": "distance,time",
                "d_time": 3,
            }
        )
        assert out["distance_units"] == "m"
        assert out["distance_step"] == 2
        assert out["time_step"] == 3

    def test_translate_legacy_attrs_ignores_non_mapping_coords(self):
        """Undecodable string coord payloads are ignored once opted in."""
        with config_context(allow_dasdae_format_unpickle=True):
            out = translate_legacy_attrs({"coords": "pickled-coords-placeholder"})
        assert "coords" not in out

    def test_translate_legacy_attrs_decodes_pickled_coord_payload(self):
        """Legacy pickled coord payloads should still restore coord metadata."""
        payload = pickle.dumps({"distance": {"min": 0, "max": 1, "units": "m"}})
        with config_context(allow_dasdae_format_unpickle=True):
            out = translate_legacy_attrs({"coords": payload.decode("latin1")})
        assert out["distance_units"] == "m"
        assert out["distance_min"] == 0
        assert out["distance_max"] == 1

    def test_translate_legacy_attrs_never_unpickles_without_opt_in(self):
        """The opt-in gate must fire before any pickle.loads call runs."""
        executed = []

        class _Payload:
            def __reduce__(self):
                return (executed.append, ("pickle ran",))

        payload = pickle.dumps(_Payload()).decode("latin1")
        with config_context(allow_dasdae_format_unpickle=False):
            with pytest.raises(InvalidFiberFileError, match="unpickle"):
                translate_legacy_attrs({"coords": payload})
        assert not executed, "pickle.loads ran before the security gate"

    def test_scan_preserves_legacy_coord_units_from_attr_payload(self):
        """Legacy attr coord units should backfill missing coord-node units."""
        with config_context(allow_dasdae_format_unpickle=True):
            summary = dc.scan(fetch("UoU_lf_urban.hdf5"))[0]
        assert str(summary.coords["distance"].units) == "1 m"

    def test_read_legacy_coord_payload_requires_opt_in(self):
        """Legacy pickled coord metadata should fail closed by default."""
        with config_context(allow_dasdae_format_unpickle=False):
            with pytest.raises(
                InvalidFiberFileError, match="allow_dasdae_format_unpickle=True"
            ):
                dc.read(fetch("UoU_lf_urban.hdf5"))

    def test_scan_prefers_exact_coord_over_legacy_step_metadata(self, tmp_path):
        """Exact coord scans should not invent steps from legacy summary metadata."""
        path = tmp_path / "legacy_coord_step.h5"
        payload = pickle.dumps(
            {
                "distance": {
                    "min": 0.0,
                    "max": 3.0,
                    "step": 1.0,
                    "units": "m",
                    "dtype": "float64",
                }
            }
        )
        with h5py.File(path, "w") as h5:
            h5.attrs["__format__"] = "DASDAE"
            h5.attrs["__DASDAE_version__"] = "1"
            waveforms = h5.create_group("waveforms")
            group = waveforms.create_group("patch")
            group.attrs["_dims"] = "distance"
            group.attrs["_attrs_coords"] = np.bytes_(payload)
            group.attrs["_cdims_distance"] = "distance"
            group.create_dataset("_coord_distance", data=np.array([0.0, 1.0, 3.0]))
            summary = _get_scan_payload_from_group(group)
        assert summary["coords"]["distance"].step is None

    def test_decode_legacy_attr_bytes_falls_back_to_text(self):
        """Undecodable legacy bytes should fall back to plain text."""
        assert _decode_legacy_attr_value({}, "key", b"abc") == "abc"

    def test_decode_legacy_attr_pickled_bytes_need_the_attrs(self):
        """A payload can only be told from text by the stored attribute."""
        payload = pickle.dumps(("a", "b"), 0)
        assert isinstance(_decode_legacy_attr_value({}, "history", payload), str)

    def test_decode_legacy_attr_unboxes_scalar_arrays(self):
        """Scalar legacy arrays should be unpacked back to scalars."""
        assert _decode_legacy_attr_value({}, "key", np.asarray(5)) == 5


class TestPyTablesAttrPayloads:
    """PyTables pickled attr values HDF5 had no native type for."""

    @pytest.fixture
    def pytables_attrs(self, tmp_path):
        """Read attrs back from a group whose attrs mimic PyTables output."""

        def _read(legacy=True, **values):
            path = tmp_path / "pytables_attrs.h5"
            with h5py.File(path, "w") as h5:
                group = h5.create_group("waveforms").create_group("patch")
                for name, value in values.items():
                    # PyTables wrote payloads as bytes, which HDF5 records
                    # with the ASCII character set, and text as UTF-8.
                    group.attrs[f"_attrs_{name}"] = np.bytes_(pickle.dumps(value, 0))
                return _get_attrs(group, legacy=legacy)

        return _read

    def test_read_decodes_none(self, pytables_attrs):
        """A pickled None should read back as None, not as its payload."""
        assert pytables_attrs(gauge_length=None)["gauge_length"] is None

    def test_read_decodes_history(self, pytables_attrs):
        """A pickled history tuple should read back as a tuple."""
        attrs = pytables_attrs(history=("first", "second"))
        assert attrs["history"] == ("first", "second")

    def test_payload_needs_the_opt_in(self, pytables_attrs):
        """Without the opt-in an attr payload stays the text it was."""
        with config_context(allow_dasdae_format_unpickle=False):
            assert pytables_attrs(gauge_length=None)["gauge_length"] == "N."

    def test_undecodable_payload_stays_text(self, tmp_path):
        """Bytes which are not a pickle keep the text they would have had."""
        path = tmp_path / "not_a_pickle.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch")
            group.attrs["_attrs_station"] = np.bytes_(b"not a pickle.")
            assert _get_attrs(group)["station"] == "not a pickle."

    def test_text_attr_is_not_decoded(self, tmp_path):
        """A real string which happens to be a valid payload stays text."""
        path = tmp_path / "text_attr.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch")
            # PyTables stored a real string as fixed-length UTF-8 and a
            # payload as raw bytes. "N." is byte-identical to a pickled
            # None, so only the character set separates the two, and the
            # value has to be written the way PyTables wrote text for the
            # test to reach that comparison at all.
            group.attrs.create(
                "_attrs_station",
                np.bytes_(b"N."),
                dtype=h5py.string_dtype(encoding="utf-8", length=2),
            )
            attrs = group.attrs
            assert isinstance(attrs["_attrs_station"], np.bytes_ | bytes)
            assert attrs.get_id("_attrs_station").get_type().get_cset() == 1
            assert _get_attrs(group)["station"] == "N."

    def test_attr_name_repeating_the_prefix(self, tmp_path):
        """The stored name must be recovered exactly, not by substitution."""
        path = tmp_path / "repeated_prefix.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch")
            group.attrs["_attrs_my_attrs_thing"] = np.bytes_(pickle.dumps(None, 0))
            assert _get_attrs(group)["my_attrs_thing"] is None

    def test_modern_file_is_not_decoded(self, pytables_attrs):
        """Only legacy files hold PyTables payloads."""
        assert pytables_attrs(legacy=False, note=None)["note"] == "N."

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("example_event_1", ()),
            ("deformation_rate_event_1", ()),
        ],
    )
    def test_shipped_file_history(self, name, expected):
        """A shipped legacy file states its history, not its payload (#1078)."""
        assert dc.get_example_patch(name).attrs.history == expected

    def test_shipped_file_history_text(self):
        """A shipped legacy file with real history states it as entries."""
        history = dc.get_example_patch("febus_dss_mine_tight").attrs.history
        assert isinstance(history, tuple)
        assert history and all(isinstance(x, str) for x in history)
        assert not history[0].startswith("(V")

    def test_shipped_file_gauge_length(self):
        """The example the strain docstrings use states no gauge length."""
        patch = dc.get_example_patch("deformation_rate_event_1")
        assert getattr(patch.attrs, "gauge_length", None) is None


class TestLegacyUnitCompanions:
    """Released files stored ``{name}_units`` beside now fixed-unit attrs."""

    def test_companion_converts_value_and_is_dropped(self):
        """A stored companion converts the value to the contract unit."""
        out = translate_legacy_attrs({"pulse_width": 50.0, "pulse_width_units": "ns"})
        assert out["pulse_width"] == pytest.approx(50.0e-9)
        assert "pulse_width_units" not in out

    def test_empty_companion_is_dropped_without_conversion(self):
        """Released attrs defaulted the companion to '', meaning unstated."""
        out = translate_legacy_attrs({"pulse_rate": 500.0, "pulse_rate_units": ""})
        assert out["pulse_rate"] == 500.0
        assert "pulse_rate_units" not in out

    def test_value_without_companion_is_untouched(self):
        """No companion means the value already speaks the contract unit."""
        out = translate_legacy_attrs({"gauge_length": 10.0})
        assert out["gauge_length"] == 10.0

    def test_companion_without_value_is_dropped(self):
        """A stray companion beside no value is legacy noise."""
        out = translate_legacy_attrs({"pulse_width_units": "m"})
        assert out == {}

    def test_length_pulse_width_migrates_to_pulse_length(self):
        """Febus cached the pulse as meters; that quantity is pulse_length."""
        out = translate_legacy_attrs({"pulse_width": 10.0, "pulse_width_units": "m"})
        assert "pulse_width" not in out
        assert "pulse_width_units" not in out
        assert out["pulse_length"] == 10.0
        assert "pulse_length_units" not in out

    def test_pulse_width_ns_key_migrates(self):
        """xml_binary spelled the units into the key name."""
        out = translate_legacy_attrs({"pulse_width_ns": 50.0})
        assert "pulse_width_ns" not in out
        assert out["pulse_width"] == pytest.approx(50.0e-9)

    def test_empty_companion_keeps_pulse_width_temporal(self):
        """Unstated units mean seconds already; the name must not migrate."""
        out = translate_legacy_attrs({"pulse_width": 5e-8, "pulse_width_units": ""})
        assert out == {"pulse_width": 5e-8}

    def test_length_pulse_width_never_clobbers_pulse_length(self):
        """An existing pulse_length wins; the ambiguous value drops loudly."""
        legacy = {"pulse_width": 7.0, "pulse_width_units": "m", "pulse_length": 99.0}
        with pytest.warns(UserWarning, match="pulse_width"):
            out = translate_legacy_attrs(legacy)
        assert out["pulse_length"] == 99.0
        assert "pulse_width" not in out

    def test_coord_named_like_companion_attr_is_left_alone(self):
        """A coordinate sharing an attr name keeps its coord metadata."""
        out = translate_legacy_attrs(
            {
                "coords": {"gauge_length": {"min": 0.0, "max": 1.0, "units": "ft"}},
                "gauge_length": 10.0,
            }
        )
        assert out["gauge_length"] == 10.0  # not rescaled by the coord's units
        assert out["gauge_length_units"] == "ft"  # kept for coord reconstruction

    def test_stored_coord_names_protect_flat_units_keys(self):
        """A flat-key-era coordinate's units key is not a value companion."""
        legacy = {"pulse_rate_min": 0.0, "pulse_rate_max": 5.0}
        legacy["pulse_rate_units"] = "1/s"
        out = translate_legacy_attrs(legacy, coord_names={"pulse_rate"})
        assert out["pulse_rate_units"] == "1/s"

    def test_legacy_cache_matches_fresh_parse(self, random_patch, tmp_path):
        """A patch from an old cache must equal one freshly parsed."""
        path = tmp_path / "legacy_units.h5"
        patch = random_patch.update_attrs(gauge_length=32.0, gauge_length_units="ft")
        patch.io.write(path, "dasdae")
        # simulate a legacy file: strip the root and group markers
        with h5py.File(path, "a") as h5:
            del h5.attrs[_SEPARATE_ATTRS_KEY]
            for group in h5["waveforms"].values():
                del group.attrs[_SEPARATE_ATTRS_KEY]
        back = dc.read(path, file_format="DASDAE")[0]
        assert back.attrs["gauge_length"] == pytest.approx(32.0 * 0.3048)
        assert back.attrs.get("gauge_length_units") is None
        summary = dc.scan(path, file_format="DASDAE")[0]
        assert summary.attrs["gauge_length"] == pytest.approx(32.0 * 0.3048)


class TestDASDAEInternalHelpers:
    """Direct tests for h5py-backed DASDAE helper branches."""

    def test_save_array_overwrites_existing_dataset(self, tmp_path):
        """Saving to an existing dataset name should replace it."""
        path = tmp_path / "overwrite_array.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms")
            _save_array(np.arange(2), "data", group=group)
            _save_array(np.arange(3), "data", group=group)
            assert np.array_equal(group["data"][:], np.arange(3))

    def test_save_patch_overwrites_existing_group(self, random_patch, tmp_path):
        """Saving a patch with the same name should replace the old group."""
        path = tmp_path / "overwrite_patch.h5"
        with h5py.File(path, "w") as h5:
            waveforms = h5.create_group("waveforms")
            _save_patch(random_patch, waveforms, "patch_0")
            _save_patch(random_patch.update_attrs(tag="new"), waveforms, "patch_0")
            attrs = _get_attrs(waveforms["patch_0"])
            assert attrs["tag"] == "new"

    def test_get_patch_summary_unpacks_scalar_attr_array(self, tmp_path):
        """Scalar encoded attrs should be unpacked in patch summaries."""
        path = tmp_path / "summary_scalar.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_attrs_station"] = np.asarray("A01", dtype=h5py.string_dtype())
            group.attrs["_cdims_time"] = "time"
            group.create_dataset("_coord_time", data=np.array([0, 1]))
            summary = _get_scan_payload_from_group(group)
        assert summary["attrs"].station == "A01"
        assert summary["dtype"] == ""

    def test_get_attrs_unpacks_scalar_attr_arrays(self, monkeypatch):
        """Scalar arrays returned by attr decoding should be unpacked."""

        class _Group:
            attrs: ClassVar[dict[str, str]] = {"_attrs_station": "unused"}

        monkeypatch.setattr(
            dasdae_utils,
            "_decode_attr_value",
            lambda *_args, **_kwargs: np.asarray("A01"),
        )
        assert _get_attrs(_Group()) == {"station": "A01"}

    def test_get_patch_summary_unpacks_scalar_arrays_from_decoder(
        self, tmp_path, monkeypatch
    ):
        """Patch summaries should unpack scalar arrays returned by decoding."""
        path = tmp_path / "summary_scalar_decoder.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_attrs_station"] = "unused"
            group.attrs["_cdims_time"] = "time"
            group.create_dataset("_coord_time", data=np.array([0, 1]))
            monkeypatch.setattr(
                dasdae_utils,
                "_decode_attr_value",
                lambda *_args, **_kwargs: np.asarray("A01"),
            )
            summary = _get_scan_payload_from_group(group)
        assert summary["attrs"].station == "A01"

    def test_get_patch_summary_preserves_empty_dims_and_shape(self, tmp_path):
        """Empty stored dims should remain empty tuples in summaries."""
        path = tmp_path / "summary_empty_dims.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = ""
            group.create_dataset("data", data=np.arange(6).reshape(2, 3))
            summary = _get_scan_payload_from_group(group)
        assert summary["dims"] == ()
        assert summary["shape"] == (2, 3)

    def test_get_contents_from_patch_groups_returns_empty_without_waveforms(
        self, tmp_path
    ):
        """Files without waveforms should scan as empty."""
        path = tmp_path / "empty_scan.h5"
        with h5py.File(path, "w") as h5:
            out = _get_contents_from_patch_groups_generic(h5)
            assert out == []

    def test_get_coords_range_like_node_skips_full_array_read(
        self, tmp_path, monkeypatch
    ):
        """Range-like coord nodes should reconstruct without materializing arrays."""
        path = tmp_path / "range_coord_fast_path.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_cdims_time"] = "time"
            node = group.create_dataset("_coord_time", data=np.array([10, 20, 30]))
            node.attrs["step"] = 10
            node.attrs["step_is_timedelta64"] = False

            def _forbid_full_read(*_args, **_kwargs):
                raise AssertionError("full coord reads should be skipped")

            monkeypatch.setattr(dasdae_utils, "_read_array", _forbid_full_read)
            coords = _get_coords(group, ("time",), {})

        coord = coords.get_coord("time")
        assert coord.__class__.__name__ == "CoordRange"
        assert len(coord) == 3
        assert coord.start == 10
        assert coord.step == 10

    def test_get_coords_range_like_node_restores_timedelta_sample(
        self, tmp_path, monkeypatch
    ):
        """Range fast path should restore timedelta step/sample metadata."""
        path = tmp_path / "range_coord_timedelta_fast_path.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_cdims_time"] = "time"
            node = group.create_dataset(
                "_coord_time", data=np.array([1, 3, 5], dtype="int64")
            )
            node.attrs["is_timedelta64"] = True
            node.attrs["step"] = 2
            node.attrs["step_is_timedelta64"] = True

            def _forbid_full_read(*_args, **_kwargs):
                raise AssertionError("full coord reads should be skipped")

            monkeypatch.setattr(dasdae_utils, "_read_array", _forbid_full_read)
            coords = _get_coords(group, ("time",), {})

        coord = coords.get_coord("time")
        assert coord.start == np.timedelta64(1, "ns")
        assert coord.step == np.timedelta64(2, "ns")

    def test_read_array_sample_restores_string_scalar(self, tmp_path):
        """Sample restoration should decode stored string scalars."""
        path = tmp_path / "string_coord_sample_path.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            node = group.create_dataset(
                "_coord_station", data=np.array([b"alpha", b"beta"], dtype="S5")
            )
            node.attrs["is_string"] = True
            node.attrs["original_string_dtype"] = "<U8"
            sample = dasdae_utils._read_array_sample(node, 0)
        assert sample == "alpha"

    @pytest.mark.parametrize(
        ("key", "value", "expected_type", "expected_value"),
        [
            ("history", ("a", "b"), "history_json", ("a", "b")),
            (
                "value",
                np.timedelta64(5, "ns"),
                "timedelta64[ns]",
                np.timedelta64(5, "ns"),
            ),
        ],
    )
    def test_encode_decode_attr_value_round_trip(
        self, key, value, expected_type, expected_value
    ):
        """Canonical attr encoding should round-trip supported rich types."""
        encoded, attr_type = _encode_attr_value(key, value)
        decoded = _decode_attr_value(
            {f"_attr_type_{key}": attr_type},
            key,
            encoded,
        )
        assert attr_type == expected_type
        assert decoded == expected_value

    def test_encode_attr_value_empty_history(self):
        """Empty history should still use the dedicated JSON history branch."""
        encoded, attr_type = _encode_attr_value("history", [])
        assert attr_type == "history_json"
        assert encoded == "[]"

    def test_encode_attr_value_string_history_uses_single_entry_json(self):
        """String history should serialize as a one-entry JSON list."""
        encoded, attr_type = _encode_attr_value("history", "one step")
        assert attr_type == "history_json"
        assert encoded == '["one step"]'

    def test_encode_attr_value_generic_sequence_is_not_special_cased(self):
        """Non-history sequences should use default passthrough handling."""
        encoded, attr_type = _encode_attr_value("value", [1, 2])
        assert attr_type is None
        assert encoded == [1, 2]

    def test_decode_attr_value_unknown_type_returns_value(self):
        """Unknown attr types should pass values through unchanged."""
        value = "abc"
        assert (
            _decode_attr_value({"_attr_type_value": "mystery"}, "value", value) == value
        )

    def test_get_file_version_reads_dasdae_attr(self, tmp_path):
        """The DASDAE version helper should read the file-level version attr."""
        path = tmp_path / "versioned.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["__DASDAE_version__"] = "9"
            assert _get_file_version(h5) == "9"


class TestRoundTrips:
    """Tests for round-tripping various patches/spools."""

    formatter = DASDAEV1()

    def test_write_patch_with_lat_lon(
        self, random_patch_with_lat_lon, tmp_path_factory
    ):
        """
        DASDAE should support writing patches with non-dimensional
        coords.
        """
        new_path = tmp_path_factory.mktemp("dasdae_append") / "tmp.h5"
        shape = random_patch_with_lat_lon.shape
        dims = random_patch_with_lat_lon.dims
        # add time deltas to ensure they are also serialized/deserialized.
        dist_shape = shape[dims.index("distance")]
        time_deltas = dc.to_timedelta64(random_state.random(dist_shape))
        patch = random_patch_with_lat_lon.update_coords(
            delta_times=("distance", time_deltas),
        )
        dc.write(patch, new_path, "DASDAE")
        spool = dc.read(new_path, file_format="DASDAE")
        assert len(spool) == 1
        new_patch = spool[0]
        assert patch.equals(new_patch)

    def test_roundtrip_empty_time_patch(self, tmp_path_factory, random_patch):
        """A patch with a dimension of length 0 should roundtrip."""
        path = tmp_path_factory.mktemp("round_trip_time_degenerate") / "out.h5"
        patch = random_patch
        # get degenerate patch
        time = patch.get_coord("time")
        time_max = time.max() + 3 * time.step
        empty_patch = patch.select(time=(time_max, ...))
        empty_patch.io.write(path, "dasdae")
        spool = self.formatter.read(path)
        new_patch = spool[0]
        assert empty_patch.shape == new_patch.shape
        assert np.equal(empty_patch.data, new_patch.data).all()
        assert empty_patch.get_coord("distance") == new_patch.get_coord("distance")
        assert len(new_patch.get_coord("time")) == 0

    def test_roundtrip_dim_1_patch(self, tmp_path_factory, random_patch):
        """A patch with length 1 time axis should roundtrip."""
        path = tmp_path_factory.mktemp("round_trip_dim_1") / "out.h5"
        patch = dc.get_example_patch(
            "random_das",
            time_step=0.999767552,
            shape=(100, 1),
            time_min="2023-06-13T15:38:00.49953408",
        )
        patch.io.write(path, "dasdae")

        spool = self.formatter.read(path)
        new_patch = spool[0]
        assert patch.equals(new_patch)

    def test_roundtrip_datetime_coord(self, tmp_path_factory, random_patch):
        """Ensure a patch with an attached datetime coord works."""
        path = tmp_path_factory.mktemp("roundtrip_datetime_coord") / "out.h5"
        dist = random_patch.get_coord("distance")
        dt = dc.to_datetime64(np.zeros_like(dist))
        dt[0] = dc.to_datetime64("2017-09-17")
        new = random_patch.update_coords(dt=("distance", dt))
        new.io.write(path, "dasdae")
        patch = dc.spool(path, file_format="DASDAE")[0]
        assert patch == new

    def test_roundtrip_nullish_datetime_coord(self, tmp_path_factory, random_patch):
        """Ensure a patch with an attached datetime coord with nulls works."""
        path = tmp_path_factory.mktemp("roundtrip_datetime_coord") / "out.h5"
        dist = random_patch.get_coord("distance")
        dt = dc.to_datetime64(np.zeros_like(dist))
        dt[~dt.astype(bool)] = np.datetime64("nat")
        dt[0] = dc.to_datetime64("2017-09-17")
        dt[-4] = dc.to_datetime64("2020-01-03")
        new = random_patch.update_coords(dt=("distance", dt))
        new.io.write(path, "dasdae")
        patch = dc.spool(path, file_format="DASDAE")[0]
        assert isinstance(patch, dc.Patch)

    def test_roundtrip_coord_multiple_dims(
        self, tmp_path_factory, multi_dim_coords_patch
    ):
        """
        Ensure a patch with a non-dimensional coordinate that is associated
        with two dims can round-trip.
        """
        patch = multi_dim_coords_patch
        folder = tmp_path_factory.mktemp("dasdae_multi_dim_coord")
        path = folder / "multidimcoord.hdf"
        patch.io.write(path, "dasdae")

        # Ensure we can read it from a directory
        patch2 = dc.spool(folder).update()[0]
        # And from a single file
        patch3 = dc.spool(path)[0]
        # All of the patches should be equal.
        assert patch == patch2 == patch3

    # Frustratingly, it doesn't seem pytables can store NaN values using
    # create_array, even when specifying an Atom with dflt=np.nan. See
    # https://github.com/PyTables/PyTables/issues/423
    @pytest.mark.xfail(reason="Pytables issue 423")
    def test_roundtrip_len_1_non_coord(self, random_spool, tmp_path_factory):
        """Ensure we can round-trip Non-coords."""
        path = tmp_path_factory.mktemp("roundtrip_non_coord") / "out.h5"
        # create a spool that has all non coords
        spool = dc.spool([x.mean("time") for x in random_spool])
        in_patch = spool[0]
        in_patch.io.write(path, "dasdae")
        new_spool = dc.spool(path, file_format="DASDAE")
        out_patch = new_spool[0]
        assert in_patch == out_patch

    def test_roundtrip_string_aux_coord(self, random_patch, tmp_path_factory):
        """Attached string coordinates should round-trip through DASDAE."""
        path = tmp_path_factory.mktemp("roundtrip_string_coord") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"sensor_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(sensor=("distance", labels))
        patch.io.write(path, "dasdae")
        out = dc.read(path, file_format="DASDAE")[0]
        coord = out.get_coord("sensor")
        assert isinstance(coord, CoordString)
        assert np.array_equal(coord.values, labels)

    def test_roundtrip_string_dim_coord(self, random_patch, tmp_path_factory):
        """String dimension coordinates should round-trip through DASDAE."""
        path = tmp_path_factory.mktemp("roundtrip_string_dim") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"ch_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(distance=labels)
        patch.io.write(path, "dasdae")
        out = dc.read(path, file_format="DASDAE")[0]
        coord = out.get_coord("distance")
        assert isinstance(coord, CoordString)
        assert np.array_equal(coord.values, labels)

    def test_scan_includes_string_coords(self, random_patch, tmp_path_factory):
        """String coordinates should appear in lossy scan summaries."""
        path = tmp_path_factory.mktemp("scan_string_coord") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"sensor_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(sensor=("distance", labels))
        patch.io.write(path, "dasdae")
        summary = dc.scan(path)[0]
        assert "sensor" in summary.coords
        assert summary.coords["sensor"].min == "sensor_000"
        assert summary.coords["sensor"].step is None

    def test_scan_to_df_includes_string_coord_columns(
        self, random_patch, tmp_path_factory
    ):
        """Flattened scan results should expose string coord summary fields."""
        path = tmp_path_factory.mktemp("scan_string_coord_df") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"sensor_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(sensor=("distance", labels))
        patch.io.write(path, "dasdae")
        df = dc.scan_to_df(path)
        row = df.iloc[0]
        assert row["sensor_min"] == "sensor_000"
        assert row["sensor_max"] == labels[-1]
        assert pd.isnull(row["sensor_step"])


class TestStringArrayHelpers:
    """Tests for DASDAE string-array integration paths."""

    def test_non_string_object_array_not_converted_to_bytes(
        self, tmp_path, monkeypatch
    ):
        """Object arrays with non-string content should not be stringified."""
        path = tmp_path / "object_array.h5"
        data = np.array([1, 2], dtype=object)

        def _raise_if_called(data):
            msg = "non-string object arrays should not be string-converted"
            raise AssertionError(msg)

        monkeypatch.setattr(dasdae_utils, "convert_strings_to_bytes", _raise_if_called)
        with h5py.File(path, mode="w") as h5:
            group = h5.create_group("waveforms")
            with pytest.raises(TypeError, match=r"Object dtype|object arrays"):
                _save_array(data, "obj", group=group)
