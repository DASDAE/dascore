"""
Tests for hive-style path attributes on directory spools.

key=value directory segments become string attrs in the index,
override file-declared attrs, stamp onto loaded patches, and survive
directory renames without content rescans. The source's own name --
the last path segment -- is never parsed.
"""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.core.spool import Spool
from dascore.examples import spool_to_directory
from dascore.io.index import ingest
from dascore.io.index.schema import INDEX_VERSION
from dascore.utils.paths import parse_hive_path_attrs
from tests.test_io.test_xml_binary.test_xml_binary import metadata

# The literal version stamped on indexes built while file names still
# contributed attrs. Spelled out rather than derived from INDEX_VERSION,
# so that retiring the parse without moving the version is a failure.
_SOURCE_NAME_INDEX_VERSION = 10


def _parse_including_name(rel_posix):
    """The parser as it was when file names counted."""
    out = parse_hive_path_attrs(rel_posix)
    name = rel_posix.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return out | parse_hive_path_attrs(f"{name}/x")


def _stamp_index_version(directory, version):
    """Write an index version onto a directory's index, then let go."""
    # close explicitly: sqlite3's context manager only wraps the
    # transaction, and a lingering handle blocks the rebuild's
    # unlink on Windows.
    con = sqlite3.connect(directory / ".dascore_index.sqlite3")
    try:
        con.execute("UPDATE meta_data SET index_version = ?", (version,))
        con.commit()
    finally:
        con.close()


@pytest.fixture()
def scan_calls(monkeypatch):
    """Count dc.scan invocations (the content-rescan choke point)."""
    calls = []
    real = dc.scan

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(dc, "scan", wrapper)
    return calls


@pytest.fixture()
def hive_dir(tmp_path):
    """One patch under network=XX/station=A/cable=north__tag=raw."""
    sub = tmp_path / "network=XX" / "station=A" / "cable=north__tag=raw"
    sub.mkdir(parents=True)
    patch = dc.get_example_patch()
    patch.io.write(sub / "das_file.h5", "dasdae")
    return tmp_path


@pytest.fixture()
def hive_spool(hive_dir):
    """An updated directory spool over the hive tree."""
    spool = Spool.from_directory(hive_dir).update(progress=None)
    yield spool
    spool.indexer.close()


class TestParseHivePathAttrs:
    """The pure path parser."""

    def test_directory_segments(self):
        """Plain hive layout parses each directory."""
        out = parse_hive_path_attrs("network=XX/station=A/file.h5")
        assert out == {"network": "XX", "station": "A"}

    def test_several_pairs_in_one_segment(self):
        """One directory can hold several __-separated pairs."""
        out = parse_hive_path_attrs("cable=north__tag=raw/file.h5")
        assert out == {"cable": "north", "tag": "raw"}

    def test_source_name_is_not_parsed(self):
        """The last segment names the source, so it contributes nothing."""
        assert parse_hive_path_attrs("cable=north.h5") == {}
        assert parse_hive_path_attrs("dir/cable=north") == {}

    def test_value_ending_in_an_extension_survives(self):
        """
        A directory value keeps a trailing dotted token.

        Telling "XX.R2D1..RAW" from a file extension is the ambiguity
        which keeps the source's own name out of the parse.
        """
        out = parse_hive_path_attrs("acquisition_key=XX.R2D1..RAW/f.h5")
        assert out == {"acquisition_key": "XX.R2D1..RAW"}

    def test_percent_decoding(self):
        """Keys/values decode after splitting so %3D survives."""
        out = parse_hive_path_attrs("a%20b=c%3Dd/file.h5")
        assert out == {"a b": "c=d"}

    def test_deepest_wins(self):
        """A repeated key takes the deepest value."""
        out = parse_hive_path_attrs("station=A/station=B/file.h5")
        assert out == {"station": "B"}

    def test_empty_key_or_value_skipped(self):
        """'=x' and 'x=' segments contribute nothing."""
        assert parse_hive_path_attrs("=x/y=/file.h5") == {}

    def test_hive_null_sentinel_skipped(self):
        """Hive's NULL partition sentinel means missing."""
        out = parse_hive_path_attrs("station=__HIVE_DEFAULT_PARTITION__/f.h5")
        assert out == {}

    def test_no_pairs(self):
        """Paths without = parse to nothing."""
        assert parse_hive_path_attrs("plain/dir/file.h5") == {}
        assert parse_hive_path_attrs(".") == {}
        assert parse_hive_path_attrs("file.h5") == {}

    def test_value_keeps_second_equals(self):
        """Only the first = splits; the rest stays in the value."""
        assert parse_hive_path_attrs("a=b=c/f.h5") == {"a": "b=c"}

    def test_numeric_value_keeps_fraction(self):
        """Nothing is stripped from a directory value."""
        assert parse_hive_path_attrs("depth=1.5/f.h5") == {"depth": "1.5"}
        assert parse_hive_path_attrs("depth=1.5/deeper/f.h5") == {"depth": "1.5"}


class TestHiveIndexing:
    """Hive attrs land in the index and drive selection."""

    def test_contents_columns(self, hive_spool):
        """Every directory segment appears as a string column."""
        df = hive_spool.get_contents()
        row = df.iloc[0]
        assert row["network"] == "XX"
        assert row["station"] == "A"
        assert row["cable"] == "north"
        assert row["tag"] == "raw"

    def test_select_equality_and_glob(self, hive_spool):
        """Select supports equality and glob on hive attrs."""
        assert len(hive_spool.select(station="A")) == 1
        assert len(hive_spool.select(station="nope")) == 0
        assert len(hive_spool.select(cable="nor*")) == 1

    def test_non_hive_spool_unaffected(self, tmp_path):
        """A plain directory spool gains no hive columns."""
        patch = dc.get_example_patch()
        patch.io.write(tmp_path / "plain_file.h5", "dasdae")
        spool = Spool.from_directory(tmp_path).update(progress=None)
        try:
            df = spool.get_contents()
            assert df["_path_attrs"].isnull().all()
            assert "cable" not in df.columns
        finally:
            spool.indexer.close()


class TestHiveWins:
    """Path attrs override file-declared attrs."""

    @pytest.fixture()
    def conflict_spool(self, tmp_path):
        """File says station=B but lives under station=A."""
        sub = tmp_path / "station=A"
        sub.mkdir()
        patch = dc.get_example_patch().update_attrs(station="B")
        patch.io.write(sub / "conflict.h5", "dasdae")
        with pytest.warns(UserWarning, match="override attrs"):
            spool = Spool.from_directory(tmp_path).update(progress=None)
        yield spool
        spool.indexer.close()

    def test_index_shows_path_value(self, conflict_spool):
        """The contents show the path's value."""
        assert conflict_spool.get_contents()["station"].iloc[0] == "A"

    def test_loaded_patch_shows_path_value(self, conflict_spool):
        """The loaded patch also shows the path's value."""
        assert conflict_spool[0].attrs.station == "A"

    def test_override_names_the_key_and_a_path(self, tmp_path):
        """
        The warning says which name and where, once for the archive.

        A layout disagreeing with every file it holds is a legitimate
        correction and a common mistake, and the two look identical from
        inside; naming one path is enough to go and look.
        """
        sub = tmp_path / "station=A"
        sub.mkdir()
        patch = dc.get_example_patch().update_attrs(station="B")
        for name in ("one.h5", "two.h5"):
            patch.io.write(sub / name, "dasdae")
        with pytest.warns(UserWarning, match="override attrs") as record:
            spool = Spool.from_directory(tmp_path).update(progress=None)
        overrides = [x for x in record if "override attrs" in str(x.message)]
        assert len(overrides) == 1
        assert "'station'" in str(overrides[0].message)
        assert "station=A" in str(overrides[0].message)
        spool.indexer.close()

    def test_agreeing_path_is_silent(self, tmp_path):
        """Restating what the file already says overrides nothing."""
        sub = tmp_path / "station=A"
        sub.mkdir()
        patch = dc.get_example_patch().update_attrs(station="A")
        patch.io.write(sub / "agree.h5", "dasdae")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spool = Spool.from_directory(tmp_path).update(progress=None)
        assert not [x for x in caught if "override attrs" in str(x.message)]
        assert spool.get_contents()["station"].iloc[0] == "A"
        spool.indexer.close()

    def test_unstated_attr_is_silent(self, tmp_path):
        """A name the file never states is not one the path overrides."""
        sub = tmp_path / "cable=north"
        sub.mkdir()
        dc.get_example_patch().io.write(sub / "quiet.h5", "dasdae")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spool = Spool.from_directory(tmp_path).update(progress=None)
        assert not [x for x in caught if "override attrs" in str(x.message)]
        assert spool.get_contents()["cable"].iloc[0] == "north"
        spool.indexer.close()


class TestPatchStamping:
    """Hive attrs reach loaded patches in every load path."""

    def test_getitem(self, hive_spool):
        """spool[i] carries the hive attrs."""
        patch = hive_spool[0]
        assert patch.attrs.network == "XX"
        assert patch.attrs.station == "A"
        assert patch.attrs["cable"] == "north"

    def test_chunked(self, tmp_path):
        """Patches from a chunked spool keep hive attrs."""
        sub = tmp_path / "network=XX"
        sub.mkdir()
        spool_to_directory(dc.get_example_spool("random_das"), path=sub)
        spool = Spool.from_directory(tmp_path).update(progress=None)
        try:
            merged = spool.chunk(time=None)
            assert merged[0].attrs.network == "XX"
        finally:
            spool.indexer.close()

    def test_union(self, hive_spool, tmp_path_factory):
        """Union spools keep hive attrs in contents and patches."""
        other_dir = tmp_path_factory.mktemp("other") / "station=Z"
        other_dir.mkdir()
        dc.get_example_patch().io.write(other_dir / "other.h5", "dasdae")
        other = Spool.from_directory(other_dir.parent).update(progress=None)
        try:
            union = hive_spool + other
            stations = set(union.get_contents()["station"])
            assert stations == {"A", "Z"}
            assert {p.attrs.station for p in union} == {"A", "Z"}
        finally:
            other.indexer.close()

    def test_pickle_roundtrip(self, hive_spool):
        """Pickled spools keep hive attrs on loaded patches."""
        loaded = pickle.loads(pickle.dumps(hive_spool))
        assert loaded[0].attrs.station == "A"


class TestMoveDetection:
    """Renames rewrite the index without content rescans."""

    def test_directory_rename_no_rescan(self, hive_spool, hive_dir, scan_calls):
        """Renaming a partition directory never re-reads file contents."""
        df_before = hive_spool.get_contents()
        (hive_dir / "network=XX" / "station=A").rename(
            hive_dir / "network=XX" / "station=Q"
        )
        updated = hive_spool.update(progress=None)
        assert not scan_calls
        df = updated.get_contents()
        assert df["station"].iloc[0] == "Q"
        assert df["source_path"].iloc[0].startswith("network=XX/station=Q/")
        # patch/coord rows survived: same patch identity
        assert list(df["_patch_id"]) == list(df_before["_patch_id"])
        assert updated[0].attrs.station == "Q"

    def test_added_key_is_a_pure_move(self, hive_spool, hive_dir, scan_calls):
        """Adding a key to an existing segment never re-reads contents."""
        old = hive_dir / "network=XX" / "station=A" / "cable=north__tag=raw"
        old.rename(old.with_name("cable=north__tag=raw__phase=2"))
        updated = hive_spool.update(progress=None)
        assert not scan_calls
        assert updated.get_contents()["phase"].iloc[0] == "2"

    def test_removed_key_triggers_rescan(self, hive_spool, hive_dir, scan_calls):
        """Dropping a hive key needs the file's own value back: rescan."""
        old = hive_dir / "network=XX" / "station=A" / "cable=north__tag=raw"
        old.rename(old.with_name("cable=north"))
        updated = hive_spool.update(progress=None)
        assert len(scan_calls) == 1
        df = updated.get_contents()
        # the file itself declares a tag; with the path key gone it returns
        assert df["tag"].iloc[0] == dc.get_example_patch().attrs.tag

    def test_ambiguous_stats_fall_back_to_rescan(self, tmp_path, scan_calls):
        """Twin files with identical stats are rescanned, not guessed."""
        patch = dc.get_example_patch()
        patch.io.write(tmp_path / "twin_a.h5", "dasdae")
        patch.io.write(tmp_path / "twin_b.h5", "dasdae")
        stat = (tmp_path / "twin_a.h5").stat()
        os.utime(tmp_path / "twin_b.h5", ns=(stat.st_atime_ns, stat.st_mtime_ns))
        if (tmp_path / "twin_a.h5").stat().st_size != (
            tmp_path / "twin_b.h5"
        ).stat().st_size:
            pytest.skip("twin files did not serialize to equal sizes")
        spool = Spool.from_directory(tmp_path).update(progress=None)
        try:
            scan_calls.clear()
            sub = tmp_path / "tag=x"
            sub.mkdir()
            (tmp_path / "twin_a.h5").rename(sub / "twin_a.h5")
            (tmp_path / "twin_b.h5").rename(sub / "twin_b.h5")
            updated = spool.update(progress=None)
            assert len(scan_calls) == 1  # one scan covering both twins
            assert set(updated.get_contents()["tag"]) == {"x"}
        finally:
            spool.indexer.close()

    def test_plain_rename_without_hive_keys(self, tmp_path, scan_calls):
        """A rename with no hive keys anywhere is still a cheap move."""
        dc.get_example_patch().io.write(tmp_path / "plain_a.h5", "dasdae")
        spool = Spool.from_directory(tmp_path).update(progress=None)
        try:
            scan_calls.clear()
            (tmp_path / "plain_a.h5").rename(tmp_path / "plain_b.h5")
            updated = spool.update(progress=None)
            assert not scan_calls
            assert updated.get_contents()["source_path"].iloc[0] == "plain_b.h5"
        finally:
            spool.indexer.close()

    def test_detect_moves_skips_unstatted_and_root(self, hive_spool, hive_dir):
        """Sources without stored stats and the root unit never match."""
        files = {"new.h5": (1, 2, hive_dir / "new.h5")}
        stored = {"gone.h5": None, ".": (1, 2)}
        moves = hive_spool.indexer._detect_moves(["gone.h5", "."], stored, files)
        assert moves == {}

    def test_non_fiber_file_rename(self, hive_spool, hive_dir, scan_calls):
        """A moved non-fiber file stays quietly tracked."""
        (hive_dir / "notes.txt").write_text("hello")
        spool = hive_spool.update(progress=None)
        scan_calls.clear()
        (hive_dir / "notes.txt").rename(hive_dir / "network=XX" / "notes.txt")
        spool = spool.update(progress=None)
        assert not scan_calls
        assert len(spool) == 1


class TestEdgeCases:
    """Reserved names, coord collisions, and version rebuilds."""

    def test_reserved_key_warns_and_skips(self, tmp_path):
        """A hive key colliding with a structural column warns."""
        sub = tmp_path / "time_min=5"
        sub.mkdir()
        dc.get_example_patch().io.write(sub / "file.h5", "dasdae")
        with pytest.warns(UserWarning, match="hive-style path key"):
            spool = Spool.from_directory(tmp_path).update(progress=None)
        spool.indexer.close()

    def test_restricted_update_still_moves(self, hive_spool, hive_dir, scan_calls):
        """update(paths=...) applies moves outside the restriction."""
        (hive_dir / "network=XX" / "station=A").rename(
            hive_dir / "network=XX" / "station=R"
        )
        hive_spool.indexer.update(paths=[hive_dir / "does_not_exist"], progress=None)
        assert not scan_calls
        assert hive_spool.get_contents()["station"].iloc[0] == "R"

    def test_old_index_version_rebuilds(self, hive_dir):
        """An index stamped with an older version rebuilds transparently."""
        spool = Spool.from_directory(hive_dir).update(progress=None)
        spool.indexer.close()
        _stamp_index_version(hive_dir, 3)
        reopened = Spool.from_directory(hive_dir).update(progress=None)
        try:
            assert reopened.get_contents()["station"].iloc[0] == "A"
        finally:
            reopened.indexer.close()

    def test_rebuild_drops_source_name_attrs(self, tmp_path, monkeypatch):
        """
        An index built when file names counted loses those attrs.

        The rebuild is what retires them, so the index version has to
        move with the parse: an index left at its old number would keep
        serving attrs the current parser would never produce.
        """
        assert INDEX_VERSION > _SOURCE_NAME_INDEX_VERSION
        sub = tmp_path / "station=A"
        sub.mkdir()
        dc.get_example_patch().io.write(sub / "cable=north.h5", "dasdae")
        monkeypatch.setattr(ingest, "parse_hive_path_attrs", _parse_including_name)
        spool = Spool.from_directory(tmp_path).update(progress=None)
        assert spool.get_contents()["cable"].iloc[0] == "north"
        spool.indexer.close()
        monkeypatch.undo()
        _stamp_index_version(tmp_path, _SOURCE_NAME_INDEX_VERSION)
        reopened = Spool.from_directory(tmp_path).update(progress=None)
        try:
            df = reopened.get_contents()
            assert "cable" not in df.columns
            assert json.loads(df["_path_attrs"].iloc[0]) == {"station": "A"}
        finally:
            reopened.indexer.close()

    def test_source_file_name_is_not_indexed(self, tmp_path):
        """A key=value file name under a partition contributes nothing."""
        sub = tmp_path / "station=A"
        sub.mkdir()
        dc.get_example_patch().io.write(sub / "cable=north.h5", "dasdae")
        spool = Spool.from_directory(tmp_path).update(progress=None)
        try:
            df = spool.get_contents()
            assert df["station"].iloc[0] == "A"
            assert "cable" not in df.columns
            assert spool[0].attrs.get("cable") is None
        finally:
            spool.indexer.close()

    def test_directory_format_unit_name_is_not_indexed(self, tmp_path):
        """A directory-format source is named, not partitioned, by its dir."""
        unit = tmp_path / "cable=north" / "tag=ignored"
        unit.mkdir(parents=True)
        (unit / "metadata.xml").write_text(metadata)
        rand = np.random.default_rng(0).random((5000, 10)).astype("float32")
        (unit / "DAS_20240530T011500_000000Z.raw").write_bytes(rand.tobytes())
        spool = Spool.from_directory(tmp_path).update(progress=None)
        try:
            stored = spool.get_contents()["_path_attrs"].iloc[0]
            assert json.loads(stored) == {"cable": "north"}
        finally:
            spool.indexer.close()


class TestHiveAcquisitionKey:
    """A path may only stamp an id a patch can carry."""

    def _write(self, path, segment):
        """Write one patch under a hive segment."""
        sub = path / segment
        sub.mkdir(parents=True)
        dc.get_example_patch().io.write(sub / "patch.h5", "DASDAE")
        return dc.spool(path)

    def test_valid_id_is_stamped(self, tmp_path):
        """A complete id reaches the index and the patch."""
        spool = self._write(tmp_path, "acquisition_key=XX.R2D1..RAW").update()
        assert spool.get_contents()["acquisition_key"].iloc[0] == "XX.R2D1..RAW"
        assert spool[0].attrs.acquisition_key == "XX.R2D1..RAW"

    @pytest.mark.parametrize("value", ["XX", "XX.R2D1.RAW", "XX.R2D1..RA_W"])
    def test_invalid_id_warns_and_is_skipped(self, tmp_path, value):
        """
        An id the patch would reject is refused at indexing.

        Stamping it anyway would index cleanly and then fail at every
        load, where the fix (renaming a directory) is far from the error.
        """
        with pytest.warns(UserWarning, match="acquisition_key"):
            spool = self._write(tmp_path, f"acquisition_key={value}").update()
        assert "acquisition_key" not in spool.get_contents().columns
        assert spool[0].attrs.acquisition_key == ""


class TestOverrideComparesMeaning:
    """The warning is about a disagreement, not about a spelling."""

    def _index(self, tmp_path, segment, **attrs):
        """Index one patch stating attrs under a hive segment."""
        sub = tmp_path / segment
        sub.mkdir(parents=True)
        dc.get_example_patch().update_attrs(**attrs).io.write(sub / "f.h5", "dasdae")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spool = Spool.from_directory(tmp_path).update(progress=None)
        spool.indexer.close()
        return [x for x in caught if "override attrs" in str(x.message)]

    def test_a_number_restated_differently_is_silent(self, tmp_path):
        """`10` and `10.0` are the same gauge length."""
        assert not self._index(tmp_path, "gauge_length=10", gauge_length=10.0)

    def test_a_different_number_warns(self, tmp_path):
        """A path stating another value is the disagreement to report."""
        assert self._index(tmp_path, "gauge_length=20", gauge_length=10.0)

    def test_an_unreadable_number_warns(self, tmp_path):
        """A path which is not a number at all changes what the attr says."""
        assert self._index(tmp_path, "gauge_length=ten", gauge_length=10.0)

    def test_a_bool_restated_differently_is_silent(self, tmp_path):
        """Hive spells a flag in lower case; it is still the same flag."""
        segment = "closed_fiber_loop=true"
        assert not self._index(tmp_path, segment, closed_fiber_loop=True)

    def test_a_flipped_bool_warns(self, tmp_path):
        """And the opposite flag is a disagreement."""
        segment = "closed_fiber_loop=false"
        assert self._index(tmp_path, segment, closed_fiber_loop=True)

    def test_a_restated_instant_is_silent(self, tmp_path):
        """
        A time is stored as integer nanoseconds, not as its own spelling.

        Comparing that integer's text against the path segment would call
        every datetime attr a disagreement.
        """
        stamp = np.datetime64("2020-01-01", "ns")
        cases = [
            ("observed=2020-01-01", False),
            ("observed=2021-06-05", True),
            ("observed=nonsense", True),
        ]
        for number, (segment, warns) in enumerate(cases):
            root = tmp_path / str(number)
            assert bool(self._index(root, segment, observed=stamp)) is warns, segment
