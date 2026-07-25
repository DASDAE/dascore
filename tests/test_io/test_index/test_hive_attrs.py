"""
Tests for hive-style path attributes on directory spools.

key=value path segments (directories and file names) become string
attrs in the index, override file-declared attrs, stamp onto loaded
patches, and survive directory renames without content rescans.
"""

from __future__ import annotations

import os
import pickle

import pytest

import dascore as dc
from dascore.core.spool import Spool
from dascore.utils.paths import parse_hive_path_attrs


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
    """One patch under network=XX/station=A with filename attrs."""
    sub = tmp_path / "network=XX" / "station=A"
    sub.mkdir(parents=True)
    patch = dc.get_example_patch()
    patch.io.write(sub / "cable=north__tag=raw.h5", "dasdae")
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

    def test_filename_pairs(self):
        """The file name participates, extension stripped, __ separated."""
        out = parse_hive_path_attrs("cable=north__tag=raw.h5")
        assert out == {"cable": "north", "tag": "raw"}

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
        """A trailing .5 is a value fragment, not an extension."""
        assert parse_hive_path_attrs("depth=1.5/f.h5") == {"depth": "1.5"}
        assert parse_hive_path_attrs("depth=1.5") == {"depth": "1.5"}


class TestHiveIndexing:
    """Hive attrs land in the index and drive selection."""

    def test_contents_columns(self, hive_spool):
        """Directory and filename attrs appear as string columns."""
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
            df = spool.indexer.get_contents()
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
        spool = Spool.from_directory(tmp_path).update(progress=None)
        yield spool
        spool.indexer.close()

    def test_index_shows_path_value(self, conflict_spool):
        """The contents show the path's value."""
        assert conflict_spool.get_contents()["station"].iloc[0] == "A"

    def test_loaded_patch_shows_path_value(self, conflict_spool):
        """The loaded patch also shows the path's value."""
        assert conflict_spool[0].attrs.station == "A"


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
        from dascore.examples import spool_to_directory

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
        df_before = hive_spool.indexer.get_contents()
        (hive_dir / "network=XX" / "station=A").rename(
            hive_dir / "network=XX" / "station=Q"
        )
        updated = hive_spool.update(progress=None)
        assert not scan_calls
        df = updated.indexer.get_contents()
        assert df["station"].iloc[0] == "Q"
        assert df["path"].iloc[0].startswith("network=XX/station=Q/")
        # patch/coord rows survived: same patch identity
        assert list(df["_patch_id"]) == list(df_before["_patch_id"])
        assert updated[0].attrs.station == "Q"

    def test_file_rename_adds_attr_no_rescan(self, hive_spool, hive_dir, scan_calls):
        """Adding a key via the file name is also a pure move."""
        old = hive_dir / "network=XX" / "station=A" / "cable=north__tag=raw.h5"
        old.rename(old.with_name("cable=north__tag=raw__phase=2.h5"))
        updated = hive_spool.update(progress=None)
        assert not scan_calls
        assert updated.get_contents()["phase"].iloc[0] == "2"

    def test_removed_key_triggers_rescan(self, hive_spool, hive_dir, scan_calls):
        """Dropping a hive key needs the file's own value back: rescan."""
        old = hive_dir / "network=XX" / "station=A" / "cable=north__tag=raw.h5"
        old.rename(old.with_name("cable=north.h5"))
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
        assert hive_spool.indexer.get_contents()["station"].iloc[0] == "R"

    def test_old_index_version_rebuilds(self, hive_dir):
        """An index stamped with an older version rebuilds transparently."""
        import sqlite3

        spool = Spool.from_directory(hive_dir).update(progress=None)
        spool.indexer.close()
        index_path = hive_dir / ".dascore_index.sqlite3"
        with sqlite3.connect(index_path) as con:
            con.execute("UPDATE meta_data SET index_version = 3")
        reopened = Spool.from_directory(hive_dir).update(progress=None)
        try:
            assert reopened.get_contents()["station"].iloc[0] == "A"
        finally:
            reopened.indexer.close()
