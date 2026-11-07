"""Tests for the DASCore Arrow patch format."""

from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.coords import _blank_coord, concat_coords, get_coord
from dascore.exceptions import (
    InvalidFiberFileError,
    MissingOptionalDependencyError,
    PatchError,
)
from dascore.io.arrow import ArrowV1
from dascore.io.arrow import core as arrow_core
from dascore.io.arrow.utils import METADATA_KEY, data_batches, patch_document
from dascore.io.sr4731.utils import SR4731PatchAttrs
from dascore.units import get_quantity

pa = pytest.importorskip("pyarrow")

T0 = np.datetime64("2020-01-01T00:00:00", "ns")
BIG = np.ones((5000, 1000), dtype=np.float32)


def _roundtrip(patch, tmp_path, **kwargs):
    """Write a patch and read it back."""
    path = tmp_path / ("p.arrows" if kwargs.get("stream") else "p.arrow")
    dc.write(patch, path, "ARROW", **kwargs)
    return dc.read(path)[0]


def _assert_exact(new, old):
    """The properties the format promises, compared directly."""
    assert new.dims == old.dims
    assert new.dtype == old.dtype.newbyteorder("=")
    assert np.array_equal(new.data, old.data, equal_nan=new.dtype.kind in "fmM")
    for name, coord in old.coords.coord_map.items():
        other = new.coords.coord_map[name]
        assert other == coord, name
        assert other.dtype == coord.dtype and other.step == coord.step, name
        assert new.coords.dim_map[name] == old.coords.dim_map[name]
    assert new.attrs.data_id == old.attrs.data_id
    assert new.attrs.origin_id == old.attrs.origin_id
    assert type(new.attrs) is type(old.attrs)


def _with_coord(coord):
    """A small patch whose time coordinate is `coord`."""
    data = np.arange(len(coord) * 2, dtype=np.float32).reshape(len(coord), 2)
    coords = {"time": coord, "distance": np.arange(2)}
    return dc.Patch(data=data, coords=coords, dims=("time", "distance"))


def _plain(data):
    """A patch with only the data's dims."""
    return dc.Patch(data=data, dims=("a", "b")[: data.ndim], coords={})


def _write_raw(path, patch, *, change=None, text=None, field=None, batches=None):
    """Write a patch with parts of it replaced, as a malformed writer would."""
    doc = patch_document(patch)
    if change:
        change(doc)
    text = json.dumps(doc) if text is None else text
    field = field or pa.field("data", pa.from_numpy_dtype(patch.dtype), nullable=False)
    schema = pa.schema([field], metadata={METADATA_KEY: text.encode()})
    batches = batches or list(data_batches(patch, schema))
    with pa.ipc.new_stream(path, schema) as writer:
        for batch in batches:
            writer.write_batch(pa.record_batch(batch.columns, schema=schema))


class TestRoundTrip:
    """Patches read back as written, ids included."""

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("compression", [None, "zstd", "lz4"])
    def test_example_patch(self, tmp_path, stream, compression):
        """The example patch, as a file or stream, compressed or not."""
        patch = dc.get_example_patch()
        new = _roundtrip(patch, tmp_path, stream=stream, compression=compression)
        _assert_exact(new, patch)
        raw = (tmp_path / ("p.arrows" if stream else "p.arrow")).read_bytes()
        assert raw.startswith(b"ARROW1") is not stream

    @pytest.mark.parametrize("compression", ["zstd", "lz4"])
    def test_compression_applies(self, tmp_path, compression):
        """Compressed files are smaller than uncompressed ones."""
        patch = _plain(BIG)
        dc.write(patch, tmp_path / "plain.arrow", "ARROW")
        dc.write(patch, tmp_path / "small.arrow", "ARROW", compression=compression)
        sizes = [(tmp_path / x).stat().st_size for x in ("small.arrow", "plain.arrow")]
        assert sizes[0] < sizes[1] / 10

    @pytest.mark.parametrize(
        "coord",
        [
            get_coord(start=T0, step=(1, 1024), shape=(4096,))[1::3],
            get_coord(start=T0, step=(1, 1024), shape=(4096,))[::-1],
            get_coord(start=0.0, stop=1.0, step=0.1)[1::3],
            get_coord(
                start=np.float32(0.1), stop=np.float32(2.0), step=np.float32(0.1)
            )[2::5],
            get_coord(start=np.float32(9.512447), step=0.7613785622883626, shape=(72,)),
            get_coord(data=np.array([0.0, 1.1, 1.95, 3.02]), step=1.0),
            concat_coords(
                get_coord(start=0, stop=10, step=1),
                get_coord(start=15, stop=25, step=1),
            ),
            get_coord(data=np.array([3, 1, 7, 2])),
            get_coord(data=T0.astype("M8[us]") + np.array([0, 976, 1953], "m8[us]")),
            get_coord(
                start=np.datetime64("2020-01-01", "s"),
                step=np.timedelta64(3, "s"),
                shape=(5,),
            ),
            get_coord(data=np.array([0.0, 1.5, 2.0, 4.5, 5.0, 7.7]))[4:0:-2],
            _blank_coord((4,), start=1.0, stop=9.0, step=None),
            get_coord(data=np.arange(4).astype("M8[10ms]")),
        ],
        ids=[
            "exact-strided",
            "exact-reversed",
            "float-strided",
            "float32-sliced",
            "mixed-float-types",
            "jittered",
            "gapped",
            "integer-labels",
            "microseconds",
            "seconds",
            "label-window",
            "blank-with-ends",
            "unit-multiple",
        ],
    )
    def test_dim_coords(self, tmp_path, coord):
        """Each kind of dimension coordinate reads back exactly."""
        patch = _with_coord(coord)
        _assert_exact(_roundtrip(patch, tmp_path), patch)

    def test_big_endian_labels(self, tmp_path):
        """Big-endian time labels keep their values and dtype."""
        labels = np.array([0, 5, 7, 20], dtype=">i8").astype(">M8[ns]")
        new = _roundtrip(_with_coord(get_coord(data=labels)), tmp_path)
        coord = new.get_coord("time")
        assert coord.dtype == labels.dtype
        assert np.array_equal(coord.values, labels)

    def test_other_coords(self, tmp_path):
        """String, bool, multi-dim, dimensionless and blank coordinates."""
        rng = np.random.default_rng(0)
        patch = dc.get_example_patch().update_coords(
            station=("distance", np.array(["A01"] * 150 + ["B2"] * 150)),
            good=("distance", np.arange(300) % 2 == 0),
            grid=(("distance", "time"), rng.random((300, 2000))),
            extra=((), np.arange(5.0)),
            scalar=((), np.array(42.0)),
            label=((), np.array("foo")),
        )
        _assert_exact(_roundtrip(patch, tmp_path), patch)
        blank = _plain(np.zeros((3, 4)))
        _assert_exact(_roundtrip(blank, tmp_path), blank)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([[True, False], [False, True]]),
            np.arange(6, dtype=np.float16).reshape(2, 3),
            np.arange(6, dtype=">f4").reshape(2, 3),
            np.arange(6, dtype=np.uint64).reshape(2, 3),
            np.array([[T0, np.datetime64("NaT", "ns")]]),
            np.array([[1, -2]], dtype="m8[ms]"),
            np.zeros((0, 3)),
        ],
        ids=["bool", "float16", "big-endian", "uint64", "nat", "timedelta", "empty"],
    )
    def test_data_dtypes(self, tmp_path, data):
        """Each supported data dtype reads back, NaT included."""
        patch = _plain(data)
        _assert_exact(_roundtrip(patch, tmp_path), patch)

    def test_attrs(self, tmp_path):
        """Attr values JSON lacks keep their values and types."""
        values = {
            "quantity": get_quantity("5 m/s"),
            "unit": get_quantity("m/s").units,
            "millis": np.timedelta64(5, "ms"),
            "gap": np.timedelta64(3, "10ms"),
            "nested": {"a": {"b": [1, 2.0]}},
            "f32": np.float32(0.1),
            "f64": np.float64(0.1),
            "flag": np.bool_(True),
            "big": 2**60,
        }
        patch = dc.get_example_patch().update_attrs(
            when=datetime.datetime(2020, 1, 1, 1, 2, 3),
            nanos=pd.Timestamp("2020-01-01T00:00:00.123456789"),
            span=pd.Timedelta(3, "s"),
            missing=float("nan"),
            **values,
        )
        new = _roundtrip(patch, tmp_path).attrs.model_dump()
        old = patch.attrs.model_dump()
        for name in values:
            assert new[name] == old[name], name
            assert type(new[name]) is type(old[name]), name
        assert np.isnan(new["missing"])
        # Python and pandas times come back as numpy times at their precision.
        assert new["span"] == np.timedelta64(3_000_000_000, "ns")
        assert new["when"] == np.datetime64("2020-01-01T01:02:03", "us")
        assert new["nanos"] == np.datetime64("2020-01-01T00:00:00.123456789", "ns")

    def test_attrs_class(self, tmp_path):
        """A registered attrs subclass comes back as itself."""
        patch = dc.get_example_patch()
        attrs = SR4731PatchAttrs.from_dict(patch.attrs.model_dump())
        patch = patch.update(attrs=attrs)
        assert type(patch.attrs) is SR4731PatchAttrs
        _assert_exact(_roundtrip(patch, tmp_path), patch)

    def test_batches(self, tmp_path):
        """A large patch is split into whole first-dim slices."""
        patch = _plain(BIG)
        schema = pa.schema([pa.field("data", pa.float32(), nullable=False)])
        lengths = [len(x) for x in data_batches(patch, schema)]
        assert len(lengths) > 1 and all(x % 1000 == 0 for x in lengths)
        _assert_exact(_roundtrip(patch, tmp_path, stream=True), patch)

    def test_snap(self, tmp_path):
        """Reading with snap puts a jittered dimension on its grid."""
        coord = get_coord(data=np.array([0.0, 1.1, 1.95, 3.02]), step=1.0)
        path = tmp_path / "p.arrow"
        dc.write(_with_coord(coord), path, "ARROW")
        assert dc.read(path, snap=True)[0].get_coord("time").evenly_sampled
        assert dc.read(path)[0].get_coord("time").jittered


class TestRefuses:
    """What the format can't hold is refused when written."""

    @pytest.mark.parametrize(
        "patch",
        [
            _plain(np.ones((2, 2), dtype=complex)),
            _plain(np.array([[T0]]).astype("M8[D]")),
            dc.get_example_patch().update_coords(c=("distance", np.ones(300) * 1j)),
            dc.get_example_patch().update_attrs(when=pd.Timestamp("2020", tz="UTC")),
            dc.get_example_patch().update_attrs(thing={(1, 2): "pair"}),
            dc.get_example_patch().update_attrs(thing=object()),
            dc.get_example_patch().update_attrs(raw=b"xy"),
            _plain(np.ma.masked_array(np.ones((2, 2)), mask=[[0, 1], [0, 0]])),
            dc.get_example_patch().isel(distance=0, time=0),
        ],
        ids=[
            "complex",
            "days",
            "complex-coord",
            "tz-aware",
            "tuple-key",
            "object",
            "bytes",
            "masked",
            "no-dims",
        ],
    )
    def test_unsupported(self, tmp_path, patch):
        """Complex data, unsupported dtypes and unencodable attrs raise."""
        with pytest.raises(PatchError):
            dc.write(patch, tmp_path / "p.arrow", "ARROW")

    def test_several_patches(self, tmp_path):
        """One patch per file."""
        with pytest.raises(PatchError, match="one patch"):
            ArrowV1().write(dc.get_example_spool(), tmp_path / "p.arrow")


def _set(path, value):
    """Return a change setting a document path to a value."""

    def change(doc):
        *parents, last = path
        for key in parents:
            doc = doc[key]
        doc[last] = value

    return change


TIME_RUN = ("coords", "time", "runs", 0)


class TestReadChecks:
    """Malformed files raise rather than read wrong."""

    @pytest.mark.parametrize(
        "change",
        [
            _set(("version",), "2.0"),
            _set(("format",), "other"),
            _set(("shape",), [1, 2]),
            _set(("shape",), [-300, 2000]),
            _set(("dims",), ["time", "time"]),
            _set(("dtype",), "<i8"),
            _set(TIME_RUN, {"spiral": {}}),
            _set(TIME_RUN, {"grid": {}, "values": []}),
        ],
        ids=[
            "version",
            "format",
            "shape",
            "negative-shape",
            "repeated-dims",
            "dtype",
            "run-kind",
            "two-member-run",
        ],
    )
    def test_bad_document(self, tmp_path, change):
        """A document breaking the format's rules is refused."""
        path = tmp_path / "bad.arrows"
        _write_raw(path, dc.get_example_patch(), change=change)
        with pytest.raises(InvalidFiberFileError):
            ArrowV1().read(path)

    @pytest.mark.parametrize(
        "text",
        ['"attrs": {{', '"attrs": {"x": {"$scalar": ["|O", 1]}, '],
        ids=["not-json", "object-scalar"],
    )
    def test_bad_json(self, tmp_path, text):
        """Broken JSON and unsupported dtypes are refused."""
        patch = dc.get_example_patch()
        good = json.dumps(patch_document(patch))
        path = tmp_path / "bad.arrows"
        _write_raw(path, patch, text=good.replace('"attrs": {', text, 1))
        with pytest.raises(InvalidFiberFileError):
            ArrowV1().read(path)

    @pytest.mark.parametrize(
        "field",
        [
            pa.field("data", pa.float64(), nullable=True),
            pa.field("data", pa.float64(), nullable=False, metadata={b"k": b"v"}),
            pa.field("other", pa.float64(), nullable=False),
        ],
        ids=["nullable", "field-metadata", "name"],
    )
    def test_bad_field(self, tmp_path, field):
        """The data field is one non-nullable column named data, nothing more."""
        path = tmp_path / "bad.arrows"
        _write_raw(path, dc.get_example_patch(), field=field)
        with pytest.raises(InvalidFiberFileError, match="non-nullable"):
            ArrowV1().read(path)

    def test_nulls(self, tmp_path):
        """Nulls in the data are refused."""
        batch = pa.record_batch([pa.array([1.0, None, 1.0, 1.0])], names=["data"])
        path = tmp_path / "bad.arrows"
        _write_raw(path, _plain(np.ones((2, 2))), batches=[batch])
        with pytest.raises(InvalidFiberFileError, match="null"):
            ArrowV1().read(path)

    @pytest.mark.parametrize(
        "sizes",
        [[2_000_000], [5_000_000, 10]],
        ids=["missing", "extra"],
    )
    def test_batch_accounting(self, tmp_path, sizes):
        """Too few or too many values are refused."""
        flat = pa.array(np.ones(sum(sizes), dtype=np.float32))
        starts = np.cumsum([0, *sizes])
        batches = [
            pa.record_batch([flat.slice(a, n)], names=["data"])
            for a, n in zip(starts, sizes, strict=False)
        ]
        path = tmp_path / "bad.arrows"
        _write_raw(path, _plain(BIG), batches=batches)
        with pytest.raises(InvalidFiberFileError):
            ArrowV1().read(path)

    def test_truncated(self, tmp_path):
        """A file cut short is invalid."""
        path = tmp_path / "p.arrows"
        dc.write(_plain(BIG), path, "ARROW", stream=True)
        path.write_bytes(path.read_bytes()[:1_000_000])
        with pytest.raises(InvalidFiberFileError):
            ArrowV1().read(path)

    def test_not_this_format(self, tmp_path):
        """Other Arrow files, and broken ones, aren't claimed."""
        plain, broken, other = (
            tmp_path / x for x in ("a.arrow", "b.arrow", "c.arrows")
        )
        table = pa.table({"x": [1, 2, 3]})
        with pa.ipc.new_file(plain, table.schema) as writer:
            writer.write_table(table)
        broken.write_bytes(b"ARROW1 and then nothing useful")
        _write_raw(other, dc.get_example_patch(), text="not json")
        for path in (plain, broken, other):
            assert ArrowV1().get_format(path) is False
        with pytest.raises(InvalidFiberFileError, match="no dascore:patch"):
            ArrowV1().read(plain)

    def test_coordinate_shape(self, tmp_path):
        """A coordinate shorter than its dim is refused, not trimmed to."""
        path = tmp_path / "bad.arrows"
        patch = _with_coord(get_coord(start=0, stop=3, step=1))

        def change(doc):
            doc["coords"]["time"] = patch_document(patch.select(time=(0, 1)))["coords"][
                "time"
            ]

        _write_raw(path, patch, change=change)
        with pytest.raises(InvalidFiberFileError, match="fit"):
            ArrowV1().read(path)

    def test_huge_first_message(self, tmp_path):
        """A file starting like a stream with a huge message isn't read."""
        path = tmp_path / "big.bin"
        path.write_bytes(b"\xff\xff\xff\xff" + (2**31 - 1).to_bytes(4, "little"))
        assert ArrowV1().get_format(path) is False

    def test_window_is_a_copy(self, tmp_path):
        """A windowed read doesn't hold the whole patch."""
        path = tmp_path / "p.arrow"
        dc.write(_plain(BIG), path, "ARROW")
        part = ArrowV1().read_array(path, windows=(slice(0, 2),))
        assert part.base is None or part.base.size == part.size


class TestDetection:
    """Format detection."""

    def test_detects_without_pyarrow(self, tmp_path, monkeypatch):
        """Without pyarrow, Arrow files are still claimed, so reading says why."""
        path = tmp_path / "p.arrow"
        dc.write(dc.get_example_patch(), path, "ARROW")

        def missing(name, **kwargs):
            raise MissingOptionalDependencyError(f"{name} is not installed")

        monkeypatch.setattr(arrow_core, "optional_import", missing)
        assert ArrowV1().get_format(path) == ("ARROW", "1")
