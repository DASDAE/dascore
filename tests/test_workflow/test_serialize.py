"""Tests for the canonical workflow serializer."""

from __future__ import annotations

import datetime
from enum import Enum
from functools import partial
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import InvalidModelTagError, ParameterError
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning
from dascore.workflow.serialize import (
    DOCUMENT,
    canonical_json,
    combine_hashes,
    decode,
    digest,
    encode,
)


def round_trip(value):
    """Encode a value as a document and read it back."""
    return decode(encode(value, mode=DOCUMENT))


def _normalized(dtype):
    """Return the dtype an array of this one is stored and read back as."""
    # Times are normalized to nanoseconds, as scalar ones are.
    kind = np.dtype(dtype).kind
    names = {"M": "datetime64[ns]", "m": "timedelta64[ns]"}
    return names.get(kind, dtype)


class Color(Enum):
    """An enum whose members stand for strings."""

    red = "red"
    blue = "blue"


class TestDigest:
    """Tests for the digest itself."""

    def test_hard_coded(self):
        """
        The digest of a fixed value never changes.

        A stored id is only useful if the same call gives the same answer in
        a later release, so this pins the whole chain: the encoding, the
        canonical text, and blake2b.
        """
        assert digest({"a": 1, "b": [1.5, "x"]}) == "3cdfb6194bc1acd6"

    def test_mode_matters(self):
        """The two modes encode an array differently, so they digest so."""
        array = np.arange(3)
        assert digest(array) != digest(array, mode=DOCUMENT)

    def test_key_order_ignored(self):
        """A mapping digests the same however it was built."""
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_combine_hashes_ordered(self):
        """Combining digests keeps the order they were given in."""
        assert combine_hashes(["a", "b"]) != combine_hashes(["b", "a"])

    def test_combine_hashes_repeats(self):
        """A digest repeated is not the same as a digest given once."""
        assert combine_hashes(["a", "a"]) != combine_hashes(["a"])


class TestCanonicalJson:
    """Tests for the canonical text a digest is taken of."""

    def test_sorted_and_tight(self):
        """Keys are sorted and no whitespace is written."""
        assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_ascii_only(self):
        """Non-ascii characters are escaped rather than written."""
        assert canonical_json("é") == '"\\u00e9"'


class TestScalars:
    """Tests for values JSON nearly spells itself."""

    def test_bool_is_not_int(self):
        """True and 1 are equal in python and are not the same parameter."""
        assert digest(True) != digest(1)
        assert digest(False) != digest(0)

    def test_bool_round_trip(self):
        """A bool comes back a bool."""
        assert round_trip(True) is True

    def test_numpy_bool(self):
        """A numpy bool is the same parameter as a python one."""
        assert digest(np.bool_(True)) == digest(True)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats(self, value):
        """The floats JSON cannot spell are still encodable."""
        out = round_trip(value)
        assert (out != out and value != value) or out == value

    def test_non_finite_floats_differ(self):
        """The three of them are three different values."""
        digests = {digest(x) for x in [float("nan"), float("inf"), float("-inf")]}
        assert len(digests) == 3

    def test_numpy_scalars(self):
        """A numpy scalar is the same parameter as the python value."""
        assert digest(np.int64(3)) == digest(3)
        assert digest(np.float64(1.5)) == digest(1.5)

    def test_none(self):
        """None encodes as itself."""
        assert encode(None) is None

    def test_enum_by_value(self):
        """An enum member stands for its value."""
        assert digest(Color.red) == digest("red")
        assert digest(Color.red) != digest(Color.blue)


class TestTimes:
    """Tests for numpy's time types."""

    def test_unit_normalized(self):
        """The unit a time was written in does not change its digest."""
        day = np.datetime64("2020-01-01")
        nanosecond = np.datetime64("2020-01-01T00:00:00.000000000")
        assert digest(day) == digest(nanosecond)

    def test_datetime_round_trip(self):
        """A datetime comes back as itself."""
        value = np.datetime64("2020-01-01T00:00:01")
        assert round_trip(value) == value

    def test_timedelta_round_trip(self):
        """A timedelta comes back as itself."""
        value = np.timedelta64(5, "s")
        assert round_trip(value) == value

    def test_datetime_is_not_timedelta(self):
        """Two times with the same count are two different values."""
        assert digest(np.datetime64(10, "ns")) != digest(np.timedelta64(10, "ns"))


class TestPythonTimes:
    """Tests for the times which are not numpy's."""

    def test_datetime(self):
        """A python datetime is the instant it names."""
        value = datetime.datetime(2020, 1, 1, 0, 0, 1)
        assert digest(value) == digest(np.datetime64("2020-01-01T00:00:01"))
        assert round_trip(value) == np.datetime64(value)

    def test_datetime_values_matter(self):
        """Two different datetimes are two different parameters."""
        first = datetime.datetime(2020, 1, 1)
        assert digest(first) != digest(datetime.datetime(1999, 5, 5))

    def test_aware_datetime_moved_to_utc(self):
        """A datetime carrying a zone is the instant it stands for."""
        zone = datetime.timezone(datetime.timedelta(hours=2))
        aware = datetime.datetime(2020, 1, 1, 2, tzinfo=zone)
        assert digest(aware) == digest(np.datetime64("2020-01-01T00:00:00"))

    def test_date(self):
        """A bare date is midnight on that day."""
        assert digest(datetime.date(2020, 1, 1)) == digest(np.datetime64("2020-01-01"))

    def test_timestamp(self):
        """A pandas timestamp is the same instant as numpy's."""
        assert digest(pd.Timestamp("2020-01-01")) == digest(np.datetime64("2020-01-01"))

    def test_timedelta(self):
        """A python timedelta is the span it names."""
        value = datetime.timedelta(seconds=5)
        assert digest(value) == digest(np.timedelta64(5, "s"))
        assert round_trip(value) == np.timedelta64(5, "s")

    def test_pandas_timedelta(self):
        """A pandas timedelta is the same span as numpy's."""
        assert digest(pd.Timedelta("5s")) == digest(np.timedelta64(5, "s"))

    @pytest.mark.parametrize("value", ["1500-01-01", "2300-01-01"])
    def test_out_of_range_time(self, value):
        """
        A time nanoseconds cannot hold is refused.

        Numpy wraps such a time silently on some versions and raises on
        others, so hashing whatever it wrapped to would make two centuries
        apart the same parameter.
        """
        with pytest.raises(ParameterError, match="nanoseconds"):
            digest(np.datetime64(value))

    def test_sub_nanosecond_time(self):
        """A span too fine for nanoseconds is refused, not truncated."""
        with pytest.raises(ParameterError, match="nanoseconds"):
            digest(np.timedelta64(5, "ps"))

    def test_not_a_time_round_trips(self):
        """A missing time is a value like any other."""
        assert np.isnat(round_trip(np.datetime64("NaT", "s")))
        assert np.isnat(round_trip(np.timedelta64("NaT", "s")))


class TestNumbersAndBytes:
    """Tests for the values JSON has no spelling for."""

    def test_complex_round_trip(self):
        """A complex number comes back as itself."""
        assert round_trip(complex(1.5, -2)) == complex(1.5, -2)

    def test_complex_values_matter(self):
        """The two halves of a complex number both count."""
        assert digest(complex(1, 2)) != digest(complex(2, 1))

    def test_complex_is_not_a_pair(self):
        """A complex number is not the tuple of its parts."""
        assert digest(complex(1, 2)) != digest((1.0, 2.0))

    def test_bytes_round_trip(self):
        """Bytes come back as themselves."""
        assert round_trip(b"\x00\xff") == b"\x00\xff"

    def test_bytes_values_matter(self):
        """Two different byte strings are two different parameters."""
        assert digest(b"a") != digest(b"b")

    def test_bytes_is_not_a_string(self):
        """Bytes are not the string which spells them."""
        assert digest(b"a") != digest("a")


class TestArrays:
    """Tests for array parameters."""

    @pytest.fixture
    def array(self):
        """A small array of known values."""
        return np.arange(6).reshape(2, 3)

    def test_values_matter(self, array):
        """Two arrays with different values digest differently."""
        assert digest(array) != digest(array + 1)

    def test_shape_matters(self, array):
        """The same values in another shape are another array."""
        assert digest(array) != digest(array.reshape(3, 2))

    def test_dtype_matters(self, array):
        """The same values in another dtype are another array."""
        assert digest(array) != digest(array.astype(np.float64))

    def test_endianness_ignored(self, array):
        """The byte order an array is stored in is not part of its value."""
        swapped = array.astype(array.dtype.newbyteorder(">"))
        assert digest(array) == digest(swapped)

    def test_non_contiguous(self, array):
        """A view digests as the values it presents."""
        assert digest(array[:, ::2]) == digest(np.ascontiguousarray(array[:, ::2]))

    def test_object_array(self):
        """An object array holds python values, and encodes as them."""
        array = np.array([1, "a", None], dtype=object)
        assert encode(array) == [1, "a", None]

    def test_round_trip(self, array):
        """A document holds an array's values, not its digest."""
        out = round_trip(array)
        assert np.array_equal(out, array)
        assert out.dtype == array.dtype

    @pytest.mark.parametrize(
        "dtype",
        [
            "int64",
            "float64",
            "bool",
            "datetime64[s]",
            "timedelta64[s]",
            "complex128",
            "<U3",
        ],
    )
    def test_round_trip_every_dtype(self, dtype):
        """A document holds an array's values whatever they are."""
        original = np.arange(3).astype(dtype)
        out = round_trip(original)
        assert np.array_equal(out, original.astype(_normalized(dtype)))

    def test_time_array_unit_normalized(self):
        """An array of times digests as the instants it holds."""
        seconds = np.array([1, 2], dtype="datetime64[s]")
        assert digest(seconds) == digest(seconds.astype("datetime64[ns]"))

    def test_time_array_with_missing_values(self):
        """An array of times may hold missing ones."""
        times = np.array(["2020-01-01", "NaT"], dtype="datetime64[s]")
        out = round_trip(times)
        assert out[0] == times[0] and np.isnat(out[1])

    def test_out_of_range_time_array(self):
        """An array holding a time nanoseconds cannot hold is refused."""
        times = np.array(["1500-01-01"], dtype="datetime64[s]")
        with pytest.raises(ParameterError, match="nanoseconds"):
            digest(times)

    def test_zero_dimensional_object_array(self):
        """A zero dimensional object array holds one value, not none."""
        assert encode(np.array("a", dtype=object)) == "a"

    def test_fingerprint_cannot_round_trip(self, array):
        """An array reduced to a digest refuses to come back."""
        with pytest.raises(ParameterError, match="cannot be read back"):
            decode(encode(array))

    def test_foreign_array_matches_numpy(self, array):
        """An array from another backend digests as its values."""
        strict = pytest.importorskip("array_api_strict")
        foreign = strict.asarray(np.asarray(array, dtype=np.float64))
        assert digest(foreign) == digest(array.astype(np.float64))


class TestCollections:
    """Tests for the containers a parameter may be."""

    def test_tuple_and_list_alike(self):
        """A tuple and a list of the same values are the same parameter."""
        assert digest([1, 2]) == digest((1, 2))

    def test_order_matters(self):
        """A sequence keeps the order it was given in."""
        assert digest([1, 2]) != digest([2, 1])

    def test_set_sorted(self):
        """A set has no order, so its encoding is sorted."""
        # Strings, whose sets do not iterate in sorted order by luck the way
        # a set of small ints does.
        assert digest({"b", "a", "c"}) == digest({"c", "a", "b"})
        assert encode({"c", "a", "b"}) == ["a", "b", "c"]

    def test_frozenset(self):
        """A frozenset is a set."""
        assert digest(frozenset({1, 2})) == digest({1, 2})

    def test_nested(self):
        """Containers nest."""
        assert digest({"a": [1, {"b": (2,)}]}) == digest({"a": [1, {"b": [2]}]})

    def test_none_dropped_from_fingerprint(self):
        """A parameter left at None is the same call as one left out."""
        assert digest({"a": 1, "b": None}) == digest({"a": 1})

    def test_none_kept_in_document(self):
        """A document keeps what it was given, so it can give it back."""
        assert round_trip({"a": 1, "b": None}) == {"a": 1, "b": None}

    def test_tag_shaped_key_is_not_a_tag(self):
        """A mapping which spells a tag is not read back as one."""
        spoof = {"$datetime64": 5}
        assert digest(spoof) != digest(np.datetime64(5, "ns"))
        assert round_trip(spoof) == spoof

    def test_tag_shaped_key_survives_nesting(self):
        """The escape holds for the tag the escape itself uses."""
        spoof = {"$dict": [["a", 1]]}
        assert round_trip(spoof) == spoof

    def test_non_string_keys(self):
        """A mapping keyed by something other than strings round trips."""
        value = {(1, 2): "x", 3: "y"}
        assert round_trip(value) == value

    def test_non_string_keys_sorted(self):
        """Such a mapping digests the same however it was built."""
        assert digest({1: "a", 2: "b"}) == digest({2: "b", 1: "a"})


class TestPaths:
    """Tests for path parameters."""

    def test_posix_string(self):
        """A path encodes as its posix spelling."""
        assert encode(Path("/a/b")) == "/a/b"

    def test_windows_separator_normalized(self):
        """A windows path digests as the same path written for posix."""
        assert digest(PureWindowsPath(r"a\b")) == digest("a/b")


class TestQuantities:
    """Tests for pint quantities."""

    def test_round_trip(self):
        """A quantity comes back as itself."""
        value = dc.get_quantity("10 m")
        assert round_trip(value) == value

    def test_dimensionless_round_trip(self):
        """A quantity with no units comes back as itself."""
        value = dc.get_quantity("2.5")
        assert round_trip(value) == value

    def test_unit_not_normalized(self):
        """The same length written two ways is not the same call."""
        # Magnitudes which survive a conversion unchanged, so the units are
        # the only thing left to tell the two apart.
        assert digest(dc.get_quantity("1.0 m")) != digest(dc.get_quantity("1.0 km"))

    def test_array_magnitude(self):
        """A quantity over an array round trips too."""
        value = np.arange(3) * dc.get_quantity("m/s")
        out = round_trip(value)
        assert np.array_equal(out.magnitude, value.magnitude)
        assert out.units == value.units


class TestOddballs:
    """Tests for the values which have no natural encoding."""

    def test_slice(self):
        """A slice round trips, holes and all."""
        assert round_trip(slice(1, None, 2)) == slice(1, None, 2)

    def test_slice_values_matter(self):
        """Two different slices are two different parameters."""
        assert digest(slice(1, 2)) != digest(slice(1, 3))

    def test_ellipsis(self):
        """An ellipsis round trips."""
        assert round_trip(...) is ...

    def test_named_callable(self):
        """A function is encoded by where it is defined."""
        assert encode(np.mean) == {"$callable": {"path": "numpy:mean"}}

    def test_callable_cannot_round_trip(self):
        """A document does not import whatever it names."""
        with pytest.raises(ParameterError, match="cannot be rebuilt"):
            decode(encode(np.mean, mode=DOCUMENT))

    def test_lambdas_differ_by_source(self):
        """Two lambdas which do different things are different parameters."""
        first = lambda x: x  # noqa: E731
        second = lambda x: x + 1  # noqa: E731
        assert digest(first) != digest(second)

    def test_named_callable_has_no_source(self):
        """A function which can be found by name is not hashed by source."""
        assert "source" not in encode(np.sin)["$callable"]

    def test_sourceless_callable(self):
        """A callable whose source cannot be read is still encodable."""
        # Built by eval, so there is no file holding the text of it.
        assert isinstance(digest(eval("lambda x: x")), str)

    def test_partial(self):
        """A partial is what it wraps and what it wraps it with."""
        value = partial(np.mean, axis=0)
        assert digest(value) != digest(partial(np.mean, axis=1))
        assert digest(value) != digest(np.mean)

    def test_partial_round_trip_refused(self):
        """A partial holds a function, so a document cannot carry it."""
        with pytest.raises(ParameterError, match="cannot be rebuilt"):
            decode(encode(partial(np.mean, axis=0), mode=DOCUMENT))

    def test_opaque_value(self):
        """A value nothing else describes is named by its class."""
        with suppress_warnings(DASCoreWarning):
            assert encode(object()) == {"$opaque": "builtins.object"}

    def test_opaque_warns(self):
        """Reducing a value to its type name is said out loud."""
        with pytest.warns(DASCoreWarning, match="only its type is hashed"):
            digest(object())

    def test_opaque_holds_no_address(self):
        """An address changes every run and would never digest alike."""
        with suppress_warnings(DASCoreWarning):
            assert "0x" not in canonical_json(object())

    def test_opaque_cannot_round_trip(self):
        """A value encoded by name only cannot be rebuilt."""
        with (
            suppress_warnings(DASCoreWarning),
            pytest.raises(ParameterError, match="cannot be rebuilt"),
        ):
            decode(encode(object()))

    def test_plain_value_decodes_to_itself(self):
        """Anything untagged decodes to what it already is."""
        assert decode([1, {"a": "b"}]) == [1, {"a": "b"}]


class TestModels:
    """Tests for dascore models as parameters."""

    def test_fields_matter(self):
        """Two models holding different values digest differently."""
        assert digest(dc.PatchAttrs(tag="a")) != digest(dc.PatchAttrs(tag="b"))

    def test_class_matters(self):
        """Two classes holding the same fields are not the same value."""
        model = dc.PatchAttrs(tag="a")
        assert digest(model) != digest({"tag": "a"})

    def test_round_trip(self):
        """A model comes back as itself."""
        model = dc.PatchAttrs(tag="a", station="wow")
        assert round_trip(model) == model

    def test_extra_fields_included(self):
        """A model's extras are part of what it holds."""
        assert digest(dc.PatchAttrs(my_extra=1)) != digest(dc.PatchAttrs())

    def test_unknown_tag_refused(self):
        """A document naming no known model refuses to decode."""
        document = {"$model": {"object_type": "NotAModelAnyoneHas", "fields": {}}}
        with pytest.raises(InvalidModelTagError, match="Nothing registers"):
            decode(document)


class TestDataFrames:
    """Tests for dataframe parameters."""

    @pytest.fixture
    def df(self):
        """A small dataframe."""
        return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def test_values_matter(self, df):
        """Two frames holding different values digest differently."""
        assert digest(df) != digest(df.assign(a=[1, 3]))

    def test_equal_frames_agree(self, df):
        """Two frames holding the same values digest alike."""
        assert digest(df) == digest(df.copy())

    def test_column_names_matter(self, df):
        """Two frames holding the same numbers under other names differ."""
        assert digest(df) != digest(df.rename(columns={"a": "z"}))

    def test_dtypes_matter(self, df):
        """A column read as floats is not the same column read as ints."""
        assert digest(df) != digest(df.assign(a=df["a"].astype(float)))

    def test_series(self, df):
        """A series digests by its values, like the frame it came from."""
        assert digest(df["a"]) != digest(df["a"] + 1)

    def test_no_document_form(self, df):
        """A frame cannot be written into a document."""
        with pytest.raises(ParameterError, match="cannot be written"):
            encode(df, mode=DOCUMENT)
