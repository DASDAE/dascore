"""Tests for the canonical workflow serializer."""

from __future__ import annotations

import gc
from enum import Enum
from functools import partial
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.workflow import serialize
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


class Color(Enum):
    """An enum whose members stand for strings."""

    red = "red"
    blue = "blue"


class TestDigest:
    """Tests for the digest itself."""

    def test_length(self):
        """The digest is 8 bytes written as hex."""
        assert len(digest("something")) == 16

    def test_stable(self):
        """Equal values digest alike, whatever built them."""
        assert digest({"dim": "time"}) == digest({"dim": "" + "time"})

    def test_hard_coded(self):
        """
        The digest of a fixed value never changes.

        A stored id is only useful if the same call gives the same answer in
        a later release, so this pins the whole chain: the encoding, the
        canonical text, and blake2b.
        """
        assert digest({"a": 1, "b": [1.5, "x"]}) == "3cdfb6194bc1acd6"

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

    def test_fingerprint_cannot_round_trip(self, array):
        """An array reduced to a digest refuses to come back."""
        with pytest.raises(ParameterError, match="cannot be read back"):
            decode(encode(array))

    def test_foreign_array_matches_numpy(self, array):
        """An array from another backend digests as its values."""
        strict = pytest.importorskip("array_api_strict")
        foreign = strict.asarray(np.asarray(array, dtype=np.float64))
        assert digest(foreign) == digest(array.astype(np.float64))

    def test_repeat_operand_cached(self, array, monkeypatch):
        """An array given twice is hashed once."""
        calls = []
        original = np.ascontiguousarray

        def _counted(value, *args, **kwargs):
            calls.append(value)
            return original(value, *args, **kwargs)

        monkeypatch.setattr(np, "ascontiguousarray", _counted)
        assert digest(array) == digest(array)
        assert len(calls) == 1

    def test_cache_released(self, array):
        """The cache holds no array alive once its owner drops it."""
        held = np.arange(4)
        digest(held)
        key = id(held)
        assert key in serialize._ARRAY_DIGESTS
        del held
        gc.collect()
        assert key not in serialize._ARRAY_DIGESTS

    def test_unweakreferenceable_array(self, monkeypatch):
        """An array which refuses a weak reference is still hashed."""

        def _refuse(*args, **kwargs):
            raise TypeError("cannot weakly reference this")

        monkeypatch.setattr(serialize.weakref, "ref", _refuse)
        array = np.arange(3) + 100
        assert digest(array) == digest(np.arange(3) + 100)


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
        assert digest({1, 2, 3}) == digest({3, 2, 1})
        assert encode({3, 1, 2}) == [1, 2, 3]

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
        assert digest(dc.get_quantity("1 m")) != digest(dc.get_quantity("100 cm"))

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
        assert encode(object()) == {"$opaque": "builtins.object"}

    def test_opaque_holds_no_address(self):
        """An address changes every run and would never digest alike."""
        assert "0x" not in canonical_json(object())

    def test_opaque_cannot_round_trip(self):
        """A value encoded by name only cannot be rebuilt."""
        with pytest.raises(ParameterError, match="cannot be rebuilt"):
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
        with pytest.raises(ParameterError, match="Nothing registers"):
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

    def test_series(self, df):
        """A series is a frame's column."""
        assert isinstance(digest(df["a"]), str)

    def test_no_document_form(self, df):
        """A frame cannot be written into a document."""
        with pytest.raises(ParameterError, match="cannot be written"):
            encode(df, mode=DOCUMENT)
