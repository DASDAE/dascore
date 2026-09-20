"""Tests for the one hash, its encoder, and the ids a patch carries."""

from __future__ import annotations

import datetime
import os
import pickle
import subprocess
import sys
from enum import Enum
from functools import partial
from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.config import config_context
from dascore.exceptions import ParameterError
from dascore.units import get_quantity, get_unit
from dascore.utils.array import hash_array
from dascore.utils.identity import (
    H,
    PatchMarker,
    derive,
    encode,
    extract_patches,
    fold_origin_ids,
    merge_operation,
    new_id,
    operation_id,
    origin_id_for,
    result_ids,
    stamp,
    try_operation_id,
    with_ids,
)
from dascore.utils.patch_registry import _as_key, _signature, call_operation_id
from dascore.warnings import DASCoreWarning


def digest(value) -> str:
    """The id of a value in a domain of the tests' own."""
    return H("test", value)


class Color(Enum):
    """An enum whose members stand for strings."""

    red = "red"
    blue = "blue"


def module_level(x):
    """A function a path can find."""
    return x


def make_closure(value):
    """Return a function holding state its source does not show."""

    def _inner(x):
        return x + value

    return _inner


@pytest.fixture()
def patch():
    """A small patch with both ids."""
    return dc.get_example_patch()


class TestH:
    """The one hash."""

    def test_is_32_hex(self):
        """Sixteen bytes, written as hex."""
        out = digest({"a": 1})
        assert len(out) == 32
        assert int(out, 16) >= 0

    def test_domains_keep_payloads_apart(self):
        """The same payload is a different thing in a different domain."""
        assert H("coord", {"a": 1}) != H("operation", {"a": 1})

    def test_key_order_is_not_part_of_it(self):
        """Mappings are canonical."""
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    @pytest.mark.skipif(
        sys.platform == "emscripten", reason="emscripten does not support processes"
    )
    def test_same_in_another_process(self):
        """Nothing process-salted reaches an id."""
        code = (
            "from dascore.utils.identity import H;"
            "print(H('test', {'a': 1, 'b': [1.5, 'x', None], 'c': {'p', 'q', 'r'}}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"PYTHONHASHSEED": "12345"},
        )
        # The set is what a hash seed would reorder.
        payload = {"a": 1, "b": [1.5, "x", None], "c": {"p", "q", "r"}}
        assert out.stdout.strip() == digest(payload)

    def test_a_written_down_id_holds(self):
        """An id recorded last week names the same payload today."""
        assert digest({"a": 1, "b": [1.5, "x"]}) == H("test", {"b": [1.5, "x"], "a": 1})
        assert digest({"a": 1}) == "c0e9167a5ad11d336977ce54316819f9"


class TestScalars:
    """Values which would otherwise read alike."""

    def test_bool_is_not_int(self):
        """True is not 1."""
        assert digest(True) != digest(1)
        assert digest(np.bool_(True)) == digest(True)

    def test_int_is_not_float(self):
        """1 and 1.0 are two calls."""
        assert digest(1) != digest(1.0)

    def test_signed_zero(self):
        """The sign of a zero survives."""
        assert digest(0.0) != digest(-0.0)

    def test_the_floats_json_cannot_spell(self):
        """Each is itself."""
        assert len({digest(float(x)) for x in ("nan", "inf", "-inf")}) == 3

    def test_numpy_scalars_are_their_values(self):
        """The width a number arrived in is not part of it."""
        assert digest(np.int64(3)) == digest(3)
        assert digest(np.float64(1.5)) == digest(1.5)

    def test_enum_is_its_value(self):
        """An enum member stands for what it holds."""
        assert digest(Color.red) == digest("red") != digest(Color.blue)

    def test_complex_and_bytes(self):
        """Neither reads as the things they are made of."""
        assert digest(complex(1, 2)) != digest((1.0, 2.0))
        assert digest(b"a") != digest("a")

    def test_none_is_kept(self):
        """A None is a value: `{}` is not `{"x": None}`."""
        assert digest({}) != digest({"x": None})
        assert digest({"a": {"x": None}}) != digest({"a": {}})

    def test_missing_and_patterns(self):
        """`pd.NA` and a compiled pattern are values `select` is given."""
        import re  # noqa: PLC0415

        assert digest(pd.NA) != digest(None) != digest(float("nan"))
        assert digest(re.compile("a.*")) == digest(re.compile("a.*"))
        assert digest(re.compile("a.*")) != digest(re.compile("b.*"))
        assert digest(re.compile("a", re.IGNORECASE)) != digest(re.compile("a"))

    def test_paths(self):
        """A path is its posix spelling."""
        assert digest(PureWindowsPath(r"a\b")) == digest("a/b")


class TestTimes:
    """Times are nanoseconds, however they were written."""

    def test_unit_is_not_part_of_it(self):
        """A second is a billion nanoseconds."""
        assert digest(np.timedelta64(1, "s")) == digest(np.timedelta64(10**9, "ns"))

    def test_datetime_is_not_timedelta(self):
        """Equal counts, different things."""
        assert digest(np.datetime64(10, "ns")) != digest(np.timedelta64(10, "ns"))

    def test_python_and_pandas_times(self):
        """Every spelling of one instant is one id."""
        expected = digest(np.datetime64("2020-01-01"))
        assert digest(datetime.datetime(2020, 1, 1)) == expected
        assert digest(datetime.date(2020, 1, 1)) == expected
        assert digest(pd.Timestamp("2020-01-01")) == expected
        aware = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)
        assert digest(aware) == expected
        assert digest(pd.Timedelta("5s")) == digest(np.timedelta64(5, "s"))

    def test_out_of_range_is_refused(self):
        """A time nanoseconds cannot hold would wrap to another one."""
        with pytest.raises(ParameterError, match="nanoseconds"):
            digest(np.datetime64("3000-01-01"))
        with pytest.raises(ParameterError, match="nanoseconds"):
            digest(np.array(["3000-01-01"], dtype="datetime64[s]"))


class TestArrays:
    """Arrays by dtype, shape and bytes."""

    def test_values_shape_and_dtype_count(self):
        """Each is part of which array it is."""
        array = np.arange(6)
        assert digest(array) != digest(array + 1)
        assert digest(array) != digest(array.reshape(3, 2))
        assert digest(array) != digest(array.astype(np.float64))

    def test_layout_is_not_part_of_it(self):
        """Byte order and strides are how it is stored, not what it holds."""
        array = np.arange(12.0).reshape(3, 4)
        assert digest(array) == digest(array.astype(">f8"))
        assert digest(array[:, ::2]) == digest(np.ascontiguousarray(array[:, ::2]))

    def test_time_arrays_normalize(self):
        """As scalar times do."""
        seconds = np.array(["2020-01-01"], dtype="datetime64[s]")
        assert digest(seconds) == digest(seconds.astype("datetime64[ns]"))

    def test_object_array_keeps_its_frame(self):
        """An object array is not the list of its elements."""
        values = [0, 2]
        assert digest(np.array(values, dtype=object)) != digest(values)
        first = np.array([[1, 2]], dtype=object)
        assert digest(first) != digest(first.reshape(2, 1))

    def test_foreign_arrays(self):
        """An array from another library is its values."""
        torch = pytest.importorskip("torch")
        array = np.arange(4.0)
        assert digest(torch.asarray(array)) == digest(array)


class TestHashArray:
    """The array-bytes primitive is framed."""

    def test_shape_cannot_read_as_data(self):
        """One int64 zero is not an empty (1, 0) array."""
        first = np.array([0], dtype="i8")
        second = np.empty((1, 0), dtype="i8")
        assert hash_array(first) != hash_array(second)

    def test_the_same_bytes_as_another_dtype(self):
        """Eight zero bytes are two int32s or one int64."""
        assert hash_array(np.zeros(2, "i4")) != hash_array(np.zeros(1, "i8"))
        assert hash_array(np.zeros((2, 1), "i4")) != hash_array(np.zeros((1, 2), "i4"))

    def test_structured_dtypes(self):
        """Two record layouts of one width are two dtypes."""
        first = np.zeros(2, dtype=[("x", "i8")])
        second = np.zeros(2, dtype=[("y", "f8")])
        assert hash_array(first) != hash_array(second)

    def test_object_fields_refused(self):
        """Pointers are not values."""
        with pytest.raises(ParameterError, match="object"):
            hash_array(np.zeros(2, dtype=[("x", object)]))


class TestCollections:
    """Containers are framed so no two of them read alike."""

    def test_list_and_tuple_are_one_spelling(self):
        """On purpose: `(1, 2)` and `[1, 2]` make the same call."""
        assert digest([1, 2]) == digest((1, 2))
        assert digest([1, 2]) != digest([2, 1])

    def test_a_set_is_not_a_list(self):
        """And has no order of its own."""
        assert digest({"b", "a"}) == digest({"a", "b"})
        assert digest({"a", "b"}) != digest(["a", "b"])

    def test_a_tag_spelled_as_data(self):
        """A mapping which looks like a tag is not read as one."""
        spoof = {"$datetime64": 5}
        assert digest(spoof) != digest(np.datetime64(5, "ns"))

    def test_odd_keys(self):
        """Keys which are not strings survive, unordered."""
        assert digest({1: "a", 2: "b"}) == digest({2: "b", 1: "a"})
        assert digest({1: "a"}) != digest({"1": "a"})

    def test_slices_and_ellipsis(self):
        """Both are values."""
        assert digest(slice(1, 2)) != digest(slice(1, 3))
        assert digest(...) != digest(None)


class TestUnits:
    """Quantities and units are the call, not the physics."""

    def test_quantity_spelling_counts(self):
        """A metre is not a hundred centimetres to an operation."""
        assert digest(get_quantity("1 m")) != digest(get_quantity("100 cm"))

    def test_units_differ(self):
        """A unit is not named by its class."""
        assert digest(get_unit("m")) != digest(get_unit("s"))
        assert digest(get_unit("m")) == digest(get_unit("meter"))


class TestRefusals:
    """What cannot be spelled faithfully raises."""

    def test_an_unknown_type(self):
        """Never hashed by its class."""
        with pytest.raises(ParameterError, match="no encoding"):
            digest(object())

    def test_a_patch_is_not_a_parameter(self, patch):
        """It is an input; only its marker is encoded."""
        with pytest.raises(ParameterError):
            digest(patch)
        assert encode(PatchMarker(2)) == {"$patch": 2}

    def test_object_dataframe(self):
        """Pandas hashes 1 and "1" alike in an object column."""
        with pytest.raises(ParameterError, match="object"):
            digest(pd.DataFrame({"a": np.array([1, "1"], dtype=object)}))

    def test_object_index_and_categories(self):
        """The same trap, one level down."""
        index = pd.Index([1, "1"], dtype=object)
        with pytest.raises(ParameterError, match="object"):
            digest(pd.DataFrame({"a": [1, 2]}, index=index))
        mixed = pd.Categorical([1, "1"])
        with pytest.raises(ParameterError, match="object"):
            digest(pd.DataFrame({"a": mixed}))

    def test_a_callable_object(self):
        """Its state is not its name."""

        class Scale:
            def __init__(self, factor):
                self.factor = factor

            def __call__(self, x):
                return x * self.factor

        with pytest.raises(ParameterError, match="callable object"):
            digest(Scale(2))

    def test_no_path_and_no_source(self):
        """Made from text: nothing to resolve and nothing to read."""
        scope: dict = {}
        exec("def made(x):\n    return x", scope)
        with pytest.raises(ParameterError, match="no source"):
            digest(scope["made"])

    def test_a_lambda(self):
        """Nothing tells it from the one beside it."""
        with pytest.raises(ParameterError, match="lambda"):
            digest(lambda x: x)

    def test_a_closure(self):
        """Two closures read alike and behave differently."""
        assert make_closure(1)(1) != make_closure(2)(1)
        with pytest.raises(ParameterError, match="closes over"):
            digest(make_closure(1))

    def test_a_bound_method(self):
        """The instance is state the source does not show."""
        with pytest.raises(ParameterError, match="bound"):
            digest([1, 2].count)

    def test_an_inherited_classmethod(self):
        """Its path names the parent's; which class it reads is state."""

        class Parent:
            factor = 1

            @classmethod
            def scale(cls, x):
                return x * cls.factor

        class Child(Parent):
            factor = 2

        with pytest.raises(ParameterError, match="bound"):
            digest(Child.scale)

    def test_self_reference(self):
        """A value which contains itself has no finite spelling."""
        deep: dict = {}
        deep["self"] = deep
        with pytest.raises(RecursionError):
            digest(deep)


class TestCallables:
    """Callables which can be named."""

    def test_a_function_its_path_finds(self):
        """The path says which it is."""
        assert digest(module_level) == digest(module_level)
        assert digest(np.mean) != digest(np.sum)

    def test_generated_bypasses_are_apart(self):
        """Closures whose paths resolve are named by them."""
        assert digest(dc.proc.abs.raw_function) != digest(dc.proc.imag.raw_function)

    def test_local_functions_by_source_and_defaults(self):
        """A function defined in a call is its source and its defaults."""

        def make(k):
            def local(x, k=k):
                return x + k

            return local

        assert digest(make(1)) == digest(make(1))
        assert digest(make(1)) != digest(make(2))

    def test_partial(self):
        """A partial is what it wraps and what it wraps it with."""
        assert digest(partial(np.mean, axis=0)) != digest(partial(np.mean, axis=1))
        assert digest(partial(np.mean, axis=0)) != digest(np.mean)


class TestModelsAndFrames:
    """Models by tag and fields; frames by labels, dtypes and values."""

    def test_models(self):
        """Fields and extras count, and a model is not a mapping."""
        assert digest(dc.PatchAttrs(tag="a")) != digest(dc.PatchAttrs(tag="b"))
        assert digest(dc.PatchAttrs(tag="a")) != digest({"tag": "a"})
        assert digest(dc.PatchAttrs(my_extra=1)) != digest(dc.PatchAttrs())

    def test_dataclasses(self):
        """By class and fields, as a model is."""
        from dascore.utils.gaps import GapTolerance  # noqa: PLC0415

        assert digest(GapTolerance(count=1.5)) == digest(GapTolerance(count=1.5))
        assert digest(GapTolerance(count=1.5)) != digest(GapTolerance(count=2.5))

    def test_frames(self):
        """Values, labels and dtypes each count."""
        df = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
        assert digest(df) == digest(df.copy())
        assert digest(df) != digest(df.assign(a=[1, 3]))
        assert digest(df) != digest(df.rename(columns={"a": "z"}))
        assert digest(df) != digest(df.assign(a=df["a"].astype(float)))
        assert digest(df["a"]) != digest(df["a"] + 1)
        # A label is its value, and the index is named.
        assert digest(df.rename(columns={"a": 1})) != digest(
            df.rename(columns={"a": "1"})
        )
        assert digest(df.rename_axis("i")) != digest(df.rename_axis("j"))


class TestCoordIdentity:
    """A coordinate has one id wherever it appears."""

    def test_physical_id_ignores_unit_spelling(self):
        """The same length in metres and centimetres is one physical id."""
        metres = dc.get_coord(start=0, stop=10, step=1, units="m")
        centimetres = dc.get_coord(start=0, stop=1000, step=100, units="cm")
        assert metres._physical_id() == centimetres._physical_id()
        assert len(metres._physical_id()) == 32

    def test_identity_is_exact(self):
        """As a parameter they are two: the same select cuts them differently."""
        metres = dc.get_coord(start=0, stop=10, step=1, units="m")
        centimetres = dc.get_coord(start=0, stop=1000, step=100, units="cm")
        assert digest(metres) != digest(centimetres)
        again = dc.get_coord(start=0, stop=10, step=1, units="m")
        assert digest(metres) == digest(again)
        assert encode(metres) == {"$id": list(metres._identity())}

    def test_as_an_operation_parameter(self, patch):
        """Installing either coordinate is a different operation."""
        dist = patch.get_coord("distance")
        other = dist.convert_units("cm")
        first = patch.update_coords(distance=dist).attrs.data_id
        second = patch.update_coords(distance=other).attrs.data_id
        assert first != second


class TestOperationId:
    """Which operation, with which parameters."""

    def test_name_version_and_params_count(self):
        """Each is part of it."""
        base = operation_id("a", {"x": 1})
        assert base == operation_id("a", {"x": 1})
        assert base != operation_id("b", {"x": 1})
        assert base != operation_id("a", {"x": 2})
        assert base != operation_id("a", {"x": 1}, version="2.0")

    def test_a_restated_default_is_the_same_call(self):
        """So a parameter added later changes no id."""
        func = dc.proc.pass_filter
        left_out = call_operation_id(func, (), {"time": (1, 10)})
        restated = call_operation_id(func, (), {"time": (1, 10), "corners": 4})
        changed = call_operation_id(func, (), {"time": (1, 10), "corners": 5})
        assert left_out == restated != changed

    def test_a_list_restates_a_tuple_default(self):
        """The encoder reads them alike, so the default check does too."""
        func = dc.proc.slope_mute
        slopes = {"slopes": (1.0, 2.0)}
        restated = slopes | {"dims": ["distance", "time"]}
        assert call_operation_id(func, (), slopes) == call_operation_id(
            func, (), restated
        )
        other = slopes | {"dims": ["time", "distance"]}
        assert call_operation_id(func, (), slopes) != call_operation_id(func, (), other)

    def test_a_tuple_restates_a_list_default(self):
        """Either way round."""

        @dc.patch_function()
        def listed(patch, names=["a", "b"]):
            """Take a list default."""
            return patch.new(data=patch.data)

        base = call_operation_id(listed, (), {})
        assert base == call_operation_id(listed, (), {"names": ("a", "b")})
        assert base != call_operation_id(listed, (), {"names": ("b", "a")})

    def test_none_is_not_the_default(self):
        """`filter_type=None` slices raw; leaving it out filters first."""
        func = dc.proc.decimate
        assert call_operation_id(func, (), {"time": 10}) != call_operation_id(
            func, (), {"time": 10, "filter_type": None}
        )

    def test_signed_zero_is_not_the_default(self):
        """A default of 0.0 is not restated by -0.0."""

        @dc.patch_function()
        def shifted(patch, amount=0.0):
            """Shift nothing."""
            return patch.new(data=patch.data)

        assert call_operation_id(shifted, (), {}) == call_operation_id(shifted, (0.0,))
        assert call_operation_id(shifted, (), {}) != call_operation_id(shifted, (-0.0,))


class TestRegistryKeys:
    """What the operation-id cache will and will not key."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([0] * 64, id="a long sequence"),
            pytest.param({str(x): x for x in range(64)}, id="a big mapping"),
        ],
    )
    def test_a_big_argument_is_not_worth_keying(self, value):
        """`TypeError` is how `_as_key` says "do not cache this"."""
        with pytest.raises(TypeError):
            _as_key(value)

    def test_a_callable_which_cannot_be_hashed(self):
        """Its signature is asked for the slow way rather than cached."""

        class Unhashable:
            """A callable which refuses to be a dict key."""

            __hash__ = None

            def __call__(self, patch, factor=1):
                """Do nothing."""
                return patch

        assert _signature(Unhashable()) is not None

    def test_try_operation_id(self):
        """None is how a refused operation is spelled."""
        assert try_operation_id("x", {"a": 1}) == operation_id("x", {"a": 1})
        assert try_operation_id("x", {"a": object()}) is None


class TestExtractPatches:
    """Patches are inputs, numbered by the walk which encodes them."""

    def test_roles_survive(self, patch):
        """Swapping which patch fills which argument is another operation."""
        other = patch * 2
        first, inputs_1 = extract_patches({"cond": patch, "other": other})
        second, inputs_2 = extract_patches({"cond": other, "other": patch})
        assert first == second == {"cond": PatchMarker(0), "other": PatchMarker(1)}
        assert inputs_1[0] is patch and inputs_2[0] is other

    def test_where_roles(self):
        """The live shape: `cond` and `other` swapped give other data."""
        coords = {"x": np.arange(3)}
        base = dc.Patch(data=np.array([10, 20, 30]), coords=coords, dims=("x",))
        a = dc.Patch(data=np.array([True, False, True]), coords=coords, dims=("x",))
        b = dc.Patch(data=np.array([False, True, True]), coords=coords, dims=("x",))
        first = base.where(cond=a, other=b)
        second = base.where(cond=b, other=a)
        assert not np.array_equal(first.data, second.data)
        assert first.attrs.data_id != second.attrs.data_id

    def test_nested_and_reordered(self, patch):
        """A mapping's insertion order does not decide who is input zero."""
        other = patch * 2
        first = extract_patches({"group": {"a": patch, "b": other}})
        second = extract_patches({"group": {"b": other, "a": patch}})
        assert first[0] == second[0]
        assert [x is y for x, y in zip(first[1], second[1])] == [True, True]
        swapped = extract_patches({"group": {"a": other, "b": patch}})
        assert swapped[1][0] is other

    def test_top_level_order_is_by_name(self, patch):
        """The order the arguments were given in does not number them."""
        other = patch * 2
        first = extract_patches({"b": other, "a": patch})
        assert first[0] == {"a": PatchMarker(0), "b": PatchMarker(1)}
        assert first[1][0] is patch

    def test_in_a_namedtuple(self, patch):
        """Which cannot be rebuilt from an iterable, and need not be."""
        from collections import namedtuple  # noqa: PLC0415

        pair = namedtuple("pair", ["first", "second"])
        params, found = extract_patches({"items": pair(patch, 3)})
        assert params == {"items": [PatchMarker(0), 3]} and found == [patch]

    def test_in_a_list(self, patch):
        """Order in a sequence is order among the inputs."""
        params, found = extract_patches({"items": [patch, 3, patch]})
        assert params == {"items": [PatchMarker(0), 3, PatchMarker(1)]}
        assert len(found) == 2


class TestDerive:
    """Which array an operation's result is."""

    def test_inputs_order_and_operation_count(self):
        """Each is part of it."""
        assert derive(["a"], "op") == derive(["a"], "op")
        assert derive(["a", "b"], "op") != derive(["b", "a"], "op")
        assert derive(["a"], "op") != derive(["a"], "other")
        assert derive(["a", "a"], "op") != derive(["a"], "op")

    def test_doing_it_twice_is_not_doing_it_once(self):
        """The id commits to everything upstream."""
        once = derive(["a"], "op")
        assert derive([once], "op") != once

    def test_output_position(self):
        """Present only for a result which is one of several."""
        assert derive(["a"], "op", 0) != derive(["a"], "op")
        assert derive(["a"], "op", 0) != derive(["a"], "op", 1)


class TestFoldOriginIds:
    """Which stored data a combination came from."""

    def test_one_origin_is_itself(self):
        """However often it is met: windows of a file are still that file."""
        assert fold_origin_ids(["a"]) == "a"
        assert fold_origin_ids(["a", "a", "a"]) == "a"

    def test_several_fold_in_order(self):
        """First seen first."""
        assert fold_origin_ids(["a", "b"]) != fold_origin_ids(["b", "a"])
        assert fold_origin_ids(["a", "b", "a"]) == fold_origin_ids(["a", "b"])
        assert len(fold_origin_ids(["a", "b"])) == 32

    def test_nothing_folds_to_nothing(self):
        """Data which named no origin does not acquire one."""
        assert fold_origin_ids([]) == ""
        assert fold_origin_ids(["", ""]) == ""
        assert fold_origin_ids(["", "a"]) == "a"


class TestResultIds:
    """The one stamping rule."""

    def test_derived_from_parents(self, patch):
        """The plain case."""
        out = result_ids([patch.attrs], "op")
        assert out["origin_id"] == patch.attrs.origin_id
        assert out["data_id"] == derive([patch.attrs.data_id], "op")

    def test_a_refusing_operation_is_random(self, patch):
        """Never the input's id, never twice the same."""
        first = result_ids([patch.attrs], None)["data_id"]
        second = result_ids([patch.attrs], None)["data_id"]
        assert len({first, second, patch.attrs.data_id}) == 3

    def test_a_parent_with_no_id_is_random(self, patch):
        """Two unnamed inputs must not lead to one id."""
        blank = patch.attrs.update(origin_id="", data_id="")
        first = result_ids([blank], "op")["data_id"]
        assert first and first != result_ids([blank], "op")["data_id"]
        # One named parent does not make the other's data known.
        mixed = [patch.attrs, blank]
        assert result_ids(mixed, "op")["data_id"] != result_ids(mixed, "op")["data_id"]
        assert result_ids(mixed, "op")["origin_id"] == patch.attrs.origin_id

    def test_disabled_clears(self, patch):
        """What changed the data was not recorded, so it claims no id."""
        with config_context(patch_provenance="disabled"):
            out = stamp(patch.attrs, [patch.attrs], "op")
        assert out.origin_id == out.data_id == ""


class TestOriginIdFor:
    """The id of a stored patch."""

    def test_every_part_counts(self):
        """Format, version, path, key, size and mtime."""
        from dascore.core.source import ArraySource  # noqa: PLC0415

        base = dict(path="/a.h5", format="DASDAE", version="1", key="k")
        expected = origin_id_for(ArraySource(**base), 10, 20)
        for name in base:
            changed = ArraySource(**(base | {name: "other"}))
            assert origin_id_for(changed, 10, 20) != expected
        assert origin_id_for(ArraySource(**base), 11, 20) != expected
        assert origin_id_for(ArraySource(**base), 10, 21) != expected

    def test_ordinal_stands_in_for_a_key(self):
        """Only when the source names none."""
        from dascore.core.source import ArraySource  # noqa: PLC0415

        keyless = ArraySource(path="/a.h5", format="F", version="1")
        assert origin_id_for(keyless, ordinal=0) != origin_id_for(keyless, ordinal=1)
        keyed = ArraySource(path="/a.h5", format="F", version="1", key="k")
        assert origin_id_for(keyed, ordinal=0) == origin_id_for(keyed, ordinal=1)


class TestPatchRules:
    """The rules, on real patches."""

    def test_new_data_is_its_own_origin(self, patch):
        """A patch nothing was done to is its origin."""
        assert patch.attrs.origin_id == patch.attrs.data_id
        assert len(patch.attrs.origin_id) == 32
        assert dc.get_example_patch().attrs.origin_id != patch.attrs.origin_id

    def test_an_operation_derives(self, patch):
        """Origin kept, data id moved, the same route the same id."""
        first = patch.pass_filter(time=(1, 10))
        second = patch.pass_filter(time=(1, 10))
        assert first.attrs.origin_id == patch.attrs.origin_id
        assert first.attrs.data_id == second.attrs.data_id != patch.attrs.data_id
        assert patch.pass_filter(time=(1, 11)).attrs.data_id != first.attrs.data_id

    def test_order_of_operations(self, patch):
        """Two routes, two ids."""
        one = patch.abs().pass_filter(time=(1, 10))
        two = patch.pass_filter(time=(1, 10)).abs()
        assert one.attrs.data_id != two.attrs.data_id

    def test_a_no_op_keeps_its_ids(self, patch):
        """Nothing was done, and nothing is what it records."""
        out = patch.transpose(*patch.dims)
        assert out.attrs.data_id == patch.attrs.data_id

    def test_units_tell_calls_apart(self, patch):
        """The probe which found one id for two units."""
        first = patch.set_units(get_unit("m"))
        second = patch.set_units(get_unit("s"))
        assert first.attrs.data_id != second.attrs.data_id

    def test_a_refused_call_keeps_every_origin(self, patch):
        """The data id is random; where the data came from is still known."""
        from decimal import Decimal  # noqa: PLC0415

        other = dc.get_example_patch() > 0
        out = patch.where(other, other=Decimal("0"))
        assert out.attrs.origin_id == fold_origin_ids(
            [patch.attrs.origin_id, other.attrs.origin_id]
        )
        assert out.attrs.data_id not in ("", patch.attrs.data_id)

    def test_a_refused_call_keeps_nested_origins(self, patch):
        """A patch in a list still says where the data came from."""
        other = dc.get_example_patch()

        @dc.patch_function()
        def combine(patch, items=None, odd=None):
            """Add the patches in a list."""
            return patch.new(data=patch.data + items[0].data)

        out = combine(patch, items=[other], odd=object())
        assert out.attrs.origin_id == fold_origin_ids(
            [patch.attrs.origin_id, other.attrs.origin_id]
        )

    def test_former_names_are_not_part_of_equality(self, patch):
        """Left on unvalidated attrs, they do not make equal patches unequal."""
        stale = patch.attrs.model_copy()
        stale.__pydantic_extra__ = dict(stale.__pydantic_extra__ or {}, patch_id="abc")
        assert patch.new(attrs=stale).equals(patch, only_required_attrs=False)

    def test_a_factory_made_patch_function(self, patch):
        """Closures over different values never share an id."""

        def make(factor):
            @dc.patch_function()
            def scale(patch):
                """Scale the data."""
                return patch.new(data=patch.data * factor)

            return scale

        ids = {make(x)(patch).attrs.data_id for x in (1, 2, 2)}
        assert len(ids) == 3 and "" not in ids

    def test_a_closure_argument_is_random(self, patch):
        """The call works; its id is nobody else's."""

        @dc.patch_function()
        def apply(patch, fn=None):
            """Apply a function to the data."""
            return patch.new(data=fn(patch.data))

        with pytest.warns(DASCoreWarning, match="random data_id"):
            first = apply(patch, fn=make_closure(1))
        second = apply(patch, fn=make_closure(1))
        ids = {first.attrs.data_id, second.attrs.data_id, patch.attrs.data_id, ""}
        assert len(ids) == 4
        assert first.attrs.origin_id == patch.attrs.origin_id

    def test_arithmetic_and_array_functions(self, patch):
        """Operations which are not patch functions are named too."""
        assert (patch + 1).attrs.data_id != (patch - 1).attrs.data_id
        assert (patch + 1).attrs.data_id == (patch + 1).attrs.data_id
        # Each site derives, rather than falling back to a random id.
        assert np.abs(patch).attrs.data_id == np.abs(patch).attrs.data_id
        first, second = np.mean(patch, axis=0), np.mean(patch, axis=0)
        assert first.attrs.data_id == second.attrs.data_id
        assert np.abs(patch).attrs.data_id != np.sqrt(np.abs(patch)).attrs.data_id
        by_axis = {np.mean(patch, axis=x).attrs.data_id for x in (0, 1)}
        assert len(by_axis) == 2

    def test_a_dtype_argument(self, patch):
        """A numpy dtype is spelled out, so it tells two calls apart."""
        first = np.sum(patch, axis=0, dtype=np.dtype("float32"))
        again = np.sum(patch, axis=0, dtype=np.dtype("float32"))
        other = np.sum(patch, axis=0, dtype=np.dtype("float64"))
        assert first.attrs.data_id == again.attrs.data_id != other.attrs.data_id

    def test_disabled_hashes_nothing(self, patch, monkeypatch):
        """With ids off, no argument is encoded."""
        import dascore.utils.identity as identity  # noqa: PLC0415

        def _fail(*args, **kwargs):
            raise AssertionError("an id was computed with ids disabled")

        with config_context(patch_provenance="disabled"):
            monkeypatch.setattr(identity, "H", _fail)
            assert (patch + 1).attrs.data_id == ""
            assert patch.pass_filter(time=(1, 10)).attrs.data_id == ""
            assert np.abs(patch).attrs.data_id == ""

    def test_two_patches(self, patch):
        """Both are inputs, in order; one origin stays itself."""
        other = patch * 2
        assert (patch - other).attrs.data_id != (other - patch).attrs.data_id
        assert (patch - other).attrs.origin_id == patch.attrs.origin_id
        fresh = dc.get_example_patch()
        assert (patch + fresh).attrs.origin_id == fold_origin_ids(
            [patch.attrs.origin_id, fresh.attrs.origin_id]
        )

    def test_disabled_then_changed(self, patch):
        """An id nobody kept up to date is not carried."""
        with config_context(patch_provenance="disabled"):
            out = patch.abs()
        assert out.attrs.data_id == out.attrs.origin_id == ""
        # Turned back on, the unnamed parent gives a random id, not a shared one.
        assert out.abs().attrs.data_id != out.abs().attrs.data_id

    def test_ids_are_not_part_of_equality(self, patch):
        """Two patches holding the same data are equal."""
        assert patch == patch.update_attrs(origin_id=new_id(), data_id=new_id())

    def test_pickle(self, patch):
        """The ids travel."""
        out = pickle.loads(pickle.dumps(patch.abs()))
        assert out.attrs.data_id == patch.abs().attrs.data_id

    def test_attrs_from_before_the_fields(self, patch):
        """An old pickle's attrs gain both on the way in."""
        attrs = patch.attrs.model_copy()
        for name in ("origin_id", "data_id"):
            del attrs.__dict__[name]
        out = with_ids(attrs)
        assert out.origin_id == out.data_id and len(out.origin_id) == 32


class TestCombinations:
    """Concatenate, stack and merge."""

    @pytest.fixture()
    def members(self):
        """Three patches of one acquisition, adjacent in time."""
        return list(dc.get_example_spool("random_das"))

    def test_processed_then_joined_shares_the_origin(self, members):
        """The same stored data, another array."""
        done = [x.abs().pass_filter(time=(1, 10)) for x in members]
        joined = dc.spool(done).concatenate(time=None)[0]
        plain = dc.spool(members).concatenate(time=None)[0]
        assert joined.attrs.origin_id == plain.attrs.origin_id
        assert joined.attrs.data_id != plain.attrs.data_id
        assert plain.attrs.origin_id == fold_origin_ids(
            [x.attrs.origin_id for x in members]
        )

    def test_merge_is_an_operation(self, members):
        """A merged patch is derived from its members, in order."""
        merged = dc.spool(members).chunk(time=None)[0]
        assert merged.attrs.data_id not in {x.attrs.data_id for x in members}
        assert (
            merged.attrs.data_id == dc.spool(members).chunk(time=None)[0].attrs.data_id
        )
        backwards = derive([x.attrs.data_id for x in members[::-1]], merge_operation())
        assert merged.attrs.data_id != backwards

    def test_merge_options_are_part_of_it(self, members):
        """`snap_coords` changes the merged coordinate, so it changes the id."""
        snapped = dc.spool(members).chunk(time=None)[0]
        exact = dc.spool(members).chunk(time=None, snap_coords=False)[0]
        assert snapped.attrs.data_id != exact.attrs.data_id

    def test_unencodable_merge_options(self, members):
        """The merge still happens; its id is nobody else's."""
        attrs = [x.attrs for x in members]
        kwargs = {"merge_params": {"odd": object()}}
        first = dc.utils.attrs.combine_patch_attrs(attrs, **kwargs)
        second = dc.utils.attrs.combine_patch_attrs(attrs, **kwargs)
        assert first.data_id != second.data_id
        assert first.origin_id == second.origin_id

    def test_former_id_names(self, members):
        """Attrs from before the rename read as the ids they were."""
        attrs = dc.PatchAttrs(patch_id="abc", processing_id="def")
        assert (attrs.origin_id, attrs.data_id) == ("abc", "def")
        assert "patch_id" not in attrs.model_dump()
        # Left on an unvalidated (unpickled) instance, they decide no merge.
        old = []
        for index, member in enumerate(members):
            stale = member.attrs.model_copy()
            extra = dict(stale.__pydantic_extra__ or {}, patch_id=str(index))
            stale.__pydantic_extra__ = extra
            old.append(member.new(attrs=stale))
        assert len(dc.spool(old).chunk(time=None)) == 1

    def test_a_single_member_merge_is_the_member(self, members):
        """Nothing was combined."""
        out = dc.spool(members[:1]).chunk(time=None)[0]
        assert out.attrs.data_id == members[0].attrs.data_id

    def test_stack(self, members):
        """Stacking derives from the kept members."""
        same = [members[0], members[0].update_attrs(tag="x")]
        out = dc.utils.patch.stack_patches(same, dim_vary="time")
        assert out.attrs.origin_id == members[0].attrs.origin_id
        assert out.attrs.data_id not in {x.attrs.data_id for x in same}


class TestSeveralOutputs:
    """A patch function which returns several patches."""

    def test_each_has_its_place(self, patch):
        """The position is part of the id; the input itself is left alone."""

        @dc.patch_function()
        def halves(patch):
            """Return the patch, and two new ones."""
            return [patch, patch.new(data=patch.data), patch.new(data=patch.data)]

        same, first, second = halves(patch)
        assert same is patch
        op = call_operation_id(halves, (), {})
        assert first.attrs.data_id == derive([patch.attrs.data_id], op, 1)
        assert second.attrs.data_id == derive([patch.attrs.data_id], op, 2)

    def test_a_spool_comes_back_a_spool(self, patch):
        """And its members are stamped by position."""

        @dc.patch_function()
        def as_spool(patch):
            """Return a spool of two new patches."""
            return dc.spool([patch.new(data=patch.data), patch.new(data=patch.data)])

        out = as_spool(patch)
        assert isinstance(out, dc.BaseSpool)
        op = call_operation_id(as_spool, (), {})
        expected = [derive([patch.attrs.data_id], op, x) for x in (0, 1)]
        assert [x.attrs.data_id for x in out] == expected

    def test_a_namedtuple_comes_back_one(self, patch):
        """A tuple which takes its members one by one."""
        from collections import namedtuple  # noqa: PLC0415

        pair = namedtuple("pair", ["low", "high"])

        @dc.patch_function()
        def split(patch):
            """Return two new patches by name."""
            return pair(patch.new(data=patch.data), patch.new(data=patch.data))

        out = split(patch)
        assert isinstance(out, pair)
        assert out.low.attrs.data_id != out.high.attrs.data_id


class TestStoredIds:
    """Ids through files and the index."""

    def test_dasdae_round_trip(self, patch, tmp_path):
        """Both ids stored win over what would be derived."""
        done = patch.pass_filter(time=(1, 10))
        path = tmp_path / "a.h5"
        done.io.write(path, "dasdae")
        out = dc.read(path)[0]
        assert out.attrs.origin_id == done.attrs.origin_id
        assert out.attrs.data_id == done.attrs.data_id
        assert out.abs().attrs.data_id == done.abs().attrs.data_id

    def test_a_derived_origin_names_the_file(self, tmp_path):
        """Read twice the same; renamed or touched, another."""
        import os  # noqa: PLC0415
        import shutil  # noqa: PLC0415

        from dascore.utils.downloader import fetch  # noqa: PLC0415

        path = tmp_path / "a.hdf5"
        shutil.copy2(fetch("terra15_das_1_trimmed.hdf5"), path)
        first = dc.read(path)[0]
        assert first.attrs.data_id == first.attrs.origin_id
        assert dc.read(path)[0].attrs.origin_id == first.attrs.origin_id
        moved = tmp_path / "b.hdf5"
        os.rename(path, moved)
        renamed = dc.read(moved)[0]
        assert renamed.attrs.origin_id != first.attrs.origin_id
        stat = os.stat(moved)
        os.utime(moved, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        assert dc.read(moved)[0].attrs.origin_id != renamed.attrs.origin_id

    def test_two_indexes_agree(self, tmp_path, patch):
        """An id is derived from the source, not from a database row."""
        import shutil  # noqa: PLC0415

        from dascore.utils.downloader import fetch  # noqa: PLC0415

        data = tmp_path / "data"
        data.mkdir()
        shutil.copy2(fetch("terra15_das_1_trimmed.hdf5"), data / "a.hdf5")
        ids = []
        for name in ("one", "two"):
            index = tmp_path / f"{name}.sqlite"
            spool = dc.spool(data, index_path=index).update()
            ids.append(spool.get_contents()["origin_id"].iloc[0])
        assert ids[0] == ids[1] == dc.read(data / "a.hdf5")[0].attrs.origin_id
