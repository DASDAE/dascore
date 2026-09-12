"""
Tests for the two ids a patch carries.

These cover the rules themselves; the tests which pin where they are
applied live beside the things which apply them.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import replace

import numpy as np
import pytest

import dascore as dc
import dascore.utils.patch_registry as registry_module
from dascore.core.source import PatchSource
from dascore.units import get_unit
from dascore.utils.identity import (
    NOTHING_DONE,
    advance,
    fold_ids,
    fold_patch_ids,
    fold_processing_ids,
    new_patch_id,
    operation_fingerprint,
    patch_id_of,
    processing_id_of,
    source_patch_id,
    stamp_combination,
    with_patch_id,
)
from dascore.utils.patch import concatenate_patches, concatenate_planned, stack_patches
from dascore.utils.patch_registry import _as_key, _signature, fingerprint_call
from dascore.utils.serialize import PATCH_ARGUMENT, encode
from dascore.warnings import DASCoreWarning


class TestNewDataId:
    """Data which names no source."""

    def test_each_is_its_own(self):
        """Two arrays are two data, however alike they look."""
        assert new_patch_id() != new_patch_id()

    def test_assignment_preserves_attrs(self):
        """Minting an internal ID preserves validated fields and the input attrs."""
        attrs = dc.PatchAttrs(data_units="m/s", tag="measurement", vendor_value=3)
        updated = with_patch_id(attrs)
        assert attrs.patch_id == ""
        assert updated.patch_id
        assert updated.model_dump(exclude={"patch_id"}) == attrs.model_dump(
            exclude={"patch_id"}
        )

    def test_legacy_attrs_restore_defaults(self):
        """Unpickled attrs missing identity fields receive their current defaults."""
        attrs = dc.PatchAttrs(data_units="m/s")
        del attrs.__dict__["patch_id"]
        del attrs.__dict__["processing_id"]
        updated = with_patch_id(attrs)
        assert updated.patch_id
        assert updated.processing_id == ""
        assert updated.data_units == attrs.data_units
        assert not hasattr(attrs, "patch_id")

    def test_it_is_a_hex_string(self):
        """The same shape as every other id, so nothing can tell them apart."""
        made = new_patch_id()
        assert isinstance(made, str)
        assert int(made, 16) >= 0


_SOURCE = PatchSource(format="DASDAE", version="1", path="/data/one.h5")


class TestSourceDataId:
    """A source object retains the original digest inputs."""

    @pytest.mark.parametrize(
        "key, ordinal, expected",
        [
            ("", 0, "2b45cb0df5f48f50"),
            ("0", None, "e64c07af748c4576"),
            ("channel-3", None, "eb088c90676d1ede"),
            ("", None, "fcbe7acb28166aad"),
        ],
    )
    def test_pre_migration_digest(self, key, ordinal, expected):
        """Pinned hashes distinguish native strings from integer ordinal fallbacks."""
        source = replace(_SOURCE, key=key)
        assert source_patch_id(source, 40, 12345, ordinal=ordinal) == expected

    @pytest.mark.parametrize(
        "changed",
        [
            {"format": "TERRA15"},
            {"version": "2"},
            {"path": "/data/two.h5"},
            {"key": "1"},
            {"ordinal": 1},
            {"ordinal": None},
            {"size_bytes": 41},
            {"mtime_ns": 999},
            {"size_bytes": None},
            {"mtime_ns": None},
        ],
    )
    def test_every_part_counts(self, changed):
        """Each identity field names data the others would call the same."""
        source_fields = {
            k: v
            for k, v in changed.items()
            if k in {"format", "version", "path", "key"}
        }
        stats = {"size_bytes": 40, "mtime_ns": 12345, "ordinal": 0}
        stats.update({k: v for k, v in changed.items() if k not in source_fields})
        assert source_patch_id(
            replace(_SOURCE, **source_fields), **stats
        ) != source_patch_id(_SOURCE, 40, 12345, ordinal=0)

    def test_missing_stats(self):
        """Omitted stats and explicit missing stats name the same source."""
        made = source_patch_id(_SOURCE, ordinal=0)
        assert made == source_patch_id(_SOURCE, None, None, ordinal=0)
        assert len(made) == len(source_patch_id(_SOURCE, 40, 12345, ordinal=0))


class TestAdvance:
    """What was done."""

    def test_nothing_done_is_the_starting_point(self):
        """Data which arrived and has not been touched says so."""
        assert NOTHING_DONE == ""

    def test_an_operation_moves_it(self):
        """Which is the whole point of the id."""
        assert advance(NOTHING_DONE, "0123456789abcdef") != NOTHING_DONE

    def test_the_same_route_gives_the_same_answer(self):
        """So two patches processed alike can be told to be alike."""
        once = advance(NOTHING_DONE, "aaaa")
        assert advance(once, "bbbb") == advance(advance(NOTHING_DONE, "aaaa"), "bbbb")

    def test_order_matters(self):
        """Filtering then decimating is not decimating then filtering."""
        first = advance(advance(NOTHING_DONE, "aaaa"), "bbbb")
        second = advance(advance(NOTHING_DONE, "bbbb"), "aaaa")
        assert first != second

    def test_doing_it_twice_is_not_doing_it_once(self):
        """A fold of a fold, so a repeated operation is two operations."""
        once = advance(NOTHING_DONE, "aaaa")
        assert advance(once, "aaaa") != once

    def test_a_different_operation_is_a_different_answer(self):
        """The fingerprint is what distinguishes them."""
        assert advance(NOTHING_DONE, "aaaa") != advance(NOTHING_DONE, "bbbb")


class TestFoldDataIds:
    """Which data, when there was more than one."""

    def test_one_folds_to_itself(self):
        """Combining a patch with nothing leaves the id where it was."""
        assert fold_patch_ids(["only"]) == "only"

    def test_order_is_part_of_it(self):
        """Concatenating a before b is not concatenating b before a."""
        assert fold_patch_ids(["a", "b"]) != fold_patch_ids(["b", "a"])

    def test_repeats_are_part_of_it(self):
        """Stacking a patch with itself is not the patch."""
        assert fold_patch_ids(["a", "a"]) != fold_patch_ids(["a"])

    def test_it_is_derived(self):
        """So the same combination gives the same answer twice."""
        assert fold_patch_ids(["a", "b"]) == fold_patch_ids(["a", "b"])

    def test_it_takes_any_sequence(self):
        """Callers hold their members in whatever they hold them in."""
        assert fold_patch_ids(("a", "b")) == fold_patch_ids(["a", "b"])


class TestFoldProcessingIds:
    """What was done, when the inputs disagree."""

    def test_a_common_route_survives(self):
        """Sixty windows of one file have one history, not sixty."""
        assert fold_processing_ids(["x", "x", "x"]) == "x"

    def test_distinct_routes_fold(self):
        """Combining differently processed data says so."""
        folded = fold_processing_ids(["x", "y"])
        assert folded not in {"x", "y"}

    def test_the_fold_is_stable(self):
        """And says it the same way every time."""
        assert fold_processing_ids(["x", "y"]) == fold_processing_ids(["x", "y"])

    def test_first_seen_order(self):
        """Two orders of the same routes are two answers."""
        assert fold_processing_ids(["x", "y"]) != fold_processing_ids(["y", "x"])

    def test_repeats_do_not_count(self):
        """It is the distinct routes which matter, not how many took each."""
        assert fold_processing_ids(["x", "y", "x"]) == fold_processing_ids(["x", "y"])

    def test_nothing_folds_to_nothing_done(self):
        """An operation given no inputs has nothing to carry forward."""
        assert fold_processing_ids([]) == NOTHING_DONE

    def test_untouched_inputs_stay_untouched(self):
        """Reading two files and putting them together is not processing."""
        assert fold_processing_ids([NOTHING_DONE, NOTHING_DONE]) == NOTHING_DONE


# A change here moves every id these calls record; make it deliberately.
_RECORDED_IDS = {
    "pass_filter": (
        "a55c04918ad2fd1b",
        lambda p, q, q2: p.pass_filter(time=(1, 10)),
    ),
    "where_patch": ("ef18744a9d94e433", lambda p, q, q2: p.where(p > 0)),
    "add": ("a6a849bc6d25158b", lambda p, q, q2: p + 1),
    "rsub": ("2b6260154c90dae0", lambda p, q, q2: 1 - p),
    "add_patch": ("da85a67f770cffb9", lambda p, q, q2: p + q),
    "multiply_array": (
        "0b91efb1fa4f6953",
        lambda p, q, q2: p * np.ones(p.shape),
    ),
    "unary": ("c81e34e4b05acf4b", lambda p, q, q2: np.abs(p)),
    "array_function": ("5d1f21ba2e75b9ba", lambda p, q, q2: np.mean(p, axis=0)),
    "concatenate": (
        "f1dcdfa083a08f29",
        lambda p, q, q2: concatenate_patches([p, q2], time=None)[0],
    ),
    "concatenate_new_dim": (
        "8d129d6050095ddc",
        lambda p, q, q2: concatenate_patches([p, q], new=None)[0],
    ),
    "concatenate_planned": (
        "ec7d67711b1c9679",
        lambda p, q, q2: concatenate_planned([p, q2], "time", conflict="drop"),
    ),
    "stack": ("5be94a7e6ec6af39", lambda p, q, q2: stack_patches([p, q])),
    "stack_dim_vary": (
        "1356f20526435262",
        lambda p, q, q2: stack_patches([p, q2], dim_vary="time"),
    ),
}


class TestOperationFingerprints:
    """The operations which are not patch functions still have names."""

    @pytest.fixture(scope="class")
    @classmethod
    def pair(cls):
        """Two patches with fixed ids, the second following the first in time."""
        base = dc.get_example_patch()
        first = base.update_attrs(patch_id="a" * 16, processing_id="")
        second = base.update_attrs(patch_id="b" * 16, processing_id="")
        time = first.get_coord("time")
        return first, second, second.update_coords(time_min=time.max() + time.step)

    @pytest.mark.filterwarnings("ignore:Concatenating patches whose histories")
    @pytest.mark.parametrize("name", sorted(_RECORDED_IDS))
    def test_recorded_ids_hold(self, pair, name):
        """Each call stamps the id recorded for it."""
        expected, call = _RECORDED_IDS[name]
        assert call(*pair).attrs.processing_id == expected

    def test_arguments_tell_calls_apart(self, pair):
        """Each site records its arguments, not just its kind."""
        p, q, _ = pair
        ids = [
            (p + 1).attrs.processing_id,
            (p + 2).attrs.processing_id,
            (1 - p).attrs.processing_id,
            np.mean(p, axis=0).attrs.processing_id,
            np.mean(p, axis=1).attrs.processing_id,
            np.mean(p, 0).attrs.processing_id,
            np.mean(p, 1).attrs.processing_id,
            stack_patches([p, q]).attrs.processing_id,
            stack_patches([p, q], dim_vary="time").attrs.processing_id,
        ]
        assert len(set(ids)) == len(ids)

    def test_a_unit_operand_does_not_warn(self, pair):
        """A pint unit has no encoding of its own; stamping it stays quiet."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DASCoreWarning)
            _ = pair[0] * get_unit("m")
            _ = np.multiply(pair[0], get_unit("m"))

    def test_an_array_operand_is_not_frozen(self, pair):
        """Fingerprinting an operand leaves the caller's array writable."""
        values = np.ones(pair[0].shape)
        _ = pair[0] * values
        assert values.flags.writeable

    def test_kind_and_params_count(self):
        """A kind or a parameter apart is another operation."""
        base = operation_fingerprint("Ufunc", {"name": "add"})
        assert operation_fingerprint("Ufunc", {"name": "subtract"}) != base
        assert operation_fingerprint("ArrayFunc", {"name": "add"}) != base
        assert operation_fingerprint("Ufunc", {"name": "add", "reversed": True}) != base
        assert operation_fingerprint("Ufunc", {"name": "add"}, version="2") != base

    def test_concatenate_names_what_it_was_given(self, pair):
        """
        Concatenating along time is not concatenating along distance.

        `time=None` is the documented call, and the serializer drops a
        `None` mapping value, so the dimensions are held as pairs.
        """
        p, q, q2 = pair
        distance = q.get_coord("distance")
        beside = q.update_coords(distance_min=distance.max() + distance.step)
        along_time = concatenate_patches([p, q2], time=None)[0]
        along_distance = concatenate_patches([p, beside], distance=None)[0]
        assert along_time.attrs.processing_id != along_distance.attrs.processing_id


class TestTheRulesOnRealPatches:
    """The rules, where they are actually applied."""

    @pytest.fixture(scope="class")
    def patch(self):
        """A patch to operate on."""
        return dc.get_example_patch("random_das")

    def test_data_arrives_with_an_id(self, patch):
        """Everything downstream needs something to carry forward."""
        assert patch.attrs.patch_id
        assert patch.attrs.processing_id == NOTHING_DONE

    def test_two_patches_are_two_data(self):
        """Nothing derives an id from values, so nothing claims they are one."""
        first = dc.get_example_patch("random_das")
        assert first.attrs.patch_id != dc.get_example_patch("random_das").attrs.patch_id

    @pytest.mark.parametrize(
        "builder",
        [
            lambda p: p.new(data=p.data),
            lambda p: p.new(data=np.asarray(p.data) * 2),
            lambda p: p.update(data=np.asarray(p.data) * 2),
            lambda p: p.update_attrs(tag="rebuilt"),
        ],
    )
    def test_building_a_patch_is_not_operating_on_one(self, patch, builder):
        """
        New and update preserve both IDs to avoid double-counting processor operations.

        Direct data changes through these methods therefore also preserve IDs.
        """
        # Operated on first: an unprocessed patch states no route, so
        # preserving it would be satisfied by dropping it.
        processed = patch.abs()
        assert processed.attrs.processing_id != NOTHING_DONE
        out = builder(processed)
        assert out.attrs.patch_id == processed.attrs.patch_id
        assert out.attrs.processing_id == processed.attrs.processing_id

    def test_a_patch_built_from_arrays_is_new_data(self, patch):
        """Naming no source, it is not the same data as anything else."""
        built = dc.Patch(
            data=np.asarray(patch.data), coords=patch.coords, dims=patch.dims
        )
        assert built.attrs.patch_id != patch.attrs.patch_id

    def test_an_operation_advances_what_was_done(self, patch):
        """Which is what the id is for."""
        out = patch.normalize("time")
        assert out.attrs.processing_id != patch.attrs.processing_id

    def test_an_operation_leaves_which_data_alone(self, patch):
        """Filtering data does not make it other data."""
        assert patch.normalize("time").attrs.patch_id == patch.attrs.patch_id

    def test_a_function_which_rebuilds_still_leaves_it_alone(self, patch):
        """
        Even one which builds its result from scratch.

        Without the wrapper carrying the id across from the input, a body
        which returns a patch it built itself would mint a fresh one and
        claim the data had changed.
        """

        @dc.patch_function()
        def rebuilds(patch):
            """Return a patch built from nothing but the data."""
            return dc.Patch(data=patch.data, coords=patch.coords, dims=patch.dims)

        assert rebuilds(patch).attrs.patch_id == patch.attrs.patch_id
        # And a real one which does the same thing.
        assert patch.fbe(0.5, time=(10, 100)).attrs.patch_id == patch.attrs.patch_id

    def test_the_same_route_gives_the_same_id(self, patch):
        """So two patches processed alike can be told to be alike."""
        first = patch.normalize("time").decimate(time=2)
        second = patch.normalize("time").decimate(time=2)
        assert first.attrs.processing_id == second.attrs.processing_id

    def test_a_different_route_does_not(self, patch):
        """Order included."""
        forward = patch.normalize("time").decimate(time=2)
        backward = patch.decimate(time=2).normalize("time")
        assert forward.attrs.processing_id != backward.attrs.processing_id

    def test_different_arguments_are_a_different_route(self, patch):
        """The operation's fingerprint is what distinguishes them."""
        assert patch.normalize("time").attrs.processing_id != (
            patch.normalize("distance").attrs.processing_id
        )

    def test_a_no_op_records_nothing(self, patch):
        """
        An operation which handed the patch straight back did nothing.

        `select` with no bounds is the live case: it returns the patch it
        was given, and a `processing_id` which moved would say otherwise.
        """
        assert patch.select(time=None).attrs.processing_id == patch.attrs.processing_id

    def test_the_ids_are_not_part_of_equality(self, patch):
        """Two patches with the same data are equal however they were made."""
        assert patch.equals(patch.update_attrs(patch_id="x", processing_id="y"))

    def test_a_raw_constructor_keeps_what_it_was_given(self, patch):
        """You edited it; that is on you."""
        made = patch.update_attrs(patch_id="kept", processing_id="also")
        assert made.attrs.patch_id == "kept"
        assert made.new(data=made.data).attrs.patch_id == "kept"

    def test_combining_patches_folds_which_data(self, patch):
        """Two sources make a third answer, not either of the two."""
        other = patch.update_attrs(patch_id="other")
        merged = dc.utils.attrs.combine_patch_attrs([patch.attrs, other.attrs])
        assert merged.patch_id == fold_patch_ids([patch.attrs.patch_id, "other"])
        # Spelled out, because `not in {...}` is also true of the empty
        # string, which is what dropping the fold entirely would leave.
        assert merged.patch_id
        assert merged.patch_id not in {patch.attrs.patch_id, "other"}

    def test_combining_patches_keeps_a_common_route(self, patch):
        """Windows of one file have one history, not one each."""
        first = patch.normalize("time")
        merged = dc.utils.attrs.combine_patch_attrs([first.attrs, first.attrs])
        assert merged.processing_id == first.attrs.processing_id

    def test_the_ids_survive_a_pickle(self, patch):
        """A patch handed to another process is the same data."""
        out = patch.normalize("time")
        assert pickle.loads(pickle.dumps(out)).attrs.patch_id == out.attrs.patch_id
        assert pickle.loads(pickle.dumps(out)).attrs.processing_id == (
            out.attrs.processing_id
        )

    def test_they_can_be_turned_off(self, patch):
        """A process which does not want them does not pay for them."""
        with dc.config_context(patch_provenance="disabled"):
            made = dc.Patch(data=patch.data, coords=patch.coords, dims=patch.dims)
            assert made.attrs.patch_id == ""
            assert made.normalize("time").attrs.processing_id == ""

    def test_a_summary_carries_them(self, patch):
        """
        The index stores them, so a summary which left them behind would
        make scanning a patch and reading it disagree about which data it
        is -- and would leave the spool nothing to find a patch by.
        """
        summary = patch.summary
        assert summary.attrs.patch_id == patch.attrs.patch_id
        assert summary.attrs.processing_id == patch.attrs.processing_id


class TestTheAwkwardCases:
    """The branches the ordinary path never reaches."""

    def test_no_members_folds_to_nothing(self):
        """A fold given nothing has nothing to say."""
        assert fold_ids([]) == {}

    def test_disabled_folds_to_nothing(self):
        """A process which is not keeping ids does not invent them."""
        patch = dc.get_example_patch()
        with dc.config_context(patch_provenance="disabled"):
            assert fold_ids([patch.attrs, patch.attrs]) == {}

    def test_a_function_defined_inside_a_call_is_still_named(self):
        """
        It cannot be named in a document, but it still did something.

        A `processing_id` which did not move would say it had not.
        """
        patch = dc.get_example_patch()

        @dc.patch_function()
        def only_here(patch):
            """Exist only for the length of this test."""
            return patch.new(data=patch.data + 1)

        out = only_here(patch)
        assert out.attrs.processing_id != patch.attrs.processing_id

    def test_a_callable_which_cannot_be_hashed(self):
        """Its signature is asked for the slow way rather than cached."""

        class Unhashable:
            """A callable which refuses to be a dict key."""

            __hash__ = None

            def __call__(self, patch, factor=1):
                """Do nothing."""
                return patch

        assert _signature(Unhashable()) is not None


class TestOperationsWhichAreNotPatchFunctions:
    """Concatenating, stacking and ufuncs still have to say what they did."""

    @pytest.fixture(scope="class")
    def patch(self):
        """A patch to operate on."""
        return dc.get_example_patch("random_das")

    @pytest.fixture(scope="class")
    def spool(self):
        """A spool whose patches are different data."""
        return dc.get_example_spool()

    def test_arithmetic_says_which_operation(self, patch):
        """
        `patch + 1` and `patch - (-1)` give the same data and are not the
        same operation; without the `Ufunc` task they shared an id.
        """
        made = {
            (patch * 2).attrs.processing_id,
            (patch * 3).attrs.processing_id,
            (patch + 5).attrs.processing_id,
            (patch - 7).attrs.processing_id,
            (patch + 1).attrs.processing_id,
            (patch - (-1)).attrs.processing_id,
        }
        assert len(made) == 6

    def test_arithmetic_leaves_which_data_alone(self, patch):
        """Scaling data does not make it other data."""
        assert (patch * 2).attrs.patch_id == patch.attrs.patch_id

    def test_two_patches_fold(self, patch):
        """Adding two patches is data from two sources."""
        other = patch.update_attrs(patch_id="other")
        assert (patch + other).attrs.patch_id == fold_patch_ids(
            [patch.attrs.patch_id, "other"]
        )

    def test_a_unary_ufunc_says_which_one(self, patch):
        """`np.abs` is not `np.sqrt`."""
        assert np.abs(patch).attrs.processing_id != patch.attrs.processing_id
        assert np.abs(patch).attrs.processing_id != (
            np.sqrt(np.abs(patch)).attrs.processing_id
        )

    def test_an_array_function_says_which_one(self, patch):
        """And what it was given."""
        first = np.mean(patch, axis=0).attrs.processing_id
        assert first != patch.attrs.processing_id
        assert first != np.mean(patch, axis=1).attrs.processing_id

    def test_a_patch_argument_is_an_input_not_a_parameter(self, patch):
        """
        Which patch was handed in is said by the ids, not the fingerprint.

        Encoding it would hash a whole patch on every call, and warn that
        a patch has no encoding of its own.
        """
        one = patch.new(data=patch.data > 0.5)
        two = patch.new(data=patch.data < 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            first = patch.where(one, 0.0)
        assert first.attrs.processing_id == patch.where(two, 0.0).attrs.processing_id

    def test_concatenating_folds_which_data(self, spool):
        """Taking the first patch's id would claim it was the only source."""
        members = {x.attrs.patch_id for x in spool}
        merged = spool.chunk(time=None)[0]
        assert merged.attrs.patch_id not in members
        assert merged.attrs.patch_id

    def test_stacking_folds_which_data(self, spool):
        """As does adding them together."""
        members = {x.attrs.patch_id for x in spool}
        stacked = spool.stack(dim_vary="time")
        assert stacked.attrs.patch_id not in members
        assert stacked.attrs.processing_id != NOTHING_DONE

    def test_the_raw_function_bypass_is_not_recorded(self, patch):
        """
        A body which calls `.raw_function` skips the wrapper, so nothing
        is stamped -- which is what those bypasses are for, and worth
        pinning so a refactor which removes one is a visible change.
        """
        wrapped = dc.proc.detrend(patch, "time")
        raw = dc.proc.detrend.raw_function(patch, "time")
        assert wrapped.attrs.processing_id != raw.attrs.processing_id
        assert raw.attrs.processing_id == patch.attrs.processing_id


class TestCombinationEdges:
    """The branches a combination reaches only in odd cases."""

    def test_disabled_leaves_a_combination_alone(self):
        """A process not keeping ids does not stamp one on a merge."""
        patch = dc.get_example_patch()
        with dc.config_context(patch_provenance="disabled"):
            out = stamp_combination(patch.attrs, [patch.attrs], "abc")
        assert out is patch.attrs

    def test_no_members_leaves_a_combination_alone(self):
        """Nothing went in, so there is nothing to fold."""
        patch = dc.get_example_patch()
        assert stamp_combination(patch.attrs, [], "abc") is patch.attrs

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([0] * 64, id="a long sequence"),
            pytest.param({str(x): x for x in range(64)}, id="a big mapping"),
        ],
    )
    def test_a_big_argument_is_not_worth_keying(self, value):
        """
        Working the key out would cost more than the digest it saves.

        `TypeError` is how `_as_key` says "do not cache this".
        """
        with pytest.raises(TypeError):
            _as_key(value)


class TestWhatTheReviewsFound:
    """Cases the first round of this PR got wrong."""

    @pytest.fixture(scope="class")
    def patch(self):
        """A patch to operate on."""
        return dc.get_example_patch("random_das")

    def test_every_patch_argument_counts(self, patch):
        """
        `where(cond, other)` uses all of them, so all of them fold.

        Copying the primary patch's id would say a result built from three
        sources was only the first.
        """
        one = patch.new(data=patch.data > 0.5).update_attrs(patch_id="a")
        two = patch.new(data=patch.data < 0.5).update_attrs(patch_id="b")
        assert patch.where(one, 0.0).attrs.patch_id != (
            patch.where(two, 0.0).attrs.patch_id
        )

    def test_provenance_never_fails_a_call(self, patch):
        """
        It is metadata about the work, not the work.

        A self-referential argument cannot be encoded; that is a reason to
        say nothing about the call, not to fail one which otherwise worked.
        """
        loop: dict = {}
        loop["self"] = loop

        @dc.patch_function()
        def takes_anything(patch, thing=None):
            """Accept whatever it is given."""
            return patch.new(data=patch.data)

        out = takes_anything(patch, thing=loop)
        assert out.attrs.processing_id == patch.attrs.processing_id

    def test_attrs_from_before_these_fields(self, patch):
        """
        An old pickle restores a `PatchAttrs` which has neither.

        It must still be usable: unpickling bypasses the constructor, so
        nothing fills the defaults in.
        """
        bare = dc.PatchAttrs()
        held = dict(bare.model_dump())
        held.pop("patch_id"), held.pop("processing_id")
        legacy = dc.PatchAttrs.model_construct(**held)
        made = dc.Patch(data=patch.data, coords=patch.coords, dims=patch.dims)
        assert patch_id_of(legacy) == NOTHING_DONE
        assert processing_id_of(legacy) == NOTHING_DONE
        # The fold reads them without raising, and says the result is data
        # from two places even though one of them could not say which.
        folded = fold_ids([legacy, made.attrs])["patch_id"]
        assert folded and folded != made.attrs.patch_id

    def test_equality_ignores_the_ids_either_way(self, patch):
        """Both spellings of `equals`, not just the default."""
        other = patch.update_attrs(patch_id="x", processing_id="y")
        assert patch.equals(other)
        assert patch.equals(other, only_required_attrs=False)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param(0.0, -0.0, id="signed zero"),
            pytest.param(1, True, id="int and bool"),
            pytest.param(1, 1.0, id="int and float"),
        ],
    )
    def test_the_cache_cannot_confuse_two_arguments(self, first, second):
        """
        Python calls these equal; the serializer does not.

        Caching on one would give a call two answers depending on which
        ran first, which is the one thing an id must never do.
        """
        assert _as_key(first) != _as_key(second)

    def test_a_quantity_is_never_cached(self):
        """`1 * m == 100 * cm` and the two hash alike; they encode apart."""
        from dascore.units import m  # noqa: PLC0415

        with pytest.raises(TypeError):
            _as_key(1 * m)

    def test_two_closures_are_two_operations(self, patch):
        """
        A factory gives every function it makes one module and qualname.

        Naming them by where they were written alone would make
        `make(2)` and `make(3)` one operation.
        """

        def make(factor):
            """Return a patch function which scales by a fixed amount."""

            @dc.patch_function()
            def scale(patch):
                """Scale by whatever this closure captured."""
                return patch.new(data=patch.data * factor)

            return scale

        first, second = make(2), make(3)
        assert first(patch).attrs.processing_id != second(patch).attrs.processing_id

    def test_a_dtype_argument_is_spelled_out(self, patch):
        """
        The serializer has no encoding for a `np.dtype`.

        Hashed by its class it would give every dtype one fingerprint and
        warn on every call, so it is recorded as its string instead.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            single = np.mean(patch, axis=0, dtype=np.dtype("float32"))
            double = np.mean(patch, axis=0, dtype=np.dtype("float64"))
        assert single.attrs.processing_id != double.attrs.processing_id

    def test_the_cache_holds_the_function_it_named(self):
        """
        Cache keys retain functions to prevent fingerprint collisions from reused IDs.

        Assert ownership directly: waiting for address reuse would not reliably expose
        the regression.
        """
        patch = dc.get_example_patch()

        @dc.patch_function()
        def unnameable(patch):
            """Be defined inside a call, so it takes no tag."""
            return patch.new(data=patch.data)

        unnameable(patch)
        held = [k[0] for k in registry_module._FINGERPRINTS]
        assert any(x is unnameable for x in held)

    def test_a_patch_argument_is_not_the_string_that_stands_for_it(self):
        """
        A caller may pass the marker's own spelling as an ordinary value.

        If the marker were that string the two calls would be one
        operation, though their operands are entirely different.
        """
        patch = dc.get_example_patch()
        by_patch = fingerprint_call(dc.proc.where, (patch,), {})
        assert by_patch != fingerprint_call(dc.proc.where, ("$patch",), {})
        assert by_patch != fingerprint_call(dc.proc.where, ({"$patch": True},), {})
        assert encode(PATCH_ARGUMENT) == {"$patch": True}
        assert "patch argument" in repr(PATCH_ARGUMENT)
