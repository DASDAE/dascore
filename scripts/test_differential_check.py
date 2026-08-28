"""Tests for the differential check script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from differential_check import (
    MATRIX_CALLS,
    check_dascore_path,
    compare,
    digest,
    get_calls,
    get_matrix_calls,
    make_arrays,
)

import dascore as dc
from dascore.exceptions import ParameterError


@pytest.fixture(scope="module")
def patch():
    """An example patch."""
    return dc.get_example_patch()


class TestDigest:
    """Tests for fingerprinting a patch."""

    def test_stable(self, patch):
        """The same patch gives the same fingerprint."""
        assert digest(patch) == digest(patch.new())

    def test_detects_tiny_data_change(self, patch):
        """A change too small to see is still a change."""
        other = patch.new(data=np.asarray(patch.data) + 1e-12)
        assert digest(patch) != digest(other)

    def test_detects_coord_change(self, patch):
        """So is a change to a coordinate."""
        other = patch.update_coords(distance=patch.get_array("distance") + 1)
        assert digest(patch) != digest(other)

    def test_ignores_history(self, patch):
        """History records the call, not the answer."""
        assert digest(patch) == digest(patch.update_attrs(history=["hello"]))


class TestCompare:
    """Tests for comparing two sets of fingerprints."""

    def test_no_report_when_equal(self, patch):
        """Identical results have nothing to report."""
        assert compare({"a": digest(patch)}, {"a": digest(patch)}) == []

    def test_reports_changed_field(self, patch):
        """A changed result names the field which changed."""
        other = patch.new(data=np.asarray(patch.data) + 1)
        report = compare({"a": digest(patch)}, {"a": digest(other)})
        assert report and "data_hash" in report[0]

    def test_reports_missing_call(self, patch):
        """A call which only one side has is reported."""
        assert compare({"a": digest(patch)}, {}) == ["a: only in before"]


# dump() records the error instead of the fingerprint when a call raises,
# so the comparison covers what each version says about a bad argument --
# the class included, since it names the error it writes. Naming them here
# keeps a fourth from joining quietly, which would drop that call from the
# comparison.
RAISERS = {
    "norm_bad": ValueError,
    "transpose_bad_dim": ParameterError,
    "rename_missing": KeyError,
}


class TestCalls:
    """Tests for the calls which get compared."""

    def test_calls_defined(self):
        """The comparison covers the converted patch functions."""
        calls = get_calls()
        assert len(calls) > 50
        assert {"add_scalar", "agg_mean", "pad_tuple"}.issubset(set(calls))

    def test_calls_run(self):
        """Every call fingerprints, save the ones which exist to raise."""
        for name, call in get_calls().items():
            if (error := RAISERS.get(name)) is not None:
                with pytest.raises(error):
                    call()
                continue
            assert digest(call()), f"{name} returned nothing"


class TestMatrix:
    """Tests for running every call against every kind of array."""

    def test_dtypes_covered(self):
        """The arrays cover the dtypes patch data can hold."""
        dtypes = {str(x.dtype) for x in make_arrays().values()}
        assert {"float64", "float32", "int64", "bool", "complex128"} <= dtypes

    def test_special_values_covered(self):
        """And the values implementations disagree about."""
        arrays = make_arrays()
        assert np.isnan(arrays["with_nan"]).any()
        assert np.isinf(arrays["with_inf"]).any()
        assert np.isnan(arrays["all_nan"]).all()

    def test_every_call_against_every_array(self):
        """Each call is compared for each array, plus the input itself."""
        calls = get_matrix_calls()
        arrays = make_arrays()
        assert len(calls) == len(arrays) * (len(MATRIX_CALLS) + 1)

    def test_calls_run(self):
        """Every call either returns something or records why it did not."""
        for name, call in get_matrix_calls().items():
            try:
                assert digest(call()), f"{name} returned nothing"
            except Exception:
                pass


class TestDascorePath:
    """Tests for the guard against measuring the wrong dascore."""

    def test_accepts_the_worktree(self):
        """The dascore inside the worktree is the one wanted."""
        check_dascore_path(str(Path("/tmp/tree/dascore")), Path("/tmp/tree"))

    def test_rejects_another(self):
        """Any other one is a mistake, not a comparison."""
        with pytest.raises(RuntimeError, match="expected the dascore"):
            check_dascore_path("/somewhere/else/dascore", Path("/tmp/tree"))
