"""
Tests for filters.
"""
import numpy as np
import pandas as pd
import pytest

import dascore
from dascore.exceptions import FilterValueError


class TestFilterChecks:
    """Test that bad filter checks raise appropriate errors."""

    def test_no_kwargs_raises(self, random_patch):
        """Ensure ValueError is raised with no dim parameters."""
        with pytest.raises(FilterValueError):
            _ = random_patch.pass_filter()

    def test_wrong_kwarg_length_raises(self, random_patch):
        """only len two sequence is allowed."""
        with pytest.raises(FilterValueError, match="length two sequence"):
            _ = random_patch.pass_filter(time=[1])
        with pytest.raises(FilterValueError, match="length two sequence"):
            _ = random_patch.pass_filter(time=[1, 3, 3])

    def test_all_null_kwarg_raises(self, random_patch):
        """There must be one Non-null kwarg."""
        with pytest.raises(FilterValueError, match="at least one filter"):
            _ = random_patch.pass_filter(time=[None, np.NAN])

    def test_unordered_params(self, random_patch):
        """Ensure a low parameter greater than a high parameter raises."""
        with pytest.raises(FilterValueError):
            _ = random_patch.pass_filter(time=[10, 1])

    def test_bad_low_param(self, random_patch):
        """Ensure a low parameter above niquest raises."""
        with pytest.raises(FilterValueError, match="possible filter bounds"):
            _ = random_patch.pass_filter(time=[1000, None])

    def test_bad_high_param(self, random_patch):
        """Ensure a high parameter above niquest raises."""
        with pytest.raises(FilterValueError, match="possible filter bounds"):
            _ = random_patch.pass_filter(distance=[None, 10])

    def test_filt_param_less_than_0(self, random_patch):
        """Ensure a high parameter above niquest raises."""
        with pytest.raises(FilterValueError, match="possible filter bounds"):
            _ = random_patch.pass_filter(distance=[None, -10])


class TestFilterBasics:
    """Simple tests to make sure filter logic runs."""

    def test_time_bandpass_runs(self, random_patch):
        """Ensure filtering along time_axis works"""
        out = random_patch.pass_filter(time=(10, 100))
        assert isinstance(out, dascore.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_time_lowpass_runs(self, random_patch):
        """Ensure stopfilter on time works."""
        out = random_patch.pass_filter(time=(None, 100.22))
        assert isinstance(out, dascore.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_time_highpass_runs(self, random_patch):
        """Ensure a highpass on time runs"""
        out = random_patch.pass_filter(time=(10.2, None))
        assert isinstance(out, dascore.Patch)
        assert not np.any(pd.isnull(out.data))

    def test_apply_distance_filter(self, random_patch):
        """Apply bandpass along distance dimension."""
        out = random_patch.pass_filter(distance=(0.1, 0.2))
        assert isinstance(out, dascore.Patch)
        assert not np.any(pd.isnull(out.data))
