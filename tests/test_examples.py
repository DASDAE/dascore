"""
tests for example data
"""

import dascore as dc


class TestWavelet:
    """
    test for wavelet moveout example
    """

    def testcase(self):
        """
        test that trace is loaded
        """
        patch = dc.get_example_patch("wavelet_mo")
        assert isinstance(patch, dc.Patch)
