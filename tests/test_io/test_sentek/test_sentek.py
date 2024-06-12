"""
Tests specific to the Sentek format.
"""

import numpy as np

from dascore.compat import random_state
from dascore.io.sentek import SentekV5


class TestSentekV5:
    """Tests for Sentek format that aren;t covered by common tests."""

    def test_das_extension_not_sentek(self, tmp_path_factory):
        """Ensure a non-sentek file with a das extension isn't id as sentek."""
        path = tmp_path_factory.mktemp("sentek_test") / "not_sentek.das"
        ar = random_state.random(10)
        with path.open("wb") as fi:
            np.save(fi, ar)
        sentek = SentekV5()
        assert not sentek.get_format(path)
