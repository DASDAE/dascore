"""
Simple tests for FrozenDict.
"""
from collections.abc import Mapping

import pytest

from dascore.utils.mapping import FrozenDict


@pytest.fixture(scope="session")
def frozen_dict():
    """Return an example frozen dict."""
    return FrozenDict({"bob": 1, "bill": 2})


class TestFrozenDict:
    """Test chases for frozen dict"""

    def test_is_mapping(self, frozen_dict):
        """Frozen dict should follow mapping ABC."""
        assert isinstance(frozen_dict, Mapping)

    def test_init_on_dict(self):
        """Ensure a dict can be used to init frozendict."""
        out = FrozenDict({"bob": 1})
        assert isinstance(out, FrozenDict)
        assert "bob" in out

    def test_len(self, frozen_dict):
        """Ensure len works."""
        assert len(frozen_dict) == 2

    def test_contains(self, frozen_dict):
        """Ensure contains works"""
        assert "bob" in frozen_dict
        assert "bill" in frozen_dict

    def test_hash(self, frozen_dict):
        """A frozen dict should be a valid key in a dict/set."""
        out = {frozen_dict: 1}
        assert frozen_dict in out

    def test_init_on_keys(self):
        """Ensure dict can be inited with keys as well."""
        out = FrozenDict(bob=1, bill=2)
        assert isinstance(out, FrozenDict)

    def test_cant_add_keys(self, frozen_dict):
        """Ensure keys can't be added to the dict."""
        with pytest.raises(TypeError, match="not support item assignment"):
            frozen_dict["bob"] = 1

        with pytest.raises(TypeError, match="not support item assignment"):
            frozen_dict["new"] = 1

    def test_cant_mutate_original(self, frozen_dict):
        """
        Ensure the original dict can be changed and this does not affect frozen's
        contents.
        """
        original = {"one": 1, "two": 2}
        froz = FrozenDict(original)
        # test adding new key
        assert "three" not in froz
        original["three"] = 3
        assert "three" not in froz
        # test modifying existing key
        original["one"] = 11
        assert froz["one"] == 1
