"""
Recipes for arrays which are read a member at a time.

A recipe holds no data: each of its members says which window of which
stored source fills which box of the output.
"""

from __future__ import annotations

from dascore.lazy.recipe import LazyArray, LazyTable, concat, stack
