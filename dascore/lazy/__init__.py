"""
Recipes for arrays which are read a member at a time.

A recipe holds no data: each of its members says which window of which
stored source fills which box of the output. Import names from the module
which defines them, such as `dascore.lazy.recipe`; this package imports
none of its modules.
"""

from __future__ import annotations
