"""
A module for handling deprecations.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from typing_extensions import deprecated as dep

F = TypeVar("F", bound=Callable[..., Any])


def deprecate(
    info: str = "",
    *,
    since: str | None = None,
    removed_in: str | None = None,
) -> Callable[[F], F]:
    """
    Mark a function as deprecated.

    - Raises a runtime warning when called.
    - Annotates the function so editors/type checkers show deprecation.
    - Augments the docstring.

    Parameters
    ----------
    info
        Short message shown in warnings and editor hints. It is useful to specify
        what should be used in place of the deprecated function.
    since
        Version/date when deprecation started (for the message only).
    removed_in
        Version/date when the function will be removed (for the message only).

    Examples
    --------
    >>> from dascore.utils.deprecate import deprecate
    >>>
    >>> # Deprecate function so it issues a warning when used.
    >>> @deprecate(info="This function is deprecated.")
    ... def foo():
    ...     pass
    """

    def _build_msg(func):
        """Build the message to emmit."""
        # Build a clear message for both runtime and typing hint
        qual = f"{func.__module__}.{getattr(func, '__qualname__', func.__name__)}"
        since_str = f" since {since}" if since else ""
        removed_str = f" and will be removed in {removed_in}" if removed_in else ""
        info_str = f" {info}" if info else ""
        msg = f"{qual} is deprecated{since_str}{removed_str}.{info_str}"
        return msg

    def _decorate(func: F) -> F:
        # Wrap to emit a runtime warning on call
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            return func(*args, **kwargs)

        # Add a simple marker attribute some tools may inspect
        setattr(wrapper, "__deprecated__", True)

        # Prepend/augment the docstring so it shows in help() / tooltips
        dep_header = f"\n\n.. deprecated:: {since or ''}\n   {info}"
        if removed_in:
            dep_header += f" (removal in {removed_in})"
        wrapper.__doc__ = (func.__doc__ or "").rstrip() + dep_header

        # Apply typing-level deprecation *to the wrapper* so editors see it
        msg = _build_msg(func)
        return dep(msg)(wrapper)  # type: ignore[return-value]

    return _decorate
