"""Trace recursive calls as an indented call tree in the terminal."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar
from functools import wraps
from numbers import Number
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")

_TRACE_DEPTH: ContextVar[int] = ContextVar("trace_depth", default=0)


def _format_call(
    function_name: str, arguments: tuple[object, ...], keywords: dict[str, object]
) -> str:
    """Return a deterministic representation of a function call."""
    parts = [repr(argument) for argument in arguments]
    parts.extend(f"{name}={value!r}" for name, value in keywords.items())
    return f"{function_name}({', '.join(parts)})"


def trace_calls(function: Callable[P, R]) -> Callable[P, R]:
    """Print entry, return, and exception lines for every decorated call."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        depth = _TRACE_DEPTH.get()
        indent = "  " * depth
        call = _format_call(function.__name__, args, kwargs)
        print(f"{indent}→ {call}")
        token = _TRACE_DEPTH.set(depth + 1)

        try:
            result = function(*args, **kwargs)
        except Exception as error:
            _TRACE_DEPTH.reset(token)
            print(f"{indent}! {call} raised {type(error).__name__}: {error}")
            raise
        else:
            _TRACE_DEPTH.reset(token)
            print(f"{indent}← {call} = {result!r}")
            return result

    return wrapper


def _validate_non_negative_integer(value: int, name: str) -> None:
    """Validate a non-negative integer argument."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@trace_calls
def factorial(n: int) -> int:
    """Return n! recursively."""
    _validate_non_negative_integer(n, "n")
    if n == 0:
        return 1
    return n * factorial(n - 1)


@trace_calls
def fibonacci(n: int) -> int:
    """Return Fibonacci(n) using the direct recursive recurrence."""
    _validate_non_negative_integer(n, "n")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@trace_calls
def recursive_sum(values: Sequence[Number], index: int = 0) -> Number:
    """Return the sum of values[index:] recursively without slicing."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("index must be an integer")
    if index < 0 or index > len(values):
        raise IndexError("index outside sequence")
    if index == len(values):
        return 0
    return values[index] + recursive_sum(values, index + 1)


@trace_calls
def power(base: Number, exponent: int) -> Number:
    """Return base raised to an integer exponent using linear recursion."""
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise TypeError("exponent must be an integer")
    if exponent == 0:
        return 1
    if exponent < 0:
        return 1 / power(base, -exponent)
    return base * power(base, exponent - 1)
