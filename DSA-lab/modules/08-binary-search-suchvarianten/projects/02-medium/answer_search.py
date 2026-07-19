"""Reusable integer answer search with observable interval transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


Decision = Literal["discard_left", "keep_left"]


@dataclass(frozen=True)
class AnswerStep:
    """One transition while searching for the first feasible integer."""

    problem: str
    low: int
    high: int
    middle: int
    feasible: bool
    decision: Decision
    next_low: int
    next_high: int

    @property
    def size(self) -> int:
        """Return the inclusive candidate count before the transition."""

        return self.high - self.low + 1

    @property
    def next_size(self) -> int:
        """Return the inclusive candidate count after the transition."""

        return self.next_high - self.next_low + 1


def first_true(
    low: int,
    high: int,
    predicate: Callable[[int], bool],
    *,
    problem: str = "first_true",
    trace: list[AnswerStep] | None = None,
) -> int:
    """Return the first integer satisfying a false-to-true predicate.

    Preconditions:
    - ``low <= high``;
    - ``predicate(high)`` is true;
    - once the predicate is true, it remains true for all larger candidates.

    Invariant: the inclusive interval ``[low, high]`` contains the first true
    candidate, and ``high`` is a true candidate.
    """

    if low > high:
        raise ValueError("low must not exceed high")
    if not predicate(high):
        raise ValueError("high must be a feasible candidate")

    while low < high:
        middle = low + (high - low) // 2
        feasible = predicate(middle)
        if feasible:
            next_low, next_high = low, middle
            decision: Decision = "keep_left"
        else:
            next_low, next_high = middle + 1, high
            decision = "discard_left"
        if trace is not None:
            trace.append(
                AnswerStep(
                    problem=problem,
                    low=low,
                    high=high,
                    middle=middle,
                    feasible=feasible,
                    decision=decision,
                    next_low=next_low,
                    next_high=next_high,
                )
            )
        low, high = next_low, next_high
    return low
