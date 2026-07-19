"""Stateful sliding-window exercises with independent brute-force references."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


Action = Literal["expand", "shrink", "settled", "fixed_window"]


@dataclass(frozen=True)
class WindowStep:
    """One observable state of a variable or fixed window."""

    problem: str
    action: Action
    left: int
    right: int
    valid: bool
    state: tuple[tuple[str, int], ...]

    @property
    def length(self) -> int:
        """Return the current inclusive window length."""

        return max(0, self.right - self.left + 1)

    def value(self, name: str) -> int:
        """Return one named integer from the immutable state snapshot."""

        for key, value in self.state:
            if key == name:
                return value
        raise KeyError(name)


def _record(
    trace: list[WindowStep] | None,
    *,
    problem: str,
    action: Action,
    left: int,
    right: int,
    valid: bool,
    state: dict[str, int],
) -> None:
    if trace is not None:
        trace.append(
            WindowStep(
                problem=problem,
                action=action,
                left=left,
                right=right,
                valid=valid,
                state=tuple(sorted(state.items())),
            )
        )


def minimum_covering_window_brute(text: str, target: str) -> str:
    """Return the earliest shortest substring covering target multiplicities."""

    if not target:
        return ""
    required = Counter(target)
    best_start = 0
    best_length = len(text) + 1
    for left in range(len(text)):
        counts: Counter[str] = Counter()
        for right in range(left, len(text)):
            counts[text[right]] += 1
            if all(counts[character] >= amount for character, amount in required.items()):
                length = right - left + 1
                if length < best_length:
                    best_start, best_length = left, length
                break
    if best_length == len(text) + 1:
        return ""
    return text[best_start : best_start + best_length]


def minimum_covering_window(
    text: str, target: str, *, trace: list[WindowStep] | None = None
) -> str:
    """Return the earliest shortest substring covering target in linear time.

    Invariant: ``need[c]`` equals required count minus current-window count.
    ``missing`` is the sum of positive deficits, so the window covers the target
    exactly when ``missing == 0``.
    """

    if not target:
        return ""
    need = Counter(target)
    missing = len(target)
    left = 0
    best_start = 0
    best_length = len(text) + 1

    for right, entering in enumerate(text):
        if need[entering] > 0:
            missing -= 1
        need[entering] -= 1
        _record(
            trace,
            problem="minimum_covering_window",
            action="expand",
            left=left,
            right=right,
            valid=missing == 0,
            state={"missing": missing},
        )

        while missing == 0:
            length = right - left + 1
            if length < best_length:
                best_start, best_length = left, length
            leaving = text[left]
            need[leaving] += 1
            if need[leaving] > 0:
                missing += 1
            left += 1
            _record(
                trace,
                problem="minimum_covering_window",
                action="shrink",
                left=left,
                right=right,
                valid=missing == 0,
                state={"missing": missing},
            )

    if best_length == len(text) + 1:
        return ""
    return text[best_start : best_start + best_length]


def longest_ones_with_flips_brute(values: Sequence[int], flips: int) -> int:
    """Return the longest window containing at most flips zeros by enumeration."""

    _validate_binary_input(values, flips)
    best = 0
    for left in range(len(values)):
        zeros = 0
        for right in range(left, len(values)):
            zeros += values[right] == 0
            if zeros > flips:
                break
            best = max(best, right - left + 1)
    return best


def longest_ones_with_flips(
    values: Sequence[int],
    flips: int,
    *,
    trace: list[WindowStep] | None = None,
) -> int:
    """Return the longest window containing at most flips zeros in linear time.

    Invariant after shrinking: the current window contains at most ``flips``
    zeros and is the longest valid suffix ending at ``right``.
    """

    _validate_binary_input(values, flips)
    left = 0
    zeros = 0
    best = 0
    for right, value in enumerate(values):
        zeros += value == 0
        while zeros > flips:
            zeros -= values[left] == 0
            left += 1
        best = max(best, right - left + 1)
        _record(
            trace,
            problem="longest_ones_with_flips",
            action="settled",
            left=left,
            right=right,
            valid=zeros <= flips,
            state={"flips": flips, "zeros": zeros},
        )
    return best


def _validate_binary_input(values: Sequence[int], flips: int) -> None:
    if not isinstance(flips, int) or isinstance(flips, bool) or flips < 0:
        raise ValueError("flips must be a non-negative integer")
    if any(value not in (0, 1) or isinstance(value, bool) for value in values):
        raise ValueError("values must contain only integer zeros and ones")


def anagram_starts_brute(text: str, pattern: str) -> list[int]:
    """Return anagram-window starts by rebuilding every frequency map."""

    width = len(pattern)
    if width == 0 or width > len(text):
        return []
    required = Counter(pattern)
    return [
        start
        for start in range(len(text) - width + 1)
        if Counter(text[start : start + width]) == required
    ]


def anagram_starts(
    text: str, pattern: str, *, trace: list[WindowStep] | None = None
) -> list[int]:
    """Return anagram-window starts with O(1) state updates per movement.

    ``balance[c]`` equals window count minus required count. ``nonzero`` counts
    characters whose balance differs from zero. A full-width window is an
    anagram exactly when ``nonzero == 0``.
    """

    width = len(pattern)
    if width == 0 or width > len(text):
        return []
    balance = {character: -amount for character, amount in Counter(pattern).items()}
    nonzero = len(balance)

    def adjust(character: str, change: int) -> None:
        nonlocal nonzero
        old = balance.get(character, 0)
        if old != 0:
            nonzero -= 1
        new = old + change
        if new != 0:
            nonzero += 1
            balance[character] = new
        else:
            balance.pop(character, None)

    for character in text[:width]:
        adjust(character, 1)

    starts: list[int] = []
    for left in range(len(text) - width + 1):
        right = left + width - 1
        valid = nonzero == 0
        if valid:
            starts.append(left)
        _record(
            trace,
            problem="anagram_starts",
            action="fixed_window",
            left=left,
            right=right,
            valid=valid,
            state={"nonzero": nonzero},
        )
        next_right = right + 1
        if next_right < len(text):
            adjust(text[left], -1)
            adjust(text[next_right], 1)
    return starts
