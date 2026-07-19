"""Demonstrate three stateful sliding-window algorithms."""

from __future__ import annotations

from stateful_windows import (
    WindowStep,
    anagram_starts,
    longest_ones_with_flips,
    minimum_covering_window,
)


def summarize(title: str, result: object, trace: list[WindowStep]) -> None:
    """Print a compact summary and the first state transitions."""

    print(f"{title}: {result}")
    print("action        left right valid state")
    print("-" * 52)
    for step in trace[:12]:
        print(
            f"{step.action:<13} {step.left:>4} {step.right:>5} "
            f"{str(step.valid):>5} {dict(step.state)}"
        )
    if len(trace) > 12:
        print(f"... {len(trace) - 12} additional transitions")
    print()


def main() -> None:
    """Run one standard example for every state invariant."""

    cover_trace: list[WindowStep] = []
    cover = minimum_covering_window("ADOBECODEBANC", "ABC", trace=cover_trace)
    summarize("Minimum covering window", cover, cover_trace)

    flips_trace: list[WindowStep] = []
    ones = longest_ones_with_flips(
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2, trace=flips_trace
    )
    summarize("Longest ones with two flips", ones, flips_trace)

    anagram_trace: list[WindowStep] = []
    starts = anagram_starts("cbaebabacd", "abc", trace=anagram_trace)
    summarize("Anagram starts", starts, anagram_trace)


if __name__ == "__main__":
    main()
