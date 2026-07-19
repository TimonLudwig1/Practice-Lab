"""Print compact traces for exact and boundary searches."""

from __future__ import annotations

from binary_search import SearchStep, binary_search, lower_bound, upper_bound


def print_trace(title: str, steps: list[SearchStep[int]], result: int) -> None:
    """Render interval transitions as a readable table."""

    print(title)
    print("left right middle value decision       next interval")
    print("-" * 59)
    for step in steps:
        closing = "]" if step.interval == "closed" else ")"
        next_interval = f"[{step.next_left}, {step.next_right}{closing}"
        print(
            f"{step.left:>4} {step.right:>5} {step.middle:>6} "
            f"{step.value:>5} {step.decision:<14} {next_interval}"
        )
    print(f"result: {result}\n")


def main() -> None:
    """Run one exact search and both duplicate-boundary searches."""

    exact_values = [3, 7, 11, 15, 18, 23, 29, 31, 42]
    exact_trace: list[SearchStep[int]] = []
    exact_result = binary_search(exact_values, 23, trace=exact_trace)
    print_trace("Exact search for 23", exact_trace, exact_result)

    duplicate_values = [2, 4, 4, 4, 7, 9]
    lower_trace: list[SearchStep[int]] = []
    lower_result = lower_bound(duplicate_values, 4, trace=lower_trace)
    print_trace("First position >= 4", lower_trace, lower_result)

    upper_trace: list[SearchStep[int]] = []
    upper_result = upper_bound(duplicate_values, 4, trace=upper_trace)
    print_trace("First position > 4", upper_trace, upper_result)


if __name__ == "__main__":
    main()
