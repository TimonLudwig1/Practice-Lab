"""Demonstrate three answer-space searches and their interval traces."""

from __future__ import annotations

from answer_search import AnswerStep
from problems import integer_square_root, kth_smallest, minimum_shipping_capacity


def print_trace(title: str, steps: list[AnswerStep], result: int) -> None:
    """Print one answer-space search as a compact transition table."""

    print(title)
    print("low high middle feasible decision       next")
    print("-" * 56)
    for step in steps:
        print(
            f"{step.low:>3} {step.high:>4} {step.middle:>6} "
            f"{str(step.feasible):>8} {step.decision:<14} "
            f"[{step.next_low}, {step.next_high}]"
        )
    print(f"result: {result}\n")


def main() -> None:
    """Run the standard scenario for each problem."""

    shipping_trace: list[AnswerStep] = []
    capacity = minimum_shipping_capacity(
        [3, 2, 2, 4, 1, 4], 3, trace=shipping_trace
    )
    print_trace("Minimum shipping capacity", shipping_trace, capacity)

    root_trace: list[AnswerStep] = []
    root = integer_square_root(200, trace=root_trace)
    print_trace("Integer square root of 200", root_trace, root)

    matrix_trace: list[AnswerStep] = []
    matrix = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15],
    ]
    value = kth_smallest(matrix, 8, trace=matrix_trace)
    print_trace("8th smallest matrix value", matrix_trace, value)


if __name__ == "__main__":
    main()
