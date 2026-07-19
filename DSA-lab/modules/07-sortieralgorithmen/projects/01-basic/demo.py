"""Compare results and metrics of all five workshop algorithms."""

from sorting import ALGORITHMS, SortMetrics, TraceStep, insertion_sort


def main() -> None:
    values = [5, 2, 4, 1, 3]
    print(f"Input: {values}\n")
    print("algorithm  result              comparisons swaps writes depth")
    print("-" * 67)
    for name, algorithm in ALGORITHMS.items():
        metrics = SortMetrics()
        result = algorithm(values, metrics=metrics)
        print(
            f"{name:<11}{str(result):<20}{metrics.comparisons:>11}"
            f"{metrics.swaps:>6}{metrics.writes:>7}"
            f"{metrics.max_recursion_depth:>6}"
        )

    trace: list[TraceStep] = []
    insertion_sort([5, 2, 4, 1], trace=trace)
    print("\nInsertion trace")
    for step in trace:
        print(f"{step.action:<8} indices={step.indices!s:<8} state={list(step.state)}")


if __name__ == "__main__":
    main()
