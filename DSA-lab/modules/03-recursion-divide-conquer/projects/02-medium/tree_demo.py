"""Print recursion trees and results for the divide-and-conquer toolkit."""

from divide_conquer import binary_power, count_inversions, maximum_subarray


def _show(title: str, trace: list[str], result: object) -> None:
    """Print one labeled trace and its final result."""
    print(f"\n{title}")
    print("=" * len(title))
    print("\n".join(trace))
    print(f"Ergebnis: {result}")


def main() -> None:
    """Run one representative input for every algorithm."""
    power_trace: list[str] = []
    power_result = binary_power(3, 13, trace=power_trace)
    _show("Binäre Potenzierung: 3^13", power_trace, power_result)

    subarray_values = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    subarray_trace: list[str] = []
    subarray_result = maximum_subarray(subarray_values, trace=subarray_trace)
    _show("Maximum Subarray", subarray_trace, subarray_result)

    inversion_values = [2, 4, 1, 3, 5]
    inversion_trace: list[str] = []
    inversion_result = count_inversions(inversion_values, trace=inversion_trace)
    _show("Inversionen zählen", inversion_trace, inversion_result)


if __name__ == "__main__":
    main()
