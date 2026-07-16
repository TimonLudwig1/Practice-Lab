"""Tests for binary power, maximum subarray, and inversion counting."""

from __future__ import annotations

import itertools
import math
import random
import unittest

from divide_conquer import (
    SubarrayResult,
    binary_power,
    count_inversions,
    maximum_subarray,
)


class BinaryPowerTests(unittest.TestCase):
    def test_zero_exponent(self) -> None:
        self.assertEqual(binary_power(17, 0), 1)

    def test_positive_even_and_odd_exponents(self) -> None:
        self.assertEqual(binary_power(2, 10), 1024)
        self.assertEqual(binary_power(3, 7), 2187)

    def test_negative_exponent(self) -> None:
        self.assertAlmostEqual(binary_power(2, -5), 1 / 32)

    def test_negative_and_complex_bases(self) -> None:
        self.assertEqual(binary_power(-2, 5), -32)
        self.assertEqual(binary_power(1 + 2j, 3), (1 + 2j) ** 3)

    def test_matches_builtin_for_range(self) -> None:
        for base, exponent in itertools.product(range(-3, 4), range(0, 12)):
            with self.subTest(base=base, exponent=exponent):
                self.assertEqual(binary_power(base, exponent), base**exponent)

    def test_trace_depth_is_logarithmic(self) -> None:
        trace: list[str] = []
        binary_power(2, 1_000, trace=trace)
        calls = [line for line in trace if line.lstrip().startswith("power(")]
        self.assertEqual(len(calls), math.floor(math.log2(1_000)) + 2)
        self.assertIn("power(e=1000)", trace[0])

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            binary_power("2", 3)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            binary_power(2, 3.0)  # type: ignore[arg-type]
        with self.assertRaises(ZeroDivisionError):
            binary_power(0, -1)


class MaximumSubarrayTests(unittest.TestCase):
    def test_classic_example(self) -> None:
        result = maximum_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        self.assertEqual(result, SubarrayResult(3, 7, 6))

    def test_all_negative_values(self) -> None:
        self.assertEqual(
            maximum_subarray([-8, -3, -6, -2, -5, -4]),
            SubarrayResult(3, 4, -2),
        )

    def test_all_positive_values(self) -> None:
        self.assertEqual(maximum_subarray([1, 2, 3]), SubarrayResult(0, 3, 6))

    def test_single_value(self) -> None:
        self.assertEqual(maximum_subarray([7]), SubarrayResult(0, 1, 7))

    def test_tie_uses_earliest_then_shortest_range(self) -> None:
        self.assertEqual(maximum_subarray([1, -1, 1]), SubarrayResult(0, 1, 1))
        self.assertEqual(maximum_subarray([0, 0]), SubarrayResult(0, 1, 0))

    def test_supports_floating_point_values(self) -> None:
        result = maximum_subarray([-1.5, 2.25, -0.25, 1.0])
        self.assertEqual((result.start, result.end), (1, 4))
        self.assertAlmostEqual(result.total, 3.0)

    def test_does_not_mutate_input(self) -> None:
        values = [3, -2, 5]
        maximum_subarray(values)
        self.assertEqual(values, [3, -2, 5])

    def test_trace_contains_complete_root_decision(self) -> None:
        trace: list[str] = []
        maximum_subarray([2, -1, 3], trace=trace)
        self.assertEqual(trace[0], "segment[0:3)")
        self.assertTrue(trace[-1].startswith("choose [0:3) sum=4"))

    def test_rejects_empty_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            maximum_subarray([])
        with self.assertRaises(TypeError):
            maximum_subarray([1, True])
        with self.assertRaises(ValueError):
            maximum_subarray([1.0, math.inf])


class InversionCountTests(unittest.TestCase):
    @staticmethod
    def brute_force(values: list[int]) -> int:
        return sum(
            values[left] > values[right]
            for left in range(len(values))
            for right in range(left + 1, len(values))
        )

    def test_known_example(self) -> None:
        sorted_values, count = count_inversions([2, 4, 1, 3, 5])
        self.assertEqual(sorted_values, [1, 2, 3, 4, 5])
        self.assertEqual(count, 3)

    def test_sorted_input_has_no_inversions(self) -> None:
        self.assertEqual(count_inversions([1, 2, 3, 4]), ([1, 2, 3, 4], 0))

    def test_reverse_input_has_maximum_inversions(self) -> None:
        values = [5, 4, 3, 2, 1]
        self.assertEqual(count_inversions(values), ([1, 2, 3, 4, 5], 10))

    def test_duplicates_are_not_inversions(self) -> None:
        self.assertEqual(count_inversions([2, 2, 1]), ([1, 2, 2], 2))

    def test_empty_and_singleton_inputs(self) -> None:
        self.assertEqual(count_inversions([]), ([], 0))
        self.assertEqual(count_inversions([7]), ([7], 0))

    def test_supports_other_comparable_values(self) -> None:
        self.assertEqual(count_inversions(["c", "a", "b"]), (["a", "b", "c"], 2))

    def test_does_not_mutate_input(self) -> None:
        values = [3, 1, 2]
        count_inversions(values)
        self.assertEqual(values, [3, 1, 2])

    def test_matches_brute_force_on_seeded_inputs(self) -> None:
        random_source = random.Random(20260716)
        for size in range(12):
            values = [random_source.randrange(6) for _ in range(size)]
            with self.subTest(values=values):
                sorted_values, count = count_inversions(values)
                self.assertEqual(sorted_values, sorted(values))
                self.assertEqual(count, self.brute_force(values))

    def test_trace_reports_root_total(self) -> None:
        trace: list[str] = []
        count_inversions([3, 1, 2], trace=trace)
        self.assertEqual(trace[0], "sort [3, 1, 2]")
        self.assertEqual(trace[-1], "merge -> [1, 2, 3], inversions=2")


if __name__ == "__main__":
    unittest.main()
