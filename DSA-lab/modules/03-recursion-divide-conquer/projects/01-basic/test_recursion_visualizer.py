"""Tests for recursive results and the terminal call-tree decorator."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from recursion_visualizer import (
    factorial,
    fibonacci,
    power,
    recursive_sum,
    trace_calls,
)


def capture(function: object, *arguments: object, **keywords: object) -> tuple[object, str]:
    """Call a traced function and return both result and captured output."""
    stream = io.StringIO()
    with redirect_stdout(stream):
        result = function(*arguments, **keywords)  # type: ignore[operator]
    return result, stream.getvalue()


class TraceDecoratorTests(unittest.TestCase):
    def test_preserves_function_metadata(self) -> None:
        self.assertEqual(factorial.__name__, "factorial")
        self.assertIn("Return n!", factorial.__doc__ or "")

    def test_prints_indented_entry_and_return_lines(self) -> None:
        result, output = capture(factorial, 2)
        self.assertEqual(result, 2)
        self.assertEqual(
            output.splitlines(),
            [
                "→ factorial(2)",
                "  → factorial(1)",
                "    → factorial(0)",
                "    ← factorial(0) = 1",
                "  ← factorial(1) = 1",
                "← factorial(2) = 2",
            ],
        )

    def test_formats_keyword_arguments(self) -> None:
        result, output = capture(recursive_sum, [1, 2], index=1)
        self.assertEqual(result, 2)
        self.assertTrue(output.startswith("→ recursive_sum([1, 2], index=1)"))

    def test_prints_exception_and_resets_depth(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            with self.assertRaises(ValueError):
                factorial(-1)
            factorial(0)

        lines = stream.getvalue().splitlines()
        self.assertEqual(lines[0], "→ factorial(-1)")
        self.assertIn("! factorial(-1) raised ValueError", lines[1])
        self.assertEqual(lines[2], "→ factorial(0)")

    def test_nested_different_decorated_functions_share_depth(self) -> None:
        @trace_calls
        def inner() -> str:
            return "done"

        @trace_calls
        def outer() -> str:
            return inner()

        result, output = capture(outer)
        self.assertEqual(result, "done")
        self.assertIn("  → inner()", output)
        self.assertIn("  ← inner() = 'done'", output)


class FactorialTests(unittest.TestCase):
    def test_base_case(self) -> None:
        result, _ = capture(factorial, 0)
        self.assertEqual(result, 1)

    def test_recursive_case(self) -> None:
        result, _ = capture(factorial, 6)
        self.assertEqual(result, 720)

    def test_rejects_negative_input(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            factorial(-1)

    def test_rejects_non_integer_and_boolean(self) -> None:
        for value in (2.5, True):
            with self.subTest(value=value), redirect_stdout(io.StringIO()):
                with self.assertRaises(TypeError):
                    factorial(value)  # type: ignore[arg-type]


class FibonacciTests(unittest.TestCase):
    def test_two_base_cases(self) -> None:
        self.assertEqual(capture(fibonacci, 0)[0], 0)
        self.assertEqual(capture(fibonacci, 1)[0], 1)

    def test_recursive_values(self) -> None:
        expected = [0, 1, 1, 2, 3, 5, 8, 13]
        actual = [capture(fibonacci, n)[0] for n in range(len(expected))]
        self.assertEqual(actual, expected)

    def test_trace_contains_both_branches(self) -> None:
        _, output = capture(fibonacci, 3)
        self.assertEqual(output.count("→ fibonacci(1)"), 2)
        self.assertIn("  → fibonacci(2)", output)

    def test_rejects_invalid_input(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            fibonacci(-1)


class RecursiveSumTests(unittest.TestCase):
    def test_sums_integer_values(self) -> None:
        result, _ = capture(recursive_sum, [2, 4, 6])
        self.assertEqual(result, 12)

    def test_sums_floating_point_values(self) -> None:
        result, _ = capture(recursive_sum, [0.5, 1.25, -0.25])
        self.assertAlmostEqual(result, 1.5)

    def test_empty_sequence_is_base_case(self) -> None:
        result, output = capture(recursive_sum, [])
        self.assertEqual(result, 0)
        self.assertEqual(len(output.splitlines()), 2)

    def test_start_index_selects_suffix(self) -> None:
        result, _ = capture(recursive_sum, [10, 20, 30], 1)
        self.assertEqual(result, 50)

    def test_rejects_invalid_index(self) -> None:
        for index in (-1, 3):
            with self.subTest(index=index), redirect_stdout(io.StringIO()):
                with self.assertRaises(IndexError):
                    recursive_sum([1, 2], index)


class PowerTests(unittest.TestCase):
    def test_zero_exponent(self) -> None:
        self.assertEqual(capture(power, 9, 0)[0], 1)

    def test_positive_exponent(self) -> None:
        self.assertEqual(capture(power, 3, 4)[0], 81)

    def test_negative_exponent(self) -> None:
        self.assertAlmostEqual(capture(power, 2, -3)[0], 0.125)

    def test_negative_base(self) -> None:
        self.assertEqual(capture(power, -2, 3)[0], -8)

    def test_zero_to_negative_exponent_raises(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(ZeroDivisionError):
            power(0, -1)

    def test_rejects_non_integer_exponent(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(TypeError):
            power(2, 1.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
