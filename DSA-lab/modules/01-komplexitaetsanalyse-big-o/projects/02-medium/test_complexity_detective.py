"""Tests for the complexity detective project."""

import csv
import tempfile
import unittest
from pathlib import Path

from complexity_detective import (
    BenchmarkConfig,
    Measurement,
    case_01,
    case_02,
    case_03,
    case_04,
    case_05,
    case_06,
    case_07,
    case_08,
    case_09,
    case_10,
    estimate_log_log_slope,
    run_benchmarks,
    select_cases,
    write_csv,
)


class CaseTests(unittest.TestCase):
    """Verify that every detective case computes its intended result."""

    def test_case_01_returns_fixed_position(self):
        self.assertEqual(case_01(2), 40)
        self.assertEqual(case_01(10_000), 40)

    def test_case_02_counts_halvings(self):
        self.assertEqual(case_02(2), 1)
        self.assertEqual(case_02(8), 3)
        self.assertEqual(case_02(16), 4)

    def test_case_03_combines_two_passes(self):
        self.assertEqual(case_03(4), 10)
        self.assertEqual(case_03(10), 55)

    def test_case_04_sums_geometrically_shrinking_ranges(self):
        self.assertEqual(case_04(2), 3)
        self.assertEqual(case_04(8), 15)
        self.assertEqual(case_04(10), 18)

    def test_case_05_finds_all_values_in_set(self):
        self.assertEqual(case_05(2), 2)
        self.assertEqual(case_05(127), 127)

    def test_case_06_counts_all_list_misses(self):
        self.assertEqual(case_06(2), 2)
        self.assertEqual(case_06(127), 127)

    def test_case_07_sums_values_while_slicing(self):
        self.assertEqual(case_07(2), 1)
        self.assertEqual(case_07(10), 45)

    def test_case_08_builds_reversed_list(self):
        self.assertEqual(case_08(2), 3)
        self.assertEqual(case_08(10), 19)

    def test_case_09_returns_combined_lengths(self):
        self.assertEqual(case_09(2), 4)
        self.assertEqual(case_09(127), 254)

    def test_case_10_is_deterministic(self):
        self.assertEqual(case_10(2), 1_444_343_149)
        self.assertEqual(case_10(32), 4_213_176_899)
        self.assertEqual(case_10(64), 4_198_688_481)


class BenchmarkTests(unittest.TestCase):
    """Verify benchmark controls and data processing."""

    def test_config_rejects_invalid_values(self):
        invalid_configs = [
            BenchmarkConfig(sizes=()),
            BenchmarkConfig(sizes=(2,)),
            BenchmarkConfig(sizes=(1, 2)),
            BenchmarkConfig(sizes=(4, 4)),
            BenchmarkConfig(sizes=(8, 4)),
            BenchmarkConfig(sizes=(2, 4), repeats=0),
            BenchmarkConfig(sizes=(2, 4), min_sample_seconds=0),
            BenchmarkConfig(sizes=(2, 4), max_iterations=0),
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()

    def test_select_cases_rejects_unknown_identifier(self):
        with self.assertRaises(ValueError):
            select_cases(["01", "99"])

    def test_log_log_slope_recovers_linear_growth(self):
        points = [
            Measurement("X", size, 0.25 * size, 1)
            for size in (2, 4, 8, 16)
        ]
        self.assertAlmostEqual(estimate_log_log_slope(points), 1.0, places=12)

    def test_small_benchmark_returns_all_points(self):
        config = BenchmarkConfig(
            sizes=(8, 16),
            repeats=2,
            min_sample_seconds=0.0001,
            max_iterations=1_024,
        )
        measurements = run_benchmarks(
            {"01": case_01, "03": case_03},
            config,
        )

        self.assertEqual(len(measurements), 4)
        self.assertTrue(all(point.seconds_per_call > 0 for point in measurements))
        self.assertTrue(
            all(point.iterations_per_sample >= 1 for point in measurements)
        )

    def test_csv_contains_header_and_all_rows(self):
        points = [
            Measurement("01", 8, 0.001, 16),
            Measurement("02", 8, 0.002, 8),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "measurements.csv"
            write_csv(points, destination)
            with destination.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(
            rows[0],
            [
                "case",
                "input_size",
                "seconds_per_call",
                "iterations_per_sample",
            ],
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0:2], ["01", "8"])
        self.assertEqual(rows[2][0:2], ["02", "8"])


if __name__ == "__main__":
    unittest.main()
