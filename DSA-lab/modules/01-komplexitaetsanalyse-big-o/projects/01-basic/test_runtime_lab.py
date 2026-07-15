"""Tests for the deterministic parts of the runtime lab."""

import csv
import tempfile
import unittest
from pathlib import Path

from runtime_lab import (
    BenchmarkConfig,
    Measurement,
    curve_a,
    curve_b,
    curve_c,
    curve_d,
    curve_e,
    estimate_log_log_slope,
    run_benchmarks,
    write_csv,
)


class CurveTests(unittest.TestCase):
    """Verify the exact abstract work represented by every curve."""

    def test_curve_a_is_independent_of_input_size(self):
        self.assertEqual(curve_a(2), 32)
        self.assertEqual(curve_a(10_000), 32)

    def test_curve_b_counts_halving_levels(self):
        self.assertEqual(curve_b(2), 32)
        self.assertEqual(curve_b(8), 96)
        self.assertEqual(curve_b(16), 128)

    def test_curve_c_visits_every_input_item(self):
        self.assertEqual(curve_c(2), 16)
        self.assertEqual(curve_c(127), 1_016)

    def test_curve_d_combines_linear_and_halving_work(self):
        self.assertEqual(curve_d(2), 2)
        self.assertEqual(curve_d(8), 24)
        self.assertEqual(curve_d(16), 64)

    def test_curve_e_counts_unordered_pairs(self):
        for size in (2, 3, 10, 25):
            self.assertEqual(curve_e(size), size * (size - 1) // 2)


class BenchmarkTests(unittest.TestCase):
    """Verify configuration, measurement, regression, and export behavior."""

    def test_config_rejects_invalid_values(self):
        invalid_configs = [
            BenchmarkConfig(sizes=()),
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

    def test_log_log_slope_recovers_quadratic_growth(self):
        points = [
            Measurement("X", size, 3.0 * size**2, 1)
            for size in (2, 4, 8, 16)
        ]
        self.assertAlmostEqual(estimate_log_log_slope(points), 2.0, places=12)

    def test_small_benchmark_returns_positive_measurements(self):
        config = BenchmarkConfig(
            sizes=(8, 16),
            repeats=2,
            min_sample_seconds=0.0001,
            max_iterations=1_024,
        )
        measurements = run_benchmarks({"A": curve_a, "C": curve_c}, config)

        self.assertEqual(len(measurements), 4)
        self.assertTrue(all(point.seconds_per_call > 0 for point in measurements))
        self.assertTrue(
            all(point.iterations_per_sample >= 1 for point in measurements)
        )

    def test_csv_contains_header_and_all_measurements(self):
        points = [
            Measurement("A", 8, 0.001, 16),
            Measurement("B", 8, 0.002, 8),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "measurements.csv"
            write_csv(points, destination)

            with destination.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

        self.assertEqual(
            rows[0],
            [
                "curve",
                "input_size",
                "seconds_per_call",
                "iterations_per_sample",
            ],
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0:2], ["A", "8"])
        self.assertEqual(rows[2][0:2], ["B", "8"])


if __name__ == "__main__":
    unittest.main()
