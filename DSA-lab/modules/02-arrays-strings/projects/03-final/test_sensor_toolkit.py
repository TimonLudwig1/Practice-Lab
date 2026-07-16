"""Tests for generation, list algorithms, integration, and NumPy agreement."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from benchmark import run_benchmark, write_benchmark_csv
from generate_sensor_data import (
    ANOMALY_OFFSETS,
    generate_sensor_data,
    write_sensor_csv,
)
from sensor_toolkit import (
    PrefixSumIndex,
    analyze_sensor_readings,
    detect_zscore_outliers,
    moving_average,
)


class SensorGeneratorTests(unittest.TestCase):
    def test_same_seed_produces_same_series(self) -> None:
        first = generate_sensor_data(50, seed=123)
        second = generate_sensor_data(50, seed=123)
        self.assertEqual(first, second)

    def test_different_seed_changes_noise(self) -> None:
        first = generate_sensor_data(50, seed=123)
        second = generate_sensor_data(50, seed=124)
        self.assertNotEqual(first.readings, second.readings)
        self.assertEqual(first.anomaly_indices, second.anomaly_indices)

    def test_explicit_anomalies_have_exact_offsets(self) -> None:
        plain = generate_sensor_data(20, seed=5, anomaly_indices=())
        spiked = generate_sensor_data(20, seed=5, anomaly_indices=(2, 10, 18))
        differences = [
            spiked.readings[index] - plain.readings[index] for index in (2, 10, 18)
        ]
        for actual, expected in zip(differences, ANOMALY_OFFSETS, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_rejects_invalid_size_and_seed(self) -> None:
        with self.assertRaises(ValueError):
            generate_sensor_data(0)
        with self.assertRaises(TypeError):
            generate_sensor_data(10.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            generate_sensor_data(10, seed=True)

    def test_rejects_invalid_anomaly_indices(self) -> None:
        with self.assertRaises(IndexError):
            generate_sensor_data(10, anomaly_indices=(10,))
        with self.assertRaises(ValueError):
            generate_sensor_data(10, anomaly_indices=(2, 2))
        with self.assertRaises(TypeError):
            generate_sensor_data(10, anomaly_indices=(2.5,))  # type: ignore[arg-type]

    def test_writes_csv_with_ground_truth(self) -> None:
        dataset = generate_sensor_data(5, seed=1, anomaly_indices=(2,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "sensor.csv"
            write_sensor_csv(dataset, output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[2]["is_injected_outlier"], "1")
        self.assertEqual(rows[1]["is_injected_outlier"], "0")


class MovingAverageTests(unittest.TestCase):
    def test_computes_typical_windows(self) -> None:
        self.assertEqual(moving_average([1, 2, 3, 4, 5], 3), [2.0, 3.0, 4.0])

    def test_window_one_returns_values_as_floats(self) -> None:
        self.assertEqual(moving_average([2, -1, 4], 1), [2.0, -1.0, 4.0])

    def test_full_window_returns_one_average(self) -> None:
        self.assertEqual(moving_average([1, 3, 8, 4], 4), [4.0])

    def test_supports_floating_point_values(self) -> None:
        result = moving_average([0.1, 0.2, 0.3], 2)
        self.assertAlmostEqual(result[0], 0.15)
        self.assertAlmostEqual(result[1], 0.25)

    def test_rejects_invalid_windows(self) -> None:
        for window in (0, -1, 4):
            with self.subTest(window=window), self.assertRaises(ValueError):
                moving_average([1, 2, 3], window)
        with self.assertRaises(TypeError):
            moving_average([1, 2], 1.5)  # type: ignore[arg-type]

    def test_rejects_non_finite_and_non_numeric_values(self) -> None:
        with self.assertRaises(ValueError):
            moving_average([1.0, math.nan], 1)
        with self.assertRaises(ValueError):
            moving_average([1.0, math.inf], 1)
        with self.assertRaises(TypeError):
            moving_average([1.0, "2"], 1)  # type: ignore[list-item]


class PrefixSumIndexTests(unittest.TestCase):
    def test_answers_half_open_ranges(self) -> None:
        index = PrefixSumIndex.from_readings([4, -1, 7, 3, 2])
        self.assertEqual(index.range_sum(0, 5), 15.0)
        self.assertEqual(index.range_sum(1, 4), 9.0)
        self.assertEqual(index.range_sum(3, 3), 0.0)

    def test_answers_batch_in_input_order(self) -> None:
        index = PrefixSumIndex.from_readings([1, 2, 3, 4])
        ranges = ((2, 4), (0, 1), (1, 3))
        self.assertEqual(index.batch_range_sums(ranges), [7.0, 1.0, 5.0])

    def test_supports_empty_series_query(self) -> None:
        index = PrefixSumIndex.from_readings([])
        self.assertEqual(len(index), 0)
        self.assertEqual(index.range_sum(0, 0), 0.0)

    def test_snapshot_is_independent_of_source(self) -> None:
        source = [1.0, 2.0, 3.0]
        index = PrefixSumIndex.from_readings(source)
        source[0] = 100.0
        self.assertEqual(index.range_sum(0, 3), 6.0)

    def test_rejects_invalid_boundaries(self) -> None:
        index = PrefixSumIndex.from_readings([1, 2, 3])
        for query in ((-1, 2), (2, 1), (0, 4)):
            with self.subTest(query=query), self.assertRaises(IndexError):
                index.range_sum(*query)

    def test_rejects_non_integer_boundaries(self) -> None:
        index = PrefixSumIndex.from_readings([1, 2])
        with self.assertRaises(TypeError):
            index.range_sum(0.0, 1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            index.range_sum(False, 1)

    def test_rejects_invalid_readings(self) -> None:
        with self.assertRaises(TypeError):
            PrefixSumIndex.from_readings([True, 2])
        with self.assertRaises(ValueError):
            PrefixSumIndex.from_readings([1, -math.inf])


class OutlierDetectionTests(unittest.TestCase):
    def test_detects_positive_and_negative_outliers(self) -> None:
        values = [0.0] * 20
        values[3] = 10.0
        values[15] = -10.0
        outliers = detect_zscore_outliers(values, threshold=3.0)
        self.assertEqual([outlier.index for outlier in outliers], [3, 15])
        self.assertGreater(outliers[0].z_score, 0)
        self.assertLess(outliers[1].z_score, 0)

    def test_generated_anomalies_are_detected(self) -> None:
        dataset = generate_sensor_data(10_000, seed=42)
        detected = {
            outlier.index
            for outlier in detect_zscore_outliers(dataset.readings, threshold=4.0)
        }
        self.assertTrue(set(dataset.anomaly_indices).issubset(detected))

    def test_constant_and_empty_series_have_no_outliers(self) -> None:
        self.assertEqual(detect_zscore_outliers([2.0] * 10), [])
        self.assertEqual(detect_zscore_outliers([]), [])

    def test_threshold_is_inclusive(self) -> None:
        values = [-1.0, 1.0]
        outliers = detect_zscore_outliers(values, threshold=1.0)
        self.assertEqual([outlier.index for outlier in outliers], [0, 1])

    def test_rejects_invalid_threshold(self) -> None:
        for threshold in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                detect_zscore_outliers([1, 2], threshold)
        with self.assertRaises(TypeError):
            detect_zscore_outliers([1, 2], True)

    def test_rejects_invalid_readings(self) -> None:
        with self.assertRaises(ValueError):
            detect_zscore_outliers([1.0, math.nan])
        with self.assertRaises(TypeError):
            detect_zscore_outliers([1.0, False])


class AnalysisPipelineTests(unittest.TestCase):
    def test_combines_all_analyses(self) -> None:
        values = [0.0] * 20
        values[10] = 10.0
        result = analyze_sensor_readings(
            values,
            window=4,
            ranges=((0, 4), (9, 12)),
            outlier_threshold=3.0,
        )
        self.assertEqual(len(result.moving_averages), 17)
        self.assertEqual(result.range_sums, (0.0, 10.0))
        self.assertEqual([outlier.index for outlier in result.outliers], [10])

    def test_consumes_one_shot_range_iterable_once(self) -> None:
        ranges = ((start, start + 1) for start in range(3))
        result = analyze_sensor_readings([1, 2, 3], window=2, ranges=ranges)
        self.assertEqual(result.range_sums, (1.0, 2.0, 3.0))


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = run_benchmark(
            generate_sensor_data(2_000, seed=7),
            window=16,
            query_count=300,
            threshold=4.0,
            repetitions=1,
            query_seed=8,
        )

    def test_covers_all_operations(self) -> None:
        self.assertEqual(
            [row.operation for row in self.results],
            [
                "moving_average",
                "prefix_build",
                "range_queries",
                "outlier_detection",
            ],
        )

    def test_timings_are_positive_and_results_agree(self) -> None:
        for row in self.results:
            with self.subTest(operation=row.operation):
                self.assertGreater(row.python_seconds, 0.0)
                self.assertGreater(row.numpy_seconds, 0.0)
                self.assertGreater(row.speedup, 0.0)
                self.assertLess(row.max_abs_error, 1e-8)

    def test_rejects_invalid_benchmark_configuration(self) -> None:
        dataset = generate_sensor_data(20)
        with self.assertRaises(ValueError):
            run_benchmark(dataset, window=2, query_count=0)
        with self.assertRaises(ValueError):
            run_benchmark(dataset, window=2, query_count=2, repetitions=0)

    def test_writes_benchmark_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results" / "benchmark.csv"
            write_benchmark_csv(self.results, output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["operation"], "moving_average")
        self.assertIn("numpy_speedup", rows[0])


if __name__ == "__main__":
    unittest.main()
