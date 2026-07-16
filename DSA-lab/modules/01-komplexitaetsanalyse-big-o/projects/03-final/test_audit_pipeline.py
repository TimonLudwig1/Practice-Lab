"""Tests for the performance audit project."""

import csv
import tempfile
import unittest
from pathlib import Path

from audit_pipeline import (
    CustomerSummary,
    Event,
    inefficient_pipeline,
    load_events,
    optimized_pipeline,
    write_summaries,
)
from data.generate_data import generate_events
from run_audit import (
    AuditMeasurement,
    benchmark_pipelines,
    validate_benchmark,
    write_measurements,
    write_report,
)


def sample_events() -> list[Event]:
    """Return a small input with completed and ignored events."""
    return [
        Event(
            "E1",
            "C2",
            "books",
            2_000,
            200,
            "completed",
            "2025-01-02T10:00",
        ),
        Event(
            "E2",
            "C1",
            "sports",
            4_000,
            500,
            "completed",
            "2025-01-01T10:00",
        ),
        Event(
            "E3",
            "C1",
            "books",
            2_500,
            0,
            "completed",
            "2025-01-03T10:00",
        ),
        Event(
            "E4",
            "C2",
            "toys",
            9_000,
            0,
            "cancelled",
            "2025-01-04T10:00",
        ),
    ]


class GeneratorAndLoadingTests(unittest.TestCase):
    """Verify deterministic CSV generation and parsing."""

    def test_same_seed_produces_identical_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.csv"
            second = Path(temporary_directory) / "second.csv"
            generate_events(first, rows=50, seed=123)
            generate_events(second, rows=50, seed=123)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_generator_writes_requested_number_of_loadable_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "events.csv"
            generate_events(destination, rows=37, seed=456)
            events = load_events(destination)

        self.assertEqual(len(events), 37)
        self.assertTrue(all(event.event_id.startswith("EVT-") for event in events))

    def test_generator_rejects_nonpositive_row_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "events.csv"
            with self.assertRaises(ValueError):
                generate_events(destination, rows=0)


class PipelineTests(unittest.TestCase):
    """Verify business results and equivalence of both implementations."""

    def test_pipelines_match_on_constructed_events(self):
        baseline = inefficient_pipeline(sample_events())
        optimized = optimized_pipeline(sample_events())
        self.assertEqual(baseline, optimized)

    def test_expected_summary_values(self):
        summaries = optimized_pipeline(sample_events())
        self.assertEqual(
            summaries,
            [
                CustomerSummary(
                    customer_id="C1",
                    completed_events=2,
                    net_revenue_cents=6_000,
                    average_net_cents=3_000,
                    unique_categories=2,
                    latest_event_timestamp="2025-01-03T10:00",
                ),
                CustomerSummary(
                    customer_id="C2",
                    completed_events=1,
                    net_revenue_cents=1_800,
                    average_net_cents=1_800,
                    unique_categories=1,
                    latest_event_timestamp="2025-01-02T10:00",
                ),
            ],
        )

    def test_noncompleted_events_are_ignored(self):
        ignored = [
            Event("E1", "C1", "books", 100, 0, "pending", "2025-01-01"),
            Event("E2", "C2", "toys", 100, 0, "cancelled", "2025-01-02"),
        ]
        self.assertEqual(inefficient_pipeline(ignored), [])
        self.assertEqual(optimized_pipeline(ignored), [])

    def test_pipelines_match_on_generated_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "events.csv"
            generate_events(destination, rows=250, seed=789)
            events = load_events(destination)

        self.assertEqual(
            inefficient_pipeline(events),
            optimized_pipeline(events),
        )

    def test_summary_csv_has_one_row_per_customer(self):
        summaries = optimized_pipeline(sample_events())
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "summary.csv"
            write_summaries(summaries, destination)
            with destination.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["customer_id"] for row in rows], ["C1", "C2"])


class AuditTests(unittest.TestCase):
    """Verify benchmark validation and result serialization."""

    def test_validate_benchmark_rejects_invalid_settings(self):
        events = sample_events()
        invalid_settings = [
            ((), 1),
            ((0, 2), 1),
            ((2, 2), 1),
            ((3, 2), 1),
            ((2, 5), 1),
            ((2, 3), 0),
        ]
        for sizes, repeats in invalid_settings:
            with self.subTest(sizes=sizes, repeats=repeats):
                with self.assertRaises(ValueError):
                    validate_benchmark(events, sizes, repeats)

    def test_small_benchmark_verifies_and_measures_both_pipelines(self):
        events = sample_events()
        measurements = benchmark_pipelines(events, (2, 4), repeats=2)
        self.assertEqual(len(measurements), 2)
        self.assertTrue(
            all(point.baseline_seconds > 0 for point in measurements)
        )
        self.assertTrue(
            all(point.optimized_seconds > 0 for point in measurements)
        )
        self.assertTrue(all(point.speedup > 0 for point in measurements))

    def test_measurement_csv_and_markdown_report_are_written(self):
        measurements = [
            AuditMeasurement(100, 0.02, 0.005, 25),
            AuditMeasurement(200, 0.08, 0.01, 50),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            csv_path = directory / "audit.csv"
            report_path = directory / "report.md"
            write_measurements(measurements, csv_path)
            write_report(measurements, report_path, seed=123, repeats=3)

            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["speedup"], "8.000000")
        self.assertIn("Seed: 123", report)
        self.assertIn("8.00x", report)


if __name__ == "__main__":
    unittest.main()
