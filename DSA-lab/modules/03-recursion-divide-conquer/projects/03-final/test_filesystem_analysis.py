"""Tests for deterministic generation and both filesystem traversals."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from benchmark import run_benchmark, write_benchmark_csv
from filesystem_analysis import (
    DepthStats,
    analyze_iterative,
    analyze_recursive,
)
from generate_tree import generate_tree


class TemporaryTreeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "tree"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()


class GeneratorTests(TemporaryTreeTestCase):
    def test_same_seed_produces_same_manifest(self) -> None:
        first = generate_tree(self.root, seed=123, max_depth=3)
        other_root = Path(self.temporary_directory.name) / "other"
        second = generate_tree(other_root, seed=123, max_depth=3)
        self.assertEqual(first, second)

    def test_different_seed_changes_manifest(self) -> None:
        first = generate_tree(self.root, seed=123, max_depth=2)
        other_root = Path(self.temporary_directory.name) / "other"
        second = generate_tree(other_root, seed=124, max_depth=2)
        self.assertNotEqual(first.files, second.files)

    def test_manifest_matches_actual_analysis(self) -> None:
        summary = generate_tree(self.root, seed=10, max_depth=3)
        analysis = analyze_recursive(self.root)
        self.assertEqual(analysis.directory_count, summary.directory_count)
        self.assertEqual(analysis.file_count, summary.file_count)
        self.assertEqual(analysis.total_bytes, summary.total_bytes)
        self.assertEqual(analysis.max_depth, summary.max_depth)

    def test_respects_zero_max_depth(self) -> None:
        summary = generate_tree(self.root, max_depth=0)
        self.assertEqual(summary.directory_count, 1)
        self.assertEqual(summary.max_depth, 0)

    def test_refuses_nonempty_root_without_overwrite(self) -> None:
        generate_tree(self.root, max_depth=1)
        with self.assertRaises(FileExistsError):
            generate_tree(self.root, max_depth=1)

    def test_overwrite_replaces_existing_tree(self) -> None:
        generate_tree(self.root, seed=1, max_depth=1)
        extra = self.root / "not_generated.txt"
        extra.write_text("old", encoding="utf-8")
        second = generate_tree(self.root, seed=2, max_depth=1, overwrite=True)
        self.assertFalse(extra.exists())
        self.assertEqual(analyze_recursive(self.root).file_count, second.file_count)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            generate_tree(self.root, max_depth=-1)
        with self.assertRaises(ValueError):
            generate_tree(self.root, max_subdirectories=0)
        with self.assertRaises(TypeError):
            generate_tree(self.root, seed=True)


class AnalysisTests(TemporaryTreeTestCase):
    def _build_known_tree(self) -> None:
        (self.root / "nested" / "deep").mkdir(parents=True)
        (self.root / "root.csv").write_bytes(b"1234")
        (self.root / "readme").write_bytes(b"ab")
        (self.root / "nested" / "report.csv").write_bytes(b"123456")
        (self.root / "nested" / "notes.txt").write_bytes(b"abc")
        (self.root / "nested" / "deep" / "events.log").write_bytes(b"12345")

    def test_recursive_aggregates_known_tree(self) -> None:
        self._build_known_tree()
        result = analyze_recursive(self.root, "*.csv")
        self.assertEqual(result.total_bytes, 20)
        self.assertEqual(result.file_count, 5)
        self.assertEqual(result.directory_count, 3)
        self.assertEqual(result.max_depth, 2)
        self.assertEqual(
            result.depth_stats,
            (
                DepthStats(0, 1, 2, 6),
                DepthStats(1, 1, 2, 9),
                DepthStats(2, 1, 1, 5),
            ),
        )

    def test_extension_totals_include_missing_extension(self) -> None:
        self._build_known_tree()
        result = analyze_recursive(self.root)
        self.assertEqual(
            dict(result.bytes_by_extension),
            {".csv": 10, ".log": 5, ".txt": 3, "<none>": 2},
        )

    def test_pattern_matches_name_or_relative_path(self) -> None:
        self._build_known_tree()
        csv_matches = analyze_recursive(self.root, "*.csv").matches
        deep_matches = analyze_recursive(self.root, "nested/deep/*").matches
        self.assertEqual(
            [record.relative_path for record in csv_matches],
            ["nested/report.csv", "root.csv"],
        )
        self.assertEqual(
            [record.relative_path for record in deep_matches],
            ["nested/deep/events.log"],
        )

    def test_pattern_is_case_sensitive(self) -> None:
        self.root.mkdir()
        (self.root / "Report.CSV").write_bytes(b"x")
        self.assertEqual(analyze_recursive(self.root, "*.csv").matches, ())

    def test_recursive_and_iterative_results_are_identical(self) -> None:
        generate_tree(self.root, seed=55, max_depth=4)
        self.assertEqual(
            analyze_recursive(self.root, "report*"),
            analyze_iterative(self.root, "report*"),
        )

    def test_empty_directory(self) -> None:
        self.root.mkdir()
        result = analyze_recursive(self.root)
        self.assertEqual(result.total_bytes, 0)
        self.assertEqual(result.file_count, 0)
        self.assertEqual(result.directory_count, 1)
        self.assertEqual(result.depth_stats, (DepthStats(0, 1, 0, 0),))

    def test_rejects_missing_or_file_root(self) -> None:
        with self.assertRaises(FileNotFoundError):
            analyze_recursive(self.root)
        self.root.write_text("not a directory", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            analyze_iterative(self.root)

    def test_rejects_invalid_pattern(self) -> None:
        self.root.mkdir()
        with self.assertRaises(ValueError):
            analyze_recursive(self.root, "")
        with self.assertRaises(TypeError):
            analyze_iterative(self.root, 42)  # type: ignore[arg-type]

    def test_symlinks_are_ignored(self) -> None:
        self.root.mkdir()
        target = self.root / "target.txt"
        target.write_bytes(b"abc")
        link = self.root / "link.txt"
        try:
            os.symlink(target, link)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        result = analyze_recursive(self.root)
        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.total_bytes, 3)


class BenchmarkTests(TemporaryTreeTestCase):
    def test_benchmark_validates_and_times_both_strategies(self) -> None:
        generate_tree(self.root, seed=8, max_depth=3)
        result = run_benchmark(self.root, pattern="*.json", repetitions=2)
        self.assertGreater(result.recursive_seconds, 0.0)
        self.assertGreater(result.iterative_seconds, 0.0)
        self.assertGreater(result.recursive_over_iterative, 0.0)
        self.assertEqual(result.repetitions, 2)

    def test_rejects_invalid_repetition_count(self) -> None:
        self.root.mkdir()
        with self.assertRaises(ValueError):
            run_benchmark(self.root, repetitions=0)
        with self.assertRaises(TypeError):
            run_benchmark(self.root, repetitions=True)

    def test_writes_csv_report(self) -> None:
        generate_tree(self.root, seed=9, max_depth=2)
        result = run_benchmark(self.root, repetitions=1)
        output = Path(self.temporary_directory.name) / "results" / "benchmark.csv"
        write_benchmark_csv(result, output)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["files"]), result.analysis.file_count)
        self.assertIn("recursive_over_iterative", rows[0])


if __name__ == "__main__":
    unittest.main()
