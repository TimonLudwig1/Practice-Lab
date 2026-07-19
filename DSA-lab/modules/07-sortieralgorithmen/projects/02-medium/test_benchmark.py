"""Tests for algorithms, workloads, measurements and output artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_algorithms import ALGORITHMS, ComparisonCounter
from run_benchmark import (
    create_plot,
    measure_algorithm,
    run_benchmark,
    run_pipeline,
    write_csv,
    write_report,
)
from workloads import INPUT_TYPES, generate_workloads


@pytest.mark.parametrize(("name", "algorithm"), list(ALGORITHMS.items()))
@pytest.mark.parametrize(
    "values",
    [
        [],
        [1],
        [3, 1, 2],
        [5, 5, 5, 5],
        list(range(20)),
        list(range(20, 0, -1)),
        [-3, 9, 0, -3, 2],
    ],
)
def test_all_algorithms_match_sorted(name, algorithm, values) -> None:
    del name
    counter = ComparisonCounter()
    original = values.copy()

    result = algorithm(values, counter)

    assert result == sorted(values)
    assert values == original
    assert counter.comparisons >= 0


def test_selection_comparison_count_is_input_independent() -> None:
    counts = []
    for values in (list(range(20)), list(range(20, 0, -1)), [1] * 20):
        counter = ComparisonCounter()
        ALGORITHMS["selection"](values, counter)
        counts.append(counter.comparisons)

    assert counts == [20 * 19 // 2] * 3


def test_bubble_uses_linear_comparisons_on_sorted_input() -> None:
    counter = ComparisonCounter()

    ALGORITHMS["bubble"](list(range(50)), counter)

    assert counter.comparisons == 49


def test_insertion_exposes_reverse_quadratic_case() -> None:
    counter = ComparisonCounter()

    ALGORITHMS["insertion"](list(range(30, 0, -1)), counter)

    assert counter.comparisons == 30 * 29 // 2


def test_three_way_quick_handles_identical_values_in_one_partition() -> None:
    counter = ComparisonCounter()

    result = ALGORITHMS["quick_3way"]([7] * 100, counter)

    assert result == [7] * 100
    assert counter.comparisons == 200


def test_workloads_are_reproducible_and_have_expected_shapes() -> None:
    first = generate_workloads(100, seed=4)
    second = generate_workloads(100, seed=4)

    assert first == second
    assert tuple(first) == INPUT_TYPES
    assert first["reversed"] == list(range(100, 0, -1))
    assert sorted(first["nearly_sorted"]) == list(range(100))
    assert len(set(first["many_duplicates"])) <= 8
    assert all(len(values) == 100 for values in first.values())


@pytest.mark.parametrize("size", [0, -1])
def test_workloads_reject_non_positive_size(size: int) -> None:
    with pytest.raises(ValueError):
        generate_workloads(size)


def test_measure_algorithm_returns_valid_row() -> None:
    row = measure_algorithm("merge", [4, 1, 3, 2], repetitions=2)

    assert row.algorithm == "merge"
    assert row.size == 4
    assert row.repetitions == 2
    assert row.median_ms >= 0
    assert row.comparisons > 0


def test_measure_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="unknown"):
        measure_algorithm("missing", [1])
    with pytest.raises(ValueError, match="repetitions"):
        measure_algorithm("merge", [1], repetitions=0)


def test_small_benchmark_contains_full_cartesian_product() -> None:
    rows = run_benchmark((20, 40), repetitions=1, seed=2)

    assert len(rows) == 2 * len(INPUT_TYPES) * len(ALGORITHMS)
    assert {row.size for row in rows} == {20, 40}
    assert {row.input_type for row in rows} == set(INPUT_TYPES)
    assert {row.algorithm for row in rows} == set(ALGORITHMS)


def test_benchmark_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        run_benchmark(())
    with pytest.raises(ValueError):
        run_benchmark((10, 0))


def test_writers_create_nonempty_artifacts(tmp_path: Path) -> None:
    rows = run_benchmark((20,), repetitions=1, seed=3)

    csv_path = write_csv(tmp_path / "benchmark.csv", rows)
    plot_path = create_plot(tmp_path / "plot.png", rows)
    report_path = write_report(tmp_path / "REPORT.md", rows)

    for path in (csv_path, plot_path, report_path):
        assert path.exists()
        assert path.stat().st_size > 0
    assert csv_path.read_text(encoding="utf-8").count("\n") == len(rows) + 1


def test_full_pipeline_with_default_workload(tmp_path: Path) -> None:
    rows = run_pipeline(tmp_path)

    assert len(rows) == 5 * len(INPUT_TYPES) * len(ALGORITHMS)
    assert (tmp_path / "benchmark.csv").exists()
    assert (tmp_path / "sorting_benchmark.png").exists()
    assert (tmp_path / "REPORT.md").exists()
