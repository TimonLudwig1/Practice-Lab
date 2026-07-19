"""Example-based and reproducible property tests for all algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from case_generator import DEFAULT_SEED, generate_random_cases, structured_cases
from sorting import (
    ALGORITHMS,
    SortMetrics,
    TraceStep,
    bubble_sort,
    insertion_sort,
    merge_sort,
    quick_sort,
    selection_sort,
)


ALGORITHM_PARAMETERS = list(ALGORITHMS.items())


@pytest.mark.parametrize(("name", "algorithm"), ALGORITHM_PARAMETERS)
@pytest.mark.parametrize(
    "values",
    [
        [],
        [1],
        [2, 1],
        [5, 2, 4, 1, 3],
        [3, 3, 1, 2, 1],
        [-5, 0, 7, -2, 7],
        list(range(20)),
        list(range(20, 0, -1)),
    ],
)
def test_known_cases_match_sorted(name, algorithm, values) -> None:
    del name
    original = values.copy()

    result = algorithm(values)

    assert result == sorted(values)
    assert values == original


@pytest.mark.parametrize(("name", "algorithm"), ALGORITHM_PARAMETERS)
def test_reproducible_random_property(name, algorithm) -> None:
    del name
    cases = generate_random_cases(seed=DEFAULT_SEED)
    for values in cases:
        original = values.copy()
        assert algorithm(values) == sorted(values)
        assert values == original


@pytest.mark.parametrize(("name", "algorithm"), ALGORITHM_PARAMETERS)
def test_structured_property_cases(name, algorithm) -> None:
    del name
    for values in structured_cases():
        assert algorithm(values) == sorted(values)


@pytest.mark.parametrize(("name", "algorithm"), ALGORITHM_PARAMETERS)
def test_key_function_matches_python_sorted(name, algorithm) -> None:
    del name
    words = ["pear", "fig", "banana", "kiwi", "plum"]

    result = algorithm(words, key=len)

    assert [len(word) for word in result] == sorted(map(len, words))
    assert sorted(result) == sorted(words)


@dataclass(frozen=True)
class Record:
    rank: int
    label: str


@pytest.mark.parametrize("algorithm", [bubble_sort, insertion_sort, merge_sort])
def test_stable_algorithms_preserve_equal_key_order(algorithm) -> None:
    records = [
        Record(2, "A"),
        Record(1, "B"),
        Record(2, "C"),
        Record(1, "D"),
        Record(2, "E"),
    ]

    result = algorithm(records, key=lambda record: record.rank)

    assert [record.label for record in result] == ["B", "D", "A", "C", "E"]


def test_bubble_early_exit_on_sorted_input() -> None:
    metrics = SortMetrics()

    assert bubble_sort(list(range(10)), metrics=metrics) == list(range(10))
    assert metrics.comparisons == 9
    assert metrics.swaps == 0


def test_selection_always_performs_triangular_comparisons() -> None:
    for values in (list(range(12)), list(range(12, 0, -1))):
        metrics = SortMetrics()
        selection_sort(values, metrics=metrics)
        assert metrics.comparisons == 12 * 11 // 2


def test_insertion_reverse_input_counts_all_pair_comparisons() -> None:
    metrics = SortMetrics()

    insertion_sort(list(range(10, 0, -1)), metrics=metrics)

    assert metrics.comparisons == 10 * 9 // 2


def test_merge_metrics_show_logarithmic_depth() -> None:
    metrics = SortMetrics()

    result = merge_sort(list(range(32, 0, -1)), metrics=metrics)

    assert result == list(range(1, 33))
    assert metrics.max_recursion_depth == 6
    assert metrics.recursive_calls == 63


def test_quick_sort_exposes_last_pivot_worst_case_depth() -> None:
    metrics = SortMetrics()

    result = quick_sort(list(range(20)), metrics=metrics)

    assert result == list(range(20))
    assert metrics.max_recursion_depth == 20
    assert metrics.comparisons == 20 * 19 // 2


@pytest.mark.parametrize("algorithm", [bubble_sort, selection_sort, insertion_sort, quick_sort])
def test_mutating_algorithms_emit_trace_steps(algorithm) -> None:
    trace: list[TraceStep] = []

    result = algorithm([3, 1, 2], trace=trace)

    assert result == [1, 2, 3]
    assert trace
    assert all(step.algorithm for step in trace)
    assert list(trace[-1].state) == result


def test_merge_emits_one_trace_for_each_merge() -> None:
    trace: list[TraceStep] = []

    result = merge_sort([4, 3, 2, 1], trace=trace)

    assert result == [1, 2, 3, 4]
    assert len(trace) == 3
    assert list(trace[-1].state) == result


def test_case_generator_is_reproducible() -> None:
    assert generate_random_cases(seed=7) == generate_random_cases(seed=7)
    assert generate_random_cases(seed=7) != generate_random_cases(seed=8)


@pytest.mark.parametrize(
    "arguments",
    [
        {"case_count": 0},
        {"case_count": -1},
        {"max_size": -1},
        {"minimum": 5, "maximum": 4},
    ],
)
def test_case_generator_rejects_invalid_configuration(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        generate_random_cases(**arguments)
