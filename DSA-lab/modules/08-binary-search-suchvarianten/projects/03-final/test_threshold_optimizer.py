"""Tests for classifier-threshold indexing and optimization."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from generate_data import generate_score_data
from threshold_optimizer import (
    ScoreRecord,
    ThresholdIndex,
    optimize_threshold_binary,
    optimize_threshold_exhaustive,
    optimize_threshold_grid,
    read_score_csv,
)


def sample_records() -> list[ScoreRecord]:
    return [
        ScoreRecord("p-high", 1, 0.9),
        ScoreRecord("n-high", 0, 0.8),
        ScoreRecord("p-low", 1, 0.6),
        ScoreRecord("n-low", 0, 0.2),
    ]


@pytest.mark.parametrize(
    ("threshold", "counts", "recall", "fpr", "precision"),
    [
        (0.0, (2, 2, 0, 0), 1.0, 1.0, 0.5),
        (0.6, (2, 1, 1, 0), 1.0, 0.5, 2 / 3),
        (0.8, (1, 1, 1, 1), 0.5, 0.5, 0.5),
        (0.85, (1, 0, 2, 1), 0.5, 0.0, 1.0),
        (1.0, (0, 0, 2, 2), 0.0, 0.0, 0.0),
    ],
)
def test_metric_evaluation(threshold, counts, recall, fpr, precision) -> None:
    metrics = ThresholdIndex(sample_records()).evaluate(threshold)
    assert (
        metrics.true_positives,
        metrics.false_positives,
        metrics.true_negatives,
        metrics.false_negatives,
    ) == counts
    assert metrics.recall == pytest.approx(recall)
    assert metrics.false_positive_rate == pytest.approx(fpr)
    assert metrics.precision == pytest.approx(precision)


def test_candidate_thresholds_include_scores_and_safe_sentinel() -> None:
    candidates = ThresholdIndex(sample_records()).candidate_thresholds
    assert candidates[:-1] == (0.2, 0.6, 0.8, 0.9)
    assert candidates[-1] > 1.0


@pytest.mark.parametrize(
    ("max_fpr", "expected_threshold", "expected_recall"),
    [
        (1.0, 0.2, 1.0),
        (0.5, 0.6, 1.0),
        (0.49, 0.9, 0.5),
        (0.0, 0.9, 0.5),
    ],
)
def test_binary_optimization_known_boundaries(
    max_fpr: float, expected_threshold: float, expected_recall: float
) -> None:
    result = optimize_threshold_binary(ThresholdIndex(sample_records()), max_fpr)
    assert result.metrics.threshold == expected_threshold
    assert result.metrics.recall == expected_recall
    assert result.metrics.false_positive_rate <= max_fpr


def test_binary_trace_preserves_boundary_and_shrinks() -> None:
    index = ThresholdIndex(sample_records())
    result = optimize_threshold_binary(index, 0.0)
    boundary = index.candidate_thresholds.index(0.9)
    assert result.trace
    for step in result.trace:
        assert step.low_index <= boundary <= step.high_index
        assert step.next_low_index <= boundary <= step.next_high_index
        assert step.next_size < step.size
        assert step.feasible == (step.metrics.false_positive_rate <= 0.0)


def test_selected_candidate_is_minimally_feasible() -> None:
    index = ThresholdIndex(sample_records())
    result = optimize_threshold_binary(index, 0.0)
    selected_index = index.candidate_thresholds.index(result.metrics.threshold)
    assert result.metrics.false_positive_rate <= 0.0
    assert selected_index > 0
    predecessor = index.evaluate(index.candidate_thresholds[selected_index - 1])
    assert predecessor.false_positive_rate > 0.0


def test_binary_matches_exhaustive_on_seeded_datasets() -> None:
    rng = random.Random(80831)  # Fixed seed makes failures reproducible.
    for case in range(300):
        record_count = rng.randrange(1, 80)
        records = [
            ScoreRecord(
                f"{case}-{index}",
                rng.randrange(2),
                rng.randrange(0, 21) / 20,
            )
            for index in range(record_count)
        ]
        max_fpr = rng.randrange(0, 11) / 10
        threshold_index = ThresholdIndex(records)
        binary = optimize_threshold_binary(threshold_index, max_fpr)
        exhaustive = optimize_threshold_exhaustive(threshold_index, max_fpr)
        assert binary.metrics == exhaustive.metrics
        assert binary.evaluations <= math.ceil(
            math.log2(len(threshold_index.candidate_thresholds))
        ) + 1


def test_fpr_and_recall_are_non_increasing_over_candidates() -> None:
    index = ThresholdIndex(sample_records())
    metrics = [index.evaluate(threshold) for threshold in index.candidate_thresholds]
    assert all(
        left.false_positive_rate >= right.false_positive_rate
        for left, right in zip(metrics, metrics[1:])
    )
    assert all(left.recall >= right.recall for left, right in zip(metrics, metrics[1:]))


def test_precision_is_not_a_monotone_search_predicate() -> None:
    index = ThresholdIndex(
        [
            ScoreRecord("positive-high", 1, 0.9),
            ScoreRecord("negative-middle", 0, 0.8),
            ScoreRecord("positive-low", 1, 0.7),
        ]
    )
    precisions = [index.evaluate(threshold).precision for threshold in (0.7, 0.8, 0.9)]
    assert precisions == pytest.approx([2 / 3, 1 / 2, 1.0])
    assert precisions[0] > precisions[1] < precisions[2]


def test_grid_is_feasible_and_exact_search_is_not_worse() -> None:
    index = ThresholdIndex(sample_records())
    binary = optimize_threshold_binary(index, 0.0)
    grid = optimize_threshold_grid(index, 0.0, grid_intervals=10)
    assert grid.metrics.false_positive_rate <= 0.0
    assert binary.metrics.recall >= grid.metrics.recall
    assert grid.evaluations == 12
    assert binary.evaluations < grid.evaluations


def test_grid_validation() -> None:
    index = ThresholdIndex(sample_records())
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            optimize_threshold_grid(index, 0.05, grid_intervals=invalid)


@pytest.mark.parametrize("invalid", [-0.1, 1.1, math.inf, math.nan])
def test_fpr_limit_validation(invalid: float) -> None:
    index = ThresholdIndex(sample_records())
    with pytest.raises(ValueError):
        optimize_threshold_binary(index, invalid)
    with pytest.raises(ValueError):
        optimize_threshold_exhaustive(index, invalid)
    with pytest.raises(ValueError):
        optimize_threshold_grid(index, invalid)


def test_no_negative_labels_makes_every_threshold_feasible() -> None:
    index = ThresholdIndex(
        [ScoreRecord("p1", 1, 0.3), ScoreRecord("p2", 1, 0.8)]
    )
    result = optimize_threshold_binary(index, 0.0)
    assert result.metrics.threshold == 0.3
    assert result.metrics.recall == 1.0
    assert result.metrics.false_positive_rate == 0.0


def test_no_positive_labels_has_zero_recall() -> None:
    index = ThresholdIndex(
        [ScoreRecord("n1", 0, 0.3), ScoreRecord("n2", 0, 0.8)]
    )
    result = optimize_threshold_binary(index, 0.0)
    assert result.metrics.threshold > 1.0
    assert result.metrics.recall == 0.0
    assert result.metrics.precision == 0.0


@pytest.mark.parametrize(
    "records",
    [
        [],
        [ScoreRecord("", 1, 0.5)],
        [ScoreRecord("same", 1, 0.5), ScoreRecord("same", 0, 0.4)],
        [ScoreRecord("x", 2, 0.5)],
        [ScoreRecord("x", True, 0.5)],
        [ScoreRecord("x", 1.0, 0.5)],
        [ScoreRecord("x", 1, -0.1)],
        [ScoreRecord("x", 1, 1.1)],
        [ScoreRecord("x", 1, math.nan)],
        [ScoreRecord("x", 1, True)],
    ],
)
def test_index_rejects_invalid_records(records) -> None:
    with pytest.raises(ValueError):
        ThresholdIndex(records)


def test_metrics_and_trace_are_immutable() -> None:
    result = optimize_threshold_binary(ThresholdIndex(sample_records()), 0.0)
    with pytest.raises(FrozenInstanceError):
        result.metrics.threshold = 0.1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.trace[0].low_index = 99  # type: ignore[misc]


def test_generator_is_reproducible(tmp_path: Path) -> None:
    first = generate_score_data(tmp_path / "first.csv", record_count=200, seed=42)
    second = generate_score_data(tmp_path / "second.csv", record_count=200, seed=42)
    third = generate_score_data(tmp_path / "third.csv", record_count=200, seed=43)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != third.read_bytes()
    records = read_score_csv(first)
    assert len(records) == 200
    assert {record.label for record in records} == {0, 1}


@pytest.mark.parametrize(
    ("record_count", "positive_rate"),
    [(0, 0.3), (-1, 0.3), (True, 0.3), (10, 0.0), (10, 1.0), (10, -0.1)],
)
def test_generator_rejects_invalid_configuration(
    tmp_path: Path, record_count, positive_rate
) -> None:
    with pytest.raises(ValueError):
        generate_score_data(
            tmp_path / "scores.csv",
            record_count=record_count,
            positive_rate=positive_rate,
        )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "id,label,score\nx,1,0.5\n",
        "record_id,label,score\nx,yes,0.5\n",
        "record_id,label,score\nx,1,nope\n",
        "record_id,label,score\nx,1\n",
        "record_id,label,score\nx,1,0.5,extra\n",
        "record_id,label,score\nx,2,0.5\n",
    ],
)
def test_csv_reader_rejects_invalid_data(tmp_path: Path, content: str) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_score_csv(path)


def test_csv_reader_accepts_quoted_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("record_id", "label", "score"))
        writer.writerow(("record, one", 1, 0.75))
        writer.writerow(("record two", 0, 0.25))
    records = read_score_csv(path)
    assert records == [
        ScoreRecord("record, one", 1, 0.75),
        ScoreRecord("record two", 0, 0.25),
    ]
