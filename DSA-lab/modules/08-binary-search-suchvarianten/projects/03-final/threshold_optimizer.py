"""Threshold metrics and exact binary-search optimization for classifiers."""

from __future__ import annotations

import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ScoreRecord:
    """One binary-classification label and model score."""

    record_id: str
    label: int
    score: float


@dataclass(frozen=True)
class ThresholdMetrics:
    """Confusion counts and derived metrics at one inclusive threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def recall(self) -> float:
        """Return true-positive rate, or zero when no positives exist."""

        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Return false-positive rate, or zero when no negatives exist."""

        denominator = self.false_positives + self.true_negatives
        return self.false_positives / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        """Return precision, or zero when no positive predictions exist."""

        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def predicted_positive(self) -> int:
        """Return the number of positive predictions."""

        return self.true_positives + self.false_positives


@dataclass(frozen=True)
class ThresholdSearchStep:
    """One transition in the candidate-threshold index space."""

    low_index: int
    high_index: int
    middle_index: int
    metrics: ThresholdMetrics
    feasible: bool
    next_low_index: int
    next_high_index: int

    @property
    def size(self) -> int:
        """Return the inclusive candidate count before the transition."""

        return self.high_index - self.low_index + 1

    @property
    def next_size(self) -> int:
        """Return the inclusive candidate count after the transition."""

        return self.next_high_index - self.next_low_index + 1


@dataclass(frozen=True)
class OptimizationResult:
    """Selected threshold, metrics, and deterministic search effort."""

    method: str
    metrics: ThresholdMetrics
    evaluations: int
    candidate_count: int
    trace: tuple[ThresholdSearchStep, ...] = ()


class ThresholdIndex:
    """Evaluate thresholds with binary counts over sorted class-specific scores."""

    def __init__(self, records: Iterable[ScoreRecord]) -> None:
        materialized = tuple(records)
        if not materialized:
            raise ValueError("records must not be empty")
        identifiers: set[str] = set()
        for record in materialized:
            if (
                not isinstance(record.record_id, str)
                or not record.record_id
                or record.record_id in identifiers
            ):
                raise ValueError("record IDs must be non-empty and unique")
            identifiers.add(record.record_id)
            if (
                not isinstance(record.label, int)
                or isinstance(record.label, bool)
                or record.label not in (0, 1)
            ):
                raise ValueError("labels must be integers equal to 0 or 1")
            if (
                not isinstance(record.score, (int, float))
                or isinstance(record.score, bool)
                or not math.isfinite(record.score)
                or not 0.0 <= record.score <= 1.0
            ):
                raise ValueError("scores must be finite values in [0, 1]")

        self._positive_scores = sorted(
            record.score for record in materialized if record.label == 1
        )
        self._negative_scores = sorted(
            record.score for record in materialized if record.label == 0
        )
        observed = sorted({record.score for record in materialized})
        self._candidate_thresholds = tuple(
            [*observed, math.nextafter(1.0, math.inf)]
        )
        self._record_count = len(materialized)

    @property
    def record_count(self) -> int:
        """Return the indexed record count."""

        return self._record_count

    @property
    def positive_count(self) -> int:
        """Return the number of positive labels."""

        return len(self._positive_scores)

    @property
    def negative_count(self) -> int:
        """Return the number of negative labels."""

        return len(self._negative_scores)

    @property
    def candidate_thresholds(self) -> tuple[float, ...]:
        """Return all exact decision boundaries plus an always-feasible sentinel."""

        return self._candidate_thresholds

    def evaluate(self, threshold: float) -> ThresholdMetrics:
        """Evaluate the inclusive rule ``score >= threshold`` in O(log n)."""

        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        positive_split = bisect_left(self._positive_scores, threshold)
        negative_split = bisect_left(self._negative_scores, threshold)
        true_positives = self.positive_count - positive_split
        false_positives = self.negative_count - negative_split
        return ThresholdMetrics(
            threshold=threshold,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=negative_split,
            false_negatives=positive_split,
        )


def _validate_fpr_limit(max_false_positive_rate: float) -> None:
    if (
        isinstance(max_false_positive_rate, bool)
        or not isinstance(max_false_positive_rate, (int, float))
        or not math.isfinite(max_false_positive_rate)
        or not 0.0 <= max_false_positive_rate <= 1.0
    ):
        raise ValueError("max_false_positive_rate must be in [0, 1]")


def optimize_threshold_binary(
    index: ThresholdIndex, max_false_positive_rate: float
) -> OptimizationResult:
    """Find the smallest exact threshold satisfying an FPR upper bound.

    FPR is non-increasing as the threshold rises. Recall is also non-increasing,
    so the first feasible threshold maximizes recall under the constraint.
    """

    _validate_fpr_limit(max_false_positive_rate)
    candidates = index.candidate_thresholds
    low, high = 0, len(candidates) - 1
    trace: list[ThresholdSearchStep] = []

    while low < high:
        middle = low + (high - low) // 2
        metrics = index.evaluate(candidates[middle])
        feasible = metrics.false_positive_rate <= max_false_positive_rate
        if feasible:
            next_low, next_high = low, middle
        else:
            next_low, next_high = middle + 1, high
        trace.append(
            ThresholdSearchStep(
                low_index=low,
                high_index=high,
                middle_index=middle,
                metrics=metrics,
                feasible=feasible,
                next_low_index=next_low,
                next_high_index=next_high,
            )
        )
        low, high = next_low, next_high

    selected = index.evaluate(candidates[low])
    return OptimizationResult(
        method="binary_exact",
        metrics=selected,
        evaluations=len(trace) + 1,
        candidate_count=len(candidates),
        trace=tuple(trace),
    )


def optimize_threshold_exhaustive(
    index: ThresholdIndex, max_false_positive_rate: float
) -> OptimizationResult:
    """Check every exact candidate as a correctness reference."""

    _validate_fpr_limit(max_false_positive_rate)
    feasible = [
        metrics
        for threshold in index.candidate_thresholds
        if (metrics := index.evaluate(threshold)).false_positive_rate
        <= max_false_positive_rate
    ]
    selected = max(feasible, key=lambda metrics: (metrics.recall, -metrics.threshold))
    return OptimizationResult(
        method="exhaustive_exact",
        metrics=selected,
        evaluations=len(index.candidate_thresholds),
        candidate_count=len(index.candidate_thresholds),
    )


def optimize_threshold_grid(
    index: ThresholdIndex,
    max_false_positive_rate: float,
    *,
    grid_intervals: int = 1_000,
) -> OptimizationResult:
    """Evaluate a uniform grid as an intentionally naive baseline."""

    _validate_fpr_limit(max_false_positive_rate)
    if (
        not isinstance(grid_intervals, int)
        or isinstance(grid_intervals, bool)
        or grid_intervals < 1
    ):
        raise ValueError("grid_intervals must be a positive integer")
    thresholds = [position / grid_intervals for position in range(grid_intervals + 1)]
    thresholds.append(math.nextafter(1.0, math.inf))
    feasible = [
        metrics
        for threshold in thresholds
        if (metrics := index.evaluate(threshold)).false_positive_rate
        <= max_false_positive_rate
    ]
    selected = max(feasible, key=lambda metrics: (metrics.recall, -metrics.threshold))
    return OptimizationResult(
        method="uniform_grid",
        metrics=selected,
        evaluations=len(thresholds),
        candidate_count=len(thresholds),
    )


def read_score_csv(path: str | Path) -> list[ScoreRecord]:
    """Read and validate ``record_id,label,score`` rows from a CSV file."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["record_id", "label", "score"]:
            raise ValueError("CSV header must be record_id,label,score")
        records: list[ScoreRecord] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None for field in reader.fieldnames):
                raise ValueError(f"malformed CSV row at line {line_number}")
            try:
                label = int(row["label"])
                score = float(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid value at line {line_number}") from error
            records.append(ScoreRecord(row["record_id"], label, score))
    ThresholdIndex(records)
    return records
