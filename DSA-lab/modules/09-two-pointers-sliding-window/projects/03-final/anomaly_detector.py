"""Streaming and naive rolling z-score anomaly detection."""

from __future__ import annotations

import csv
import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricPoint:
    """One timestamped metric value with optional injected ground truth."""

    timestamp: str
    value: float
    injected_anomaly: bool = False


@dataclass(frozen=True)
class Detection:
    """One metric value scored against its preceding rolling window."""

    index: int
    timestamp: str
    value: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    is_anomaly: bool
    injected_anomaly: bool


@dataclass(frozen=True)
class DetectionMetrics:
    """Confusion counts and derived anomaly-detection metrics."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0


class RollingStats:
    """Maintain fixed-window population statistics with O(1) updates."""

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 2:
            raise ValueError("capacity must be an integer of at least 2")
        self._capacity = capacity
        self._values: deque[float] = deque()
        self._total = 0.0
        self._square_total = 0.0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def full(self) -> bool:
        return self.count == self.capacity

    @property
    def values(self) -> tuple[float, ...]:
        """Return an immutable snapshot for tests and diagnostics."""

        return tuple(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            raise ValueError("statistics require at least one value")
        return self._total / self.count

    @property
    def variance(self) -> float:
        if not self._values:
            raise ValueError("statistics require at least one value")
        mean = self.mean
        return max(0.0, self._square_total / self.count - mean * mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def append(self, value: float) -> float | None:
        """Append a finite value and return the evicted value, if any."""

        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("rolling values must be finite numbers")
        numeric = float(value)
        evicted = None
        if self.full:
            evicted = self._values.popleft()
            self._total -= evicted
            self._square_total -= evicted * evicted
        self._values.append(numeric)
        self._total += numeric
        self._square_total += numeric * numeric
        return evicted


def _validate_configuration(window_size: int, z_threshold: float) -> None:
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 2:
        raise ValueError("window_size must be an integer of at least 2")
    if (
        not isinstance(z_threshold, (int, float))
        or isinstance(z_threshold, bool)
        or not math.isfinite(z_threshold)
        or z_threshold <= 0
    ):
        raise ValueError("z_threshold must be a positive finite number")


def _score(
    index: int,
    point: MetricPoint,
    mean: float,
    std: float,
    z_threshold: float,
) -> Detection:
    if std == 0.0:
        z_score = 0.0 if point.value == mean else math.copysign(math.inf, point.value - mean)
    else:
        z_score = (point.value - mean) / std
    return Detection(
        index=index,
        timestamp=point.timestamp,
        value=point.value,
        rolling_mean=mean,
        rolling_std=std,
        z_score=z_score,
        is_anomaly=abs(z_score) > z_threshold,
        injected_anomaly=point.injected_anomaly,
    )


def detect_streaming(
    points: Iterable[MetricPoint],
    *,
    window_size: int,
    z_threshold: float,
) -> list[Detection]:
    """Score each point against the preceding window with O(1) state updates."""

    _validate_configuration(window_size, z_threshold)
    state = RollingStats(window_size)
    detections: list[Detection] = []
    for index, point in enumerate(points):
        _validate_point(point)
        if state.full:
            detections.append(_score(index, point, state.mean, state.std, z_threshold))
        state.append(point.value)
    return detections


def detect_naive(
    points: Sequence[MetricPoint],
    *,
    window_size: int,
    z_threshold: float,
) -> list[Detection]:
    """Score by rebuilding mean and variance from every preceding window."""

    _validate_configuration(window_size, z_threshold)
    for point in points:
        _validate_point(point)
    detections: list[Detection] = []
    for index in range(window_size, len(points)):
        window = [point.value for point in points[index - window_size : index]]
        mean = sum(window) / window_size
        variance = max(
            0.0,
            sum(value * value for value in window) / window_size - mean * mean,
        )
        detections.append(
            _score(index, points[index], mean, math.sqrt(variance), z_threshold)
        )
    return detections


def _validate_point(point: MetricPoint) -> None:
    if not isinstance(point.timestamp, str) or not point.timestamp:
        raise ValueError("timestamps must be non-empty strings")
    if (
        not isinstance(point.value, (int, float))
        or isinstance(point.value, bool)
        or not math.isfinite(point.value)
    ):
        raise ValueError("metric values must be finite numbers")
    if not isinstance(point.injected_anomaly, bool):
        raise ValueError("injected_anomaly must be boolean")


def evaluate_detections(detections: Iterable[Detection]) -> DetectionMetrics:
    """Return confusion counts for detection flags against injected truth."""

    tp = fp = tn = fn = 0
    for detection in detections:
        if detection.is_anomaly and detection.injected_anomaly:
            tp += 1
        elif detection.is_anomaly:
            fp += 1
        elif detection.injected_anomaly:
            fn += 1
        else:
            tn += 1
    return DetectionMetrics(tp, fp, tn, fn)


def read_metric_csv(path: str | Path) -> list[MetricPoint]:
    """Read and validate timestamp,value,injected_anomaly CSV records."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["timestamp", "value", "injected_anomaly"]:
            raise ValueError("unexpected metric CSV header")
        points: list[MetricPoint] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None for field in reader.fieldnames):
                raise ValueError(f"malformed CSV row at line {line_number}")
            try:
                value = float(row["value"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid value at line {line_number}") from error
            truth = row["injected_anomaly"]
            if truth not in ("True", "False"):
                raise ValueError(f"invalid anomaly flag at line {line_number}")
            point = MetricPoint(row["timestamp"], value, truth == "True")
            _validate_point(point)
            points.append(point)
    return points
