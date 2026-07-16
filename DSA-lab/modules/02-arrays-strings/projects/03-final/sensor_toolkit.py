"""List-based algorithms for analyzing a one-dimensional sensor time series."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real


Number = int | float
Range = tuple[int, int]


def _validate_readings(values: Sequence[Number]) -> None:
    """Reject booleans, non-numeric values, infinities, and NaNs."""
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("readings must contain only real numbers")
        if not math.isfinite(float(value)):
            raise ValueError("readings must be finite")


def _validate_window(window: int, size: int) -> None:
    """Validate a rolling-window size against a series length."""
    if not isinstance(window, int) or isinstance(window, bool):
        raise TypeError("window must be an integer")
    if window <= 0:
        raise ValueError("window must be positive")
    if window > size:
        raise ValueError("window must not exceed the number of readings")


def moving_average(values: Sequence[Number], window: int) -> list[float]:
    """Compute valid moving averages in O(n) time with a rolling sum."""
    _validate_readings(values)
    _validate_window(window, len(values))

    rolling_sum = math.fsum(values[index] for index in range(window))
    averages = [rolling_sum / window]

    for right in range(window, len(values)):
        rolling_sum += values[right] - values[right - window]
        averages.append(rolling_sum / window)
    return averages


@dataclass(frozen=True)
class PrefixSumIndex:
    """Immutable prefix-sum index for repeated range-sum queries."""

    _prefix: tuple[float, ...]

    @classmethod
    def from_readings(cls, values: Sequence[Number]) -> PrefixSumIndex:
        """Build the prefix array in O(n) time."""
        _validate_readings(values)
        prefix = [0.0]
        for value in values:
            prefix.append(prefix[-1] + float(value))
        return cls(tuple(prefix))

    def __len__(self) -> int:
        return len(self._prefix) - 1

    @property
    def prefix_values(self) -> tuple[float, ...]:
        """Expose the immutable prefix values for validation and inspection."""
        return self._prefix

    def range_sum(self, start: int, end: int) -> float:
        """Return the half-open range sum readings[start:end] in O(1)."""
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise TypeError("range boundaries must be integers")
        if start < 0 or end < start or end > len(self):
            raise IndexError("range must satisfy 0 <= start <= end <= length")
        return self._prefix[end] - self._prefix[start]

    def batch_range_sums(self, ranges: Iterable[Range]) -> list[float]:
        """Answer a batch of range queries while preserving input order."""
        return [self.range_sum(start, end) for start, end in ranges]


@dataclass(frozen=True)
class Outlier:
    """A detected reading with its index, value, and signed z-score."""

    index: int
    value: float
    z_score: float


def detect_zscore_outliers(
    values: Sequence[Number], threshold: float = 4.0
) -> list[Outlier]:
    """Detect readings whose absolute population z-score reaches threshold."""
    _validate_readings(values)
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number")
    if not math.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not values:
        return []

    mean = math.fsum(values) / len(values)
    variance = math.fsum((float(value) - mean) ** 2 for value in values) / len(values)
    standard_deviation = math.sqrt(variance)
    if standard_deviation == 0.0:
        return []

    outliers: list[Outlier] = []
    for index, value in enumerate(values):
        z_score = (float(value) - mean) / standard_deviation
        if abs(z_score) >= threshold:
            outliers.append(Outlier(index, float(value), z_score))
    return outliers


@dataclass(frozen=True)
class AnalysisResult:
    """Combined immutable output of the time-series analysis pipeline."""

    moving_averages: tuple[float, ...]
    range_sums: tuple[float, ...]
    outliers: tuple[Outlier, ...]


def analyze_sensor_readings(
    values: Sequence[Number],
    *,
    window: int,
    ranges: Iterable[Range],
    outlier_threshold: float = 4.0,
) -> AnalysisResult:
    """Run all list-based analyses and return a stable result snapshot."""
    prefix_index = PrefixSumIndex.from_readings(values)
    return AnalysisResult(
        moving_averages=tuple(moving_average(values, window)),
        range_sums=tuple(prefix_index.batch_range_sums(ranges)),
        outliers=tuple(detect_zscore_outliers(values, outlier_threshold)),
    )
