"""Tests for streaming rolling statistics and anomaly detection."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from anomaly_detector import (
    MetricPoint,
    RollingStats,
    detect_naive,
    detect_streaming,
    evaluate_detections,
    read_metric_csv,
)
from generate_stream import generate_metric_stream


def points(values, anomalies=()) -> list[MetricPoint]:
    anomaly_set = set(anomalies)
    return [
        MetricPoint(f"t-{index}", float(value), index in anomaly_set)
        for index, value in enumerate(values)
    ]


def test_rolling_stats_basic_window() -> None:
    state = RollingStats(3)
    assert state.append(1) is None
    assert state.append(2) is None
    assert state.append(3) is None
    assert state.full
    assert state.values == (1.0, 2.0, 3.0)
    assert state.mean == 2.0
    assert state.variance == pytest.approx(2 / 3)
    assert state.std == pytest.approx(math.sqrt(2 / 3))
    assert state.append(4) == 1.0
    assert state.values == (2.0, 3.0, 4.0)
    assert state.mean == 3.0


def test_rolling_stats_matches_recomputation() -> None:
    rng = random.Random(90931)
    state = RollingStats(20)
    history = []
    for _ in range(1_000):
        value = rng.uniform(-100, 100)
        history.append(value)
        state.append(value)
        window = history[-20:]
        assert state.mean == pytest.approx(sum(window) / len(window), abs=1e-10)
        expected_variance = sum(item * item for item in window) / len(window) - (
            sum(window) / len(window)
        ) ** 2
        assert state.variance == pytest.approx(expected_variance, abs=1e-9)


@pytest.mark.parametrize("capacity", [0, 1, -1, 2.5, True])
def test_rolling_stats_rejects_invalid_capacity(capacity) -> None:
    with pytest.raises(ValueError):
        RollingStats(capacity)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1"])
def test_rolling_stats_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        RollingStats(3).append(value)


def test_empty_stats_have_no_mean_or_variance() -> None:
    state = RollingStats(3)
    with pytest.raises(ValueError):
        _ = state.mean
    with pytest.raises(ValueError):
        _ = state.variance


def test_known_spike_is_detected_without_lookahead() -> None:
    stream = points([10, 10, 10, 10, 10, 30, 10], anomalies=[5])
    detections = detect_streaming(stream, window_size=5, z_threshold=3)
    assert [item.index for item in detections] == [5, 6]
    assert detections[0].is_anomaly
    assert math.isinf(detections[0].z_score)
    assert not detections[1].is_anomaly


def test_streaming_matches_naive_on_random_cases() -> None:
    rng = random.Random(90932)
    for case in range(300):
        size = rng.randrange(5, 100)
        window = rng.randrange(2, min(20, size) + 1)
        stream = points([rng.gauss(0, 3) for _ in range(size)])
        streaming = detect_streaming(stream, window_size=window, z_threshold=2.5)
        naive = detect_naive(stream, window_size=window, z_threshold=2.5)
        assert len(streaming) == len(naive)
        for first, second in zip(streaming, naive):
            assert first.index == second.index
            assert first.is_anomaly == second.is_anomaly
            assert first.rolling_mean == pytest.approx(second.rolling_mean, abs=1e-10)
            assert first.rolling_std == pytest.approx(second.rolling_std, abs=1e-9)


def test_short_stream_produces_no_detections() -> None:
    stream = points([1, 2, 3])
    assert detect_streaming(stream, window_size=5, z_threshold=3) == []
    assert detect_naive(stream, window_size=5, z_threshold=3) == []


@pytest.mark.parametrize(
    ("window", "threshold"),
    [(1, 3), (0, 3), (2.5, 3), (True, 3), (3, 0), (3, -1), (3, math.nan), (3, True)],
)
def test_detectors_reject_invalid_configuration(window, threshold) -> None:
    stream = points([1, 2, 3])
    with pytest.raises(ValueError):
        detect_streaming(stream, window_size=window, z_threshold=threshold)
    with pytest.raises(ValueError):
        detect_naive(stream, window_size=window, z_threshold=threshold)


@pytest.mark.parametrize(
    "point",
    [
        MetricPoint("", 1.0),
        MetricPoint("t", math.nan),
        MetricPoint("t", math.inf),
        MetricPoint("t", True),
        MetricPoint("t", 1.0, 1),
    ],
)
def test_detectors_reject_invalid_points(point) -> None:
    stream = [MetricPoint("a", 1.0), MetricPoint("b", 2.0), point]
    with pytest.raises(ValueError):
        detect_streaming(stream, window_size=2, z_threshold=3)
    with pytest.raises(ValueError):
        detect_naive(stream, window_size=2, z_threshold=3)


def test_detection_metrics() -> None:
    stream = points([10, 10, 10, 10, 10, 30, 10, 40], anomalies=[5, 7])
    detections = detect_streaming(stream, window_size=5, z_threshold=3)
    metrics = evaluate_detections(detections)
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_generator_is_reproducible_and_labeled(tmp_path: Path) -> None:
    first = generate_metric_stream(
        tmp_path / "first.csv", record_count=1_000, anomaly_count=2, seed=42
    )
    second = generate_metric_stream(
        tmp_path / "second.csv", record_count=1_000, anomaly_count=2, seed=42
    )
    third = generate_metric_stream(
        tmp_path / "third.csv", record_count=1_000, anomaly_count=2, seed=43
    )
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() != third.read_bytes()
    loaded = read_metric_csv(first)
    assert len(loaded) == 1_000
    assert sum(point.injected_anomaly for point in loaded) == 2


@pytest.mark.parametrize(
    ("count", "anomalies"),
    [(100, 0), (500.5, 0), (True, 0), (500, -1), (500, 1.5), (500, True), (500, 2)],
)
def test_generator_rejects_invalid_configuration(tmp_path: Path, count, anomalies) -> None:
    with pytest.raises(ValueError):
        generate_metric_stream(
            tmp_path / "stream.csv", record_count=count, anomaly_count=anomalies
        )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "time,value,injected_anomaly\nt,1,False\n",
        "timestamp,value,injected_anomaly\nt,nope,False\n",
        "timestamp,value,injected_anomaly\nt,1,yes\n",
        "timestamp,value,injected_anomaly\nt,1\n",
        "timestamp,value,injected_anomaly\nt,1,False,extra\n",
    ],
)
def test_reader_rejects_invalid_csv(tmp_path: Path, content: str) -> None:
    path = tmp_path / "stream.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_metric_csv(path)
