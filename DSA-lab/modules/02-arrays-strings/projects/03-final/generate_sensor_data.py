"""Generate a reproducible synthetic temperature-sensor time series."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# A fixed seed makes tests, benchmark comparisons, and learning results reproducible.
DEFAULT_SEED: Final = 20_260_716
ANOMALY_OFFSETS: Final = (15.0, -16.0, 18.0)


@dataclass(frozen=True)
class SensorDataset:
    """Synthetic readings and the positions of deliberately injected anomalies."""

    readings: list[float]
    anomaly_indices: tuple[int, ...]


def _default_anomaly_indices(size: int) -> tuple[int, ...]:
    """Place up to three anomalies at deterministic fractions of the series."""
    candidates = (size // 5, size // 2, (4 * size) // 5)
    return tuple(sorted(set(candidates)))


def _validate_anomaly_indices(
    anomaly_indices: tuple[int, ...], size: int
) -> tuple[int, ...]:
    """Validate and normalize explicitly requested anomaly positions."""
    if any(not isinstance(index, int) or isinstance(index, bool) for index in anomaly_indices):
        raise TypeError("anomaly indices must be integers")
    if len(set(anomaly_indices)) != len(anomaly_indices):
        raise ValueError("anomaly indices must be unique")
    if any(index < 0 or index >= size for index in anomaly_indices):
        raise IndexError("anomaly index outside generated series")
    return tuple(sorted(anomaly_indices))


def generate_sensor_data(
    size: int = 10_000,
    *,
    seed: int = DEFAULT_SEED,
    anomaly_indices: tuple[int, ...] | None = None,
) -> SensorDataset:
    """Generate temperature readings with seasonality, drift, noise, and spikes."""
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    selected = (
        _default_anomaly_indices(size)
        if anomaly_indices is None
        else _validate_anomaly_indices(tuple(anomaly_indices), size)
    )
    random_source = random.Random(seed)
    readings: list[float] = []

    for index in range(size):
        baseline = 20.0 + 0.00002 * index
        daily_cycle = 2.5 * math.sin(2.0 * math.pi * index / 1_440)
        noise = random_source.gauss(0.0, 0.35)
        readings.append(baseline + daily_cycle + noise)

    for position, index in enumerate(selected):
        readings[index] += ANOMALY_OFFSETS[position % len(ANOMALY_OFFSETS)]

    return SensorDataset(readings=readings, anomaly_indices=selected)


def write_sensor_csv(dataset: SensorDataset, output_path: Path) -> None:
    """Write readings and anomaly ground truth to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_set = set(dataset.anomaly_indices)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "temperature", "is_injected_outlier"))
        for index, value in enumerate(dataset.readings):
            writer.writerow((index, f"{value:.8f}", int(index in anomaly_set)))


def main() -> None:
    """Generate the default CSV dataset from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sensor_readings.csv"),
    )
    arguments = parser.parse_args()

    dataset = generate_sensor_data(arguments.size, seed=arguments.seed)
    write_sensor_csv(dataset, arguments.output)
    print(f"Wrote {len(dataset.readings)} readings to {arguments.output}")
    print(f"Injected anomaly indices: {dataset.anomaly_indices}")


if __name__ == "__main__":
    main()
