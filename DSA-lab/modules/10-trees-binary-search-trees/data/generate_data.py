"""Generate a reproducible nonlinear binary-classification dataset."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


# A fixed seed makes the dataset, split, tests, and model comparison reproducible.
SEED = 20260720
TRAIN_SIZE = 900
TEST_SIZE = 300
FEATURE_NAMES = ("signal_x", "signal_y", "context", "noise")


@dataclass(frozen=True)
class Sample:
    """One generated observation."""

    features: tuple[float, float, float, float]
    label: int


def generate_samples(total: int = TRAIN_SIZE + TEST_SIZE) -> list[Sample]:
    """Create samples with a nonlinear, noisy, tree-friendly decision rule."""

    if total < 2:
        raise ValueError("total must be at least 2")

    rng = random.Random(SEED)
    samples: list[Sample] = []
    for _ in range(total):
        signal_x = rng.uniform(-2.0, 2.0)
        signal_y = rng.uniform(-2.0, 2.0)
        context = rng.uniform(-1.5, 1.5)
        noise = rng.gauss(0.0, 1.0)

        rule = (
            (signal_x > 0.45 and signal_y > -0.30)
            or (signal_x < -0.70 and context > 0.15)
            or (signal_y < -1.00 and context < -0.25)
        )
        label = int(rule)

        # Seven percent label noise prevents a perfectly separable toy result.
        if rng.random() < 0.07:
            label = 1 - label

        samples.append(
            Sample((signal_x, signal_y, context, noise), label)
        )
    return samples


def split_samples(
    samples: list[Sample],
    train_size: int = TRAIN_SIZE,
) -> tuple[list[Sample], list[Sample]]:
    """Create a deterministic stratified train/test split."""

    if not 0 < train_size < len(samples):
        raise ValueError("train_size must leave at least one test sample")

    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)

    rng = random.Random(SEED + 1)
    for group in by_label.values():
        rng.shuffle(group)

    target_positive = round(train_size * len(by_label[1]) / len(samples))
    target_negative = train_size - target_positive
    train = by_label[0][:target_negative] + by_label[1][:target_positive]
    test = by_label[0][target_negative:] + by_label[1][target_positive:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_csv(path: Path, samples: list[Sample]) -> None:
    """Write samples using stable decimal formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow((*FEATURE_NAMES, "label"))
        for sample in samples:
            writer.writerow(
                [*(f"{value:.8f}" for value in sample.features), sample.label]
            )


def main() -> None:
    samples = generate_samples()
    train, test = split_samples(samples)
    data_dir = Path(__file__).resolve().parent
    write_csv(data_dir / "decision_tree_train.csv", train)
    write_csv(data_dir / "decision_tree_test.csv", test)

    train_positives = sum(sample.label for sample in train)
    test_positives = sum(sample.label for sample in test)
    print(
        f"wrote {len(train)} train samples "
        f"({train_positives / len(train):.1%} positive)"
    )
    print(
        f"wrote {len(test)} test samples "
        f"({test_positives / len(test):.1%} positive)"
    )


if __name__ == "__main__":
    main()
