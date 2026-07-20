"""Train the scratch tree and compare it with scikit-learn on the same data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

from decision_tree import ScratchDecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
MODULE_DIR = PROJECT_DIR.parents[1]
DATA_DIR = MODULE_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
TRAIN_PATH = DATA_DIR / "decision_tree_train.csv"
TEST_PATH = DATA_DIR / "decision_tree_test.csv"
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 8
RANDOM_STATE = 20260720


@dataclass(frozen=True)
class Dataset:
    """Numeric features, integer labels, and their column names."""

    features: list[list[float]]
    labels: list[int]
    feature_names: list[str]


def load_dataset(path: Path) -> Dataset:
    """Load one generated CSV file."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or reader.fieldnames[-1] != "label":
            raise ValueError(f"invalid dataset header in {path}")
        feature_names = reader.fieldnames[:-1]
        features: list[list[float]] = []
        labels: list[int] = []
        for row in reader:
            features.append([float(row[name]) for name in feature_names])
            labels.append(int(row["label"]))

    if not features:
        raise ValueError(f"dataset is empty: {path}")
    return Dataset(features, labels, feature_names)


def accuracy(expected: list[int], predicted: list[int]) -> float:
    """Return the share of equal labels."""

    if len(expected) != len(predicted) or not expected:
        raise ValueError("accuracy requires equally sized, non-empty inputs")
    return sum(a == b for a, b in zip(expected, predicted)) / len(expected)


def confusion_counts(
    expected: list[int], predicted: list[int]
) -> dict[str, int]:
    """Return binary true/false positive/negative counts."""

    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for actual, guess in zip(expected, predicted):
        key = {
            (0, 0): "tn",
            (0, 1): "fp",
            (1, 0): "fn",
            (1, 1): "tp",
        }[(actual, guess)]
        counts[key] += 1
    return counts


def run_experiment() -> dict[str, float | int]:
    """Run both models, write artifacts, and return headline metrics."""

    train = load_dataset(TRAIN_PATH)
    test = load_dataset(TEST_PATH)
    if train.feature_names != test.feature_names:
        raise ValueError("train and test feature columns differ")

    scratch = ScratchDecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    ).fit(train.features, train.labels)
    reference = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    ).fit(train.features, train.labels)

    scratch_train = scratch.predict(train.features)
    scratch_test = scratch.predict(test.features)
    sklearn_train = [int(value) for value in reference.predict(train.features)]
    sklearn_test = [int(value) for value in reference.predict(test.features)]

    rows = [
        {
            "model": "scratch",
            "train_accuracy": accuracy(train.labels, scratch_train),
            "test_accuracy": accuracy(test.labels, scratch_test),
            "depth": scratch.get_depth(),
            "leaves": scratch.get_n_leaves(),
        },
        {
            "model": "sklearn",
            "train_accuracy": accuracy(train.labels, sklearn_train),
            "test_accuracy": accuracy(test.labels, sklearn_test),
            "depth": reference.get_depth(),
            "leaves": reference.get_n_leaves(),
        },
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "train_accuracy": f"{row['train_accuracy']:.6f}",
                    "test_accuracy": f"{row['test_accuracy']:.6f}",
                }
            )

    scratch_importance = scratch.feature_importances()
    with (OUTPUT_DIR / "feature_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("feature", "scratch", "sklearn"))
        for name, own_value, reference_value in zip(
            train.feature_names,
            scratch_importance,
            reference.feature_importances_,
        ):
            writer.writerow((name, f"{own_value:.6f}", f"{reference_value:.6f}"))

    (OUTPUT_DIR / "scratch_tree.txt").write_text(
        scratch.export_text(train.feature_names) + "\n",
        encoding="utf-8",
    )

    scratch_confusion = confusion_counts(test.labels, scratch_test)
    sklearn_confusion = confusion_counts(test.labels, sklearn_test)
    disagreement = sum(a != b for a, b in zip(scratch_test, sklearn_test))

    summary: dict[str, float | int] = {
        "scratch_test_accuracy": rows[0]["test_accuracy"],
        "sklearn_test_accuracy": rows[1]["test_accuracy"],
        "prediction_disagreement": disagreement,
        **{f"scratch_{key}": value for key, value in scratch_confusion.items()},
        **{f"sklearn_{key}": value for key, value in sklearn_confusion.items()},
    }
    return summary


def main() -> None:
    summary = run_experiment()
    print("DECISION-TREE COMPARISON")
    print(f"scratch test accuracy: {summary['scratch_test_accuracy']:.3f}")
    print(f"sklearn test accuracy: {summary['sklearn_test_accuracy']:.3f}")
    print(
        "prediction disagreements: "
        f"{summary['prediction_disagreement']} of 300"
    )
    print(f"artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
