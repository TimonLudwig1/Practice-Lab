"""Tests for impurity, recursive learning, prediction, and the ML comparison."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from sklearn.tree import DecisionTreeClassifier

from decision_tree import (
    ScratchDecisionTreeClassifier,
    gini_impurity,
    weighted_gini,
)
from run_experiment import TEST_PATH, TRAIN_PATH, load_dataset


MODULE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = MODULE_DIR / "data"
sys.path.insert(0, str(DATA_DIR))
from generate_data import (  # noqa: E402
    SEED,
    TRAIN_SIZE,
    generate_samples,
    split_samples,
)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([], 0.0),
        ([0], 0.0),
        ([1, 1, 1], 0.0),
        ([0, 1], 0.5),
        ([0, 0, 1, 1], 0.5),
        ([0, 0, 0, 1], 0.375),
        ([0, 1, 2], 2 / 3),
    ],
)
def test_gini_impurity(labels: list[int], expected: float) -> None:
    assert gini_impurity(labels) == pytest.approx(expected)


def test_weighted_gini() -> None:
    assert weighted_gini([0, 0, 1], [1]) == pytest.approx(1 / 3)
    assert weighted_gini([], []) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": -1}, "max_depth"),
        ({"min_samples_split": 1}, "min_samples_split"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"min_impurity_decrease": -0.01}, "min_impurity_decrease"),
    ],
)
def test_invalid_hyperparameters(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ScratchDecisionTreeClassifier(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        ([], [], "at least one"),
        ([[1.0]], [], "equal length"),
        ([[]], [0], "at least one feature"),
        ([[1.0], [1.0, 2.0]], [0, 1], "equal length"),
        ([[math.nan]], [0], "finite"),
        ([[math.inf]], [0], "finite"),
        ([[1.0]], ["zero"], "integers"),
    ],
)
def test_invalid_training_data(
    features: list[list[float]], labels: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ScratchDecisionTreeClassifier().fit(features, labels)


def test_operations_before_fit_raise() -> None:
    model = ScratchDecisionTreeClassifier()

    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[1.0]])
    with pytest.raises(RuntimeError, match="fit"):
        model.get_depth()
    with pytest.raises(RuntimeError, match="fit"):
        model.get_n_leaves()
    with pytest.raises(RuntimeError, match="fit"):
        model.feature_importances()
    with pytest.raises(RuntimeError, match="fit"):
        model.export_text()


def test_single_class_creates_root_leaf() -> None:
    model = ScratchDecisionTreeClassifier().fit([[0.0], [1.0], [2.0]], [7, 7, 7])

    assert model.root_ is not None and model.root_.is_leaf
    assert model.predict([[-10.0], [10.0]]) == [7, 7]
    assert model.get_depth() == 0
    assert model.get_n_leaves() == 1


def test_one_dimensional_split() -> None:
    features = [[0.0], [1.0], [2.0], [3.0]]
    labels = [0, 0, 1, 1]
    model = ScratchDecisionTreeClassifier(max_depth=1).fit(features, labels)

    assert model.root_ is not None
    assert model.root_.feature_index == 0
    assert model.root_.threshold == pytest.approx(1.5)
    assert model.root_.gain == pytest.approx(0.5)
    assert model.predict(features) == labels
    assert model.score(features, labels) == 1.0


def test_tie_prediction_uses_smallest_label() -> None:
    model = ScratchDecisionTreeClassifier(max_depth=0).fit([[0.0], [1.0]], [5, 2])

    assert model.predict_one([99.0]) == 2


def test_max_depth_is_respected() -> None:
    features = [[float(a), float(b)] for a in (0, 1) for b in (0, 1)]
    labels = [0, 1, 1, 0]

    shallow = ScratchDecisionTreeClassifier(max_depth=1).fit(features, labels)
    deeper = ScratchDecisionTreeClassifier(max_depth=2).fit(features, labels)

    assert shallow.get_depth() == 1
    assert deeper.get_depth() == 2
    assert shallow.score(features, labels) == 0.5
    assert deeper.score(features, labels) == 1.0


def test_min_samples_leaf_is_respected() -> None:
    features = [[float(value)] for value in range(6)]
    labels = [0, 0, 0, 0, 0, 1]

    model = ScratchDecisionTreeClassifier(min_samples_leaf=2).fit(features, labels)

    assert model.root_ is not None and not model.root_.is_leaf
    assert model.root_.threshold == pytest.approx(3.5)
    assert model.root_.left is not None and model.root_.left.samples == 4
    assert model.root_.right is not None and model.root_.right.samples == 2


def test_min_samples_split_is_respected() -> None:
    model = ScratchDecisionTreeClassifier(min_samples_split=5).fit(
        [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
    )

    assert model.root_ is not None and model.root_.is_leaf


def test_min_impurity_decrease_is_respected() -> None:
    model = ScratchDecisionTreeClassifier(min_impurity_decrease=0.51).fit(
        [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
    )

    assert model.root_ is not None and model.root_.is_leaf


def test_constant_features_cannot_split() -> None:
    model = ScratchDecisionTreeClassifier().fit([[1.0], [1.0]], [0, 1])

    assert model.root_ is not None and model.root_.is_leaf
    assert model.get_n_leaves() == 1


def test_wrong_prediction_width_and_nonfinite_value_raise() -> None:
    model = ScratchDecisionTreeClassifier().fit([[0.0, 1.0]], [0])

    with pytest.raises(ValueError, match="wrong number"):
        model.predict_one([0.0])
    with pytest.raises(ValueError, match="finite"):
        model.predict_one([0.0, math.nan])


def test_score_rejects_invalid_targets() -> None:
    model = ScratchDecisionTreeClassifier().fit([[0.0]], [0])

    with pytest.raises(ValueError, match="equal length"):
        model.score([[0.0]], [])
    with pytest.raises(ValueError, match="at least one"):
        model.score([], [])


def test_feature_importances_identify_signal_feature() -> None:
    features = [[float(value), float(value % 2)] for value in range(10)]
    labels = [0] * 5 + [1] * 5
    model = ScratchDecisionTreeClassifier(max_depth=2).fit(features, labels)

    importances = model.feature_importances()

    assert importances[0] == pytest.approx(1.0)
    assert importances[1] == pytest.approx(0.0)
    assert sum(importances) == pytest.approx(1.0)


def test_feature_importances_of_leaf_are_zero() -> None:
    model = ScratchDecisionTreeClassifier().fit([[0.0], [1.0]], [1, 1])

    assert model.feature_importances() == [0.0]


def test_export_text_contains_conditions_and_leaves() -> None:
    model = ScratchDecisionTreeClassifier(max_depth=1).fit(
        [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
    )

    text = model.export_text(["temperature"])

    assert "if temperature <= 1.500000" in text
    assert "predict=0" in text
    assert "predict=1" in text
    assert "counts={" in text


def test_export_text_validates_feature_names() -> None:
    model = ScratchDecisionTreeClassifier().fit([[0.0, 1.0]], [0])

    with pytest.raises(ValueError, match="wrong length"):
        model.export_text(["only_one_name"])


def test_refit_replaces_previous_tree() -> None:
    model = ScratchDecisionTreeClassifier(max_depth=1)
    model.fit([[0.0], [1.0]], [0, 1])
    assert model.predict([[0.0], [1.0]]) == [0, 1]

    model.fit([[0.0], [1.0]], [9, 9])
    assert model.predict([[0.0], [1.0]]) == [9, 9]
    assert model.classes_ == (9,)


def test_generation_is_reproducible_and_stratified() -> None:
    first = generate_samples(100)
    second = generate_samples(100)
    assert first == second

    train, test = split_samples(generate_samples())
    assert len(train) == TRAIN_SIZE
    assert len(test) == 300
    full_rate = sum(sample.label for sample in train + test) / 1200
    train_rate = sum(sample.label for sample in train) / len(train)
    test_rate = sum(sample.label for sample in test) / len(test)
    assert abs(train_rate - full_rate) < 0.002
    assert abs(test_rate - full_rate) < 0.005
    assert SEED == 20260720


def test_generated_csv_files_have_expected_shape() -> None:
    train = load_dataset(TRAIN_PATH)
    test = load_dataset(TEST_PATH)

    assert len(train.features) == 900
    assert len(test.features) == 300
    assert train.feature_names == ["signal_x", "signal_y", "context", "noise"]
    assert set(train.labels) == {0, 1}
    assert all(len(row) == 4 for row in train.features + test.features)


def test_scratch_model_performs_close_to_sklearn() -> None:
    train = load_dataset(TRAIN_PATH)
    test = load_dataset(TEST_PATH)
    own = ScratchDecisionTreeClassifier(max_depth=5, min_samples_leaf=8).fit(
        train.features, train.labels
    )
    reference = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=8,
        random_state=SEED,
    ).fit(train.features, train.labels)

    own_accuracy = own.score(test.features, test.labels)
    reference_accuracy = reference.score(test.features, test.labels)

    assert own_accuracy >= 0.82
    assert abs(own_accuracy - reference_accuracy) <= 0.03
    assert own.get_depth() <= 5
    assert sum(own.feature_importances()) == pytest.approx(1.0)
