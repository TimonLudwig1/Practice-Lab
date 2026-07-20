"""A small decision-tree classifier implemented from first principles."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


Label = int


def gini_impurity(labels: Iterable[Label]) -> float:
    """Return Gini impurity for a collection of class labels."""

    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def weighted_gini(
    left_labels: Sequence[Label], right_labels: Sequence[Label]
) -> float:
    """Return the sample-weighted impurity after a binary split."""

    total = len(left_labels) + len(right_labels)
    if total == 0:
        return 0.0
    return (
        len(left_labels) * gini_impurity(left_labels)
        + len(right_labels) * gini_impurity(right_labels)
    ) / total


@dataclass
class TreeNode:
    """One learned decision node or leaf."""

    prediction: Label
    samples: int
    impurity: float
    class_counts: dict[Label, int]
    feature_index: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        """Return whether this node has no split."""

        return self.feature_index is None


@dataclass(frozen=True)
class _Split:
    feature_index: int
    threshold: float
    gain: float
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]


class ScratchDecisionTreeClassifier:
    """A deterministic numeric decision-tree classifier using Gini impurity."""

    def __init__(
        self,
        *,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be non-negative")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root_: TreeNode | None = None
        self.n_features_: int | None = None
        self.classes_: tuple[Label, ...] = ()
        self._x: tuple[tuple[float, ...], ...] = ()
        self._y: tuple[Label, ...] = ()

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[Label],
    ) -> ScratchDecisionTreeClassifier:
        """Learn a tree recursively and return this estimator."""

        x, y = self._validate_training_data(features, labels)
        self._x = x
        self._y = y
        self.n_features_ = len(x[0])
        self.classes_ = tuple(sorted(set(y)))
        self.root_ = self._build(tuple(range(len(y))), depth=0)
        return self

    def predict_one(self, features: Sequence[float]) -> Label:
        """Predict the class of one feature row."""

        root = self._require_fitted()
        row = self._validate_prediction_row(features)
        node = root
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.prediction

    def predict(self, features: Sequence[Sequence[float]]) -> list[Label]:
        """Predict all rows."""

        self._require_fitted()
        return [self.predict_one(row) for row in features]

    def score(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[Label],
    ) -> float:
        """Return classification accuracy."""

        if len(features) != len(labels):
            raise ValueError("features and labels must have equal length")
        if not labels:
            raise ValueError("score requires at least one sample")
        predictions = self.predict(features)
        return sum(a == b for a, b in zip(predictions, labels)) / len(labels)

    def get_depth(self) -> int:
        """Return maximum learned depth, where a root leaf has depth zero."""

        root = self._require_fitted()

        def depth(node: TreeNode) -> int:
            if node.is_leaf:
                return 0
            assert node.left is not None and node.right is not None
            return 1 + max(depth(node.left), depth(node.right))

        return depth(root)

    def get_n_leaves(self) -> int:
        """Return the number of learned leaves."""

        root = self._require_fitted()

        def count(node: TreeNode) -> int:
            if node.is_leaf:
                return 1
            assert node.left is not None and node.right is not None
            return count(node.left) + count(node.right)

        return count(root)

    def feature_importances(self) -> list[float]:
        """Return normalized total weighted impurity decrease per feature."""

        root = self._require_fitted()
        assert self.n_features_ is not None
        importances = [0.0] * self.n_features_

        def collect(node: TreeNode) -> None:
            if node.is_leaf:
                return
            assert node.feature_index is not None
            assert node.left is not None and node.right is not None
            importances[node.feature_index] += node.samples * node.gain
            collect(node.left)
            collect(node.right)

        collect(root)
        total = sum(importances)
        if total > 0:
            importances = [value / total for value in importances]
        return importances

    def export_text(self, feature_names: Sequence[str] | None = None) -> str:
        """Return a readable preorder representation of the learned tree."""

        root = self._require_fitted()
        assert self.n_features_ is not None
        if feature_names is None:
            names = [f"feature_{index}" for index in range(self.n_features_)]
        else:
            names = list(feature_names)
            if len(names) != self.n_features_:
                raise ValueError("feature_names has the wrong length")

        lines: list[str] = []

        def render(node: TreeNode, prefix: str) -> None:
            counts = ", ".join(
                f"{label}:{count}" for label, count in sorted(node.class_counts.items())
            )
            if node.is_leaf:
                lines.append(
                    f"{prefix}predict={node.prediction} "
                    f"(n={node.samples}, gini={node.impurity:.4f}, counts={{{counts}}})"
                )
                return

            assert node.feature_index is not None and node.threshold is not None
            assert node.left is not None and node.right is not None
            name = names[node.feature_index]
            lines.append(
                f"{prefix}if {name} <= {node.threshold:.6f} "
                f"(n={node.samples}, gain={node.gain:.6f})"
            )
            render(node.left, prefix + "|   L: ")
            lines.append(f"{prefix}else {name} > {node.threshold:.6f}")
            render(node.right, prefix + "|   R: ")

        render(root, "")
        return "\n".join(lines)

    def _build(self, indices: tuple[int, ...], depth: int) -> TreeNode:
        labels = [self._y[index] for index in indices]
        counts = Counter(labels)
        prediction = max(sorted(counts), key=lambda label: counts[label])
        impurity = gini_impurity(labels)
        node = TreeNode(
            prediction=prediction,
            samples=len(indices),
            impurity=impurity,
            class_counts=dict(counts),
        )

        depth_limit_reached = self.max_depth is not None and depth >= self.max_depth
        if (
            impurity == 0.0
            or depth_limit_reached
            or len(indices) < self.min_samples_split
            or len(indices) < 2 * self.min_samples_leaf
        ):
            return node

        split = self._best_split(indices, impurity)
        if split is None or split.gain + 1e-15 < self.min_impurity_decrease:
            return node

        node.feature_index = split.feature_index
        node.threshold = split.threshold
        node.gain = split.gain
        node.left = self._build(split.left_indices, depth + 1)
        node.right = self._build(split.right_indices, depth + 1)
        return node

    def _best_split(
        self, indices: tuple[int, ...], parent_impurity: float
    ) -> _Split | None:
        assert self.n_features_ is not None
        best: _Split | None = None

        for feature_index in range(self.n_features_):
            ordered = sorted(
                (self._x[index][feature_index], self._y[index], index)
                for index in indices
            )
            left_counts: Counter[Label] = Counter()
            right_counts: Counter[Label] = Counter(label for _, label, _ in ordered)

            for position in range(len(ordered) - 1):
                value, label, _ = ordered[position]
                next_value = ordered[position + 1][0]
                left_counts[label] += 1
                right_counts[label] -= 1

                left_size = position + 1
                right_size = len(ordered) - left_size
                if value == next_value:
                    continue
                if (
                    left_size < self.min_samples_leaf
                    or right_size < self.min_samples_leaf
                ):
                    continue

                left_impurity = self._gini_from_counts(left_counts, left_size)
                right_impurity = self._gini_from_counts(right_counts, right_size)
                child_impurity = (
                    left_size * left_impurity + right_size * right_impurity
                ) / len(ordered)
                gain = max(0.0, parent_impurity - child_impurity)
                threshold = value + (next_value - value) / 2.0

                if best is None or gain > best.gain + 1e-15:
                    left_indices = tuple(item[2] for item in ordered[:left_size])
                    right_indices = tuple(item[2] for item in ordered[left_size:])
                    best = _Split(
                        feature_index,
                        threshold,
                        gain,
                        left_indices,
                        right_indices,
                    )
        return best

    @staticmethod
    def _gini_from_counts(counts: Counter[Label], total: int) -> float:
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    @staticmethod
    def _validate_training_data(
        features: Sequence[Sequence[float]], labels: Sequence[Label]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[Label, ...]]:
        if not features:
            raise ValueError("fit requires at least one sample")
        if len(features) != len(labels):
            raise ValueError("features and labels must have equal length")
        width = len(features[0])
        if width == 0:
            raise ValueError("samples must contain at least one feature")

        normalized: list[tuple[float, ...]] = []
        for row in features:
            if len(row) != width:
                raise ValueError("all feature rows must have equal length")
            numeric_row = tuple(float(value) for value in row)
            if not all(math.isfinite(value) for value in numeric_row):
                raise ValueError("features must be finite numbers")
            normalized.append(numeric_row)

        normalized_labels = tuple(labels)
        if not all(isinstance(label, int) for label in normalized_labels):
            raise ValueError("labels must be integers")
        return tuple(normalized), normalized_labels

    def _validate_prediction_row(self, features: Sequence[float]) -> tuple[float, ...]:
        assert self.n_features_ is not None
        if len(features) != self.n_features_:
            raise ValueError("prediction row has the wrong number of features")
        row = tuple(float(value) for value in features)
        if not all(math.isfinite(value) for value in row):
            raise ValueError("features must be finite numbers")
        return row

    def _require_fitted(self) -> TreeNode:
        if self.root_ is None:
            raise RuntimeError("fit must be called before this operation")
        return self.root_
