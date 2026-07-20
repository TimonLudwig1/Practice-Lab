"""Recursive algorithms for binary trees and binary search trees."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass
class Node(Generic[T]):
    """A node with references to at most two children."""

    value: T
    left: Node[T] | None = None
    right: Node[T] | None = None


@dataclass(frozen=True)
class BalanceResult:
    """Height and balance state returned by one postorder traversal."""

    height: int
    is_balanced: bool


def build_bst(values: Iterable[T]) -> Node[T] | None:
    """Build a strict BST and reject duplicate values.

    This helper exists to create examples for the algorithms. Its insertion
    logic is iterative so that the recursive behavior under study remains in
    the individual catalog functions.
    """

    root: Node[T] | None = None
    for value in values:
        if root is None:
            root = Node(value)
            continue

        current = root
        while True:
            if value == current.value:
                raise ValueError(f"duplicate value: {value!r}")
            if value < current.value:  # type: ignore[operator]
                if current.left is None:
                    current.left = Node(value)
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = Node(value)
                    break
                current = current.right
    return root


def tree_height(root: Node[Any] | None) -> int:
    """Return height measured in edges; an empty tree has height -1."""

    if root is None:
        return -1
    return 1 + max(tree_height(root.left), tree_height(root.right))


def balance_result(root: Node[Any] | None) -> BalanceResult:
    """Compute height and height-balance in one postorder traversal.

    A tree is height-balanced when the child heights differ by at most one at
    every node. Combining both questions prevents the repeated height scans of
    the straightforward O(n squared) solution.
    """

    if root is None:
        return BalanceResult(height=-1, is_balanced=True)

    left = balance_result(root.left)
    right = balance_result(root.right)
    height = 1 + max(left.height, right.height)
    balanced = (
        left.is_balanced
        and right.is_balanced
        and abs(left.height - right.height) <= 1
    )
    return BalanceResult(height=height, is_balanced=balanced)


def is_height_balanced(root: Node[Any] | None) -> bool:
    """Return whether every node has child heights differing by at most one."""

    return balance_result(root).is_balanced


def is_valid_bst(root: Node[T] | None) -> bool:
    """Validate the strict BST invariant using inherited open bounds."""

    def validate(
        node: Node[T] | None,
        lower: T | None,
        upper: T | None,
    ) -> bool:
        if node is None:
            return True
        if lower is not None and node.value <= lower:  # type: ignore[operator]
            return False
        if upper is not None and node.value >= upper:  # type: ignore[operator]
            return False
        return validate(node.left, lower, node.value) and validate(
            node.right, node.value, upper
        )

    return validate(root, None, None)


def lowest_common_ancestor(
    root: Node[T] | None,
    first: T,
    second: T,
) -> Node[T] | None:
    """Return the lowest node whose subtree contains both target values.

    The algorithm works for any binary tree, not only a BST. It performs one
    postorder traversal and returns ``None`` if either target is absent. Values
    are assumed to be unique. Equal targets are supported.
    """

    def visit(node: Node[T] | None) -> tuple[Node[T] | None, bool, bool]:
        if node is None:
            return None, False, False

        left_lca, left_first, left_second = visit(node.left)
        right_lca, right_first, right_second = visit(node.right)

        found_first = left_first or right_first or node.value == first
        found_second = left_second or right_second or node.value == second

        if left_lca is not None:
            candidate = left_lca
        elif right_lca is not None:
            candidate = right_lca
        elif found_first and found_second:
            candidate = node
        else:
            candidate = None

        return candidate, found_first, found_second

    candidate, found_first, found_second = visit(root)
    if not (found_first and found_second):
        return None
    return candidate


def serialize(root: Node[Any] | None) -> str:
    """Serialize a tree as JSON containing preorder values and null markers."""

    tokens: list[Any] = []

    def visit(node: Node[Any] | None) -> None:
        if node is None:
            tokens.append(None)
            return
        if node.value is None:
            raise ValueError("None is reserved as the missing-child marker")
        tokens.append(node.value)
        visit(node.left)
        visit(node.right)

    visit(root)
    try:
        return json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("node values must be JSON-serializable") from error


def deserialize(data: str) -> Node[Any] | None:
    """Restore a tree serialized by :func:`serialize`.

    Raises:
        ValueError: If *data* is not a complete preorder token sequence.
    """

    try:
        tokens = json.loads(data)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("invalid serialized tree") from error

    if not isinstance(tokens, list):
        raise ValueError("serialized tree must be a JSON list")

    cursor = 0

    def restore() -> Node[Any] | None:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("serialized tree ended before all children were read")

        value = tokens[cursor]
        cursor += 1
        if value is None:
            return None

        node = Node(value)
        node.left = restore()
        node.right = restore()
        return node

    root = restore()
    if cursor != len(tokens):
        raise ValueError("serialized tree contains trailing tokens")
    return root
