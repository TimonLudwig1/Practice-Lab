"""A binary search tree implemented without container shortcuts.

The tree stores every value at most once. Inserting a duplicate is a no-op and
returns ``False``. This explicit policy keeps the search invariant simple:
values in a left subtree are smaller and values in a right subtree are larger
than the node value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass
class Node(Generic[T]):
    """One node in a binary search tree."""

    value: T
    left: Node[T] | None = None
    right: Node[T] | None = None


class BinarySearchTree(Generic[T]):
    """A mutable binary search tree with unique, comparable values."""

    def __init__(self, values: Iterable[T] | None = None) -> None:
        self._root: Node[T] | None = None
        self._size = 0
        if values is not None:
            for value in values:
                self.insert(value)

    @property
    def root(self) -> Node[T] | None:
        """Return the root node for inspection."""

        return self._root

    @property
    def size(self) -> int:
        """Return the number of stored values."""

        return self._size

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._root is not None

    def __contains__(self, value: object) -> bool:
        return self.search(value) is not None  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[T]:
        yield from self._inorder(self._root)

    def insert(self, value: T) -> bool:
        """Insert *value* and return whether the tree changed.

        The iterative walk needs O(h) time and O(1) auxiliary space, where h
        is the current tree height.
        """

        if self._root is None:
            self._root = Node(value)
            self._size = 1
            return True

        current = self._root
        while True:
            if value == current.value:
                return False
            if value < current.value:  # type: ignore[operator]
                if current.left is None:
                    current.left = Node(value)
                    self._size += 1
                    return True
                current = current.left
            else:
                if current.right is None:
                    current.right = Node(value)
                    self._size += 1
                    return True
                current = current.right

    def search(self, value: T) -> Node[T] | None:
        """Return the node holding *value*, or ``None`` if it is absent."""

        current = self._root
        while current is not None:
            if value == current.value:
                return current
            if value < current.value:  # type: ignore[operator]
                current = current.left
            else:
                current = current.right
        return None

    def minimum(self) -> T:
        """Return the smallest value.

        Raises:
            ValueError: If the tree is empty.
        """

        if self._root is None:
            raise ValueError("minimum is undefined for an empty tree")
        return self._minimum_node(self._root).value

    def maximum(self) -> T:
        """Return the largest value.

        Raises:
            ValueError: If the tree is empty.
        """

        if self._root is None:
            raise ValueError("maximum is undefined for an empty tree")
        current = self._root
        while current.right is not None:
            current = current.right
        return current.value

    def delete(self, value: T) -> bool:
        """Delete *value* and return whether the tree changed.

        Leaf nodes are removed directly. A node with one child is replaced by
        that child. A node with two children receives its inorder successor,
        after which that successor is removed from the right subtree.
        """

        self._root, deleted = self._delete_node(self._root, value)
        if deleted:
            self._size -= 1
        return deleted

    def inorder(self) -> list[T]:
        """Return values in left-root-right order (sorted for a valid BST)."""

        return list(self._inorder(self._root))

    def preorder(self) -> list[T]:
        """Return values in root-left-right order."""

        result: list[T] = []

        def visit(node: Node[T] | None) -> None:
            if node is None:
                return
            result.append(node.value)
            visit(node.left)
            visit(node.right)

        visit(self._root)
        return result

    def postorder(self) -> list[T]:
        """Return values in left-right-root order."""

        result: list[T] = []

        def visit(node: Node[T] | None) -> None:
            if node is None:
                return
            visit(node.left)
            visit(node.right)
            result.append(node.value)

        visit(self._root)
        return result

    def level_order(self) -> list[T]:
        """Return values breadth-first, level by level.

        A cursor advances through the list so that removing its first item is
        unnecessary. Each node therefore enters and leaves the queue once.
        """

        if self._root is None:
            return []

        result: list[T] = []
        queue = [self._root]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            result.append(node.value)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        return result

    def to_ascii(self) -> str:
        """Return a deterministic ASCII representation of the tree."""

        if self._root is None:
            return "<empty>"

        lines = [f"ROOT: {self._root.value}"]

        def add_subtree(node: Node[T], prefix: str) -> None:
            children = [
                (label, child)
                for label, child in (("L", node.left), ("R", node.right))
                if child is not None
            ]
            for index, (label, child) in enumerate(children):
                is_last = index == len(children) - 1
                lines.append(f"{prefix}+-- {label}: {child.value}")
                continuation = "    " if is_last else "|   "
                add_subtree(child, prefix + continuation)

        add_subtree(self._root, "")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_ascii()

    @classmethod
    def _delete_node(
        cls, node: Node[T] | None, value: T
    ) -> tuple[Node[T] | None, bool]:
        if node is None:
            return None, False

        if value < node.value:  # type: ignore[operator]
            node.left, deleted = cls._delete_node(node.left, value)
            return node, deleted
        if value > node.value:  # type: ignore[operator]
            node.right, deleted = cls._delete_node(node.right, value)
            return node, deleted

        if node.left is None:
            return node.right, True
        if node.right is None:
            return node.left, True

        successor = cls._minimum_node(node.right)
        node.value = successor.value
        node.right, _ = cls._delete_node(node.right, successor.value)
        return node, True

    @staticmethod
    def _minimum_node(node: Node[T]) -> Node[T]:
        current = node
        while current.left is not None:
            current = current.left
        return current

    @classmethod
    def _inorder(cls, node: Node[T] | None) -> Iterator[T]:
        if node is None:
            return
        yield from cls._inorder(node.left)
        yield node.value
        yield from cls._inorder(node.right)
