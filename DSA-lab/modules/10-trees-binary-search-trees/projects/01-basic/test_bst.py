"""Tests for the binary search tree and its observable invariants."""

import random

import pytest

from bst import BinarySearchTree


VALUES = [8, 3, 10, 1, 6, 14, 4, 7, 13]


@pytest.fixture
def tree() -> BinarySearchTree[int]:
    return BinarySearchTree(VALUES)


def assert_matches(tree: BinarySearchTree[int], expected: set[int]) -> None:
    """Check the public consequences of the BST and size invariants."""

    assert tree.inorder() == sorted(expected)
    assert list(tree) == sorted(expected)
    assert len(tree) == len(expected)
    assert tree.size == len(expected)
    for value in expected:
        assert value in tree


def test_empty_tree() -> None:
    tree = BinarySearchTree[int]()

    assert len(tree) == 0
    assert not tree
    assert tree.root is None
    assert tree.search(42) is None
    assert tree.inorder() == []
    assert tree.preorder() == []
    assert tree.postorder() == []
    assert tree.level_order() == []
    assert tree.to_ascii() == "<empty>"


@pytest.mark.parametrize("operation", ["minimum", "maximum"])
def test_extreme_of_empty_tree_raises(operation: str) -> None:
    tree = BinarySearchTree[int]()

    with pytest.raises(ValueError, match="empty tree"):
        getattr(tree, operation)()


def test_insert_creates_expected_shape(tree: BinarySearchTree[int]) -> None:
    assert tree.root is not None
    assert tree.root.value == 8
    assert tree.root.left is not None
    assert tree.root.left.value == 3
    assert tree.root.right is not None
    assert tree.root.right.value == 10
    assert_matches(tree, set(VALUES))


def test_insert_reports_change_and_rejects_duplicate() -> None:
    tree = BinarySearchTree[int]()

    assert tree.insert(5)
    assert not tree.insert(5)
    assert_matches(tree, {5})


@pytest.mark.parametrize("value", VALUES)
def test_search_finds_every_inserted_value(
    tree: BinarySearchTree[int], value: int
) -> None:
    node = tree.search(value)

    assert node is not None
    assert node.value == value


@pytest.mark.parametrize("value", [-1, 0, 2, 5, 9, 11, 15])
def test_search_rejects_absent_values(
    tree: BinarySearchTree[int], value: int
) -> None:
    assert tree.search(value) is None
    assert value not in tree


def test_minimum_and_maximum(tree: BinarySearchTree[int]) -> None:
    assert tree.minimum() == 1
    assert tree.maximum() == 14


def test_all_traversals(tree: BinarySearchTree[int]) -> None:
    assert tree.preorder() == [8, 3, 1, 6, 4, 7, 10, 14, 13]
    assert tree.inorder() == [1, 3, 4, 6, 7, 8, 10, 13, 14]
    assert tree.postorder() == [1, 4, 7, 6, 3, 13, 14, 10, 8]
    assert tree.level_order() == [8, 3, 10, 1, 6, 14, 4, 7, 13]


def test_ascii_visualization(tree: BinarySearchTree[int]) -> None:
    assert tree.to_ascii() == "\n".join(
        [
            "ROOT: 8",
            "+-- L: 3",
            "|   +-- L: 1",
            "|   +-- R: 6",
            "|       +-- L: 4",
            "|       +-- R: 7",
            "+-- R: 10",
            "    +-- R: 14",
            "        +-- L: 13",
        ]
    )
    assert str(tree) == tree.to_ascii()


def test_delete_absent_value_does_not_change_tree(
    tree: BinarySearchTree[int],
) -> None:
    before = tree.to_ascii()

    assert not tree.delete(99)
    assert tree.to_ascii() == before
    assert_matches(tree, set(VALUES))


def test_delete_leaf(tree: BinarySearchTree[int]) -> None:
    assert tree.delete(7)
    assert_matches(tree, set(VALUES) - {7})
    assert tree.root is not None
    assert tree.root.left is not None
    assert tree.root.left.right is not None
    assert tree.root.left.right.right is None


def test_delete_node_with_only_left_child() -> None:
    tree = BinarySearchTree([8, 3, 10, 9])

    assert tree.delete(10)
    assert_matches(tree, {3, 8, 9})
    assert tree.root is not None
    assert tree.root.right is not None
    assert tree.root.right.value == 9


def test_delete_node_with_only_right_child() -> None:
    tree = BinarySearchTree([8, 3, 4, 10])

    assert tree.delete(3)
    assert_matches(tree, {4, 8, 10})
    assert tree.root is not None
    assert tree.root.left is not None
    assert tree.root.left.value == 4


def test_delete_node_with_two_children_uses_successor(
    tree: BinarySearchTree[int],
) -> None:
    assert tree.delete(3)
    assert_matches(tree, set(VALUES) - {3})
    assert tree.root is not None
    assert tree.root.left is not None
    assert tree.root.left.value == 4


def test_delete_root_with_two_children(tree: BinarySearchTree[int]) -> None:
    assert tree.delete(8)
    assert_matches(tree, set(VALUES) - {8})
    assert tree.root is not None
    assert tree.root.value == 10


def test_delete_root_until_tree_is_empty() -> None:
    tree = BinarySearchTree([2, 1, 3])

    assert tree.delete(2)
    assert tree.delete(3)
    assert tree.delete(1)
    assert not tree
    assert tree.root is None
    assert len(tree) == 0


def test_strings_are_supported() -> None:
    values = {"pear", "apple", "plum", "banana"}
    tree = BinarySearchTree(values)

    assert_matches(tree, values)
    assert tree.minimum() == "apple"
    assert tree.maximum() == "plum"


def test_seeded_operation_sequence_matches_set_reference() -> None:
    rng = random.Random(20260720)
    tree = BinarySearchTree[int]()
    reference: set[int] = set()

    for _ in range(500):
        value = rng.randrange(-50, 51)
        if rng.random() < 0.58:
            assert tree.insert(value) == (value not in reference)
            reference.add(value)
        else:
            assert tree.delete(value) == (value in reference)
            reference.discard(value)
        assert_matches(tree, reference)
