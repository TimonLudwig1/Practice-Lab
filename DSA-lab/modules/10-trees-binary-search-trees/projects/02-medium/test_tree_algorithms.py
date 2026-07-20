"""Tests for all recursive algorithms in the tree catalog."""

import json

import pytest

from tree_algorithms import (
    BalanceResult,
    Node,
    balance_result,
    build_bst,
    deserialize,
    is_height_balanced,
    is_valid_bst,
    lowest_common_ancestor,
    serialize,
    tree_height,
)


VALUES = [8, 3, 10, 1, 6, 14, 4, 7, 13]


@pytest.fixture
def root() -> Node[int]:
    result = build_bst(VALUES)
    assert result is not None
    return result


def test_build_empty_tree() -> None:
    assert build_bst([]) is None


def test_build_bst_shape(root: Node[int]) -> None:
    assert root.value == 8
    assert root.left is not None and root.left.value == 3
    assert root.right is not None and root.right.value == 10
    assert root.left.right is not None and root.left.right.value == 6
    assert root.right.right is not None and root.right.right.value == 14


def test_build_bst_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate value"):
        build_bst([4, 2, 6, 2])


@pytest.mark.parametrize(
    ("tree", "expected"),
    [
        (None, -1),
        (Node(1), 0),
        (Node(1, Node(2), Node(3)), 1),
        (Node(1, Node(2, Node(3))), 2),
    ],
)
def test_tree_height(tree: Node[int] | None, expected: int) -> None:
    assert tree_height(tree) == expected


def test_sample_tree_height(root: Node[int]) -> None:
    assert tree_height(root) == 3


def test_empty_tree_balance_result() -> None:
    assert balance_result(None) == BalanceResult(height=-1, is_balanced=True)
    assert is_height_balanced(None)


def test_balanced_tree() -> None:
    root = build_bst([4, 2, 6, 1, 3, 5, 7])

    assert balance_result(root) == BalanceResult(height=2, is_balanced=True)
    assert is_height_balanced(root)


def test_imbalance_at_root() -> None:
    root = build_bst([1, 2, 3])

    assert balance_result(root) == BalanceResult(height=2, is_balanced=False)
    assert not is_height_balanced(root)


def test_deep_imbalance_is_propagated() -> None:
    root = Node(
        8,
        Node(4, Node(2, Node(1, Node(0))), Node(6)),
        Node(12, Node(10, Node(9)), Node(14)),
    )

    assert abs(tree_height(root.left) - tree_height(root.right)) == 1
    assert not is_height_balanced(root)


@pytest.mark.parametrize(
    "tree",
    [
        None,
        Node(5),
        Node(5, Node(2), Node(8)),
        Node(0, Node(-10), Node(10)),
    ],
)
def test_valid_bsts(tree: Node[int] | None) -> None:
    assert is_valid_bst(tree)


def test_built_sample_is_valid_bst(root: Node[int]) -> None:
    assert is_valid_bst(root)


@pytest.mark.parametrize(
    "tree",
    [
        Node(5, Node(6), None),
        Node(5, None, Node(4)),
        Node(5, Node(5), None),
        Node(5, None, Node(5)),
    ],
)
def test_local_bst_violations(tree: Node[int]) -> None:
    assert not is_valid_bst(tree)


def test_deep_left_subtree_violation() -> None:
    root = Node(10, Node(5, None, Node(12)), Node(15))

    assert not is_valid_bst(root)


def test_deep_right_subtree_violation() -> None:
    root = Node(10, Node(5), Node(15, Node(8), None))

    assert not is_valid_bst(root)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (1, 7, 3),
        (4, 7, 6),
        (13, 14, 14),
        (1, 14, 8),
        (3, 7, 3),
        (6, 6, 6),
    ],
)
def test_lowest_common_ancestor(
    root: Node[int], first: int, second: int, expected: int
) -> None:
    ancestor = lowest_common_ancestor(root, first, second)

    assert ancestor is not None
    assert ancestor.value == expected


@pytest.mark.parametrize(("first", "second"), [(1, 99), (99, 1), (98, 99)])
def test_lca_returns_none_if_target_is_missing(
    root: Node[int], first: int, second: int
) -> None:
    assert lowest_common_ancestor(root, first, second) is None


def test_lca_of_empty_tree() -> None:
    assert lowest_common_ancestor(None, 1, 2) is None


def test_lca_works_for_non_bst() -> None:
    root = Node("a", Node("z", Node("x"), Node("y")), Node("b"))

    ancestor = lowest_common_ancestor(root, "x", "y")

    assert ancestor is not None
    assert ancestor.value == "z"


def test_serialize_empty_tree() -> None:
    assert serialize(None) == "[null]"
    assert deserialize("[null]") is None


def test_serialize_uses_preorder_and_null_markers() -> None:
    root = Node(2, Node(1), Node(3))

    assert serialize(root) == "[2,1,null,null,3,null,null]"


def test_round_trip_preserves_sample_tree(root: Node[int]) -> None:
    restored = deserialize(serialize(root))

    assert restored == root
    assert restored is not root


@pytest.mark.parametrize(
    "tree",
    [
        Node("root,with,commas", Node("links"), Node("rechts")),
        Node(-3.5, Node(-9.25), Node(0.0)),
        Node(True, Node(False), None),
    ],
)
def test_round_trip_preserves_json_values(tree: Node[object]) -> None:
    assert deserialize(serialize(tree)) == tree


def test_round_trip_of_degenerate_tree() -> None:
    root = build_bst(range(25))

    assert deserialize(serialize(root)) == root


def test_serialize_rejects_none_as_node_value() -> None:
    with pytest.raises(ValueError, match="reserved"):
        serialize(Node(None))


def test_serialize_rejects_non_json_value() -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        serialize(Node({1, 2, 3}))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("not json", "invalid serialized tree"),
        ("null", "JSON list"),
        ("[]", "ended before"),
        ("[1,null]", "ended before"),
        ("[null,null]", "trailing tokens"),
    ],
)
def test_deserialize_rejects_malformed_data(data: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        deserialize(data)


def test_serialized_output_is_valid_json(root: Node[int]) -> None:
    tokens = json.loads(serialize(root))

    assert isinstance(tokens, list)
    assert tokens[0] == 8
    assert len(tokens) == 2 * len(VALUES) + 1
