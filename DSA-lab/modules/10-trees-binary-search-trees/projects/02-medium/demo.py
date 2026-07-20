"""Demonstrate the recursive binary-tree algorithm catalog."""

from tree_algorithms import (
    balance_result,
    build_bst,
    deserialize,
    is_valid_bst,
    lowest_common_ancestor,
    serialize,
)


VALUES = [8, 3, 10, 1, 6, 14, 4, 7, 13]


def main() -> None:
    root = build_bst(VALUES)

    print("TREE ANALYSIS")
    result = balance_result(root)
    print(f"height (edges): {result.height}")
    print(f"height-balanced: {result.is_balanced}")
    print(f"valid BST: {is_valid_bst(root)}")

    print("\nLOWEST COMMON ANCESTORS")
    for first, second in ((1, 7), (4, 7), (13, 14), (1, 99)):
        ancestor = lowest_common_ancestor(root, first, second)
        value = None if ancestor is None else ancestor.value
        print(f"LCA({first}, {second}) = {value}")

    print("\nSERIALIZATION")
    encoded = serialize(root)
    restored = deserialize(encoded)
    print(f"tokens: {encoded}")
    print(f"round trip is structural equal: {restored == root}")


if __name__ == "__main__":
    main()
