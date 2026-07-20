"""Run a small, traceable binary search tree demonstration."""

from bst import BinarySearchTree


VALUES = [8, 3, 10, 1, 6, 14, 4, 7, 13]


def main() -> None:
    tree = BinarySearchTree[int]()

    print("INSERT")
    for value in VALUES:
        tree.insert(value)
        print(f"insert {value:>2}: inorder = {tree.inorder()}")

    print("\nTREE")
    print(tree)

    print("\nTRAVERSALS")
    print("preorder:   ", tree.preorder())
    print("inorder:    ", tree.inorder())
    print("postorder:  ", tree.postorder())
    print("level-order:", tree.level_order())

    print("\nSEARCH AND EXTREMES")
    for value in (6, 5):
        print(f"search {value}: {tree.search(value) is not None}")
    print(f"minimum: {tree.minimum()}, maximum: {tree.maximum()}")

    print("\nDELETE CASES")
    for value, case in ((7, "leaf"), (14, "one child"), (3, "two children")):
        tree.delete(value)
        print(f"after deleting {value} ({case}):")
        print(tree)


if __name__ == "__main__":
    main()
