# Trees and Binary Search Trees

An array describes a sequence, a hash map an assignment, and a graph of arbitrary
relationships. A tree, on the other hand, describes a **hierarchy**: Each element except
the root has exactly one parent node, and from the root, exactly one path leads to each
node.

This chapter first builds up a precise vocabulary, derives from it the four most
important traversals and then develops a Binary Search Tree (BST). The focus is not on
memorized code, but on the invariants that make search, insertion and deletion correct.

## 1. Learning objectives

After this chapter you can:

- Safely use tree terms on a concrete example,
- explain the preorder, inorder and postorder recursively and level order iteratively,
- predict the order of each traversal;
- formulate the BST search invariant and receive it during each operation;
- distinguish the three deletion cases and implement them correctly,
- justify runtimes over tree height rather than blanket over `n`,
- explain why sorted insertion degenerates a BST,
- classify the idea of AVL and Red-Black trees,
- Detect tree structures in data science and data format tasks.

## 2. Why Hierarchies Are Trees

Typical hierarchies are:

- a file system with folders and files,
- the structure of a JSON or XML document;
- an organigram;
- an expression tree of a compiler,
- a decision tree with questions and predictions,
- a hierarchical index in a database.

A tree is suitable if a relationship such as "contains", "consists of" or "decides
further" runs directed from top to bottom and each element has only one direct parent
node. Cross-links or several parents generally make a graph out of the structure.

### 2.1 Our Common Example Tree

All traversals use the same binary tree:

```text
          A
        /   \
       B     C
      / \     \
     D   E     F
        / \   /
       G   H I
```

A **binary tree** allows a maximum of two children per node. It is not automatically a
binary search tree: the letters above do not fulfill a search order, but serve only for
traversal.

```python
from dataclasses import dataclass


@dataclass
class TreeNode:
    """A node of a binary tree."""

    value: str
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def build_example_tree() -> TreeNode:
    """Build the example tree used throughout the chapter."""
    return TreeNode(
        "A",
        left=TreeNode(
            "B",
            left=TreeNode("D"),
            right=TreeNode("E", TreeNode("G"), TreeNode("H")),
        ),
        right=TreeNode("C", right=TreeNode("F", TreeNode("I"))),
    )


root = build_example_tree()
assert root.value == "A"
assert root.left.right.left.value == "G"
```

## 3. Terminology without ambiguity

For the example tree apply:

| Definition | Meaning | Example |
|---|---|---|
| Root | Nods without parent nodes | `A` |
| Parent nodes | Direct predecessors | `B` is a parent node of `D` and `E` |
| Child | Direct successor | `G` is a child of `E` |
| Siblings | Children of the same parent node | `G` and `H` |
| Sheet | nodes without children | `D`, `G`, `H`, `I` |
| inner node | nodes with at least one child | `A`, `B`, `C`, `E`, `F` |
| Subtree | Nodes with all descendants | Partial tree with root `E` contains `E,G,H` |
| Depth | Number of edges from root to node | Depth of `H` is 3 |
| Height of a node | longest descent in edges to a leaf | Height of `B` is 2 |
| Height of the tree | Height of its root | The height of the example tree is 3 |

This definition uses edges: A single node has height `0`, an empty tree height `-1`.
Some sources count nodes instead and call the height of a single node `1`. Both
conventions are possible; within a solution, the chosen definition must remain
consistent.

### 3.1 Depth runs down, height comes from below

The depth is passed as a parameter on the descent. The height is built from the results
of the children. This is precisely the recursive nature of a tree: every partial tree is
itself a tree again.

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def height(node: Node | None) -> int:
    """Return height measured in edges; an empty tree has height -1."""
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))


def depths(node: Node | None, depth: int = 0) -> dict[str, int]:
    """Map each node value to its depth."""
    if node is None:
        return {}
    result = {node.value: depth}
    result.update(depths(node.left, depth + 1))
    result.update(depths(node.right, depth + 1))
    return result


tree = Node("A", Node("B", Node("D"), Node("E")), Node("C"))
assert height(tree) == 2
assert depths(tree) == {"A": 0, "B": 1, "D": 2, "E": 2, "C": 1}
```

For `n` nodes, a full elevation calculation visits each node once: time `O(n)`. The
recursion memory is `O(h)`, where `h` is the tree height.

## 4. traversal: When is the node processed?

"traversal" means systematically visiting every node. For the three recursive depth
searches, only the time of processing changes:

- **Preorder:** node, left, right.
- **Inorder:** left, node, right.
- **Postorder:** left, right, node.

Level Order, on the other hand, visits level by level and uses a queue.

### 4.1 Preorder: nodes in front of its subtrees

Preorder is suitable if the parent node has to be processed before its children, for
example when copying or serializing a tree structure.

Simulation at the example tree:

| Step | Visit | Reasons |
|---:|---|---|
| 1 | A | Root First |
| 2 | B | left subtree of A |
| 3 | D | left subtree of B |
| 4 | E | right subtree of B |
| 5 | G | left subtree of E |
| 6 | H | right subtree of E |
| 7 | C | right subtree of A |
| 8 | F | right subtree of C |
| 9 | I | left subtree of F |

Result: `A B D E G H C F I`

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def preorder(node: Node | None) -> list[str]:
    """Return root-left-right traversal."""
    if node is None:
        return []
    return [node.value] + preorder(node.left) + preorder(node.right)


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert preorder(tree) == list("ABDEGHCFI")
```

### 4.2 Inorder: nodes between its subtrees

Inorder is especially important for a BST because it delivers its keys sorted. For any
binary tree, it is only a defined order of visits.

Simulation:

1. From `A` to the left as far as possible: `D`.
2. Back to `B`, then in its right subtree: `G E H`.
3. Now only `A`.
4. In the right subtree follow `C`, then `I F`.

Result: `D B G E H A C I F`

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def inorder(node: Node | None) -> list[str]:
    """Return left-root-right traversal."""
    if node is None:
        return []
    return inorder(node.left) + [node.value] + inorder(node.right)


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert inorder(tree) == list("DBGEHACIF")
```

### 4.3 Postorder: nodes according to its subtrees

Postorder matches tasks where children must be ready before their parent node: sum up
directory sizes, evaluate printout trees, or delete a tree from bottom to top.

Simulation:

- Subtree `B`: `D`, then `G H E`, last `B`.
- Subtree `C`: `I F`, last `C`.
- At the very end, the root `A` follows.

Result: `D G H E B I F C A`

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def postorder(node: Node | None) -> list[str]:
    """Return left-right-root traversal."""
    if node is None:
        return []
    return postorder(node.left) + postorder(node.right) + [node.value]


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert postorder(tree) == list("DGHEBIFCA")
```

### 4.4 Level Order: Level by level with queue

Level Order is a width search (BFS). The queue always contains the already discovered,
but not yet processed nodes. When removing a node, its children are added to the rear.

| Step | taken | then in the queue |
|---:|---|---|
| 1 | A | B, C |
| 2 | B | C, D, E |
| 3 | C | D, E, F |
| 4 | D | E, F |
| 5 | E | F, G, H |
| 6 | F | G, H, I |
| 7 | G | H, I |
| 8 | H | I |
| 9 | I | empty |

Result: `A B C D E F G H I`

```python
from collections import deque
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def level_order(root: Node | None) -> list[str]:
    """Return a breadth-first traversal."""
    if root is None:
        return []
    queue = deque([root])
    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node.value)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return result


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert level_order(tree) == list("ABCDEFGHI")
```

All four traversals cost `O(n)` time. The additional memory is different:

- Deep search: `O(h)` Recursion stack, in the worst case `O(n)`.
- Level Order: `O(w)` Queue, where `w` is the maximum width.
- The result list itself requires for all variants `O(n)` and is often viewed separately
  when specifying the auxiliary memory.

### 4.5 Recursion and explicit stack are two representations of the same idea

The Python call stack remembers in recursive traversal which subtrees are still open. An
explicit stack makes this state visible. For Preorder, the right child is first placed,
so that the left child is processed first because of LIFO.

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def iterative_preorder(root: Node | None) -> list[str]:
    """Return preorder traversal without recursion."""
    if root is None:
        return []
    stack = [root]
    result: list[str] = []
    while stack:
        node = stack.pop()
        result.append(node.value)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result


tree = Node("A", Node("B", Node("D"), Node("E")), Node("C"))
assert iterative_preorder(tree) == ["A", "B", "D", "E", "C"]
```

## 5. Binary Search Tree: Order as Invariant

A Binary Search Tree is a binary tree with an additional rule. For each node with key
`k`:

- All keys in the left subtree are smaller than `k`.
- All keys in the right subtree are larger than `k`.
- Both subtrees are BSTs again.

In this chapter, double keys are not allowed. Other contracts are possible, for example,
to place duplicates consistently on the right or to store a counter in the node. Without
a specified duplicate rule, the invariant is incomplete.

Example:

```text
             8
           /   \
          3     10
         / \      \
        1   6      14
           / \    /
          4   7  13
```

The inorder sequence is `1, 3, 4, 6, 7, 8, 10, 13, 14` and therefore strictly sorted.

### 5.1 Search: Excluding a subtree in each step

Wanted `7`:

1. `7 < 8`, so the entire right subtree can be excluded.
2. `7 > 3`, so the search continues to the right.
3. `7 > 6`, that's right again.
4. `7 == 7`: found.

The loop invariant is: **If the key exists, it is in the current subtree.** The
comparison gets this statement because the BST order excludes a complete page.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def search(root: BSTNode | None, key: int) -> BSTNode | None:
    """Find key iteratively, or return None."""
    current = root
    while current is not None:
        if key == current.key:
            return current
        current = current.left if key < current.key else current.right
    return None


root = BSTNode(8, BSTNode(3, BSTNode(1), BSTNode(6)), BSTNode(10))
assert search(root, 6).key == 6
assert search(root, 9) is None
```

Search costs `O(h)`, not basically `O(log n)`. Only a balanced tree guarantees `h =
O(log n)`. In a degenerated tree may be `h = n - 1`.

### 5.2 Insert: search to the free position

When inserting `5` into the example BST, the path is:

```text
5 < 8 -> left
5 > 3 -> right
5 < 6 -> left
Free   -> 5 becomes left child of 6
```

The method returns the root of the possibly modified subtree. This also makes the same
code work when inserting into an empty tree.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def insert(node: BSTNode | None, key: int) -> BSTNode:
    """Insert a unique key and return the subtree root."""
    if node is None:
        return BSTNode(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        raise ValueError(f"duplicate key: {key}")
    return node


def inorder(node: BSTNode | None) -> list[int]:
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


root = None
for value in [8, 3, 10, 1, 6, 14, 4, 7, 13]:
    root = insert(root, value)
assert inorder(root) == [1, 3, 4, 6, 7, 8, 10, 13, 14]
```

### 5.3 Minimum and maximum

Because of the searchin variant, the minimum is left and the maximum is right. Both
operations cost `O(h)` and do not require complete traversal.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def minimum(node: BSTNode) -> BSTNode:
    """Return the node with the smallest key in a non-empty subtree."""
    current = node
    while current.left is not None:
        current = current.left
    return current


def maximum(node: BSTNode) -> BSTNode:
    """Return the node with the largest key in a non-empty subtree."""
    current = node
    while current.right is not None:
        current = current.right
    return current


root = BSTNode(8, BSTNode(3, BSTNode(1), BSTNode(6)), BSTNode(10))
assert minimum(root).key == 1
assert maximum(root).key == 10
```

## 6. Delete: the three structural cases

Deleting is more difficult than inserting because the node disappears from the structure
and the BST invariant must still be preserved. First, the key is found as in the search.
Then there are exactly three cases.

### 6.1 Case 1: Delete sheet

When deleting `1`, the node has no children. The parent pointer can simply be set to
`None`.

```text
    3                 3
   / \      ->         \
  1   6                 6
```

### 6.2 Case 2: Delete nodes with exactly one child

When deleting `10` in the large example, the node has only the right child `14`. The
parent node is connected directly to this child.

```text
  8                   8
   \                   \
   10       ->         14
     \                 /
     14               13
    /
   13
```

The entire preserved subtree remains in the correct size relation.

### 6.3 Case 3: Delete nodes with two children

When deleting `3`, both edges must not be removed. A suitable replacement position is
the **Inorder successor**, i.e. the minimum of the right subtree. Here's the `4`.

1. Copy the key `4` to the previous node `3`.
2. Delete the original `4` recursively in the right subtree.
3. This second deletion process is case 1 or case 2 because a minimum cannot have a left
   child.

Alternatively, the maximum of the left subtree can be used as an inorder forerunner.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def insert(node: BSTNode | None, key: int) -> BSTNode:
    if node is None:
        return BSTNode(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    return node


def minimum(node: BSTNode) -> BSTNode:
    while node.left is not None:
        node = node.left
    return node


def delete(node: BSTNode | None, key: int) -> BSTNode | None:
    """Delete key if present and return the subtree root."""
    if node is None:
        return None
    if key < node.key:
        node.left = delete(node.left, key)
    elif key > node.key:
        node.right = delete(node.right, key)
    elif node.left is None:
        return node.right
    elif node.right is None:
        return node.left
    else:
        successor = minimum(node.right)
        node.key = successor.key
        node.right = delete(node.right, successor.key)
    return node


def inorder(node: BSTNode | None) -> list[int]:
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


root = None
for value in [8, 3, 10, 1, 6, 14, 4, 7, 13]:
    root = insert(root, value)

root = delete(root, 1)   # leaf
root = delete(root, 10)  # one child
root = delete(root, 3)   # two children
assert inorder(root) == [4, 6, 7, 8, 13, 14]
```

### 6.4 Why the invariant is retained after case 3

The successor is the smallest element in the right subtree. Therefore:

- It is larger than all elements in the left subtree of the deleted node.
- No other element in the right subtree is smaller than it.
- After removing its old position, the key only exists once.

The costs are search `O(h)` plus a descent to the successor of at most `O(h)`, together
continue `O(h)`.

### 6.5 Common Delete Errors

- Do not return the new root value to the parent node.
- Copy the successor for two children, but don't delete it in the old place.
- Only compare the immediate children and ignore deeper descendants.
- Don't define the duplicate rule.
- When deleting the root, assume that the external reference remains unchanged.

## 7. Validate a BST correctly

The local test `left.key < node.key < right.key` is not enough. In the following tree
`12` is larger than its parent node `6`, but lies in the left subtree of `8` and thus
violates a limit set above:

```text
      8
     / \
    6   10
     \
     12   <- invalid
```

Each recursive call must therefore carry the entire permissible interval.

```python
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    left: "Node | None" = None
    right: "Node | None" = None


def is_valid_bst(
    node: Node | None,
    lower: int | None = None,
    upper: int | None = None,
) -> bool:
    """Validate a strict BST using inherited bounds."""
    if node is None:
        return True
    if lower is not None and node.key <= lower:
        return False
    if upper is not None and node.key >= upper:
        return False
    return (
        is_valid_bst(node.left, lower, node.key)
        and is_valid_bst(node.right, node.key, upper)
    )


valid = Node(8, Node(3, Node(1), Node(6)), Node(10))
invalid = Node(8, Node(6, right=Node(12)), Node(10))
assert is_valid_bst(valid)
assert not is_valid_bst(invalid)
```

The barriers are open because duplicates are excluded. For floating point or other key
types, do not require artificial values such as `float("inf")`; optional barriers make
the contract explicit.

## 8. runtime depends on the height

| Operation | balanced, `h = O(log n)` | degenerated, `h = O(n)` |
|---|---:|---:|
| Search | `O(log n)` | `O(n)` |
| Insert | `O(log n)` | `O(n)` |
| Delete | `O(log n)` | `O(n)` |
| Min/Max | `O(log n)` | `O(n)` |
| Complete traversal | `O(n)` | `O(n)` |

The traversal remains linear because it visits all nodes anyway. Only the path-based
operations benefit from low altitude.

### 8.1 Degeneration to the Linked-List tree

If `1, 2, 3, 4, 5, 6, 7` is inserted in this order into an ordinary BST, each node has
only one right child:

```text
1
 \
  2
   \
    3
     \
      ... 7
```

The height is `n - 1`; search and insert behave like in a chained list.

```python
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    right: "Node | None" = None


def build_sorted_chain(size: int) -> Node | None:
    """Build the shape produced by increasing BST insertions."""
    if size <= 0:
        return None
    root = Node(1)
    current = root
    for key in range(2, size + 1):
        current.right = Node(key)
        current = current.right
    return root


def height(node: Node | None) -> int:
    if node is None:
        return -1
    return 1 + height(node.right)


assert height(build_sorted_chain(7)) == 6
```

Random insertion order often produces usable but not guaranteed balanced trees. A
guaranteed barrier requires an additional balance invariant or another structure, such
as recursive selection of the center of a sorted array.

## 9. Balancing: rotations get the search order

A **rotation** changes local edges without changing the inorder order. If the left turn
is `x` its right child will rise `y`:

```text
    x                  y
   / \                / \
  A   y      ->       x   C
     / \            / \
    B   C          A   B
```

Before and after is Inorder `A, x, B, y, C`. Therefore, the BST invariant is retained.
But the height may change.

```python
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    left: "Node | None" = None
    right: "Node | None" = None


def rotate_left(x: Node) -> Node:
    """Rotate a subtree left and return its new root."""
    if x.right is None:
        raise ValueError("left rotation requires a right child")
    y = x.right
    x.right = y.left
    y.left = x
    return y


def inorder(node: Node | None) -> list[int]:
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


root = Node(10, Node(5), Node(20, Node(15), Node(30)))
before = inorder(root)
root = rotate_left(root)
assert root.key == 20
assert inorder(root) == before == [5, 10, 15, 20, 30]
```

### 9.1 AVL trees conceptually

An AVL tree stores or calculates the balance factor for each node

`height(left subtree) - height(right subtree)`.

It must be in `{-1, 0, 1}`. After insert or delete, heights are updated on the path to
the root and an imbalance is repaired by a simple or double rotation. The strict balance
delivers very fast searches, but can require more rotations in case of changes.

### 9.2 Red black trees conceptually

A red-black tree adds a color to each node and enforces rules on red neighborhoods and
the number of black nodes on root-blade thread. The rules guarantee a logarithmic
height, but allow more room for manoeuvre than AVL trees. This often makes changes
cheaper. The complete fix-up cases are deliberately not the subject of this module; the
key is the idea:

> Additional local metadata and invariants prevent linear height.

Python does not provide a built-in tree map type. In other languages, ordered maps and
sets are often implemented with balanced search trees. Which variant is used internally
is an implementation decision.

## 10. Trees in Data Science everyday life

### 10.1 Decision trees

An inner node checks a condition like `feature_2 <= 1.7`. The left or right path leads
to the next condition; a sheet contains the prediction. A single prediction costs
proportionally to the depth of the reached leaf.

```python
from dataclasses import dataclass


@dataclass
class DecisionNode:
    feature: int | None = None
    threshold: float | None = None
    prediction: int | None = None
    left: "DecisionNode | None" = None
    right: "DecisionNode | None" = None


def predict(root: DecisionNode, row: list[float]) -> int:
    """Follow decision nodes until a prediction leaf is reached."""
    node = root
    while node.prediction is None:
        if node.feature is None or node.threshold is None:
            raise ValueError("incomplete decision node")
        child = node.left if row[node.feature] <= node.threshold else node.right
        if child is None:
            raise ValueError("missing decision branch")
        node = child
    return node.prediction


model = DecisionNode(
    feature=0,
    threshold=2.5,
    left=DecisionNode(prediction=0),
    right=DecisionNode(prediction=1),
)
assert predict(model, [1.2]) == 0
assert predict(model, [4.0]) == 1
```

A ML decision tree is usually **no BST**. Its order is based on feature waves per node,
not on a global key order. But he shares the recursive structure and many traversal
techniques.

### 10.2 Hierarchical data and JSON

Nested dictionaries and lists form a tree structure as long as objects are not
referenced multiple times. A deep search can collect all leaves with their path.

```python
def flatten_leaves(value: object, path: tuple[str, ...] = ()) -> dict[str, object]:
    """Flatten dictionary leaves into dotted paths."""
    if not isinstance(value, dict):
        return {".".join(path): value}
    result: dict[str, object] = {}
    for key, child in value.items():
        result.update(flatten_leaves(child, path + (str(key),)))
    return result


record = {
    "model": {"name": "tree", "params": {"depth": 4}},
    "score": 0.91,
}
assert flatten_leaves(record) == {
    "model.name": "tree",
    "model.params.depth": 4,
    "score": 0.91,
}
```

For any Python objects, cycles or jointly referenced sub-objects can occur. Then a pure
tree traversal is not enough; a set of `visited` as with graphs is necessary.

### 10.3 Hierarchical indices

B-trees and B+ trees store multiple keys per node and are optimized for block access to
disks. They are not binary search trees, but use the same basic idea: A comparison
excludes large data ranges. For DataFrames, multi-indices are also conceptually
hierarchical, even if the concrete library can use other internal structures.

### 10.4 File systems and aggregations

Folders are internal nodes, files sheets. Size calculation is postorder: First the sizes
of all children are determined, then their sum is assigned to the folder. A tree view is
Preorder: First output the folder, then display its contents indented.

## 11. Recognize patterns and choose the right traversal

| Task Formulation | Matching pattern | Why |
|---|---|---|
| "Processing Parents Before Children" | Preorder | Node action comes first |
| "BST assorted output" | Inorder | Search invariant creates order |
| "Aggregating results from children" | Postal order | Children's results are available first |
| "Next level / minimum depth" | Level Order | BFS processed by distance |
| "Find a key in BST" | directional descent | one page is discarded per comparison |
| "Is this a BST globally?" | DFS with barriers | Ancestor conditions must be carried along |

### Decision-making issues

1. Does every node really have to be visited, or does an order allow the exclusion of a
   subtree?
2. Does a parent node have to be processed before, between or after its children?
3. Is a deep-based statement required? Then level order is often natural.
4. Can the tree be very deep? Then in Python a recursion limit threatens and an explicit
   stack can be more robust.
5. Does a condition refer only to direct children or to all descendants?

## 12. Typical Thinking Mistakes

- **Binary tree and BST confuse:** Up to two children do not yet mean a key order.
- **`O(log n)` Claim without balance:** Cost of BST operations `O(h)`.
- **Transverse depth and height:** Depth comes from the root, height from the deepest
  leaf.
- ** Implement Level Order with a Stack:** BFS needs FIFO, i.e. a queue.
- **Inorder always consider "sorted":** This applies only to the BST invariant.
- **BST only validate locally:** Limits of all ancestors must apply.
- **Delete case with two children to treat incompletely:** Replacement key must be
  removed at its old position.
- ** Forget the empty tree:** Many functions require `None` as a base case.
- **Recursion for free:** The call stack requires `O(h)` memory.

## 13. Compact Correctness Arguments

### traversal

Induction over the subtree size:

- Base: An empty tree creates an empty sequence.
- Step: The recursive calls traverse left and right subtrees correctly. Inserting the
  root at the defined position results in exactly preorder, inorder or postorder for the
  entire subtree.

### BST Search

Loop invariant: If `key` occurs in the original tree, it is in the current subtree. A
comparison with the current root finds the key or, due to the BST invariant, excludes
exactly the subtree in which it cannot lie. With `None` the set is empty and the key
does not exist.

### BST insert

The search path ends at an empty position, which meets all the ancestral conditions of
the new key. Only there will a sheet be added; existing relations of order remain
unchanged.

### BST-Delete

- In zero children, nothing is lost below the node.
- In the case of a child, the entire subtree already fulfils the limits of the deleted
  node.
- In two children, the successor lies between the left and right subtrees; its
  subsequent removal is reduced to a simpler case.

## 14. Self-control

1. Specify all four traversal sequences without code for the example tree.
2. Which queue states are created by level orders after `B` and after `E`?
3. Why can a BST with a million nodes still have linear search?
4. Insert `5` into the sample BST and mark all comparisons.
5. Delete successively `1`, `10` and `3`. Which extinguishing case does it take?
6. Construct a tree that passes the local parent-child test, but is not a valid BST.
7. Why does a rotation get the inorder order?
8. Which traversal fits the calculation of directory sizes and why?
9. Why is a ML decision tree generally not a BST?
10. Formulate the complete invariant for your own BST duplicate rule.

## 15. Executive summary

- Trees model hierarchies recursively; each subtree is again a tree.
- Preorder, Inorder and Postorder differ only by the time of node processing; level
  order uses a queue.
- The BST invariant allows the exclusion of an entire subtree per comparison.
- Search, insert, delete and min/max cost `O(h)`; only balance makes it guaranteed
  `O(log n)`.
- Delete has three structural cases. In the case of two children, the Inorder successor
  or predecessor typically takes over.
- AVL and Red-Black trees limit the height by additional invariants and rotations.
- Decision trees, file systems, JSON structures and hierarchical indices transfer the
  same thinking tools into the data science everyday.

---

# Deutsche Fassung

# Bäume und Binary Search Trees

Ein Array beschreibt eine Reihenfolge, eine Hash Map eine Zuordnung und ein Graph
beliebige Beziehungen. Ein Baum beschreibt dagegen eine **Hierarchie**: Jedes
Element außer der Wurzel hat genau einen Elternknoten, und von der Wurzel aus
führt genau ein Pfad zu jedem Knoten.

Dieses Kapitel baut zuerst ein präzises Vokabular auf, leitet daraus die vier
wichtigsten Traversierungen ab und entwickelt anschließend einen Binary Search
Tree (BST). Der Schwerpunkt liegt nicht auf auswendig gelerntem Code, sondern
auf den Invarianten, die Suche, Einfügen und Löschen korrekt machen.

## 1. Lernziele

Nach der Bearbeitung kannst du:

- Baumbegriffe an einem konkreten Beispiel sicher verwenden,
- Preorder, Inorder und Postorder rekursiv sowie Level-Order iterativ erklären,
- die Reihenfolge jeder Traversierung vorhersagen,
- die BST-Suchinvariante formulieren und bei jeder Operation erhalten,
- die drei Löschfälle unterscheiden und korrekt implementieren,
- die Laufzeiten über die Baumhöhe statt pauschal über `n` begründen,
- erklären, weshalb sortiertes Einfügen einen BST degenerieren lässt,
- die Idee von AVL- und Red-Black-Bäumen einordnen,
- Baumstrukturen in Data-Science- und Datenformat-Aufgaben erkennen.

## 2. Warum Hierarchien Bäume sind

Typische Hierarchien sind:

- ein Dateisystem mit Ordnern und Dateien,
- die Struktur eines JSON- oder XML-Dokuments,
- ein Organigramm,
- ein Ausdrucksbaum eines Compilers,
- ein Entscheidungsbaum mit Fragen und Vorhersagen,
- ein hierarchischer Index in einer Datenbank.

Ein Baum eignet sich, wenn eine Beziehung wie „enthält“, „besteht aus“ oder
„entscheidet weiter“ gerichtet von oben nach unten verläuft und jedes Element
nur einen direkten Elternknoten besitzt. Querverbindungen oder mehrere Eltern
machen aus der Struktur im Allgemeinen einen Graphen.

### 2.1 Unser gemeinsamer Beispielbaum

Alle Traversierungen verwenden denselben binären Baum:

```text
          A
        /   \
       B     C
      / \     \
     D   E     F
        / \   /
       G   H I
```

Ein **binärer Baum** erlaubt pro Knoten höchstens zwei Kinder. Er ist nicht
automatisch ein Binary Search Tree: Die Buchstaben oben erfüllen keine
Suchordnung, sondern dienen nur der Traversierung.

```python
from dataclasses import dataclass


@dataclass
class TreeNode:
    """A node of a binary tree."""

    value: str
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def build_example_tree() -> TreeNode:
    """Build the example tree used throughout the chapter."""
    return TreeNode(
        "A",
        left=TreeNode(
            "B",
            left=TreeNode("D"),
            right=TreeNode("E", TreeNode("G"), TreeNode("H")),
        ),
        right=TreeNode("C", right=TreeNode("F", TreeNode("I"))),
    )


root = build_example_tree()
assert root.value == "A"
assert root.left.right.left.value == "G"
```

## 3. Terminologie ohne Mehrdeutigkeit

Für den Beispielbaum gelten:

| Begriff | Bedeutung | Beispiel |
|---|---|---|
| Wurzel | Knoten ohne Elternknoten | `A` |
| Elternknoten | direkter Vorgänger | `B` ist Elternknoten von `D` und `E` |
| Kind | direkter Nachfolger | `G` ist Kind von `E` |
| Geschwister | Kinder desselben Elternknotens | `G` und `H` |
| Blatt | Knoten ohne Kinder | `D`, `G`, `H`, `I` |
| innerer Knoten | Knoten mit mindestens einem Kind | `A`, `B`, `C`, `E`, `F` |
| Teilbaum | Knoten mit allen Nachfahren | Teilbaum mit Wurzel `E` enthält `E,G,H` |
| Tiefe | Anzahl Kanten von der Wurzel zum Knoten | Tiefe von `H` ist 3 |
| Höhe eines Knotens | längster Abstieg in Kanten zu einem Blatt | Höhe von `B` ist 2 |
| Höhe des Baums | Höhe seiner Wurzel | Höhe des Beispielbaums ist 3 |

Diese Definition verwendet Kanten: Ein einzelner Knoten hat Höhe `0`, ein
leerer Baum Höhe `-1`. Manche Quellen zählen stattdessen Knoten und nennen die
Höhe eines einzelnen Knotens `1`. Beide Konventionen sind möglich; innerhalb
einer Lösung muss die gewählte Definition konsequent bleiben.

### 3.1 Tiefe läuft nach unten, Höhe kommt von unten

Die Tiefe wird beim Abstieg als Parameter weitergegeben. Die Höhe wird aus den
Ergebnissen der Kinder aufgebaut. Genau darin zeigt sich die rekursive Natur
eines Baums: Jeder Teilbaum ist selbst wieder ein Baum.

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def height(node: Node | None) -> int:
    """Return height measured in edges; an empty tree has height -1."""
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))


def depths(node: Node | None, depth: int = 0) -> dict[str, int]:
    """Map each node value to its depth."""
    if node is None:
        return {}
    result = {node.value: depth}
    result.update(depths(node.left, depth + 1))
    result.update(depths(node.right, depth + 1))
    return result


tree = Node("A", Node("B", Node("D"), Node("E")), Node("C"))
assert height(tree) == 2
assert depths(tree) == {"A": 0, "B": 1, "D": 2, "E": 2, "C": 1}
```

Für `n` Knoten besucht eine vollständige Höhenberechnung jeden Knoten einmal:
Zeit `O(n)`. Der Rekursionsspeicher ist `O(h)`, wobei `h` die Baumhöhe ist.

## 4. Traversierungen: Wann wird der Knoten verarbeitet?

„Traversieren“ heißt, jeden Knoten systematisch zu besuchen. Bei den drei
rekursiven Tiefensuchen ändert sich nur der Zeitpunkt der Verarbeitung:

- **Preorder:** Knoten, links, rechts.
- **Inorder:** links, Knoten, rechts.
- **Postorder:** links, rechts, Knoten.

Level-Order besucht dagegen Ebene für Ebene und verwendet eine Queue.

### 4.1 Preorder: Knoten vor seinen Teilbäumen

Preorder eignet sich, wenn der Elternknoten vor seinen Kindern verarbeitet
werden muss, etwa beim Kopieren oder Serialisieren einer Baumstruktur.

Simulation am Beispielbaum:

| Schritt | Besuch | Begründung |
|---:|---|---|
| 1 | A | Wurzel zuerst |
| 2 | B | linker Teilbaum von A |
| 3 | D | linker Teilbaum von B |
| 4 | E | rechter Teilbaum von B |
| 5 | G | linker Teilbaum von E |
| 6 | H | rechter Teilbaum von E |
| 7 | C | rechter Teilbaum von A |
| 8 | F | rechter Teilbaum von C |
| 9 | I | linker Teilbaum von F |

Ergebnis: `A B D E G H C F I`

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def preorder(node: Node | None) -> list[str]:
    """Return root-left-right traversal."""
    if node is None:
        return []
    return [node.value] + preorder(node.left) + preorder(node.right)


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert preorder(tree) == list("ABDEGHCFI")
```

### 4.2 Inorder: Knoten zwischen seinen Teilbäumen

Inorder ist für einen BST besonders wichtig, weil sie seine Schlüssel sortiert
liefert. Für einen beliebigen binären Baum ist sie lediglich eine definierte
Besuchsreihenfolge.

Simulation:

1. Vom `A` so weit wie möglich nach links: `D`.
2. Zurück zu `B`, dann in dessen rechten Teilbaum: `G E H`.
3. Jetzt erst `A`.
4. Im rechten Teilbaum folgt `C`, danach `I F`.

Ergebnis: `D B G E H A C I F`

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def inorder(node: Node | None) -> list[str]:
    """Return left-root-right traversal."""
    if node is None:
        return []
    return inorder(node.left) + [node.value] + inorder(node.right)


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert inorder(tree) == list("DBGEHACIF")
```

### 4.3 Postorder: Knoten nach seinen Teilbäumen

Postorder passt zu Aufgaben, bei denen Kinder vor ihrem Elternknoten fertig sein
müssen: Verzeichnisgrößen aufsummieren, Ausdrucksbäume auswerten oder einen Baum
von unten nach oben löschen.

Simulation:

- Teilbaum `B`: `D`, dann `G H E`, zuletzt `B`.
- Teilbaum `C`: `I F`, zuletzt `C`.
- Ganz am Ende folgt die Wurzel `A`.

Ergebnis: `D G H E B I F C A`

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def postorder(node: Node | None) -> list[str]:
    """Return left-right-root traversal."""
    if node is None:
        return []
    return postorder(node.left) + postorder(node.right) + [node.value]


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert postorder(tree) == list("DGHEBIFCA")
```

### 4.4 Level-Order: Ebene für Ebene mit Queue

Level-Order ist eine Breitensuche (BFS). Die Queue enthält stets die bereits
entdeckten, aber noch nicht verarbeiteten Knoten. Beim Entfernen eines Knotens
werden seine Kinder hinten angefügt.

| Schritt | entnommen | anschließend in der Queue |
|---:|---|---|
| 1 | A | B, C |
| 2 | B | C, D, E |
| 3 | C | D, E, F |
| 4 | D | E, F |
| 5 | E | F, G, H |
| 6 | F | G, H, I |
| 7 | G | H, I |
| 8 | H | I |
| 9 | I | leer |

Ergebnis: `A B C D E F G H I`

```python
from collections import deque
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def level_order(root: Node | None) -> list[str]:
    """Return a breadth-first traversal."""
    if root is None:
        return []
    queue = deque([root])
    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node.value)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return result


tree = Node(
    "A",
    Node("B", Node("D"), Node("E", Node("G"), Node("H"))),
    Node("C", right=Node("F", Node("I"))),
)
assert level_order(tree) == list("ABCDEFGHI")
```

Alle vier Traversierungen kosten `O(n)` Zeit. Der Zusatzspeicher unterscheidet
sich:

- Tiefensuche: `O(h)` Rekursionsstack, im schlimmsten Fall `O(n)`.
- Level-Order: `O(w)` Queue, wobei `w` die maximale Breite ist.
- Die Ergebnisliste selbst benötigt bei allen Varianten `O(n)` und wird bei der
  Angabe des Hilfsspeichers oft separat betrachtet.

### 4.5 Rekursion und expliziter Stack sind zwei Darstellungen derselben Idee

Der Python-Aufrufstack merkt sich bei der rekursiven Traversierung, welche
Teilbäume noch offen sind. Ein expliziter Stack macht diesen Zustand sichtbar.
Für Preorder wird zuerst das rechte Kind abgelegt, damit wegen LIFO das linke
Kind zuerst verarbeitet wird.

```python
from dataclasses import dataclass


@dataclass
class Node:
    value: str
    left: "Node | None" = None
    right: "Node | None" = None


def iterative_preorder(root: Node | None) -> list[str]:
    """Return preorder traversal without recursion."""
    if root is None:
        return []
    stack = [root]
    result: list[str] = []
    while stack:
        node = stack.pop()
        result.append(node.value)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return result


tree = Node("A", Node("B", Node("D"), Node("E")), Node("C"))
assert iterative_preorder(tree) == ["A", "B", "D", "E", "C"]
```

## 5. Binary Search Tree: Ordnung als Invariante

Ein Binary Search Tree ist ein binärer Baum mit einer zusätzlichen Regel. Für
jeden Knoten mit Schlüssel `k` gilt:

- Alle Schlüssel im linken Teilbaum sind kleiner als `k`.
- Alle Schlüssel im rechten Teilbaum sind größer als `k`.
- Beide Teilbäume sind selbst wieder BSTs.

In diesem Kapitel sind doppelte Schlüssel nicht erlaubt. Andere Verträge sind
möglich, etwa Duplikate konsequent rechts abzulegen oder einen Zähler im Knoten
zu speichern. Ohne festgelegte Duplikatregel ist die Invariante unvollständig.

Beispiel:

```text
             8
           /   \
          3     10
         / \      \
        1   6      14
           / \    /
          4   7  13
```

Die Inorder-Folge ist `1, 3, 4, 6, 7, 8, 10, 13, 14` und damit streng sortiert.

### 5.1 Suche: In jedem Schritt einen Teilbaum ausschließen

Gesucht sei `7`:

1. `7 < 8`, also kann der gesamte rechte Teilbaum ausgeschlossen werden.
2. `7 > 3`, also geht die Suche rechts weiter.
3. `7 > 6`, also erneut rechts.
4. `7 == 7`: gefunden.

Die Schleifeninvariante lautet: **Falls der Schlüssel existiert, liegt er im
aktuellen Teilbaum.** Der Vergleich erhält diese Aussage, weil die BST-Ordnung
eine komplette Seite ausschließt.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def search(root: BSTNode | None, key: int) -> BSTNode | None:
    """Find key iteratively, or return None."""
    current = root
    while current is not None:
        if key == current.key:
            return current
        current = current.left if key < current.key else current.right
    return None


root = BSTNode(8, BSTNode(3, BSTNode(1), BSTNode(6)), BSTNode(10))
assert search(root, 6).key == 6
assert search(root, 9) is None
```

Suche kostet `O(h)`, nicht grundsätzlich `O(log n)`. Erst ein balancierter Baum
garantiert `h = O(log n)`. In einem degenerierten Baum kann `h = n - 1` sein.

### 5.2 Insert: Suche bis zur freien Position

Beim Einfügen von `5` in den Beispiel-BST lautet der Pfad:

```text
5 < 8 -> left
5 > 3 -> right
5 < 6 -> left
Free   -> 5 becomes left child of 6
```

Die Methode gibt jeweils die Wurzel des möglicherweise veränderten Teilbaums
zurück. Dadurch funktioniert derselbe Code auch beim Einfügen in einen leeren
Baum.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def insert(node: BSTNode | None, key: int) -> BSTNode:
    """Insert a unique key and return the subtree root."""
    if node is None:
        return BSTNode(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    else:
        raise ValueError(f"duplicate key: {key}")
    return node


def inorder(node: BSTNode | None) -> list[int]:
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


root = None
for value in [8, 3, 10, 1, 6, 14, 4, 7, 13]:
    root = insert(root, value)
assert inorder(root) == [1, 3, 4, 6, 7, 8, 10, 13, 14]
```

### 5.3 Minimum und Maximum

Wegen der Suchinvariante liegt das Minimum ganz links und das Maximum ganz
rechts. Beide Operationen kosten `O(h)` und benötigen keine vollständige
Traversierung.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def minimum(node: BSTNode) -> BSTNode:
    """Return the node with the smallest key in a non-empty subtree."""
    current = node
    while current.left is not None:
        current = current.left
    return current


def maximum(node: BSTNode) -> BSTNode:
    """Return the node with the largest key in a non-empty subtree."""
    current = node
    while current.right is not None:
        current = current.right
    return current


root = BSTNode(8, BSTNode(3, BSTNode(1), BSTNode(6)), BSTNode(10))
assert minimum(root).key == 1
assert maximum(root).key == 10
```

## 6. Delete: die drei strukturellen Fälle

Löschen ist schwieriger als Einfügen, weil der Knoten aus der Struktur
verschwindet und die BST-Invariante trotzdem erhalten bleiben muss. Zuerst wird
der Schlüssel wie bei der Suche gefunden. Dann gibt es genau drei Fälle.

### 6.1 Fall 1: Blatt löschen

Beim Löschen von `1` besitzt der Knoten keine Kinder. Der Elternzeiger kann
einfach auf `None` gesetzt werden.

```text
    3                 3
   / \      ->         \
  1   6                 6
```

### 6.2 Fall 2: Knoten mit genau einem Kind löschen

Beim Löschen von `10` im großen Beispiel besitzt der Knoten nur das rechte Kind
`14`. Der Elternknoten wird direkt mit diesem Kind verbunden.

```text
  8                   8
   \                   \
   10       ->         14
     \                 /
     14               13
    /
   13
```

Der gesamte erhaltene Teilbaum bleibt in der korrekten Größenrelation.

### 6.3 Fall 3: Knoten mit zwei Kindern löschen

Beim Löschen von `3` dürfen nicht einfach beide Kanten entfernt werden. Eine
geeignete Ersatzposition ist der **Inorder-Nachfolger**, also das Minimum des
rechten Teilbaums. Hier ist das `4`.

1. Kopiere den Schlüssel `4` in den bisherigen Knoten `3`.
2. Lösche das ursprüngliche `4` rekursiv im rechten Teilbaum.
3. Dieser zweite Löschvorgang ist Fall 1 oder Fall 2, weil ein Minimum kein
   linkes Kind haben kann.

Alternativ kann das Maximum des linken Teilbaums als Inorder-Vorgänger dienen.

```python
from dataclasses import dataclass


@dataclass
class BSTNode:
    key: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def insert(node: BSTNode | None, key: int) -> BSTNode:
    if node is None:
        return BSTNode(key)
    if key < node.key:
        node.left = insert(node.left, key)
    elif key > node.key:
        node.right = insert(node.right, key)
    return node


def minimum(node: BSTNode) -> BSTNode:
    while node.left is not None:
        node = node.left
    return node


def delete(node: BSTNode | None, key: int) -> BSTNode | None:
    """Delete key if present and return the subtree root."""
    if node is None:
        return None
    if key < node.key:
        node.left = delete(node.left, key)
    elif key > node.key:
        node.right = delete(node.right, key)
    elif node.left is None:
        return node.right
    elif node.right is None:
        return node.left
    else:
        successor = minimum(node.right)
        node.key = successor.key
        node.right = delete(node.right, successor.key)
    return node


def inorder(node: BSTNode | None) -> list[int]:
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


root = None
for value in [8, 3, 10, 1, 6, 14, 4, 7, 13]:
    root = insert(root, value)

root = delete(root, 1)   # leaf
root = delete(root, 10)  # one child
root = delete(root, 3)   # two children
assert inorder(root) == [4, 6, 7, 8, 13, 14]
```

### 6.4 Warum die Invariante nach Fall 3 erhalten bleibt

Der Nachfolger ist das kleinste Element im rechten Teilbaum. Deshalb gilt:

- Er ist größer als alle Elemente im linken Teilbaum des gelöschten Knotens.
- Kein anderes Element im rechten Teilbaum ist kleiner als er.
- Nach dem Entfernen seiner alten Position existiert der Schlüssel nur einmal.

Die Kosten sind Suche `O(h)` plus ein Abstieg zum Nachfolger von höchstens
`O(h)`, zusammen weiterhin `O(h)`.

### 6.5 Häufige Delete-Fehler

- Den neuen Teilbaumwurzelwert nicht an den Elternknoten zurückgeben.
- Bei zwei Kindern den Nachfolger kopieren, aber nicht am alten Ort löschen.
- Nur die unmittelbaren Kinder vergleichen und tiefere Nachfahren ignorieren.
- Die Duplikatregel nicht definieren.
- Beim Löschen der Wurzel annehmen, dass die äußere Referenz unverändert bleibt.

## 7. Einen BST korrekt validieren

Die lokale Prüfung `left.key < node.key < right.key` genügt nicht. Im folgenden
Baum ist `12` zwar größer als sein Elternknoten `6`, liegt aber im linken
Teilbaum von `8` und verletzt damit eine weiter oben gesetzte Grenze:

```text
      8
     / \
    6   10
     \
     12   <- invalid
```

Jeder rekursive Aufruf muss deshalb das gesamte zulässige Intervall mitführen.

```python
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    left: "Node | None" = None
    right: "Node | None" = None


def is_valid_bst(
    node: Node | None,
    lower: int | None = None,
    upper: int | None = None,
) -> bool:
    """Validate a strict BST using inherited bounds."""
    if node is None:
        return True
    if lower is not None and node.key <= lower:
        return False
    if upper is not None and node.key >= upper:
        return False
    return (
        is_valid_bst(node.left, lower, node.key)
        and is_valid_bst(node.right, node.key, upper)
    )


valid = Node(8, Node(3, Node(1), Node(6)), Node(10))
invalid = Node(8, Node(6, right=Node(12)), Node(10))
assert is_valid_bst(valid)
assert not is_valid_bst(invalid)
```

Die Schranken sind offen, weil Duplikate ausgeschlossen sind. Für Fließkomma-
oder andere Schlüsseltypen sollte man keine künstlichen Werte wie
`float("inf")` voraussetzen; optionale Schranken machen den Vertrag explizit.

## 8. Laufzeit hängt von der Höhe ab

| Operation | balanciert, `h = O(log n)` | degeneriert, `h = O(n)` |
|---|---:|---:|
| Suche | `O(log n)` | `O(n)` |
| Insert | `O(log n)` | `O(n)` |
| Delete | `O(log n)` | `O(n)` |
| Min/Max | `O(log n)` | `O(n)` |
| vollständige Traversierung | `O(n)` | `O(n)` |

Die Traversierung bleibt linear, weil sie ohnehin alle Knoten besucht. Nur die
pfadbasierten Operationen profitieren von geringer Höhe.

### 8.1 Degeneration zum Linked-List-Baum

Werden `1, 2, 3, 4, 5, 6, 7` in dieser Reihenfolge in einen gewöhnlichen BST
eingefügt, besitzt jeder Knoten nur ein rechtes Kind:

```text
1
 \
  2
   \
    3
     \
      ... 7
```

Die Höhe ist `n - 1`; Suche und Insert verhalten sich wie in einer verketteten
Liste.

```python
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    right: "Node | None" = None


def build_sorted_chain(size: int) -> Node | None:
    """Build the shape produced by increasing BST insertions."""
    if size <= 0:
        return None
    root = Node(1)
    current = root
    for key in range(2, size + 1):
        current.right = Node(key)
        current = current.right
    return root


def height(node: Node | None) -> int:
    if node is None:
        return -1
    return 1 + height(node.right)


assert height(build_sorted_chain(7)) == 6
```

Zufällige Einfügereihenfolge erzeugt häufig brauchbare, aber nicht garantiert
balancierte Bäume. Eine garantierte Schranke erfordert eine zusätzliche
Balance-Invariante oder einen anderen Aufbau, etwa das rekursive Wählen der
Mitte eines sortierten Arrays.

## 9. Balancierung: Rotationen erhalten die Suchordnung

Eine **Rotation** verändert lokale Kanten, ohne die Inorder-Reihenfolge zu
ändern. Bei einer Linksdrehung um `x` steigt dessen rechtes Kind `y` auf:

```text
    x                  y
   / \                / \
  A   y      ->       x   C
     / \            / \
    B   C          A   B
```

Vorher und nachher lautet Inorder `A, x, B, y, C`. Deshalb bleibt die
BST-Invariante erhalten. Die Höhe kann sich aber ändern.

```python
from dataclasses import dataclass


@dataclass
class Node:
    key: int
    left: "Node | None" = None
    right: "Node | None" = None


def rotate_left(x: Node) -> Node:
    """Rotate a subtree left and return its new root."""
    if x.right is None:
        raise ValueError("left rotation requires a right child")
    y = x.right
    x.right = y.left
    y.left = x
    return y


def inorder(node: Node | None) -> list[int]:
    if node is None:
        return []
    return inorder(node.left) + [node.key] + inorder(node.right)


root = Node(10, Node(5), Node(20, Node(15), Node(30)))
before = inorder(root)
root = rotate_left(root)
assert root.key == 20
assert inorder(root) == before == [5, 10, 15, 20, 30]
```

### 9.1 AVL-Bäume konzeptionell

Ein AVL-Baum speichert oder berechnet für jeden Knoten den Balancefaktor

`height(left subtree) - height(right subtree)`.

Er muss in `{-1, 0, 1}` liegen. Nach Insert oder Delete werden Höhen auf dem Pfad
zur Wurzel aktualisiert und ein Ungleichgewicht durch eine einfache oder doppelte
Rotation repariert. Die strenge Balance liefert sehr schnelle Suchen, kann aber
mehr Rotationen bei Änderungen erfordern.

### 9.2 Red-Black-Bäume konzeptionell

Ein Red-Black-Baum ergänzt jeden Knoten um eine Farbe und erzwingt Regeln über
rote Nachbarschaften und die Anzahl schwarzer Knoten auf Wurzel-Blatt-Pfaden.
Die Regeln garantieren eine logarithmische Höhe, erlauben aber mehr Spielraum als
AVL-Bäume. Dadurch sind Änderungen oft günstiger. Die vollständigen Fix-up-Fälle
sind bewusst nicht Gegenstand dieses Moduls; entscheidend ist die Idee:

> Zusätzliche lokale Metadaten und Invarianten verhindern eine lineare Höhe.

Python stellt keinen eingebauten Tree-Map-Typ bereit. In anderen Sprachen werden
geordnete Maps und Sets häufig mit balancierten Suchbäumen implementiert. Welche
Variante intern verwendet wird, ist eine Implementierungsentscheidung.

## 10. Bäume im Data-Science-Alltag

### 10.1 Entscheidungsbäume

Ein innerer Knoten prüft eine Bedingung wie `feature_2 <= 1.7`. Der linke oder
rechte Pfad führt zur nächsten Bedingung; ein Blatt enthält die Vorhersage. Eine
einzelne Vorhersage kostet proportional zur Tiefe des erreichten Blatts.

```python
from dataclasses import dataclass


@dataclass
class DecisionNode:
    feature: int | None = None
    threshold: float | None = None
    prediction: int | None = None
    left: "DecisionNode | None" = None
    right: "DecisionNode | None" = None


def predict(root: DecisionNode, row: list[float]) -> int:
    """Follow decision nodes until a prediction leaf is reached."""
    node = root
    while node.prediction is None:
        if node.feature is None or node.threshold is None:
            raise ValueError("incomplete decision node")
        child = node.left if row[node.feature] <= node.threshold else node.right
        if child is None:
            raise ValueError("missing decision branch")
        node = child
    return node.prediction


model = DecisionNode(
    feature=0,
    threshold=2.5,
    left=DecisionNode(prediction=0),
    right=DecisionNode(prediction=1),
)
assert predict(model, [1.2]) == 0
assert predict(model, [4.0]) == 1
```

Ein ML-Entscheidungsbaum ist normalerweise **kein BST**. Seine Ordnung basiert
auf Feature-Schwellen pro Knoten, nicht auf einer globalen Schlüsselordnung.
Er teilt aber die rekursive Struktur und viele Traversierungstechniken.

### 10.2 Hierarchische Daten und JSON

Verschachtelte Dictionaries und Listen bilden eine Baumstruktur, solange
Objekte nicht mehrfach referenziert werden. Eine Tiefensuche kann alle Blätter
mit ihrem Pfad sammeln.

```python
def flatten_leaves(value: object, path: tuple[str, ...] = ()) -> dict[str, object]:
    """Flatten dictionary leaves into dotted paths."""
    if not isinstance(value, dict):
        return {".".join(path): value}
    result: dict[str, object] = {}
    for key, child in value.items():
        result.update(flatten_leaves(child, path + (str(key),)))
    return result


record = {
    "model": {"name": "tree", "params": {"depth": 4}},
    "score": 0.91,
}
assert flatten_leaves(record) == {
    "model.name": "tree",
    "model.params.depth": 4,
    "score": 0.91,
}
```

Bei beliebigen Python-Objekten können Zyklen oder gemeinsam referenzierte
Teilobjekte auftreten. Dann reicht eine reine Baumtraversierung nicht; eine
`visited`-Menge wie bei Graphen wird nötig.

### 10.3 Hierarchische Indizes

B-Bäume und B+-Bäume speichern mehrere Schlüssel pro Knoten und sind für
Blockzugriffe auf Datenträger optimiert. Sie sind keine binären Suchbäume, nutzen
aber dieselbe Grundidee: Ein Vergleich schließt große Datenbereiche aus. Für
DataFrames sind Multi-Indizes ebenfalls konzeptionell hierarchisch, auch wenn
die konkrete Bibliothek andere interne Strukturen einsetzen kann.

### 10.4 Dateisysteme und Aggregationen

Ordner sind innere Knoten, Dateien Blätter. Größenberechnung ist Postorder:
Zuerst werden die Größen aller Kinder ermittelt, danach ihre Summe dem Ordner
zugeordnet. Eine Baumansicht ist Preorder: Erst den Ordner ausgeben, dann seine
Inhalte eingerückt darstellen.

## 11. Muster erkennen und die richtige Traversierung wählen

| Aufgabenformulierung | Passendes Muster | Warum |
|---|---|---|
| „Eltern vor Kindern verarbeiten“ | Preorder | Knotenaktion kommt zuerst |
| „BST sortiert ausgeben“ | Inorder | Suchinvariante erzeugt Reihenfolge |
| „Ergebnis aus Kindern aggregieren“ | Postorder | Kinderergebnisse liegen zuerst vor |
| „Nächste Ebene / minimale Tiefe“ | Level-Order | BFS verarbeitet nach Distanz |
| „Einen Schlüssel im BST finden“ | gerichteter Abstieg | eine Seite wird je Vergleich verworfen |
| „Ist dies global ein BST?“ | DFS mit Schranken | Ahnenbedingungen müssen mitgeführt werden |

### Entscheidungsfragen

1. Muss wirklich jeder Knoten besucht werden, oder erlaubt eine Ordnung das
   Ausschließen eines Teilbaums?
2. Muss ein Elternknoten vor, zwischen oder nach seinen Kindern verarbeitet
   werden?
3. Ist eine tiefenbasierte Aussage gefragt? Dann ist Level-Order oft natürlich.
4. Kann der Baum sehr tief sein? Dann droht in Python ein Rekursionslimit und
   ein expliziter Stack kann robuster sein.
5. Bezieht sich eine Bedingung nur auf direkte Kinder oder auf alle Nachfahren?

## 12. Typische Denkfehler

- **Binärer Baum und BST verwechseln:** Höchstens zwei Kinder bedeuten noch
  keine Schlüsselordnung.
- **`O(log n)` ohne Balance behaupten:** BST-Operationen kosten `O(h)`.
- **Tiefe und Höhe vertauschen:** Tiefe kommt von der Wurzel, Höhe vom tiefsten
  Blatt.
- **Level-Order mit einem Stack implementieren:** BFS benötigt FIFO, also eine
  Queue.
- **Inorder immer für „sortiert“ halten:** Das gilt nur bei erfüllter
  BST-Invariante.
- **BST nur lokal validieren:** Grenzen aller Vorfahren müssen gelten.
- **Delete-Fall mit zwei Kindern unvollständig behandeln:** Ersatzschlüssel muss
  an seiner alten Position entfernt werden.
- **Leeren Baum vergessen:** Viele Funktionen benötigen `None` als Basisfall.
- **Rekursion für kostenlos halten:** Der Call Stack benötigt `O(h)` Speicher.

## 13. Kompakte Korrektheitsargumente

### Traversierung

Induktion über die Teilbaumgröße:

- Basis: Ein leerer Baum erzeugt eine leere Folge.
- Schritt: Die rekursiven Aufrufe traversieren linken und rechten Teilbaum
  korrekt. Das Einfügen der Wurzel an der definierten Position ergibt genau
  Preorder, Inorder oder Postorder für den gesamten Teilbaum.

### BST-Suche

Schleifeninvariante: Falls `key` im ursprünglichen Baum vorkommt, liegt es im
aktuellen Teilbaum. Ein Vergleich mit der aktuellen Wurzel findet den Schlüssel
oder schließt wegen der BST-Invariante genau den Teilbaum aus, in dem er nicht
liegen kann. Bei `None` ist die Menge leer und der Schlüssel existiert nicht.

### BST-Insert

Der Suchpfad endet an einer leeren Position, deren alle Ahnenbedingungen der
neue Schlüssel erfüllt. Nur dort wird ein Blatt ergänzt; bestehende
Ordnungsrelationen bleiben unverändert.

### BST-Delete

- Bei null Kindern wird nichts unterhalb des Knotens verloren.
- Bei einem Kind erfüllt der gesamte übernommene Teilbaum bereits die Schranken
  des gelöschten Knotens.
- Bei zwei Kindern liegt der Nachfolger zwischen linkem und rechtem Teilbaum;
  sein anschließendes Entfernen reduziert sich auf einen einfacheren Fall.

## 14. Selbstkontrolle

1. Gib für den Beispielbaum alle vier Traversierungsfolgen ohne Code an.
2. Welche Queue-Zustände entstehen bei Level-Order nach `B` und nach `E`?
3. Warum kann ein BST mit einer Million Knoten trotzdem lineare Suche haben?
4. Füge `5` in den Beispiel-BST ein und markiere alle Vergleiche.
5. Lösche nacheinander `1`, `10` und `3`. Welcher Löschfall greift jeweils?
6. Konstruiere einen Baum, der die lokale Eltern-Kind-Prüfung besteht, aber kein
   gültiger BST ist.
7. Warum erhält eine Rotation die Inorder-Reihenfolge?
8. Welche Traversierung passt zur Berechnung von Verzeichnisgrößen und warum?
9. Weshalb ist ein ML-Entscheidungsbaum im Allgemeinen kein BST?
10. Formuliere für eine eigene BST-Duplikatregel die vollständige Invariante.

## 15. Zusammenfassung

- Bäume modellieren Hierarchien rekursiv; jeder Teilbaum ist wieder ein Baum.
- Preorder, Inorder und Postorder unterscheiden sich nur durch den Zeitpunkt der
  Knotenverarbeitung; Level-Order verwendet eine Queue.
- Die BST-Invariante erlaubt pro Vergleich den Ausschluss eines ganzen
  Teilbaums.
- Suche, Insert, Delete und Min/Max kosten `O(h)`; nur Balance macht daraus
  garantiert `O(log n)`.
- Delete hat drei strukturelle Fälle. Bei zwei Kindern übernimmt typischerweise
  der Inorder-Nachfolger oder -Vorgänger.
- AVL- und Red-Black-Bäume begrenzen die Höhe durch zusätzliche Invarianten und
  Rotationen.
- Entscheidungsbäume, Dateisysteme, JSON-Strukturen und hierarchische Indizes
  übertragen dieselben Denkwerkzeuge in den Data-Science-Alltag.
