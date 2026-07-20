"""Ausführbare Demonstration des Graph-Grundgerüsts."""

from graph import Graph, example_graph


def format_order(order: tuple[str, ...]) -> str:
    return " -> ".join(order)


def main() -> None:
    graph = example_graph()

    print("Adjazenzliste")
    for vertex, neighbors in graph.adjacency.items():
        print(f"  {vertex}: {', '.join(neighbors) or '—'}")

    bfs_order = graph.bfs("A")
    recursive_order = graph.dfs_recursive("A")
    iterative_order = graph.dfs_iterative("A")

    print("\nTraversierungen ab A")
    print(f"  BFS:           {format_order(bfs_order)}")
    print(f"  DFS rekursiv:  {format_order(recursive_order)}")
    print(f"  DFS iterativ:  {format_order(iterative_order)}")

    disconnected: Graph[str] = Graph()
    disconnected.add_edge("Berlin", "Potsdam")
    disconnected.add_edge("Köln", "Bonn")
    disconnected.add_vertex("Insel")

    components = disconnected.connected_components()
    formatted = ["{" + ", ".join(sorted(component)) + "}" for component in components]
    print("\nZusammenhangskomponenten")
    print("  " + " | ".join(formatted))

    assert bfs_order == ("A", "B", "C", "D", "E", "F")
    assert recursive_order == ("A", "B", "D", "C", "E", "F")
    assert iterative_order == recursive_order
    assert len(components) == 3
    graph.validate()
    disconnected.validate()
    print("\nAlle Demo-Invarianten erfüllt.")


if __name__ == "__main__":
    main()
