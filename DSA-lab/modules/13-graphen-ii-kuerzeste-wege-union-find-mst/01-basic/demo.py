"""Handrechnungs-Abgleich für Dijkstra und Union-Find-Simulation."""

import math

from algorithms import UnionFind, dijkstra, hand_calculation_graph


EXPECTED_DISTANCES = {
    "A": 0.0,
    "B": 3.0,
    "C": 2.0,
    "D": 8.0,
    "E": 10.0,
    "F": 13.0,
}


def show_dijkstra() -> None:
    graph = hand_calculation_graph()
    result = dijkstra(graph, "A")

    print("Dijkstra ab A")
    print("Knoten | Handrechnung | Algorithmus | Vorgänger")
    print("-------+---------------+-------------+----------")
    for vertex in graph.vertices:
        predecessor = result.predecessors[vertex] or "—"
        print(
            f"{vertex:^6} | {EXPECTED_DISTANCES[vertex]:^13.0f} | "
            f"{result.distances[vertex]:^11.0f} | {predecessor:^9}"
        )

    print("\nFixierreihenfolge:", " -> ".join(result.settled_order))
    path = result.path_to("F")
    assert path is not None
    print("Kürzester Pfad A->F:", " -> ".join(path))
    print("Pfadkosten:", graph.path_cost(path))

    assert dict(result.distances) == EXPECTED_DISTANCES
    assert result.settled_order == ("A", "C", "B", "D", "E", "F")
    assert path == ("A", "C", "B", "D", "E", "F")
    assert math.isclose(graph.path_cost(path), 13.0)


def show_union_find() -> None:
    structure = UnionFind("ABCDEF")
    operations = (
        ("A", "B"),
        ("C", "D"),
        ("A", "C"),
        ("E", "F"),
        ("B", "D"),
    )

    print("\nUnion-Find")
    for first, second in operations:
        changed = structure.union(first, second)
        state = "vereinigt" if changed else "bereits verbunden"
        groups = ["{" + ",".join(sorted(group)) + "}" for group in structure.components()]
        print(f"union({first}, {second}): {state:18} -> {' '.join(groups)}")

    assert structure.component_count == 2
    assert structure.connected("A", "D")
    assert not structure.connected("A", "E")
    assert structure.component_size("B") == 4
    structure.validate()
    print("Alle Demo-Invarianten erfüllt.")


def main() -> None:
    show_dijkstra()
    show_union_find()


if __name__ == "__main__":
    main()
