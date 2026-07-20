"""Führt jedes Muster mit einem sichtbaren Traversierungs-Trace aus."""

from graph_patterns import (
    analyze_islands,
    bipartite_coloring,
    has_directed_cycle,
    has_undirected_cycle,
    is_valid_topological_order,
    topological_sort_kahn,
)


def show(title: str, result: object, trace: list[str]) -> None:
    print(f"\n{title}: {result}")
    for step, message in enumerate(trace, start=1):
        print(f"  {step:02d}. {message}")


def main() -> None:
    undirected = {
        "A": ("B", "C"),
        "B": ("A", "C"),
        "C": ("A", "B"),
    }
    trace: list[str] = []
    show(
        "Ungerichteter Zyklus",
        has_undirected_cycle(undirected, trace),
        trace,
    )

    directed = {
        "extract": ("clean",),
        "clean": ("train",),
        "train": ("evaluate",),
        "evaluate": (),
    }
    trace = []
    show("Gerichteter Zyklus", has_directed_cycle(directed, trace), trace)

    trace = []
    order = topological_sort_kahn(directed, trace)
    assert is_valid_topological_order(directed, order)
    show("Topologische Ordnung", order, trace)

    square = {
        "A": ("B", "D"),
        "B": ("A", "C"),
        "C": ("B", "D"),
        "D": ("A", "C"),
    }
    trace = []
    colors = bipartite_coloring(square, trace)
    show("Bipartite Färbung", colors, trace)

    grid = (
        (1, 1, 0, 0, 0),
        (1, 0, 0, 1, 1),
        (0, 0, 1, 0, 0),
        (1, 1, 0, 0, 1),
    )
    trace = []
    report = analyze_islands(grid, trace)
    show("Inselanalyse", report, trace)

    assert report.count == 5
    assert report.sizes == (3, 2, 1, 2, 1)
    print("\nAlle Demo-Ergebnisse geprüft.")


if __name__ == "__main__":
    main()
