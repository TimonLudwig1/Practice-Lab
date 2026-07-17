"""Print stepwise traces for all four linked-list pointer classics."""

from linked_algorithms import (
    build_chain,
    chain_nodes,
    chain_values,
    detect_cycle,
    merge_sorted,
    middle_node,
    reverse_in_place,
)


def show(title: str, trace: list[str], result: object) -> None:
    """Print one labeled algorithm trace."""
    print(f"\n{title}")
    print("=" * len(title))
    print("\n".join(trace))
    print(f"Ergebnis: {result}")


def main() -> None:
    """Run representative inputs for reversal, middle, cycle, and merge."""
    reverse_trace: list[str] = []
    reversed_head = reverse_in_place(build_chain([1, 2, 3, 4]), trace=reverse_trace)
    show("In-place-Reversal", reverse_trace, chain_values(reversed_head))

    middle_trace: list[str] = []
    middle = middle_node(build_chain([10, 20, 30, 40, 50, 60]), trace=middle_trace)
    show("Runner-Mitte", middle_trace, None if middle is None else middle.value)

    cycle_head = build_chain(["A", "B", "C", "D", "E"])
    cycle_nodes = chain_nodes(cycle_head)
    cycle_nodes[-1].next = cycle_nodes[2]
    cycle_trace: list[str] = []
    cycle = detect_cycle(cycle_head, trace=cycle_trace)
    cycle_result = None
    if cycle is not None:
        cycle_result = {
            "entry": cycle.entry.value,
            "cycle_length": cycle.cycle_length,
            "prefix_length": cycle.prefix_length,
        }
    show("Floyd-Zyklenerkennung", cycle_trace, cycle_result)

    merge_trace: list[str] = []
    merged = merge_sorted(
        build_chain([1, 4, 7]),
        build_chain([2, 3, 8]),
        trace=merge_trace,
    )
    show("Stabiler In-place-Merge", merge_trace, chain_values(merged))


if __name__ == "__main__":
    main()
