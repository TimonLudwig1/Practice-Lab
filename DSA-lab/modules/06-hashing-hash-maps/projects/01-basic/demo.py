"""Trace collisions, updates, deletion and rehashing."""

from hash_map import ChainedHashMap


def constant_hash(key: str) -> int:
    """Force all demo keys into the same bucket."""
    del key
    return 1


def main() -> None:
    """Print state transitions of a deliberately collision-heavy map."""
    mapping: ChainedHashMap[str, int] = ChainedHashMap(
        initial_capacity=4,
        hash_function=constant_hash,
    )
    print("operation          size capacity load  collisions max_chain")
    print("-" * 66)

    def show(operation: str) -> None:
        stats = mapping.stats()
        print(
            f"{operation:<19}{stats.size:>4}{stats.capacity:>9}"
            f"{stats.load_factor:>6.2f}{stats.collision_count:>12}"
            f"{stats.max_chain_length:>10}"
        )

    show("initial")
    for key, value in (("Ada", 10), ("Grace", 20), ("Linus", 30), ("Edsger", 40)):
        mapping.put(key, value)
        show(f"put({key})")

    mapping.put("Ada", 99)
    show("update(Ada)")
    print(f"get(Ada)           -> {mapping.get('Ada')}")
    print(f"delete(Grace)      -> {mapping.delete('Grace')}")
    show("after delete")
    mapping.check_invariants()


if __name__ == "__main__":
    main()
