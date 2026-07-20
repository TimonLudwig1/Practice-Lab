"""Demonstrate the three central heap patterns on small inputs."""

from heap_patterns import RunningMedian, merge_sorted, top_k_frequent


def main() -> None:
    events = ["api", "db", "api", "cache", "db", "api", "worker", "cache"]
    print("TOP-K FREQUENT")
    print(f"events: {events}")
    print(f"top 3: {top_k_frequent(events, 3)}")

    shards = [[1, 7, 10], [2, 3, 11], [], [4, 8]]
    print("\nK-WAY MERGE")
    print(f"shards: {shards}")
    print(f"merged: {merge_sorted(shards)}")

    print("\nRUNNING MEDIAN")
    tracker = RunningMedian()
    for value in [5, 2, 10, 4, 8, 1]:
        tracker.add(value)
        lower, upper = tracker.halves()
        print(
            f"add {value:>2}: lower={lower}, upper={upper}, "
            f"median={tracker.median():.1f}"
        )


if __name__ == "__main__":
    main()
