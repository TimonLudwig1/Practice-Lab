"""Run one representative example for every pattern in the catalog."""

from pattern_catalog import (
    PrefixSum,
    are_anagrams,
    compress_runs_in_place,
    longest_unique_substring,
    merge_sorted_in_place,
    move_zeros_to_end,
    product_except_self,
    remove_duplicates_sorted,
    rotate_right_in_place,
    spiral_order,
)


def main() -> None:
    """Execute and print all ten catalog examples."""
    rotated = [1, 2, 3, 4, 5, 6, 7]
    rotate_right_in_place(rotated, 3)

    merged = [1, 4, 7, None, None, None]
    merge_sorted_in_place(merged, 3, [2, 3, 8])

    prefix = PrefixSum.from_values([4, -1, 7, 3, 2])

    unique_values = [1, 1, 2, 2, 2, 5]
    remove_duplicates_sorted(unique_values)

    moved_zeros = [0, 3, 0, 1, 0, 8]
    move_zeros_to_end(moved_zeros)

    compressed = list("aaabcccccccccc")
    compress_runs_in_place(compressed)

    examples = [
        ("Rotation", rotated),
        ("Merge", merged),
        ("Prefix sum [1:4]", prefix.range_sum(1, 4)),
        ("Anagram", are_anagrams("listen", "silent")),
        ("Deduplizierung", unique_values),
        ("Zero-Move", moved_zeros),
        ("Product Except Self", product_except_self([1, 2, 3, 4])),
        ("Längstes eindeutiges Teilstück", longest_unique_substring("pwwkew")),
        ("Spirale", spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ("Kompression", "".join(compressed)),
    ]

    for name, result in examples:
        print(f"{name:34} {result}")


if __name__ == "__main__":
    main()
