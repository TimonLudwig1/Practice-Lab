"""Demonstrate the four hash-based problem-solving patterns."""

from patterns import (
    DuplicateDetector,
    first_unique_index_hash,
    group_anagrams_hash,
    two_sum_hash,
)


def main() -> None:
    numbers = [2, 7, 11, 15]
    print(f"Two Sum: {numbers}, target=9 -> {two_sum_hash(numbers, 9)}")

    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(f"Anagrams: {group_anagrams_hash(words)}")

    text = "swiss"
    unique_index = first_unique_index_hash(text)
    print(
        f"First unique: {text!r} -> index {unique_index}, "
        f"character {text[unique_index]!r}"
    )

    detector: DuplicateDetector[str] = DuplicateDetector()
    stream = ["evt-1", "evt-2", "evt-1", "evt-3", "evt-2"]
    print("Stream duplicates:")
    for item in stream:
        print(f"  {item}: duplicate={detector.process(item)}")
    print(
        f"Processed={detector.processed_count}, unique={detector.unique_count}, "
        f"duplicates={detector.duplicate_count}"
    )


if __name__ == "__main__":
    main()
