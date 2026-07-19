"""Tests for optimized and naive hash-map patterns."""

from pathlib import Path

import pytest

from benchmark import run_benchmark, write_csv
from patterns import (
    DuplicateDetector,
    duplicate_flags_hash,
    duplicate_flags_naive,
    first_unique_index_hash,
    first_unique_index_naive,
    group_anagrams_hash,
    group_anagrams_naive,
    two_sum_hash,
    two_sum_naive,
)


TWO_SUM_CASES = [
    ([2, 7, 11, 15], 9, (0, 1)),
    ([3, 2, 4], 6, (1, 2)),
    ([3, 3], 6, (0, 1)),
    ([-4, -1, 5, 8], 7, (1, 3)),
    ([0, 4, 3, 0], 0, (0, 3)),
    ([1, 2, 3], 99, None),
    ([], 0, None),
]


@pytest.mark.parametrize(("numbers", "target", "expected"), TWO_SUM_CASES)
@pytest.mark.parametrize("function", [two_sum_hash, two_sum_naive])
def test_two_sum(function, numbers, target, expected) -> None:
    assert function(numbers, target) == expected


def test_two_sum_does_not_reuse_one_element() -> None:
    assert two_sum_hash([3], 6) is None
    assert two_sum_naive([3], 6) is None


def normalize_groups(groups: list[list[str]]) -> list[list[str]]:
    return sorted((sorted(group) for group in groups), key=lambda group: (len(group), group))


@pytest.mark.parametrize("function", [group_anagrams_hash, group_anagrams_naive])
def test_group_anagrams(function) -> None:
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]

    assert normalize_groups(function(words)) == normalize_groups(
        [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
    )


@pytest.mark.parametrize("function", [group_anagrams_hash, group_anagrams_naive])
def test_group_anagrams_handles_empty_and_repeated_words(function) -> None:
    assert function([]) == []
    assert normalize_groups(function(["", "", "a", "a"])) == [["", ""], ["a", "a"]]


@pytest.mark.parametrize("function", [group_anagrams_hash, group_anagrams_naive])
def test_group_anagrams_is_case_sensitive(function) -> None:
    assert normalize_groups(function(["ab", "ba", "Ab"])) == [["Ab"], ["ab", "ba"]]


UNIQUE_CASES = [
    ("leetcode", 0),
    ("loveleetcode", 2),
    ("aabb", -1),
    ("", -1),
    ("swiss", 1),
    ("ääbää", 2),
]


@pytest.mark.parametrize(("text", "expected"), UNIQUE_CASES)
@pytest.mark.parametrize("function", [first_unique_index_hash, first_unique_index_naive])
def test_first_unique_index(function, text, expected) -> None:
    assert function(text) == expected


@pytest.mark.parametrize("function", [duplicate_flags_hash, duplicate_flags_naive])
def test_duplicate_flags(function) -> None:
    assert function(["A", "B", "A", "C", "B", "B"]) == [
        False,
        False,
        True,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize("function", [duplicate_flags_hash, duplicate_flags_naive])
def test_duplicate_flags_accepts_generator(function) -> None:
    assert function(value for value in [1, 2, 1]) == [False, False, True]


@pytest.mark.parametrize("function", [duplicate_flags_hash, duplicate_flags_naive])
def test_duplicate_flags_empty_stream(function) -> None:
    assert function([]) == []


def test_stateful_detector_tracks_counters() -> None:
    detector: DuplicateDetector[int] = DuplicateDetector()

    assert detector.process(10) is False
    assert detector.process(20) is False
    assert detector.process(10) is True
    assert detector.process(10) is True
    assert detector.processed_count == 4
    assert detector.unique_count == 2
    assert detector.duplicate_count == 2


def test_stateful_detector_reset_forgets_history() -> None:
    detector: DuplicateDetector[str] = DuplicateDetector()
    detector.process("A")
    assert detector.process("A") is True

    detector.reset()

    assert detector.process("A") is False
    assert detector.processed_count == 1
    assert detector.unique_count == 1
    assert detector.duplicate_count == 0


def test_hash_detector_rejects_unhashable_item_without_corrupting_counters() -> None:
    detector: DuplicateDetector[object] = DuplicateDetector()

    with pytest.raises(TypeError):
        detector.process([1, 2])

    # The item was consumed even though exact membership could not be computed.
    assert detector.processed_count == 1
    assert detector.unique_count == 0


def test_hash_and_naive_implementations_match_on_varied_inputs() -> None:
    for size in range(20):
        numbers = [(index * 7) % 13 for index in range(size)]
        for result in (
            two_sum_hash(numbers, 17),
            two_sum_naive(numbers, 17),
        ):
            if result is not None:
                left, right = result
                assert left < right
                assert numbers[left] + numbers[right] == 17
        assert duplicate_flags_hash(numbers) == duplicate_flags_naive(numbers)


def test_small_benchmark_has_every_pattern_and_strategy() -> None:
    rows = run_benchmark((20, 40), seed=1)

    assert len(rows) == 16
    assert {row.pattern for row in rows} == {
        "two_sum",
        "group_anagrams",
        "first_unique",
        "stream_duplicates",
    }
    assert {row.strategy for row in rows} == {"hash", "naive"}
    assert all(row.elapsed_ms >= 0 for row in rows)


def test_benchmark_is_reproducible_in_structure() -> None:
    first = run_benchmark((12,), seed=7)
    second = run_benchmark((12,), seed=7)

    assert [(row.pattern, row.strategy, row.item_count) for row in first] == [
        (row.pattern, row.strategy, row.item_count) for row in second
    ]


def test_benchmark_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        run_benchmark(())
    with pytest.raises(ValueError):
        run_benchmark((10, 0))


def test_benchmark_csv_contains_all_rows(tmp_path: Path) -> None:
    rows = run_benchmark((10,), seed=2)

    path = write_csv(tmp_path / "benchmark.csv", rows)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 9
    assert lines[0] == "pattern,strategy,item_count,elapsed_ms"
