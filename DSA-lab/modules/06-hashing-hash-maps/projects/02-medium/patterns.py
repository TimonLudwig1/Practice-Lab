"""Classic hash-based patterns paired with intentionally naive solutions."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar


T = TypeVar("T", bound=Hashable)


def two_sum_hash(numbers: list[int], target: int) -> tuple[int, int] | None:
    """Return the first discovered index pair whose values sum to target."""
    index_by_value: dict[int, int] = {}
    for right_index, value in enumerate(numbers):
        complement = target - value
        if complement in index_by_value:
            return index_by_value[complement], right_index
        # Retaining the first index makes the output deterministic with repeats.
        if value not in index_by_value:
            index_by_value[value] = right_index
    return None


def two_sum_naive(numbers: list[int], target: int) -> tuple[int, int] | None:
    """Return a matching pair by checking every index pair."""
    for left_index in range(len(numbers)):
        for right_index in range(left_index + 1, len(numbers)):
            if numbers[left_index] + numbers[right_index] == target:
                return left_index, right_index
    return None


def _anagram_signature(word: str) -> str:
    """Return a canonical, case-sensitive signature for a word."""
    return "".join(sorted(word))


def group_anagrams_hash(words: list[str]) -> list[list[str]]:
    """Group anagrams through a signature-to-group hash map."""
    groups: dict[str, list[str]] = {}
    for word in words:
        signature = _anagram_signature(word)
        groups.setdefault(signature, []).append(word)
    return list(groups.values())


def group_anagrams_naive(words: list[str]) -> list[list[str]]:
    """Group anagrams by linearly searching existing signatures."""
    signatures: list[str] = []
    groups: list[list[str]] = []
    for word in words:
        signature = _anagram_signature(word)
        try:
            group_index = signatures.index(signature)
        except ValueError:
            signatures.append(signature)
            groups.append([word])
        else:
            groups[group_index].append(word)
    return groups


def first_unique_index_hash(text: str) -> int:
    """Return the first character index with frequency one, or -1."""
    frequencies: dict[str, int] = {}
    for character in text:
        frequencies[character] = frequencies.get(character, 0) + 1
    for index, character in enumerate(text):
        if frequencies[character] == 1:
            return index
    return -1


def first_unique_index_naive(text: str) -> int:
    """Return the first unique index through repeated linear counting."""
    for index, character in enumerate(text):
        if text.count(character) == 1:
            return index
    return -1


class DuplicateDetector(Generic[T]):
    """Stateful exact duplicate detector for an incremental data stream."""

    def __init__(self) -> None:
        self._seen: set[T] = set()
        self._processed_count = 0
        self._duplicate_count = 0

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def unique_count(self) -> int:
        return len(self._seen)

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    def process(self, item: T) -> bool:
        """Consume one item and return whether it has appeared before."""
        self._processed_count += 1
        if item in self._seen:
            self._duplicate_count += 1
            return True
        self._seen.add(item)
        return False

    def reset(self) -> None:
        """Forget all history and reset counters."""
        self._seen.clear()
        self._processed_count = 0
        self._duplicate_count = 0


def duplicate_flags_hash(items: Iterable[T]) -> list[bool]:
    """Mark every item that was seen earlier in the stream."""
    detector: DuplicateDetector[T] = DuplicateDetector()
    return [detector.process(item) for item in items]


def duplicate_flags_naive(items: Iterable[T]) -> list[bool]:
    """Mark duplicates using a linearly searched history list."""
    history: list[T] = []
    flags: list[bool] = []
    for item in items:
        duplicate = item in history
        flags.append(duplicate)
        if not duplicate:
            history.append(item)
    return flags
