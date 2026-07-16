"""Tests for all array and string patterns in the catalog."""

import unittest

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


class RotateRightTests(unittest.TestCase):
    def test_rotates_typical_input(self) -> None:
        values = [1, 2, 3, 4, 5, 6, 7]
        rotate_right_in_place(values, 3)
        self.assertEqual(values, [5, 6, 7, 1, 2, 3, 4])

    def test_normalizes_large_step_count(self) -> None:
        values = [1, 2, 3]
        rotate_right_in_place(values, 7)
        self.assertEqual(values, [3, 1, 2])

    def test_negative_steps_rotate_left(self) -> None:
        values = [1, 2, 3, 4]
        rotate_right_in_place(values, -1)
        self.assertEqual(values, [2, 3, 4, 1])

    def test_empty_and_singleton_inputs(self) -> None:
        empty: list[int] = []
        singleton = [1]
        rotate_right_in_place(empty, 100)
        rotate_right_in_place(singleton, -100)
        self.assertEqual(empty, [])
        self.assertEqual(singleton, [1])

    def test_rejects_non_integer_steps(self) -> None:
        with self.assertRaises(TypeError):
            rotate_right_in_place([1, 2], 1.5)  # type: ignore[arg-type]


class MergeSortedTests(unittest.TestCase):
    def test_merges_interleaved_values(self) -> None:
        target = [1, 4, 7, None, None, None]
        merge_sorted_in_place(target, 3, [2, 3, 8])
        self.assertEqual(target, [1, 2, 3, 4, 7, 8])

    def test_copies_smaller_other_values(self) -> None:
        target = [4, 5, 6, None, None, None]
        merge_sorted_in_place(target, 3, [1, 2, 3])
        self.assertEqual(target, [1, 2, 3, 4, 5, 6])

    def test_handles_empty_sides(self) -> None:
        only_other = [None, None]
        merge_sorted_in_place(only_other, 0, [1, 2])
        only_target = [1, 2]
        merge_sorted_in_place(only_target, 2, [])
        self.assertEqual(only_other, [1, 2])
        self.assertEqual(only_target, [1, 2])

    def test_preserves_duplicate_values(self) -> None:
        target = [1, 2, None, None]
        merge_sorted_in_place(target, 2, [1, 2])
        self.assertEqual(target, [1, 1, 2, 2])

    def test_rejects_wrong_buffer_size(self) -> None:
        with self.assertRaises(ValueError):
            merge_sorted_in_place([1, None, None], 1, [2])

    def test_rejects_unsorted_inputs(self) -> None:
        with self.assertRaises(ValueError):
            merge_sorted_in_place([2, 1, None], 2, [3])
        with self.assertRaises(ValueError):
            merge_sorted_in_place([1, None, None], 1, [3, 2])

    def test_rejects_invalid_valid_count(self) -> None:
        with self.assertRaises(ValueError):
            merge_sorted_in_place([None], -1, [1, 2])
        with self.assertRaises(TypeError):
            merge_sorted_in_place([1], 1.0, [])  # type: ignore[arg-type]


class PrefixSumTests(unittest.TestCase):
    def test_answers_multiple_ranges(self) -> None:
        prefix = PrefixSum.from_values([4, -1, 7, 3, 2])
        self.assertEqual(prefix.range_sum(0, 5), 15)
        self.assertEqual(prefix.range_sum(1, 4), 9)
        self.assertEqual(prefix.range_sum(2, 3), 7)

    def test_empty_range_is_zero(self) -> None:
        prefix = PrefixSum.from_values([3, 4])
        self.assertEqual(prefix.range_sum(1, 1), 0)

    def test_empty_input(self) -> None:
        prefix = PrefixSum.from_values([])
        self.assertEqual(len(prefix), 0)
        self.assertEqual(prefix.range_sum(0, 0), 0)

    def test_supports_floating_point_values(self) -> None:
        prefix = PrefixSum.from_values([0.5, 1.25, -0.25])
        self.assertAlmostEqual(prefix.range_sum(0, 3), 1.5)

    def test_is_independent_of_source_mutation(self) -> None:
        source = [1, 2, 3]
        prefix = PrefixSum.from_values(source)
        source[0] = 100
        self.assertEqual(prefix.range_sum(0, 3), 6)

    def test_rejects_invalid_ranges(self) -> None:
        prefix = PrefixSum.from_values([1, 2, 3])
        for boundaries in [(-1, 2), (2, 1), (0, 4)]:
            with self.subTest(boundaries=boundaries), self.assertRaises(IndexError):
                prefix.range_sum(*boundaries)

    def test_rejects_non_integer_boundaries(self) -> None:
        prefix = PrefixSum.from_values([1, 2])
        with self.assertRaises(TypeError):
            prefix.range_sum(0.0, 2)  # type: ignore[arg-type]


class AnagramTests(unittest.TestCase):
    def test_detects_anagram(self) -> None:
        self.assertTrue(are_anagrams("listen", "silent"))

    def test_is_case_and_space_sensitive(self) -> None:
        self.assertFalse(are_anagrams("Listen", "silent"))
        self.assertTrue(are_anagrams("a b", "ab "))
        self.assertFalse(are_anagrams("a b", "abx"))

    def test_handles_unicode_and_empty_strings(self) -> None:
        self.assertTrue(are_anagrams("äöä", "ääö"))
        self.assertTrue(are_anagrams("", ""))

    def test_rejects_different_counts(self) -> None:
        self.assertFalse(are_anagrams("aab", "abb"))
        self.assertFalse(are_anagrams("abc", "ab"))

    def test_rejects_non_strings(self) -> None:
        with self.assertRaises(TypeError):
            are_anagrams("123", 123)  # type: ignore[arg-type]


class RemoveDuplicatesTests(unittest.TestCase):
    def test_removes_repeated_values(self) -> None:
        values = [1, 1, 2, 2, 2, 5]
        length = remove_duplicates_sorted(values)
        self.assertEqual(length, 3)
        self.assertEqual(values, [1, 2, 5])

    def test_handles_empty_and_unique_inputs(self) -> None:
        empty: list[int] = []
        unique = [1, 2, 3]
        self.assertEqual(remove_duplicates_sorted(empty), 0)
        self.assertEqual(remove_duplicates_sorted(unique), 3)
        self.assertEqual(unique, [1, 2, 3])

    def test_handles_all_equal_values(self) -> None:
        values = [4, 4, 4, 4]
        self.assertEqual(remove_duplicates_sorted(values), 1)
        self.assertEqual(values, [4])

    def test_rejects_unsorted_input(self) -> None:
        values = [1, 3, 2]
        with self.assertRaises(ValueError):
            remove_duplicates_sorted(values)
        self.assertEqual(values, [1, 3, 2])


class MoveZerosTests(unittest.TestCase):
    def test_moves_zeros_stably(self) -> None:
        values = [0, 3, 0, 1, 0, 8]
        non_zero_count = move_zeros_to_end(values)
        self.assertEqual(non_zero_count, 3)
        self.assertEqual(values, [3, 1, 8, 0, 0, 0])

    def test_handles_no_zeros(self) -> None:
        values = [1, 2, 3]
        self.assertEqual(move_zeros_to_end(values), 3)
        self.assertEqual(values, [1, 2, 3])

    def test_handles_only_zeros_and_empty_input(self) -> None:
        zeros = [0, 0]
        empty: list[int] = []
        self.assertEqual(move_zeros_to_end(zeros), 0)
        self.assertEqual(move_zeros_to_end(empty), 0)
        self.assertEqual(zeros, [0, 0])

    def test_treats_float_zero_as_zero(self) -> None:
        values = [0.0, 2.5, -0.0, -1.0]
        move_zeros_to_end(values)
        self.assertEqual(values, [2.5, -1.0, 0.0, -0.0])


class ProductExceptSelfTests(unittest.TestCase):
    def test_computes_products(self) -> None:
        self.assertEqual(product_except_self([1, 2, 3, 4]), [24, 12, 8, 6])

    def test_handles_one_zero(self) -> None:
        self.assertEqual(product_except_self([1, 2, 0, 4]), [0, 0, 8, 0])

    def test_handles_multiple_zeros(self) -> None:
        self.assertEqual(product_except_self([0, 2, 0, 4]), [0, 0, 0, 0])

    def test_handles_negative_values(self) -> None:
        self.assertEqual(product_except_self([-1, 2, -3]), [-6, 3, -2])

    def test_handles_empty_and_singleton_inputs(self) -> None:
        self.assertEqual(product_except_self([]), [])
        self.assertEqual(product_except_self([7]), [1])


class LongestUniqueSubstringTests(unittest.TestCase):
    def test_finds_typical_window(self) -> None:
        self.assertEqual(longest_unique_substring("abcabcbb"), "abc")

    def test_moves_start_past_last_duplicate(self) -> None:
        self.assertEqual(longest_unique_substring("abba"), "ab")
        self.assertEqual(longest_unique_substring("pwwkew"), "wke")

    def test_returns_earliest_tie(self) -> None:
        self.assertEqual(longest_unique_substring("abcaef"), "bcaef")
        self.assertEqual(longest_unique_substring("abcade"), "bcade")

    def test_handles_empty_and_repeated_inputs(self) -> None:
        self.assertEqual(longest_unique_substring(""), "")
        self.assertEqual(longest_unique_substring("aaaa"), "a")

    def test_rejects_non_string(self) -> None:
        with self.assertRaises(TypeError):
            longest_unique_substring(["a", "b"])  # type: ignore[arg-type]


class SpiralOrderTests(unittest.TestCase):
    def test_traverses_square_matrix(self) -> None:
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(spiral_order(matrix), [1, 2, 3, 6, 9, 8, 7, 4, 5])

    def test_traverses_wide_matrix(self) -> None:
        matrix = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(spiral_order(matrix), [1, 2, 3, 4, 8, 7, 6, 5])

    def test_traverses_tall_matrix(self) -> None:
        matrix = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.assertEqual(spiral_order(matrix), [1, 2, 4, 6, 8, 7, 5, 3])

    def test_handles_single_row_and_column(self) -> None:
        self.assertEqual(spiral_order([[1, 2, 3]]), [1, 2, 3])
        self.assertEqual(spiral_order([[1], [2], [3]]), [1, 2, 3])

    def test_handles_empty_shapes(self) -> None:
        self.assertEqual(spiral_order([]), [])
        self.assertEqual(spiral_order([[], []]), [])

    def test_rejects_ragged_matrix(self) -> None:
        with self.assertRaises(ValueError):
            spiral_order([[1, 2], [3]])


class CompressRunsTests(unittest.TestCase):
    def test_compresses_repeated_runs(self) -> None:
        characters = list("aabcccccaaa")
        length = compress_runs_in_place(characters)
        self.assertEqual(length, 7)
        self.assertEqual(characters, list("a2bc5a3"))

    def test_writes_multi_digit_counts(self) -> None:
        characters = list("abbbbbbbbbbbb")
        compress_runs_in_place(characters)
        self.assertEqual(characters, list("ab12"))

    def test_keeps_single_characters_unannotated(self) -> None:
        characters = list("abcd")
        self.assertEqual(compress_runs_in_place(characters), 4)
        self.assertEqual(characters, list("abcd"))

    def test_handles_empty_input(self) -> None:
        characters: list[str] = []
        self.assertEqual(compress_runs_in_place(characters), 0)
        self.assertEqual(characters, [])

    def test_rejects_non_character_elements(self) -> None:
        with self.assertRaises(ValueError):
            compress_runs_in_place(["a", "bc"])
        with self.assertRaises(ValueError):
            compress_runs_in_place(["a", 2])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
