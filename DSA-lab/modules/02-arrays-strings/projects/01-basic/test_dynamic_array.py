"""Tests for the DynamicArray implementation and growth experiment."""

import csv
import tempfile
import unittest
from pathlib import Path

from dynamic_array import DynamicArray, GrowthEvent
from growth_experiment import run_experiment, write_csv


class DynamicArrayTests(unittest.TestCase):
    """Verify storage, operations, bounds, and resize metadata."""

    def test_initial_state_uses_fixed_buffer(self):
        array: DynamicArray[int] = DynamicArray(initial_capacity=3)
        self.assertEqual(len(array), 0)
        self.assertEqual(array.capacity, 3)
        self.assertNotIsInstance(array._buffer, list)
        self.assertEqual(array.to_list(), [])

    def test_invalid_initial_capacity_is_rejected(self):
        for capacity in (0, -1, -10):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    DynamicArray(capacity)

    def test_append_preserves_order_and_doubles_capacity(self):
        array: DynamicArray[int] = DynamicArray()
        capacities = []
        for value in range(9):
            array.append(value)
            capacities.append(array.capacity)

        self.assertEqual(array.to_list(), list(range(9)))
        self.assertEqual(capacities, [1, 2, 4, 4, 8, 8, 8, 8, 16])

    def test_get_set_and_negative_indices(self):
        array: DynamicArray[str] = DynamicArray()
        for value in ("A", "B", "C"):
            array.append(value)

        self.assertEqual(array[0], "A")
        self.assertEqual(array[-1], "C")
        array[1] = "X"
        array[-1] = "Z"
        self.assertEqual(array.to_list(), ["A", "X", "Z"])

    def test_access_bounds_are_enforced(self):
        array: DynamicArray[int] = DynamicArray()
        array.append(10)
        for index in (1, -2, 100):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    _ = array[index]
                with self.assertRaises(IndexError):
                    array[index] = 0

    def test_insert_at_start_middle_and_end(self):
        array: DynamicArray[str] = DynamicArray()
        array.append("B")
        array.append("D")
        array.insert(0, "A")
        array.insert(2, "C")
        array.insert(len(array), "E")
        self.assertEqual(array.to_list(), ["A", "B", "C", "D", "E"])

    def test_insert_can_trigger_resize(self):
        array: DynamicArray[int] = DynamicArray(initial_capacity=2)
        array.append(1)
        array.append(3)
        array.insert(1, 2)

        self.assertEqual(array.to_list(), [1, 2, 3])
        self.assertEqual(array.capacity, 4)
        self.assertEqual(array.growth_events[-1].copied_elements, 2)

    def test_insert_bounds_are_enforced(self):
        array: DynamicArray[int] = DynamicArray()
        array.append(1)
        for index in (-1, 2, 99):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    array.insert(index, 0)

    def test_delete_returns_value_and_shifts_left(self):
        array: DynamicArray[str] = DynamicArray()
        for value in ("A", "B", "C", "D"):
            array.append(value)

        self.assertEqual(array.delete(1), "B")
        self.assertEqual(array.to_list(), ["A", "C", "D"])
        self.assertEqual(array.delete(-1), "D")
        self.assertEqual(array.to_list(), ["A", "C"])

    def test_delete_does_not_shrink_capacity(self):
        array: DynamicArray[int] = DynamicArray()
        for value in range(5):
            array.append(value)
        capacity_before = array.capacity

        for _ in range(4):
            array.delete(0)

        self.assertEqual(array.to_list(), [4])
        self.assertEqual(array.capacity, capacity_before)

    def test_delete_bounds_are_enforced(self):
        array: DynamicArray[int] = DynamicArray()
        for index in (0, -1):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    array.delete(index)

    def test_mixed_python_objects_are_supported(self):
        marker = object()
        array: DynamicArray[object] = DynamicArray()
        for value in (42, "DSA", None, marker):
            array.append(value)
        self.assertEqual(array[0], 42)
        self.assertEqual(array[1], "DSA")
        self.assertIsNone(array[2])
        self.assertIs(array[3], marker)

    def test_growth_events_are_exact_and_immutable(self):
        array: DynamicArray[int] = DynamicArray()
        for value in range(9):
            array.append(value)

        self.assertEqual(
            array.growth_events,
            (
                GrowthEvent(1, 1, 2, 1),
                GrowthEvent(2, 2, 4, 2),
                GrowthEvent(4, 4, 8, 4),
                GrowthEvent(8, 8, 16, 8),
            ),
        )
        self.assertIsInstance(array.growth_events, tuple)
        self.assertEqual(array.total_copied_elements, 15)

    def test_total_resize_copies_have_linear_bound(self):
        for append_count in (1, 2, 3, 8, 16, 100, 1_000):
            with self.subTest(append_count=append_count):
                array: DynamicArray[int] = DynamicArray()
                for value in range(append_count):
                    array.append(value)
                self.assertLess(array.total_copied_elements, 2 * append_count)

    def test_iteration_and_repr(self):
        array: DynamicArray[int] = DynamicArray(initial_capacity=2)
        array.append(3)
        array.append(5)
        self.assertEqual(list(array), [3, 5])
        self.assertEqual(repr(array), "DynamicArray([3, 5], capacity=2)")


class GrowthExperimentTests(unittest.TestCase):
    """Verify deterministic cost accounting and CSV output."""

    def test_first_eight_append_costs_show_resize_spikes(self):
        measurements = run_experiment(8)
        self.assertEqual(
            [point.actual_cost for point in measurements],
            [1, 2, 3, 1, 5, 1, 1, 1],
        )
        self.assertEqual(
            [point.capacity for point in measurements],
            [1, 2, 4, 4, 8, 8, 8, 8],
        )

    def test_experiment_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            run_experiment(0)
        with self.assertRaises(ValueError):
            run_experiment(10, initial_capacity=0)

    def test_csv_contains_one_row_per_append(self):
        measurements = run_experiment(5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "growth.csv"
            write_csv(measurements, destination)
            with destination.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]["capacity"], "8")
        self.assertEqual(rows[-1]["actual_cost"], "5")


if __name__ == "__main__":
    unittest.main()
