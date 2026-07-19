"""Three applications of binary search on an integer answer space."""

from __future__ import annotations

from collections.abc import Sequence

from answer_search import AnswerStep, first_true


def required_shipping_days(weights: Sequence[int], capacity: int) -> int | float:
    """Return required days, or infinity when one item exceeds capacity."""

    if capacity < 1:
        raise ValueError("capacity must be positive")
    if any(
        not isinstance(weight, int) or isinstance(weight, bool) or weight < 1
        for weight in weights
    ):
        raise ValueError("weights must contain positive integers")
    if not weights:
        return 0

    days = 1
    current_load = 0
    for weight in weights:
        if weight > capacity:
            return float("inf")
        if current_load + weight > capacity:
            days += 1
            current_load = 0
        current_load += weight
    return days


def minimum_shipping_capacity(
    weights: Sequence[int],
    day_limit: int,
    *,
    trace: list[AnswerStep] | None = None,
) -> int:
    """Return the smallest capacity that ships all weights within day_limit.

    Monotonicity: if a capacity is feasible, every larger capacity is feasible
    because it can reproduce the same day partition without adding a day.
    """

    if not weights:
        raise ValueError("weights must not be empty")
    if (
        not isinstance(day_limit, int)
        or isinstance(day_limit, bool)
        or day_limit < 1
    ):
        raise ValueError("day_limit must be a positive integer")
    if any(
        not isinstance(weight, int) or isinstance(weight, bool) or weight < 1
        for weight in weights
    ):
        raise ValueError("weights must contain positive integers")

    low, high = max(weights), sum(weights)
    return first_true(
        low,
        high,
        lambda capacity: required_shipping_days(weights, capacity) <= day_limit,
        problem="minimum_shipping_capacity",
        trace=trace,
    )


def integer_square_root(
    number: int, *, trace: list[AnswerStep] | None = None
) -> int:
    """Return floor(sqrt(number)) without floating point or math.sqrt.

    The search finds the first integer whose square is strictly greater than
    ``number``. This predicate changes exactly once from false to true; the
    preceding integer is the requested floor square root.
    """

    if not isinstance(number, int) or isinstance(number, bool) or number < 0:
        raise ValueError("number must be a non-negative integer")
    first_too_large = first_true(
        0,
        number + 1,
        lambda candidate: candidate * candidate > number,
        problem="integer_square_root",
        trace=trace,
    )
    return first_too_large - 1


def _validate_matrix(matrix: Sequence[Sequence[int]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    column_count = len(matrix[0])
    if any(len(row) != column_count for row in matrix):
        raise ValueError("matrix must be rectangular")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for row in matrix
        for value in row
    ):
        raise ValueError("matrix values must be integers")
    if any(
        row[index] > row[index + 1]
        for row in matrix
        for index in range(column_count - 1)
    ):
        raise ValueError("matrix rows must be sorted")
    if any(
        matrix[row][column] > matrix[row + 1][column]
        for row in range(len(matrix) - 1)
        for column in range(column_count)
    ):
        raise ValueError("matrix columns must be sorted")
    return len(matrix), column_count


def count_less_equal(matrix: Sequence[Sequence[int]], candidate: int) -> int:
    """Count matrix values <= candidate in O(rows + columns) time.

    Preconditions: rows and columns are sorted in non-decreasing order and the
    matrix is rectangular. Validation is intentionally left to the public
    ``kth_smallest`` entry point so this predicate stays efficient per round.
    """

    row = len(matrix) - 1
    column = 0
    column_count = len(matrix[0])
    count = 0
    while row >= 0 and column < column_count:
        if matrix[row][column] <= candidate:
            count += row + 1
            column += 1
        else:
            row -= 1
    return count


def kth_smallest(
    matrix: Sequence[Sequence[int]],
    k: int,
    *,
    trace: list[AnswerStep] | None = None,
) -> int:
    """Return the kth smallest value in a row- and column-sorted matrix.

    Monotonicity: the number of matrix entries <= candidate never decreases as
    candidate grows. Therefore ``count >= k`` changes from false to true, and
    its first true candidate is the kth smallest value.
    """

    row_count, column_count = _validate_matrix(matrix)
    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or not 1 <= k <= row_count * column_count
    ):
        raise ValueError("k must be a valid one-based rank")

    low = matrix[0][0]
    high = matrix[-1][-1]
    return first_true(
        low,
        high,
        lambda candidate: count_less_equal(matrix, candidate) >= k,
        problem="kth_smallest",
        trace=trace,
    )
