"""Tests for tokenization, Shunting-Yard conversion and stack evaluation."""

import pytest

from calculator import (
    ConversionStep,
    ExpressionEvaluationError,
    ExpressionSyntaxError,
    evaluate,
    evaluate_postfix,
    infix_to_postfix,
    parentheses_are_balanced,
    tokenize,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("12 + 3.5", ["12", "+", "3.5"]),
        (".5 * 2.", [".5", "*", "2."]),
        ("1e3 + 2.5E-2", ["1e3", "+", "2.5E-2"]),
        ("-(2 + 3)", ["-", "(", "2", "+", "3", ")"]),
    ],
)
def test_tokenize_valid_expression(expression: str, expected: list[str]) -> None:
    assert tokenize(expression) == expected


@pytest.mark.parametrize("expression", ["", "   ", "2 & 3", "2,5 + 1", "abc"])
def test_tokenize_rejects_empty_or_unknown_input(expression: str) -> None:
    with pytest.raises(ExpressionSyntaxError):
        tokenize(expression)


def test_tokenize_requires_string() -> None:
    with pytest.raises(TypeError, match="string"):
        tokenize(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("", True),
        ("1 + 2", True),
        ("((1) + (2))", True),
        ("(", False),
        (")(", False),
        ("(1 + 2))", False),
    ],
)
def test_parentheses_are_balanced(expression: str, expected: bool) -> None:
    assert parentheses_are_balanced(expression) is expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", ["2", "3", "+"]),
        ("2 + 3 * 4", ["2", "3", "4", "*", "+"]),
        ("(2 + 3) * 4", ["2", "3", "+", "4", "*"]),
        ("10 - 6 / 2", ["10", "6", "2", "/", "-"]),
        ("2^3^2", ["2", "3", "2", "^", "^"]),
        ("-2^2", ["2", "2", "^", "u-"]),
        ("2^-3", ["2", "3", "u-", "^"]),
        ("--2", ["2", "u-", "u-"]),
        ("3 * -(2 + 1)", ["3", "2", "1", "+", "u-", "*"]),
    ],
)
def test_infix_to_postfix(expression: str, expected: list[str]) -> None:
    assert infix_to_postfix(expression) == expected


def test_conversion_trace_contains_one_snapshot_per_token() -> None:
    trace: list[ConversionStep] = []

    postfix = infix_to_postfix("2 + 3 * 4", trace=trace)

    assert len(trace) == 5
    assert trace[0] == ConversionStep("2", ("2",), ())
    assert trace[-1] == ConversionStep("4", ("2", "3", "4"), ("+", "*"))
    assert postfix == ["2", "3", "4", "*", "+"]


@pytest.mark.parametrize(
    "expression",
    [
        "(1 + 2",
        "1 + 2)",
        "()",
        "2 3",
        "2(3 + 4)",
        "(2 + 3)4",
        "2 +",
        "* 2",
        "2 ** 3",
        "2 + ()",
    ],
)
def test_infix_to_postfix_rejects_invalid_syntax(expression: str) -> None:
    with pytest.raises(ExpressionSyntaxError):
        infix_to_postfix(expression)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["2", "3", "+"], 5.0),
        (["8", "2", "/", "3", "*"], 12.0),
        (["2", "3", "2", "^", "^"], 512.0),
        (["4", "u-", "2", "+"], -2.0),
        (["4", "u+"], 4.0),
    ],
)
def test_evaluate_postfix(tokens: list[str], expected: float) -> None:
    assert evaluate_postfix(tokens) == pytest.approx(expected)


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        ["2", "3"],
        ["+"],
        ["2", "+"],
        ["u-"],
        ["2", "unknown", "+"],
    ],
)
def test_evaluate_postfix_rejects_malformed_input(tokens: list[str]) -> None:
    with pytest.raises(ExpressionSyntaxError):
        evaluate_postfix(tokens)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7.0),
        ("(1 + 2) * 3", 9.0),
        ("10 / 4", 2.5),
        ("2^3^2", 512.0),
        ("-2^2", -4.0),
        ("(-2)^2", 4.0),
        ("2^-3", 0.125),
        ("1e2 + .5", 100.5),
        ("3 + 4 * 2 / (1 - 5)^2^3", 3.0001220703125),
    ],
)
def test_evaluate(expression: str, expected: float) -> None:
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize("expression", ["1 / 0", "10 / (3 - 3)"])
def test_evaluate_reports_division_by_zero(expression: str) -> None:
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        evaluate(expression)


def test_evaluate_rejects_complex_results() -> None:
    with pytest.raises(ExpressionEvaluationError, match="complex"):
        evaluate("(-1)^0.5")
