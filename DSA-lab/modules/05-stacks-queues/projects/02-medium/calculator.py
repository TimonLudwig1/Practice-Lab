"""Arithmetic expression parser based on the Shunting-Yard algorithm."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re


NUMBER_PATTERN = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
OPERATORS = {"+", "-", "*", "/", "^"}
UNARY_OPERATORS = {"u+", "u-"}
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "u+": 3, "u-": 3, "^": 4}
RIGHT_ASSOCIATIVE = {"^", "u+", "u-"}


class ExpressionSyntaxError(ValueError):
    """Raised when an infix or postfix expression is malformed."""


class ExpressionEvaluationError(ArithmeticError):
    """Raised when a valid expression has an unsupported numeric result."""


@dataclass(frozen=True)
class ConversionStep:
    """One observable state transition of the Shunting-Yard algorithm."""

    token: str
    output: tuple[str, ...]
    operator_stack: tuple[str, ...]


def tokenize(expression: str) -> list[str]:
    """Split an arithmetic expression into numbers, operators and parentheses.

    Integers, decimal numbers and scientific notation are accepted. Whitespace
    may occur between tokens but not inside a number.
    """
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")

    tokens: list[str] = []
    position = 0

    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue

        number_match = NUMBER_PATTERN.match(expression, position)
        if number_match:
            tokens.append(number_match.group())
            position = number_match.end()
            continue

        character = expression[position]
        if character in OPERATORS or character in "()":
            tokens.append(character)
            position += 1
            continue

        raise ExpressionSyntaxError(
            f"unexpected character {character!r} at position {position}"
        )

    if not tokens:
        raise ExpressionSyntaxError("expression must not be empty")
    return tokens


def parentheses_are_balanced(expression: str) -> bool:
    """Return whether parentheses are balanced and correctly ordered."""
    depth = 0
    for character in expression:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _validate_parentheses(tokens: list[str]) -> None:
    """Raise a descriptive error for mismatched tokenized parentheses."""
    opening_positions: list[int] = []
    for position, token in enumerate(tokens):
        if token == "(":
            opening_positions.append(position)
        elif token == ")":
            if not opening_positions:
                raise ExpressionSyntaxError(
                    f"closing parenthesis at token {position} has no match"
                )
            opening_positions.pop()

    if opening_positions:
        position = opening_positions[-1]
        raise ExpressionSyntaxError(
            f"opening parenthesis at token {position} has no match"
        )


def _should_pop(current: str, stacked: str) -> bool:
    """Return whether ``stacked`` must precede ``current`` in postfix output."""
    if stacked == "(":
        return False
    if current in RIGHT_ASSOCIATIVE:
        return PRECEDENCE[current] < PRECEDENCE[stacked]
    return PRECEDENCE[current] <= PRECEDENCE[stacked]


def infix_to_postfix(
    expression: str, *, trace: list[ConversionStep] | None = None
) -> list[str]:
    """Convert an infix expression to postfix notation with Shunting-Yard.

    Unary plus and minus are represented as ``u+`` and ``u-`` in the returned
    token list. If ``trace`` is provided, one immutable state is appended after
    every consumed token.
    """
    tokens = tokenize(expression)
    _validate_parentheses(tokens)

    output: list[str] = []
    operator_stack: list[str] = []
    expect_operand = True

    for token in tokens:
        if NUMBER_PATTERN.fullmatch(token):
            if not expect_operand:
                raise ExpressionSyntaxError(
                    f"missing operator before number {token!r}"
                )
            output.append(token)
            expect_operand = False

        elif token == "(":
            if not expect_operand:
                raise ExpressionSyntaxError("missing operator before '('")
            operator_stack.append(token)
            expect_operand = True

        elif token == ")":
            if expect_operand:
                raise ExpressionSyntaxError("missing operand before ')'")
            while operator_stack[-1] != "(":
                output.append(operator_stack.pop())
            operator_stack.pop()
            expect_operand = False

        else:
            if expect_operand:
                if token not in {"+", "-"}:
                    raise ExpressionSyntaxError(
                        f"operator {token!r} has no left operand"
                    )
                # Prefix operators wait for their operand. They must not pop a
                # preceding power operator in an expression such as 2^-3.
                operator_stack.append(f"u{token}")
            else:
                while operator_stack and _should_pop(token, operator_stack[-1]):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
                expect_operand = True

        if trace is not None:
            trace.append(
                ConversionStep(token, tuple(output), tuple(operator_stack))
            )

    if expect_operand:
        raise ExpressionSyntaxError("expression ends before an operand")

    while operator_stack:
        output.append(operator_stack.pop())
    return output


def _apply_binary(operator: str, left: float, right: float) -> float:
    """Apply one binary arithmetic operator."""
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return left / right

    result = left**right
    if isinstance(result, complex):
        raise ExpressionEvaluationError("complex results are not supported")
    return result


def evaluate_postfix(tokens: Iterable[str]) -> float:
    """Evaluate postfix tokens using a value stack."""
    stack: list[float] = []

    for token in tokens:
        if token in UNARY_OPERATORS:
            if not stack:
                raise ExpressionSyntaxError(
                    f"unary operator {token!r} has no operand"
                )
            operand = stack.pop()
            stack.append(operand if token == "u+" else -operand)
            continue

        if token in OPERATORS:
            if len(stack) < 2:
                raise ExpressionSyntaxError(
                    f"binary operator {token!r} has fewer than two operands"
                )
            right = stack.pop()
            left = stack.pop()
            stack.append(_apply_binary(token, left, right))
            continue

        try:
            stack.append(float(token))
        except (TypeError, ValueError) as error:
            raise ExpressionSyntaxError(f"invalid postfix token {token!r}") from error

    if len(stack) != 1:
        raise ExpressionSyntaxError(
            f"postfix expression leaves {len(stack)} values on the stack"
        )
    return stack[0]


def evaluate(expression: str) -> float:
    """Parse and evaluate one arithmetic infix expression."""
    return evaluate_postfix(infix_to_postfix(expression))
