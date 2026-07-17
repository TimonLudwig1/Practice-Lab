"""Show the conversion and evaluation of an arithmetic expression."""

from calculator import ConversionStep, evaluate_postfix, infix_to_postfix


def print_trace(expression: str) -> None:
    """Print all Shunting-Yard states for ``expression``."""
    trace: list[ConversionStep] = []
    postfix = infix_to_postfix(expression, trace=trace)

    print(f"Infix:   {expression}")
    print("\nToken      Postfix output                    Operator stack")
    print("-" * 68)
    for step in trace:
        output = " ".join(step.output) or "-"
        operators = " ".join(step.operator_stack) or "-"
        print(f"{step.token:<10} {output:<33} {operators}")

    print(f"\nPostfix: {' '.join(postfix)}")
    print(f"Result:  {evaluate_postfix(postfix):g}")


if __name__ == "__main__":
    print_trace("3 + 4 * 2 / (1 - 5)^2^3")
