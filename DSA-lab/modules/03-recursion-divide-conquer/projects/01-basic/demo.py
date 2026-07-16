"""Display representative call trees for all recursive examples."""

from recursion_visualizer import factorial, fibonacci, power, recursive_sum


def _run(title: str, function: object, *arguments: object) -> None:
    """Print a heading and invoke one traced callable."""
    print(f"\n{title}")
    print("=" * len(title))
    function(*arguments)  # type: ignore[operator]


def main() -> None:
    """Run the complete visualization demo."""
    _run("Fakultät: factorial(4)", factorial, 4)
    _run("Fibonacci: fibonacci(4)", fibonacci, 4)
    _run("Summe: recursive_sum([2, 4, 6])", recursive_sum, [2, 4, 6])
    _run("Potenz: power(2, -3)", power, 2, -3)


if __name__ == "__main__":
    main()
