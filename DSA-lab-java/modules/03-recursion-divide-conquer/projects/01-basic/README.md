# 01-basic — Recursion Visualizer

## Goal

Make the call stack visible. Four recursive methods use a shared tracer that
prints entry, return, and exception lines with two spaces per active recursive
level. Compare the generated call tree with a manual stack trace.

Java annotations cannot transparently wrap arbitrary method calls without an
additional framework. This edition therefore uses a small generic `Tracer.call`
helper. It provides the same separation between algorithm and visualization,
shares depth across traced methods, restores depth after exceptions, and uses a
`ThreadLocal` so concurrent threads do not share mutable depth.

## Tasks

1. Study the tracer's depth invariant and exception-safe `finally` restoration.
2. Implement factorial with `0! = 1` and strictly decreasing input.
3. Implement intentionally naive branching Fibonacci.
4. Sum from a progressing index without copying array suffixes.
5. Implement linear recursive power, including negative integer exponents.

Expected trace:

~~~text
→ factorial(2)
  → factorial(1)
    → factorial(0)
    ← factorial(0) = 1
  ← factorial(1) = 1
← factorial(2) = 2
~~~

## Commands

~~~bash
mvn test
mvn exec:java
~~~

Observe the deepest factorial stack, repeated Fibonacci subproblems, the reverse
order of return lines, and whether a valid top-level call starts at depth zero
after an invalid call.

## Done when

- entry, return, and exception traces are correctly indented,
- all four recursive methods produce correct results and have reachable base cases,
- errors leave the next call at depth zero,
- all JUnit tests and the demo pass,
- you can explain descent, active frames, and unwinding from the trace.
