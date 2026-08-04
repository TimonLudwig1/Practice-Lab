package lab.dsa.module03.basic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;

class RecursionVisualizerTest {
    private record Captured<T>(T result, String output) {
    }

    private static <T> Captured<T> capture(Supplier<T> operation) {
        PrintStream original = System.out; ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (PrintStream replacement = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
            System.setOut(replacement); T result = operation.get(); return new Captured<>(result, bytes.toString(StandardCharsets.UTF_8));
        } finally { System.setOut(original); }
    }

    @Test void factorialTracePrintsIndentedEntryAndReturnLines() {
        Captured<Long> captured = capture(() -> RecursionVisualizer.factorial(2));
        assertEquals(2, captured.result());
        assertEquals(List.of("→ factorial(2)", "  → factorial(1)", "    → factorial(0)",
                "    ← factorial(0) = 1", "  ← factorial(1) = 1", "← factorial(2) = 2"),
                captured.output().lines().toList());
    }

    @Test void exceptionIsPrintedAndDepthIsReset() {
        PrintStream original = System.out; ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (PrintStream replacement = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
            System.setOut(replacement);
            assertThrows(IllegalArgumentException.class, () -> RecursionVisualizer.factorial(-1));
            RecursionVisualizer.factorial(0);
        } finally { System.setOut(original); }
        List<String> lines = bytes.toString(StandardCharsets.UTF_8).lines().toList();
        assertEquals("→ factorial(-1)", lines.get(0)); assertTrue(lines.get(1).contains("raised IllegalArgumentException"));
        assertEquals("→ factorial(0)", lines.get(2));
    }

    @Test void differentTracedOperationsShareDepth() {
        Captured<String> captured = capture(() -> RecursionVisualizer.trace("outer()",
                () -> RecursionVisualizer.trace("inner()", () -> "done")));
        assertEquals("done", captured.result()); assertTrue(captured.output().contains("  → inner()"));
        assertTrue(captured.output().contains("  ← inner() = done"));
    }

    @Test void factorialHandlesBaseRecursiveAndInvalidCases() {
        assertEquals(1, capture(() -> RecursionVisualizer.factorial(0)).result());
        assertEquals(720, capture(() -> RecursionVisualizer.factorial(6)).result());
        assertThrows(IllegalArgumentException.class, () -> capture(() -> RecursionVisualizer.factorial(-1)));
    }

    @Test void fibonacciHandlesBaseCasesAndRecursiveValues() {
        List<Long> actual = java.util.stream.IntStream.range(0, 8)
                .mapToObj(number -> capture(() -> RecursionVisualizer.fibonacci(number)).result()).toList();
        assertEquals(List.of(0L, 1L, 1L, 2L, 3L, 5L, 8L, 13L), actual);
        Captured<Long> trace = capture(() -> RecursionVisualizer.fibonacci(3));
        assertEquals(2, trace.output().split("→ fibonacci\\(1\\)", -1).length - 1);
    }

    @Test void recursiveSumHandlesValuesEmptyInputAndSuffixes() {
        assertEquals(12, capture(() -> RecursionVisualizer.recursiveSum(new double[]{2, 4, 6})).result());
        assertEquals(1.5, capture(() -> RecursionVisualizer.recursiveSum(new double[]{.5, 1.25, -.25})).result());
        assertEquals(0, capture(() -> RecursionVisualizer.recursiveSum(new double[]{})).result());
        assertEquals(50, capture(() -> RecursionVisualizer.recursiveSum(new double[]{10, 20, 30}, 1)).result());
        assertThrows(IndexOutOfBoundsException.class,
                () -> capture(() -> RecursionVisualizer.recursiveSum(new double[]{1, 2}, 3)));
    }

    @Test void powerHandlesZeroPositiveNegativeAndInvalidZeroCases() {
        assertEquals(1, capture(() -> RecursionVisualizer.power(9, 0)).result());
        assertEquals(81, capture(() -> RecursionVisualizer.power(3, 4)).result());
        assertEquals(.125, capture(() -> RecursionVisualizer.power(2, -3)).result());
        assertEquals(-8, capture(() -> RecursionVisualizer.power(-2, 3)).result());
        assertThrows(ArithmeticException.class, () -> capture(() -> RecursionVisualizer.power(0, -1)));
    }
}
