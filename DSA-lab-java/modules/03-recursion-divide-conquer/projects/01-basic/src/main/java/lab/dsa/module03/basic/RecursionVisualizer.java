package lab.dsa.module03.basic;

import java.util.Arrays;
import java.util.function.Supplier;

/** Traces recursive calls as an indented call tree. */
public final class RecursionVisualizer {
    private static final ThreadLocal<Integer> TRACE_DEPTH = ThreadLocal.withInitial(() -> 0);

    private RecursionVisualizer() {
    }

    public static <T> T trace(String call, Supplier<T> body) {
        int depth = TRACE_DEPTH.get();
        String indent = "  ".repeat(depth);
        System.out.println(indent + "→ " + call);
        TRACE_DEPTH.set(depth + 1);
        try {
            T result = body.get();
            TRACE_DEPTH.set(depth);
            System.out.println(indent + "← " + call + " = " + result);
            return result;
        } catch (RuntimeException error) {
            TRACE_DEPTH.set(depth);
            System.out.println(indent + "! " + call + " raised "
                    + error.getClass().getSimpleName() + ": " + error.getMessage());
            throw error;
        }
    }

    public static long factorial(int number) {
        return trace("factorial(" + number + ")", () -> {
            validateNonNegative(number, "number");
            return number == 0 ? 1L : number * factorial(number - 1);
        });
    }

    public static long fibonacci(int number) {
        return trace("fibonacci(" + number + ")", () -> {
            validateNonNegative(number, "number");
            return number < 2 ? (long) number : fibonacci(number - 1) + fibonacci(number - 2);
        });
    }

    public static double recursiveSum(double[] values) {
        return recursiveSum(values, 0);
    }

    public static double recursiveSum(double[] values, int index) {
        String call = "recursiveSum(" + Arrays.toString(values) + ", index=" + index + ")";
        return trace(call, () -> {
            if (index < 0 || index > values.length) {
                throw new IndexOutOfBoundsException("Index outside array: " + index);
            }
            return index == values.length ? 0.0 : values[index] + recursiveSum(values, index + 1);
        });
    }

    public static double power(double base, int exponent) {
        String call = "power(" + formatNumber(base) + ", " + exponent + ")";
        return trace(call, () -> {
            if (exponent == 0) return 1.0;
            if (exponent < 0) {
                if (base == 0) throw new ArithmeticException("Zero cannot have a negative exponent");
                if (exponent == Integer.MIN_VALUE) {
                    return 1.0 / (power(base, Integer.MAX_VALUE) * base);
                }
                return 1.0 / power(base, -exponent);
            }
            return base * power(base, exponent - 1);
        });
    }

    private static void validateNonNegative(int value, String name) {
        if (value < 0) throw new IllegalArgumentException(name + " must be non-negative");
    }

    private static String formatNumber(double value) {
        return value == Math.rint(value) ? Long.toString((long) value) : Double.toString(value);
    }
}
