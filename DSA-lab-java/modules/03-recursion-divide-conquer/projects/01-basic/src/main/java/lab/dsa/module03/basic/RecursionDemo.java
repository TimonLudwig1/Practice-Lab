package lab.dsa.module03.basic;

/** Displays representative call trees for every recursive example. */
public final class RecursionDemo {
    private RecursionDemo() {
    }

    public static void main(String[] arguments) {
        run("Factorial: factorial(4)", () -> RecursionVisualizer.factorial(4));
        run("Fibonacci: fibonacci(4)", () -> RecursionVisualizer.fibonacci(4));
        run("Sum: recursiveSum([2, 4, 6])", () -> RecursionVisualizer.recursiveSum(new double[]{2, 4, 6}));
        run("Power: power(2, -3)", () -> RecursionVisualizer.power(2, -3));
    }

    private static void run(String title, Runnable example) {
        System.out.println("\n" + title); System.out.println("=".repeat(title.length())); example.run();
    }
}
