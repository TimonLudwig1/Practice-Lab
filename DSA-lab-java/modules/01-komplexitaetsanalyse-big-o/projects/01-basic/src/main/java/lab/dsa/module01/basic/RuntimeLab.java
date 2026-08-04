package lab.dsa.module01.basic;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.IntToLongFunction;
import javax.imageio.ImageIO;

/** Benchmarks five functions with growth from constant to quadratic. */
public final class RuntimeLab {
    public static final List<Integer> DEFAULT_SIZES =
            List.of(128, 256, 512, 1_024, 2_048, 4_096);
    public static final Path DEFAULT_OUTPUT_DIRECTORY = Paths.get("results");

    private static final Map<String, IntToLongFunction> FUNCTIONS = createFunctions();
    private static volatile long benchmarkSink;

    private RuntimeLab() {
    }

    /** Performs a fixed amount of work independently of the input size. */
    public static long curveA(int size) {
        long operations = 0;
        for (int index = 0; index < 32; index++) {
            operations++;
        }
        return operations;
    }

    /** Repeatedly halves the remaining problem size. */
    public static long curveB(int size) {
        long operations = 0;
        int remaining = size;
        while (remaining > 1) {
            for (int index = 0; index < 32; index++) {
                operations++;
            }
            remaining /= 2;
        }
        return operations;
    }

    /** Visits one item for every unit of input. */
    public static long curveC(int size) {
        long operations = 0;
        for (int item = 0; item < size; item++) {
            for (int index = 0; index < 8; index++) {
                operations++;
            }
        }
        return operations;
    }

    /** Performs a halving process for every input item. */
    public static long curveD(int size) {
        long operations = 0;
        for (int item = 0; item < size; item++) {
            int remaining = size;
            while (remaining > 1) {
                operations++;
                remaining /= 2;
            }
        }
        return operations;
    }

    /** Visits every unordered pair of distinct positions. */
    public static long curveE(int size) {
        long operations = 0;
        for (int left = 0; left < size; left++) {
            for (int right = 0; right < left; right++) {
                operations++;
            }
        }
        return operations;
    }

    /** Immutable configuration for a complete benchmark series. */
    public record BenchmarkConfig(
            List<Integer> sizes,
            int repeats,
            double minimumSampleSeconds,
            int maximumIterations) {

        public BenchmarkConfig {
            sizes = List.copyOf(sizes);
        }

        public static BenchmarkConfig defaults() {
            return new BenchmarkConfig(DEFAULT_SIZES, 5, 0.01, 1_048_576);
        }

        /** Rejects configurations that cannot produce valid benchmark data. */
        public void validate() {
            if (sizes.isEmpty()) {
                throw new IllegalArgumentException("At least one input size is required");
            }
            if (sizes.stream().anyMatch(size -> size < 2)) {
                throw new IllegalArgumentException("All input sizes must be at least 2");
            }
            for (int index = 1; index < sizes.size(); index++) {
                if (sizes.get(index - 1) >= sizes.get(index)) {
                    throw new IllegalArgumentException("Input sizes must be strictly increasing");
                }
            }
            if (repeats < 1) {
                throw new IllegalArgumentException("Repeats must be positive");
            }
            if (minimumSampleSeconds <= 0) {
                throw new IllegalArgumentException("Minimum sample duration must be positive");
            }
            if (maximumIterations < 1) {
                throw new IllegalArgumentException("Maximum iterations must be positive");
            }
        }
    }

    /** One median runtime measurement. */
    public record Measurement(
            String label,
            int size,
            double secondsPerCall,
            int iterationsPerSample) {
    }

    private record RuntimeResult(double secondsPerCall, int iterations) {
    }

    private record Arguments(List<Integer> sizes, int repeats, double minimumSampleSeconds,
                             Path outputDirectory, boolean help) {
    }

    private static Map<String, IntToLongFunction> createFunctions() {
        Map<String, IntToLongFunction> functions = new LinkedHashMap<>();
        functions.put("A", RuntimeLab::curveA);
        functions.put("B", RuntimeLab::curveB);
        functions.put("C", RuntimeLab::curveC);
        functions.put("D", RuntimeLab::curveD);
        functions.put("E", RuntimeLab::curveE);
        return java.util.Collections.unmodifiableMap(functions);
    }

    private static double runBatch(IntToLongFunction function, int size, int iterations) {
        long checksum = 0;
        long start = System.nanoTime();
        for (int iteration = 0; iteration < iterations; iteration++) {
            checksum ^= function.applyAsLong(size);
        }
        double elapsed = (System.nanoTime() - start) / 1_000_000_000.0;
        benchmarkSink ^= checksum;
        return elapsed;
    }

    public static int calibrateIterations(
            IntToLongFunction function,
            int size,
            double minimumSampleSeconds,
            int maximumIterations) {
        int iterations = 1;
        while (true) {
            double elapsed = runBatch(function, size, iterations);
            if (elapsed >= minimumSampleSeconds || iterations >= maximumIterations) {
                return iterations;
            }
            iterations = Math.min(iterations * 2, maximumIterations);
        }
    }

    private static RuntimeResult measureRuntime(
            IntToLongFunction function, int size, BenchmarkConfig config) {
        int iterations = calibrateIterations(
                function, size, config.minimumSampleSeconds(), config.maximumIterations());
        double[] samples = new double[config.repeats()];
        for (int repeat = 0; repeat < config.repeats(); repeat++) {
            samples[repeat] = runBatch(function, size, iterations) / iterations;
        }
        Arrays.sort(samples);
        double median = samples.length % 2 == 1
                ? samples[samples.length / 2]
                : (samples[samples.length / 2 - 1] + samples[samples.length / 2]) / 2;
        return new RuntimeResult(median, iterations);
    }

    public static List<Measurement> runBenchmarks(
            Map<String, IntToLongFunction> functions, BenchmarkConfig config) {
        config.validate();
        List<Measurement> measurements = new ArrayList<>();
        functions.forEach((label, function) -> {
            for (int size : config.sizes()) {
                RuntimeResult result = measureRuntime(function, size, config);
                measurements.add(new Measurement(
                        label, size, result.secondsPerCall(), result.iterations()));
            }
        });
        return measurements;
    }

    public static double estimateLogLogSlope(List<Measurement> measurements) {
        if (measurements.size() < 2) {
            throw new IllegalArgumentException("At least two measurements are required");
        }
        double xMean = measurements.stream().mapToDouble(point -> Math.log(point.size())).average().orElseThrow();
        double yMean = measurements.stream().mapToDouble(point -> Math.log(point.secondsPerCall())).average().orElseThrow();
        double numerator = 0;
        double denominator = 0;
        for (Measurement point : measurements) {
            double xDifference = Math.log(point.size()) - xMean;
            numerator += xDifference * (Math.log(point.secondsPerCall()) - yMean);
            denominator += xDifference * xDifference;
        }
        if (denominator == 0) {
            throw new IllegalArgumentException("Input sizes must not all be equal");
        }
        return numerator / denominator;
    }

    public static Map<String, List<Measurement>> groupByLabel(List<Measurement> measurements) {
        Map<String, List<Measurement>> grouped = new LinkedHashMap<>();
        for (Measurement measurement : measurements) {
            grouped.computeIfAbsent(measurement.label(), ignored -> new ArrayList<>()).add(measurement);
        }
        return grouped;
    }

    public static void writeCsv(List<Measurement> measurements, Path destination) throws IOException {
        Path parent = destination.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("curve,input_size,seconds_per_call,iterations_per_sample");
            writer.newLine();
            for (Measurement point : measurements) {
                writer.write(String.format(Locale.ROOT, "%s,%d,%.12g,%d",
                        point.label(), point.size(), point.secondsPerCall(), point.iterationsPerSample()));
                writer.newLine();
            }
        }
    }

    /** Creates a dependency-free PNG chart with base-2 logarithmic axes. */
    public static void createPlot(List<Measurement> measurements, Path destination) throws IOException {
        int width = 1_200;
        int height = 720;
        int left = 105;
        int right = 55;
        int top = 70;
        int bottom = 100;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, width, height);

        double minimumX = measurements.stream().mapToDouble(point -> log2(point.size())).min().orElseThrow();
        double maximumX = measurements.stream().mapToDouble(point -> log2(point.size())).max().orElseThrow();
        double minimumY = measurements.stream().mapToDouble(point -> Math.log10(point.secondsPerCall())).min().orElseThrow();
        double maximumY = measurements.stream().mapToDouble(point -> Math.log10(point.secondsPerCall())).max().orElseThrow();
        if (minimumX == maximumX) {
            maximumX = minimumX + 1;
        }
        if (minimumY == maximumY) {
            maximumY = minimumY + 1;
        }

        graphics.setColor(new Color(225, 225, 225));
        for (int line = 0; line <= 5; line++) {
            int y = top + line * (height - top - bottom) / 5;
            graphics.drawLine(left, y, width - right, y);
        }
        graphics.setColor(Color.BLACK);
        graphics.setStroke(new BasicStroke(2));
        graphics.drawLine(left, top, left, height - bottom);
        graphics.drawLine(left, height - bottom, width - right, height - bottom);
        graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 24));
        graphics.drawString("Runtime growth of five unknown complexity classes", 250, 38);
        graphics.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 16));
        graphics.drawString("Input size n (log₂)", 520, height - 40);
        graphics.rotate(-Math.PI / 2);
        graphics.drawString("Median runtime per call (log₁₀ seconds)", -500, 30);
        graphics.rotate(Math.PI / 2);

        Color[] colors = {new Color(31, 119, 180), new Color(255, 127, 14),
                new Color(44, 160, 44), new Color(214, 39, 40), new Color(148, 103, 189)};
        int colorIndex = 0;
        for (Map.Entry<String, List<Measurement>> entry : groupByLabel(measurements).entrySet()) {
            graphics.setColor(colors[colorIndex % colors.length]);
            graphics.setStroke(new BasicStroke(3));
            int previousX = -1;
            int previousY = -1;
            for (Measurement point : entry.getValue()) {
                int x = left + (int) ((log2(point.size()) - minimumX) / (maximumX - minimumX)
                        * (width - left - right));
                int y = height - bottom - (int) ((Math.log10(point.secondsPerCall()) - minimumY)
                        / (maximumY - minimumY) * (height - top - bottom));
                if (previousX >= 0) {
                    graphics.drawLine(previousX, previousY, x, y);
                }
                graphics.fillOval(x - 5, y - 5, 10, 10);
                previousX = x;
                previousY = y;
            }
            graphics.drawString("Curve " + entry.getKey(), width - 145, top + 25 * colorIndex);
            colorIndex++;
        }
        graphics.dispose();
        Path parent = destination.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        ImageIO.write(image, "png", destination.toFile());
    }

    private static double log2(double value) {
        return Math.log(value) / Math.log(2);
    }

    public static void printSummary(List<Measurement> measurements) {
        System.out.println("\nEmpirical results");
        System.out.println("=".repeat(76));
        groupByLabel(measurements).forEach((label, points) -> {
            List<String> runtimes = points.stream()
                    .map(point -> String.format(Locale.ROOT, "n=%d: %.3es", point.size(), point.secondsPerCall()))
                    .toList();
            List<String> ratios = new ArrayList<>();
            for (int index = 1; index < points.size(); index++) {
                ratios.add(String.format(Locale.ROOT, "%.2f",
                        points.get(index).secondsPerCall() / points.get(index - 1).secondsPerCall()));
            }
            System.out.println("Curve " + label);
            System.out.println("  runtimes: " + String.join(", ", runtimes));
            System.out.println("  doubling ratios: " + String.join(", ", ratios));
            System.out.printf(Locale.ROOT, "  estimated log-log slope: %.3f%n", estimateLogLogSlope(points));
        });
    }

    private static Arguments parseArguments(String[] arguments) {
        List<Integer> sizes = new ArrayList<>(DEFAULT_SIZES);
        int repeats = 5;
        double minimumSampleSeconds = 0.01;
        Path outputDirectory = DEFAULT_OUTPUT_DIRECTORY;
        for (int index = 0; index < arguments.length; index++) {
            switch (arguments[index]) {
                case "--help" -> {
                    return new Arguments(sizes, repeats, minimumSampleSeconds, outputDirectory, true);
                }
                case "--repeats" -> repeats = Integer.parseInt(requireValue(arguments, ++index, "--repeats"));
                case "--min-sample-ms" -> minimumSampleSeconds =
                        Double.parseDouble(requireValue(arguments, ++index, "--min-sample-ms")) / 1_000;
                case "--output-dir" -> outputDirectory = Paths.get(requireValue(arguments, ++index, "--output-dir"));
                case "--sizes" -> {
                    sizes = new ArrayList<>();
                    while (index + 1 < arguments.length && !arguments[index + 1].startsWith("--")) {
                        sizes.add(Integer.parseInt(arguments[++index]));
                    }
                }
                default -> throw new IllegalArgumentException("Unknown option: " + arguments[index]);
            }
        }
        return new Arguments(sizes, repeats, minimumSampleSeconds, outputDirectory, false);
    }

    private static String requireValue(String[] arguments, int index, String option) {
        if (index >= arguments.length) {
            throw new IllegalArgumentException("Missing value for " + option);
        }
        return arguments[index];
    }

    private static void printHelp() {
        System.out.println("Usage: mvn exec:java -Dexec.args=\"[options]\"");
        System.out.println("  --sizes N...          Strictly increasing sizes (default: 128..4096)");
        System.out.println("  --repeats N           Samples per function and size (default: 5)");
        System.out.println("  --min-sample-ms N     Target sample duration in milliseconds (default: 10)");
        System.out.println("  --output-dir PATH     Directory for CSV and PNG output (default: results)");
    }

    public static void main(String[] commandLineArguments) throws IOException {
        Arguments arguments = parseArguments(commandLineArguments);
        if (arguments.help()) {
            printHelp();
            return;
        }
        BenchmarkConfig config = new BenchmarkConfig(
                arguments.sizes(), arguments.repeats(), arguments.minimumSampleSeconds(), 1_048_576);
        System.out.printf("Measuring %d curves across %d sizes. Please wait...%n",
                FUNCTIONS.size(), config.sizes().size());
        List<Measurement> measurements = runBenchmarks(FUNCTIONS, config);
        Path csvPath = arguments.outputDirectory().resolve("measurements.csv");
        Path plotPath = arguments.outputDirectory().resolve("runtime_growth.png");
        writeCsv(measurements, csvPath);
        createPlot(measurements, plotPath);
        printSummary(measurements);
        System.out.println("\nArtifacts");
        System.out.println("  CSV:  " + csvPath);
        System.out.println("  Plot: " + plotPath);
    }
}
