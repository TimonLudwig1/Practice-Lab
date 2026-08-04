package lab.dsa.module01.medium;

import java.awt.BasicStroke;
import java.awt.Color;
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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.IntToLongFunction;
import javax.imageio.ImageIO;

/** Investigates ten Java methods with visible and hidden runtime costs. */
public final class ComplexityDetective {
    public static final List<Integer> DEFAULT_SIZES = List.of(64, 128, 256, 512, 1_024, 2_048, 4_096);
    public static final Map<String, IntToLongFunction> CASES = createCases();
    private static volatile long benchmarkSink;

    private ComplexityDetective() {
    }

    public static long case01(int size) {
        int[] values = {10, 20, 30, 40, 50};
        return values[3];
    }

    public static long case02(int size) {
        int steps = 0;
        for (int remaining = size; remaining > 1; remaining /= 2) {
            steps++;
        }
        return steps;
    }

    public static long case03(int size) {
        long result = 0;
        for (int value = 0; value < size; value++) {
            result += value;
        }
        for (int index = 0; index < size; index++) {
            result++;
        }
        return result;
    }

    public static long case04(int size) {
        long operations = 0;
        for (int remaining = size; remaining > 0; remaining /= 2) {
            for (int index = 0; index < remaining; index++) {
                operations++;
            }
        }
        return operations;
    }

    public static long case05(int size) {
        Set<Integer> allowed = new HashSet<>();
        for (int value = 0; value < size; value++) {
            allowed.add(value);
        }
        int matches = 0;
        for (int value = 0; value < size; value++) {
            if (allowed.contains(value)) {
                matches++;
            }
        }
        return matches;
    }

    public static long case06(int size) {
        List<Integer> values = new ArrayList<>(size);
        for (int value = 0; value < size; value++) {
            values.add(value);
        }
        int misses = 0;
        for (int attempt = 0; attempt < size; attempt++) {
            if (!values.contains(-1)) {
                misses++;
            }
        }
        return misses;
    }

    public static long case07(int size) {
        int[] values = new int[size];
        for (int index = 0; index < size; index++) {
            values[index] = index;
        }
        long result = 0;
        while (values.length > 0) {
            result += values[0];
            values = Arrays.copyOfRange(values, 1, values.length);
        }
        return result;
    }

    public static long case08(int size) {
        List<Integer> values = new ArrayList<>();
        for (int value = 0; value < size; value++) {
            values.add(0, value);
        }
        return values.getFirst() + values.size();
    }

    public static long case09(int size) {
        String result = "";
        List<String> history = new ArrayList<>();
        for (int index = 0; index < size; index++) {
            history.add(result);
            result = result + "x";
        }
        return result.length() + history.size();
    }

    public static long case10(int size) {
        long state = 42;
        long[] values = new long[size];
        for (int index = 0; index < size; index++) {
            state = (1_664_525L * state + 1_013_904_223L) & 0xffff_ffffL;
            values[index] = state;
        }
        Arrays.sort(values);
        return values[0] ^ values[values.length - 1];
    }

    public record BenchmarkConfig(List<Integer> sizes, int repeats,
                                  double minimumSampleSeconds, int maximumIterations) {
        public BenchmarkConfig {
            sizes = List.copyOf(sizes);
        }

        public static BenchmarkConfig defaults() {
            return new BenchmarkConfig(DEFAULT_SIZES, 5, 0.005, 1_048_576);
        }

        public void validate() {
            if (sizes.size() < 2) {
                throw new IllegalArgumentException("At least two input sizes are required");
            }
            if (sizes.stream().anyMatch(size -> size < 2)) {
                throw new IllegalArgumentException("All input sizes must be at least 2");
            }
            for (int index = 1; index < sizes.size(); index++) {
                if (sizes.get(index - 1) >= sizes.get(index)) {
                    throw new IllegalArgumentException("Input sizes must be strictly increasing");
                }
            }
            if (repeats < 1 || minimumSampleSeconds <= 0 || maximumIterations < 1) {
                throw new IllegalArgumentException("Repeats, sample duration, and iterations must be positive");
            }
        }
    }

    public record Measurement(String caseId, int size, double secondsPerCall, int iterationsPerSample) {
    }

    private record TimedResult(double secondsPerCall, int iterations) {
    }

    private record Arguments(List<String> caseIds, List<Integer> sizes, int repeats,
                             double minimumSampleSeconds, Path outputDirectory, boolean help) {
    }

    private static Map<String, IntToLongFunction> createCases() {
        Map<String, IntToLongFunction> cases = new LinkedHashMap<>();
        cases.put("01", ComplexityDetective::case01);
        cases.put("02", ComplexityDetective::case02);
        cases.put("03", ComplexityDetective::case03);
        cases.put("04", ComplexityDetective::case04);
        cases.put("05", ComplexityDetective::case05);
        cases.put("06", ComplexityDetective::case06);
        cases.put("07", ComplexityDetective::case07);
        cases.put("08", ComplexityDetective::case08);
        cases.put("09", ComplexityDetective::case09);
        cases.put("10", ComplexityDetective::case10);
        return java.util.Collections.unmodifiableMap(cases);
    }

    private static double runBatch(IntToLongFunction function, int size, int iterations) {
        long checksum = 0;
        long start = System.nanoTime();
        for (int iteration = 0; iteration < iterations; iteration++) {
            checksum ^= function.applyAsLong(size);
        }
        benchmarkSink ^= checksum;
        return (System.nanoTime() - start) / 1_000_000_000.0;
    }

    public static int calibrateIterations(IntToLongFunction function, int size, BenchmarkConfig config) {
        int iterations = 1;
        while (true) {
            double elapsed = runBatch(function, size, iterations);
            if (elapsed >= config.minimumSampleSeconds() || iterations >= config.maximumIterations()) {
                return iterations;
            }
            iterations = Math.min(iterations * 2, config.maximumIterations());
        }
    }

    private static TimedResult measureCase(IntToLongFunction function, int size, BenchmarkConfig config) {
        int iterations = calibrateIterations(function, size, config);
        double[] samples = new double[config.repeats()];
        for (int repeat = 0; repeat < config.repeats(); repeat++) {
            samples[repeat] = runBatch(function, size, iterations) / iterations;
        }
        Arrays.sort(samples);
        double median = samples.length % 2 == 1
                ? samples[samples.length / 2]
                : (samples[samples.length / 2 - 1] + samples[samples.length / 2]) / 2;
        return new TimedResult(median, iterations);
    }

    public static List<Measurement> runBenchmarks(
            Map<String, IntToLongFunction> cases, BenchmarkConfig config) {
        config.validate();
        List<Measurement> measurements = new ArrayList<>();
        cases.forEach((caseId, function) -> {
            for (int size : config.sizes()) {
                TimedResult result = measureCase(function, size, config);
                measurements.add(new Measurement(caseId, size, result.secondsPerCall(), result.iterations()));
            }
        });
        return measurements;
    }

    public static Map<String, List<Measurement>> groupByCase(List<Measurement> measurements) {
        Map<String, List<Measurement>> grouped = new LinkedHashMap<>();
        for (Measurement point : measurements) {
            grouped.computeIfAbsent(point.caseId(), ignored -> new ArrayList<>()).add(point);
        }
        return grouped;
    }

    public static double estimateLogLogSlope(List<Measurement> points) {
        if (points.size() < 2) {
            throw new IllegalArgumentException("At least two measurements are required");
        }
        double xMean = points.stream().mapToDouble(point -> Math.log(point.size())).average().orElseThrow();
        double yMean = points.stream().mapToDouble(point -> Math.log(point.secondsPerCall())).average().orElseThrow();
        double numerator = 0;
        double denominator = 0;
        for (Measurement point : points) {
            double xDifference = Math.log(point.size()) - xMean;
            numerator += xDifference * (Math.log(point.secondsPerCall()) - yMean);
            denominator += xDifference * xDifference;
        }
        if (denominator == 0) {
            throw new IllegalArgumentException("Input sizes must not all be equal");
        }
        return numerator / denominator;
    }

    public static Map<String, IntToLongFunction> selectCases(List<String> caseIds) {
        Map<String, IntToLongFunction> selected = new LinkedHashMap<>();
        for (String caseId : caseIds) {
            IntToLongFunction function = CASES.get(caseId);
            if (function == null) {
                throw new IllegalArgumentException("Unknown case identifier: " + caseId);
            }
            selected.put(caseId, function);
        }
        return selected;
    }

    public static void writeCsv(List<Measurement> measurements, Path destination) throws IOException {
        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("case,input_size,seconds_per_call,iterations_per_sample");
            writer.newLine();
            for (Measurement point : measurements) {
                writer.write(String.format(Locale.ROOT, "%s,%d,%.12g,%d",
                        point.caseId(), point.size(), point.secondsPerCall(), point.iterationsPerSample()));
                writer.newLine();
            }
        }
    }

    /** Draws normalized curves on a base-2 logarithmic chart without external libraries. */
    public static void createNormalizedPlot(List<Measurement> measurements, Path destination) throws IOException {
        int width = 1_200;
        int height = 720;
        int left = 90;
        int right = 50;
        int top = 70;
        int bottom = 90;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, width, height);
        graphics.setColor(Color.LIGHT_GRAY);
        for (int line = 0; line <= 6; line++) {
            int y = top + line * (height - top - bottom) / 6;
            graphics.drawLine(left, y, width - right, y);
        }
        graphics.setColor(Color.BLACK);
        graphics.drawString("Complexity detective: normalized runtime growth", 430, 35);
        graphics.drawString("Input size n (log2)", 530, height - 35);

        Map<String, List<Measurement>> grouped = groupByCase(measurements);
        double minimumX = measurements.stream().mapToDouble(point -> log2(point.size())).min().orElseThrow();
        double maximumX = measurements.stream().mapToDouble(point -> log2(point.size())).max().orElseThrow();
        double maximumY = grouped.values().stream().flatMap(List::stream)
                .mapToDouble(point -> log2(point.secondsPerCall()
                        / grouped.get(point.caseId()).getFirst().secondsPerCall())).max().orElse(1);
        maximumY = Math.max(1, maximumY);
        Color[] colors = {Color.BLUE, Color.RED, new Color(0, 140, 0), Color.MAGENTA, Color.ORANGE,
                Color.CYAN.darker(), Color.PINK.darker(), Color.GRAY, new Color(120, 80, 20), new Color(80, 40, 140)};
        int colorIndex = 0;
        for (Map.Entry<String, List<Measurement>> entry : grouped.entrySet()) {
            graphics.setColor(colors[colorIndex % colors.length]);
            graphics.setStroke(new BasicStroke(2));
            double baseline = entry.getValue().getFirst().secondsPerCall();
            int previousX = -1;
            int previousY = -1;
            for (Measurement point : entry.getValue()) {
                int x = left + (int) ((log2(point.size()) - minimumX) / (maximumX - minimumX)
                        * (width - left - right));
                double normalizedLog = Math.max(0, log2(point.secondsPerCall() / baseline));
                int y = height - bottom - (int) (normalizedLog / maximumY * (height - top - bottom));
                if (previousX >= 0) {
                    graphics.drawLine(previousX, previousY, x, y);
                }
                graphics.fillOval(x - 4, y - 4, 8, 8);
                previousX = x;
                previousY = y;
            }
            graphics.drawString("Case " + entry.getKey(), width - 110, top + colorIndex * 22);
            colorIndex++;
        }
        graphics.dispose();
        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }
        ImageIO.write(image, "png", destination.toFile());
    }

    private static double log2(double value) {
        return Math.log(value) / Math.log(2);
    }

    public static void printSummary(List<Measurement> measurements) {
        System.out.println("\nEmpirical detective report");
        System.out.println("=".repeat(66));
        groupByCase(measurements).forEach((caseId, points) -> {
            double slope = estimateLogLogSlope(points);
            double growth = points.getLast().secondsPerCall() / points.getFirst().secondsPerCall();
            System.out.printf(Locale.ROOT, "Case %s: slope=%6.3f, runtime growth=%9.2fx%n",
                    caseId, slope, growth);
        });
    }

    private static Arguments parseArguments(String[] arguments) {
        List<String> caseIds = new ArrayList<>(CASES.keySet());
        List<Integer> sizes = new ArrayList<>(DEFAULT_SIZES);
        int repeats = 5;
        double minimumSampleSeconds = 0.005;
        Path outputDirectory = Paths.get("results");
        for (int index = 0; index < arguments.length; index++) {
            switch (arguments[index]) {
                case "--help" -> { return new Arguments(caseIds, sizes, repeats, minimumSampleSeconds, outputDirectory, true); }
                case "--repeats" -> repeats = Integer.parseInt(requireValue(arguments, ++index, "--repeats"));
                case "--min-sample-ms" -> minimumSampleSeconds =
                        Double.parseDouble(requireValue(arguments, ++index, "--min-sample-ms")) / 1_000;
                case "--output-dir" -> outputDirectory = Paths.get(requireValue(arguments, ++index, "--output-dir"));
                case "--cases" -> {
                    caseIds = new ArrayList<>();
                    while (index + 1 < arguments.length && !arguments[index + 1].startsWith("--")) {
                        caseIds.add(arguments[++index]);
                    }
                }
                case "--sizes" -> {
                    sizes = new ArrayList<>();
                    while (index + 1 < arguments.length && !arguments[index + 1].startsWith("--")) {
                        sizes.add(Integer.parseInt(arguments[++index]));
                    }
                }
                default -> throw new IllegalArgumentException("Unknown option: " + arguments[index]);
            }
        }
        return new Arguments(caseIds, sizes, repeats, minimumSampleSeconds, outputDirectory, false);
    }

    private static String requireValue(String[] arguments, int index, String option) {
        if (index >= arguments.length) {
            throw new IllegalArgumentException("Missing value for " + option);
        }
        return arguments[index];
    }

    private static void printHelp() {
        System.out.println("Options: --cases ID... --sizes N... --repeats N --min-sample-ms N --output-dir PATH");
    }

    public static void main(String[] commandLineArguments) throws IOException {
        Arguments arguments = parseArguments(commandLineArguments);
        if (arguments.help()) {
            printHelp();
            return;
        }
        Map<String, IntToLongFunction> selectedCases = selectCases(arguments.caseIds());
        BenchmarkConfig config = new BenchmarkConfig(
                arguments.sizes(), arguments.repeats(), arguments.minimumSampleSeconds(), 1_048_576);
        config.validate();
        System.out.printf("Measuring %d cases across %d sizes. Please wait...%n",
                selectedCases.size(), config.sizes().size());
        List<Measurement> measurements = runBenchmarks(selectedCases, config);
        Path csvPath = arguments.outputDirectory().resolve("measurements.csv");
        Path plotPath = arguments.outputDirectory().resolve("normalized_growth.png");
        writeCsv(measurements, csvPath);
        createNormalizedPlot(measurements, plotPath);
        printSummary(measurements);
        System.out.println("\nArtifacts\n  CSV:  " + csvPath + "\n  Plot: " + plotPath);
    }
}
