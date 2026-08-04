package lab.dsa.module01.finalproject;

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
import java.util.List;
import java.util.Locale;
import javax.imageio.ImageIO;
import lab.dsa.module01.finalproject.AuditPipeline.CustomerSummary;
import lab.dsa.module01.finalproject.AuditPipeline.Event;

/** Benchmarks the baseline and optimized customer aggregation pipelines. */
public final class RunAudit {
    public static final int DEFAULT_ROWS = 8_000;
    public static final List<Integer> DEFAULT_SIZES = List.of(500, 1_000, 2_000, 4_000, 8_000);
    private static volatile int auditSink;

    private RunAudit() {
    }

    @FunctionalInterface
    public interface Pipeline {
        List<CustomerSummary> apply(List<Event> events);
    }

    public record AuditMeasurement(int size, double baselineSeconds,
                                   double optimizedSeconds, int customerCount) {
        public double speedup() {
            return baselineSeconds / optimizedSeconds;
        }
    }

    private record Arguments(int rows, List<Integer> sizes, int repeats, long seed,
                             Path dataPath, Path outputDirectory, boolean reuseData, boolean help) {
    }

    public static double medianRuntime(Pipeline pipeline, List<Event> events, int repeats) {
        double[] durations = new double[repeats];
        for (int repeat = 0; repeat < repeats; repeat++) {
            long start = System.nanoTime();
            List<CustomerSummary> summaries = pipeline.apply(events);
            durations[repeat] = (System.nanoTime() - start) / 1_000_000_000.0;
            auditSink ^= summaries.size();
        }
        Arrays.sort(durations);
        return durations.length % 2 == 1
                ? durations[durations.length / 2]
                : (durations[durations.length / 2 - 1] + durations[durations.length / 2]) / 2;
    }

    public static void validateBenchmark(List<Event> events, List<Integer> sizes, int repeats) {
        if (sizes.isEmpty()) {
            throw new IllegalArgumentException("At least one benchmark size is required");
        }
        if (sizes.stream().anyMatch(size -> size < 1)) {
            throw new IllegalArgumentException("All benchmark sizes must be positive");
        }
        for (int index = 1; index < sizes.size(); index++) {
            if (sizes.get(index - 1) >= sizes.get(index)) {
                throw new IllegalArgumentException("Benchmark sizes must be strictly increasing");
            }
        }
        if (sizes.getLast() > events.size()) {
            throw new IllegalArgumentException("Largest benchmark size exceeds available rows");
        }
        if (repeats < 1) {
            throw new IllegalArgumentException("Repeats must be positive");
        }
    }

    public static List<AuditMeasurement> benchmarkPipelines(
            List<Event> events, List<Integer> sizes, int repeats) {
        validateBenchmark(events, sizes, repeats);
        List<AuditMeasurement> measurements = new ArrayList<>();
        for (int size : sizes) {
            List<Event> subset = events.subList(0, size);
            List<CustomerSummary> baselineResult = AuditPipeline.inefficientPipeline(subset);
            List<CustomerSummary> optimizedResult = AuditPipeline.optimizedPipeline(subset);
            if (!baselineResult.equals(optimizedResult)) {
                throw new AssertionError("Pipeline outputs differ for input size " + size);
            }
            double baselineSeconds = medianRuntime(AuditPipeline::inefficientPipeline, subset, repeats);
            double optimizedSeconds = medianRuntime(AuditPipeline::optimizedPipeline, subset, repeats);
            measurements.add(new AuditMeasurement(
                    size, baselineSeconds, optimizedSeconds, optimizedResult.size()));
        }
        return measurements;
    }

    public static void writeMeasurements(List<AuditMeasurement> measurements, Path destination)
            throws IOException {
        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("input_rows,customer_count,baseline_seconds,optimized_seconds,speedup");
            writer.newLine();
            for (AuditMeasurement point : measurements) {
                writer.write(String.format(Locale.ROOT, "%d,%d,%.12g,%.12g,%.6f",
                        point.size(), point.customerCount(), point.baselineSeconds(),
                        point.optimizedSeconds(), point.speedup()));
                writer.newLine();
            }
        }
    }

    public static void writeReport(List<AuditMeasurement> measurements, Path destination,
                                   long seed, int repeats) throws IOException {
        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }
        AuditMeasurement largest = measurements.getLast();
        List<String> lines = new ArrayList<>(List.of(
                "# Performance Audit — Measurement Results", "", "## Configuration", "",
                "- Seed: " + seed,
                "- Repetitions per variant and size: " + repeats,
                "- Measurement boundary: aggregation only; CSV generation and loading excluded",
                "- Correctness condition: complete output equality before every measurement",
                "", "## Before and After", "",
                "| Rows | Customers | Baseline (s) | Optimized (s) | Speedup |",
                "|---:|---:|---:|---:|---:|"));
        for (AuditMeasurement point : measurements) {
            lines.add(String.format(Locale.ROOT, "| %d | %d | %.6f | %.6f | %.2fx |",
                    point.size(), point.customerCount(), point.baselineSeconds(),
                    point.optimizedSeconds(), point.speedup()));
        }
        lines.addAll(List.of("", "## Automatic Summary", "",
                String.format(Locale.ROOT,
                        "At %,d events, the optimized pipeline produces the same output in %.6f seconds instead of %.6f seconds, a %.2fx speedup.",
                        largest.size(), largest.optimizedSeconds(), largest.baselineSeconds(), largest.speedup()),
                "", "The baseline uses list membership and repeated full scans, giving O(nu) time and O(n²) when u is O(n). The optimized version aggregates with a hash map in expected O(n) time and sorts u outputs in O(u log u).",
                "", "## Your Interpretation", "",
                "Add doubling factors, practical relevance, limitations of the synthetic benchmark, and possible next steps."));
        Files.write(destination, lines, StandardCharsets.UTF_8);
    }

    /** Creates a dependency-free PNG with runtime and speedup panels. */
    public static void createPlot(List<AuditMeasurement> measurements, Path destination) throws IOException {
        int width = 1_300;
        int height = 560;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, width, height);
        graphics.setColor(Color.BLACK);
        graphics.drawString("Customer aggregation performance audit", 520, 25);
        drawRuntimePanel(graphics, measurements, 55, 55, 570, 440);
        drawSpeedupPanel(graphics, measurements, 680, 55, 570, 440);
        graphics.dispose();
        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }
        ImageIO.write(image, "png", destination.toFile());
    }

    private static void drawRuntimePanel(Graphics2D graphics, List<AuditMeasurement> points,
                                         int x, int y, int width, int height) {
        graphics.setColor(Color.BLACK);
        graphics.drawString("Before/after runtime growth", x + 190, y);
        graphics.drawRect(x, y + 20, width, height - 20);
        double minimum = points.stream().mapToDouble(AuditMeasurement::optimizedSeconds).min().orElseThrow();
        double maximum = points.stream().mapToDouble(AuditMeasurement::baselineSeconds).max().orElseThrow();
        plotLine(graphics, points, x, y + 20, width, height - 20, minimum, maximum, true, Color.RED);
        plotLine(graphics, points, x, y + 20, width, height - 20, minimum, maximum, false, new Color(0, 140, 0));
        graphics.setColor(Color.RED); graphics.drawString("Baseline", x + 15, y + 45);
        graphics.setColor(new Color(0, 140, 0)); graphics.drawString("Optimized", x + 15, y + 65);
    }

    private static void plotLine(Graphics2D graphics, List<AuditMeasurement> points,
                                 int x, int y, int width, int height, double minimum,
                                 double maximum, boolean baseline, Color color) {
        graphics.setColor(color);
        graphics.setStroke(new BasicStroke(2));
        int previousX = -1;
        int previousY = -1;
        double logMinimum = Math.log(Math.max(minimum, 1e-12));
        double logMaximum = Math.log(Math.max(maximum, minimum * 1.01));
        for (int index = 0; index < points.size(); index++) {
            AuditMeasurement point = points.get(index);
            double value = baseline ? point.baselineSeconds() : point.optimizedSeconds();
            int currentX = x + index * width / Math.max(1, points.size() - 1);
            int currentY = y + height - (int) ((Math.log(value) - logMinimum)
                    / (logMaximum - logMinimum) * height);
            if (previousX >= 0) graphics.drawLine(previousX, previousY, currentX, currentY);
            graphics.fillOval(currentX - 4, currentY - 4, 8, 8);
            previousX = currentX; previousY = currentY;
        }
    }

    private static void drawSpeedupPanel(Graphics2D graphics, List<AuditMeasurement> points,
                                         int x, int y, int width, int height) {
        graphics.setColor(Color.BLACK);
        graphics.drawString("Measured speedup", x + 220, y);
        graphics.drawRect(x, y + 20, width, height - 20);
        double maximum = Math.max(1, points.stream().mapToDouble(AuditMeasurement::speedup).max().orElseThrow());
        graphics.setColor(new Color(0, 120, 0));
        graphics.setStroke(new BasicStroke(2));
        int previousX = -1;
        int previousY = -1;
        for (int index = 0; index < points.size(); index++) {
            int currentX = x + index * width / Math.max(1, points.size() - 1);
            int currentY = y + 20 + height - 20 - (int) (points.get(index).speedup() / maximum * (height - 20));
            if (previousX >= 0) graphics.drawLine(previousX, previousY, currentX, currentY);
            graphics.fillOval(currentX - 4, currentY - 4, 8, 8);
            previousX = currentX; previousY = currentY;
        }
    }

    public static void printSummary(List<AuditMeasurement> measurements) {
        System.out.println("\nPerformance audit");
        System.out.println("=".repeat(78));
        System.out.printf("%8s %10s %14s %14s %10s%n", "rows", "customers", "baseline", "optimized", "speedup");
        for (AuditMeasurement point : measurements) {
            System.out.printf(Locale.ROOT, "%8d %10d %13.6fs %13.6fs %9.2fx%n",
                    point.size(), point.customerCount(), point.baselineSeconds(), point.optimizedSeconds(), point.speedup());
        }
    }

    private static Arguments parseArguments(String[] arguments) {
        int rows = DEFAULT_ROWS;
        List<Integer> sizes = new ArrayList<>(DEFAULT_SIZES);
        int repeats = 3;
        long seed = GenerateData.DEFAULT_SEED;
        Path dataPath = Paths.get("data", "events.csv");
        Path outputDirectory = Paths.get("results");
        boolean reuseData = false;
        boolean help = false;
        for (int index = 0; index < arguments.length; index++) {
            switch (arguments[index]) {
                case "--rows" -> rows = Integer.parseInt(arguments[++index]);
                case "--repeats" -> repeats = Integer.parseInt(arguments[++index]);
                case "--seed" -> seed = Long.parseLong(arguments[++index]);
                case "--data-path" -> dataPath = Paths.get(arguments[++index]);
                case "--output-dir" -> outputDirectory = Paths.get(arguments[++index]);
                case "--reuse-data" -> reuseData = true;
                case "--help" -> help = true;
                case "--sizes" -> {
                    sizes = new ArrayList<>();
                    while (index + 1 < arguments.length && !arguments[index + 1].startsWith("--")) {
                        sizes.add(Integer.parseInt(arguments[++index]));
                    }
                }
                default -> throw new IllegalArgumentException("Unknown option: " + arguments[index]);
            }
        }
        return new Arguments(rows, sizes, repeats, seed, dataPath, outputDirectory, reuseData, help);
    }

    public static void main(String[] commandLineArguments) throws IOException {
        Arguments arguments = parseArguments(commandLineArguments);
        if (arguments.help()) {
            System.out.println("Options: --rows N --sizes N... --repeats N --seed N --data-path PATH --output-dir PATH --reuse-data");
            return;
        }
        if (arguments.rows() < 1) throw new IllegalArgumentException("Rows must be positive");
        if (!arguments.reuseData()) {
            GenerateData.generateEvents(arguments.dataPath(), arguments.rows(), arguments.seed());
            System.out.printf("Generated %d events with seed %d at %s.%n",
                    arguments.rows(), arguments.seed(), arguments.dataPath());
        } else if (!Files.exists(arguments.dataPath())) {
            throw new IllegalArgumentException("Data file does not exist: " + arguments.dataPath());
        }
        List<Event> events = AuditPipeline.loadEvents(arguments.dataPath());
        List<AuditMeasurement> measurements = benchmarkPipelines(events, arguments.sizes(), arguments.repeats());
        Path summaryPath = arguments.outputDirectory().resolve("customer_summary.csv");
        Path measurementsPath = arguments.outputDirectory().resolve("performance_audit.csv");
        Path plotPath = arguments.outputDirectory().resolve("performance_audit.png");
        Path reportPath = arguments.outputDirectory().resolve("AUDIT_REPORT.md");
        AuditPipeline.writeSummaries(AuditPipeline.optimizedPipeline(events), summaryPath);
        writeMeasurements(measurements, measurementsPath);
        createPlot(measurements, plotPath);
        writeReport(measurements, reportPath, arguments.seed(), arguments.repeats());
        printSummary(measurements);
        System.out.println("\nArtifacts\n  " + summaryPath + "\n  " + measurementsPath + "\n  " + plotPath + "\n  " + reportPath);
    }
}
