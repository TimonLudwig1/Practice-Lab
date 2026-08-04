package lab.dsa.module02.finalproject;

import java.awt.Color;
import java.awt.Graphics2D;
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
import java.util.Random;
import java.util.function.Supplier;
import javax.imageio.ImageIO;
import lab.dsa.module02.finalproject.GenerateSensorData.SensorDataset;
import lab.dsa.module02.finalproject.SensorToolkit.Outlier;
import lab.dsa.module02.finalproject.SensorToolkit.Range;

/** Validates and benchmarks the toolkit against independent Java references. */
public final class SensorBenchmark {
    private static volatile int benchmarkSink;

    private SensorBenchmark() {
    }

    public record BenchmarkResult(String operation, double toolkitSeconds,
                                  double referenceSeconds, double referenceFactor,
                                  double maximumAbsoluteError) {
    }

    public static List<BenchmarkResult> runBenchmark(SensorDataset dataset, int window,
                                                     int queryCount, double threshold,
                                                     int repetitions, long querySeed) {
        if (queryCount <= 0) throw new IllegalArgumentException("Query count must be positive");
        if (repetitions <= 0) throw new IllegalArgumentException("Repetitions must be positive");
        List<Double> values = dataset.readings();
        List<Range> ranges = makeRanges(values.size(), queryCount, querySeed);

        List<Double> moving = SensorToolkit.movingAverage(values, window);
        List<Double> movingReference = referenceMovingAverage(values, window);
        double movingError = maximumError(moving, movingReference);
        assertClose(movingError, 1e-9, "Moving-average implementations disagree");

        SensorToolkit.PrefixSumIndex prefix = SensorToolkit.PrefixSumIndex.fromReadings(values);
        double[] prefixReference = referencePrefix(values);
        double prefixError = maximumError(prefix.prefixValues(), prefixReference);
        assertClose(prefixError, 1e-8, "Prefix implementations disagree");

        List<Double> rangeSums = prefix.batchRangeSums(ranges);
        List<Double> rangeReference = referenceRangeSums(prefixReference, ranges);
        double rangeError = maximumError(rangeSums, rangeReference);
        assertClose(rangeError, 1e-8, "Range-query implementations disagree");

        List<Outlier> outliers = SensorToolkit.detectZScoreOutliers(values, threshold);
        List<Outlier> outlierReference = referenceOutliers(values, threshold);
        if (!outliers.stream().map(Outlier::index).toList().equals(outlierReference.stream().map(Outlier::index).toList())) {
            throw new AssertionError("Outlier implementations disagree");
        }

        return List.of(
                result("moving_average", timed(() -> SensorToolkit.movingAverage(values, window), repetitions),
                        timed(() -> referenceMovingAverage(values, window), repetitions), movingError),
                result("prefix_build", timed(() -> SensorToolkit.PrefixSumIndex.fromReadings(values), repetitions),
                        timed(() -> referencePrefix(values), repetitions), prefixError),
                result("range_queries", timed(() -> prefix.batchRangeSums(ranges), repetitions),
                        timed(() -> referenceRangeSums(prefixReference, ranges), repetitions), rangeError),
                result("outlier_detection", timed(() -> SensorToolkit.detectZScoreOutliers(values, threshold), repetitions),
                        timed(() -> referenceOutliers(values, threshold), repetitions), 0));
    }

    private static List<Range> makeRanges(int size, int count, long seed) {
        Random random = new Random(seed); List<Range> ranges = new ArrayList<>(count);
        for (int query = 0; query < count; query++) {
            int start = random.nextInt(size); int width = 1 + random.nextInt(Math.min(500, size - start));
            ranges.add(new Range(start, start + width));
        }
        return ranges;
    }

    private static double timed(Supplier<?> operation, int repetitions) {
        double[] samples = new double[repetitions];
        for (int repeat = 0; repeat < repetitions; repeat++) {
            long start = System.nanoTime(); Object result = operation.get();
            samples[repeat] = (System.nanoTime() - start) / 1_000_000_000.0;
            benchmarkSink ^= result.hashCode();
        }
        Arrays.sort(samples);
        return samples.length % 2 == 1 ? samples[samples.length / 2]
                : (samples[samples.length / 2 - 1] + samples[samples.length / 2]) / 2;
    }

    private static BenchmarkResult result(String name, double toolkit, double reference, double error) {
        if (toolkit <= 0 || reference <= 0) throw new IllegalStateException("Timer resolution was insufficient");
        return new BenchmarkResult(name, toolkit, reference, toolkit / reference, error);
    }

    private static List<Double> referenceMovingAverage(List<Double> values, int window) {
        List<Double> result = new ArrayList<>();
        for (int start = 0; start + window <= values.size(); start++) {
            double sum = 0; for (int index = start; index < start + window; index++) sum += values.get(index);
            result.add(sum / window);
        }
        return result;
    }

    private static double[] referencePrefix(List<Double> values) {
        double[] prefix = new double[values.size() + 1];
        for (int index = 0; index < values.size(); index++) prefix[index + 1] = prefix[index] + values.get(index);
        return prefix;
    }

    private static List<Double> referenceRangeSums(double[] prefix, List<Range> ranges) {
        return ranges.stream().map(range -> prefix[range.end()] - prefix[range.start()]).toList();
    }

    private static List<Outlier> referenceOutliers(List<Double> values, double threshold) {
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double deviation = Math.sqrt(values.stream().mapToDouble(value -> Math.pow(value - mean, 2)).average().orElse(0));
        if (deviation == 0) return List.of();
        List<Outlier> result = new ArrayList<>();
        for (int index = 0; index < values.size(); index++) {
            double zScore = (values.get(index) - mean) / deviation;
            if (Math.abs(zScore) >= threshold) result.add(new Outlier(index, values.get(index), zScore));
        }
        return result;
    }

    private static double maximumError(List<Double> left, List<Double> right) {
        if (left.size() != right.size()) throw new AssertionError("Shape mismatch");
        double maximum = 0; for (int index = 0; index < left.size(); index++) maximum = Math.max(maximum, Math.abs(left.get(index) - right.get(index)));
        return maximum;
    }

    private static double maximumError(double[] left, double[] right) {
        if (left.length != right.length) throw new AssertionError("Shape mismatch");
        double maximum = 0; for (int index = 0; index < left.length; index++) maximum = Math.max(maximum, Math.abs(left[index] - right[index]));
        return maximum;
    }

    private static void assertClose(double error, double tolerance, String message) {
        if (error > tolerance) throw new AssertionError(message + ": " + error);
    }

    public static void writeBenchmarkCsv(List<BenchmarkResult> results, Path destination) throws IOException {
        if (destination.getParent() != null) Files.createDirectories(destination.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("operation,toolkit_seconds,reference_seconds,toolkit_to_reference_factor,max_abs_error"); writer.newLine();
            for (BenchmarkResult row : results) {
                writer.write(String.format(Locale.ROOT, "%s,%.9f,%.9f,%.3f,%.12g", row.operation(), row.toolkitSeconds(), row.referenceSeconds(), row.referenceFactor(), row.maximumAbsoluteError())); writer.newLine();
            }
        }
    }

    public static void createBenchmarkPlot(SensorDataset dataset, List<BenchmarkResult> results,
                                           Path destination, int window, double threshold) throws IOException {
        int width = 1_200; int height = 800; BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics(); graphics.setColor(Color.WHITE); graphics.fillRect(0, 0, width, height);
        graphics.setColor(Color.BLACK); graphics.drawString("Synthetic sensor series", 500, 25);
        List<Double> averages = SensorToolkit.movingAverage(dataset.readings(), window);
        double min = dataset.readings().stream().mapToDouble(Double::doubleValue).min().orElseThrow();
        double max = dataset.readings().stream().mapToDouble(Double::doubleValue).max().orElseThrow();
        int sampleStep = Math.max(1, dataset.readings().size() / 5_000);
        graphics.setColor(new Color(90, 130, 170));
        for (int index = sampleStep; index < dataset.readings().size(); index += sampleStep) {
            int x1 = 60 + (index - sampleStep) * 1_080 / dataset.readings().size(); int x2 = 60 + index * 1_080 / dataset.readings().size();
            int y1 = 350 - (int) ((dataset.readings().get(index - sampleStep) - min) / (max - min) * 280);
            int y2 = 350 - (int) ((dataset.readings().get(index) - min) / (max - min) * 280); graphics.drawLine(x1, y1, x2, y2);
        }
        graphics.setColor(new Color(20, 70, 110));
        for (int index = sampleStep; index < averages.size(); index += sampleStep) {
            int x1 = 60 + (index - sampleStep + window - 1) * 1_080 / dataset.readings().size(); int x2 = 60 + (index + window - 1) * 1_080 / dataset.readings().size();
            int y1 = 350 - (int) ((averages.get(index - sampleStep) - min) / (max - min) * 280);
            int y2 = 350 - (int) ((averages.get(index) - min) / (max - min) * 280); graphics.drawLine(x1, y1, x2, y2);
        }
        graphics.setColor(Color.BLACK); graphics.drawString("Toolkit and reference runtimes (seconds)", 450, 430);
        double runtimeMax = results.stream().flatMapToDouble(row -> java.util.stream.DoubleStream.of(row.toolkitSeconds(), row.referenceSeconds())).max().orElseThrow();
        int groupWidth = 1_000 / results.size();
        for (int index = 0; index < results.size(); index++) {
            BenchmarkResult row = results.get(index); int baseX = 80 + index * groupWidth;
            graphics.setColor(new Color(210, 120, 50)); graphics.fillRect(baseX, 730 - (int) (row.toolkitSeconds() / runtimeMax * 240), 60, (int) (row.toolkitSeconds() / runtimeMax * 240));
            graphics.setColor(new Color(70, 150, 100)); graphics.fillRect(baseX + 70, 730 - (int) (row.referenceSeconds() / runtimeMax * 240), 60, (int) (row.referenceSeconds() / runtimeMax * 240));
            graphics.setColor(Color.BLACK); graphics.drawString(row.operation(), baseX, 755);
        }
        graphics.dispose(); if (destination.getParent() != null) Files.createDirectories(destination.getParent()); ImageIO.write(image, "png", destination.toFile());
    }

    public static void main(String[] arguments) throws IOException {
        int size = 100_000; int queries = 20_000; int window = 64; double threshold = 4; int repetitions = 5; long seed = GenerateSensorData.DEFAULT_SEED;
        for (int index = 0; index < arguments.length; index++) {
            switch (arguments[index]) {
                case "--size" -> size = Integer.parseInt(arguments[++index]); case "--queries" -> queries = Integer.parseInt(arguments[++index]);
                case "--window" -> window = Integer.parseInt(arguments[++index]); case "--threshold" -> threshold = Double.parseDouble(arguments[++index]);
                case "--repetitions" -> repetitions = Integer.parseInt(arguments[++index]); case "--seed" -> seed = Long.parseLong(arguments[++index]);
                default -> throw new IllegalArgumentException("Unknown option: " + arguments[index]);
            }
        }
        SensorDataset dataset = GenerateSensorData.generateSensorData(size, seed, null);
        List<BenchmarkResult> results = runBenchmark(dataset, window, queries, threshold, repetitions, seed + 1);
        GenerateSensorData.writeSensorCsv(dataset, Paths.get("data", "sensor_readings.csv"));
        writeBenchmarkCsv(results, Paths.get("results", "benchmark_results.csv"));
        createBenchmarkPlot(dataset, results, Paths.get("results", "sensor_and_runtime_comparison.png"), window, threshold);
        System.out.println("Readings: " + dataset.readings().size()); System.out.println("Injected anomalies: " + dataset.anomalyIndices());
        for (BenchmarkResult row : results) System.out.printf(Locale.ROOT, "%s toolkit=%.6fs reference=%.6fs factor=%.2f error=%.3e%n", row.operation(), row.toolkitSeconds(), row.referenceSeconds(), row.referenceFactor(), row.maximumAbsoluteError());
    }
}
