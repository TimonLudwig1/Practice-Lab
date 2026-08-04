package lab.dsa.module02.finalproject;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/** Generates a reproducible synthetic temperature-sensor series. */
public final class GenerateSensorData {
    public static final long DEFAULT_SEED = 20_260_716L;
    private static final double[] ANOMALY_OFFSETS = {15.0, -16.0, 18.0};

    private GenerateSensorData() {
    }

    public record SensorDataset(List<Double> readings, List<Integer> anomalyIndices) {
        public SensorDataset {
            readings = List.copyOf(readings);
            anomalyIndices = List.copyOf(anomalyIndices);
        }
    }

    public static SensorDataset generateSensorData(int size, long seed, List<Integer> anomalyIndices) {
        if (size <= 0) throw new IllegalArgumentException("Size must be positive");
        List<Integer> selected = anomalyIndices == null
                ? new ArrayList<>(new HashSet<>(List.of(size / 5, size / 2, (4 * size) / 5)))
                : new ArrayList<>(anomalyIndices);
        selected.sort(Integer::compareTo);
        if (new HashSet<>(selected).size() != selected.size()) throw new IllegalArgumentException("Anomaly indices must be unique");
        if (selected.stream().anyMatch(index -> index < 0 || index >= size)) throw new IndexOutOfBoundsException("Anomaly index outside generated series");
        Random random = new Random(seed);
        List<Double> readings = new ArrayList<>(size);
        for (int index = 0; index < size; index++) {
            double baseline = 20.0 + 0.00002 * index;
            double dailyCycle = 2.5 * Math.sin(2.0 * Math.PI * index / 1_440);
            readings.add(baseline + dailyCycle + random.nextGaussian() * 0.35);
        }
        for (int position = 0; position < selected.size(); position++) {
            int index = selected.get(position);
            readings.set(index, readings.get(index) + ANOMALY_OFFSETS[position % ANOMALY_OFFSETS.length]);
        }
        return new SensorDataset(readings, selected);
    }

    public static void writeSensorCsv(SensorDataset dataset, Path destination) throws IOException {
        if (destination.getParent() != null) Files.createDirectories(destination.getParent());
        Set<Integer> anomalies = new HashSet<>(dataset.anomalyIndices());
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("index,temperature,is_injected_outlier"); writer.newLine();
            for (int index = 0; index < dataset.readings().size(); index++) {
                writer.write(String.format(Locale.ROOT, "%d,%.8f,%d", index, dataset.readings().get(index), anomalies.contains(index) ? 1 : 0));
                writer.newLine();
            }
        }
    }

    public static void main(String[] arguments) throws IOException {
        int size = 10_000; long seed = DEFAULT_SEED; Path output = Paths.get("data", "sensor_readings.csv");
        for (int index = 0; index < arguments.length; index++) {
            switch (arguments[index]) {
                case "--size" -> size = Integer.parseInt(arguments[++index]);
                case "--seed" -> seed = Long.parseLong(arguments[++index]);
                case "--output" -> output = Paths.get(arguments[++index]);
                default -> throw new IllegalArgumentException("Unknown option: " + arguments[index]);
            }
        }
        SensorDataset dataset = generateSensorData(size, seed, null); writeSensorCsv(dataset, output);
        System.out.println("Generated " + size + " readings at " + output + ". Anomalies: " + Arrays.toString(dataset.anomalyIndices().toArray()));
    }
}
