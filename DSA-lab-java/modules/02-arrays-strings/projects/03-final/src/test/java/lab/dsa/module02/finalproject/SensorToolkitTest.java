package lab.dsa.module02.finalproject;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import lab.dsa.module02.finalproject.GenerateSensorData.SensorDataset;
import lab.dsa.module02.finalproject.SensorToolkit.Outlier;
import lab.dsa.module02.finalproject.SensorToolkit.Range;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SensorToolkitTest {
    @Test void generationIsReproducibleAndKeepsGroundTruth() {
        SensorDataset first = GenerateSensorData.generateSensorData(1_000, 42, null);
        SensorDataset second = GenerateSensorData.generateSensorData(1_000, 42, null);
        assertEquals(first, second); assertEquals(List.of(200, 500, 800), first.anomalyIndices());
    }

    @Test void generationValidatesSizeAndAnomalyIndices() {
        assertThrows(IllegalArgumentException.class, () -> GenerateSensorData.generateSensorData(0, 1, null));
        assertThrows(IllegalArgumentException.class, () -> GenerateSensorData.generateSensorData(10, 1, List.of(2, 2)));
        assertThrows(IndexOutOfBoundsException.class, () -> GenerateSensorData.generateSensorData(10, 1, List.of(10)));
    }

    @Test void sensorCsvContainsHeaderAndRows(@TempDir Path directory) throws IOException {
        Path destination = directory.resolve("sensor.csv");
        GenerateSensorData.writeSensorCsv(GenerateSensorData.generateSensorData(5, 1, List.of(2)), destination);
        List<String> rows = Files.readAllLines(destination); assertEquals(6, rows.size());
        assertEquals("index,temperature,is_injected_outlier", rows.getFirst()); assertTrue(rows.get(3).endsWith(",1"));
    }

    @Test void movingAverageUsesValidWindows() {
        assertEquals(List.of(2.0, 3.0, 4.0), SensorToolkit.movingAverage(List.of(1.0, 2.0, 3.0, 4.0, 5.0), 3));
        assertEquals(List.of(2.5), SensorToolkit.movingAverage(List.of(2.0, 3.0), 2));
        assertThrows(IllegalArgumentException.class, () -> SensorToolkit.movingAverage(List.of(1.0), 0));
        assertThrows(IllegalArgumentException.class, () -> SensorToolkit.movingAverage(List.of(1.0), 2));
        assertThrows(IllegalArgumentException.class, () -> SensorToolkit.movingAverage(List.of(1.0, Double.NaN), 1));
    }

    @Test void prefixIndexAnswersRangesAndCopiesItsState() {
        List<Double> source = new ArrayList<>(List.of(4.0, -1.0, 7.0, 3.0, 2.0));
        SensorToolkit.PrefixSumIndex index = SensorToolkit.PrefixSumIndex.fromReadings(source); source.set(0, 100.0);
        assertEquals(15.0, index.rangeSum(0, 5)); assertEquals(9.0, index.rangeSum(1, 4)); assertEquals(0.0, index.rangeSum(1, 1));
        double[] values = index.prefixValues(); values[0] = 99; assertArrayEquals(new double[]{0, 4, 3, 10, 13, 15}, index.prefixValues());
        assertThrows(IndexOutOfBoundsException.class, () -> index.rangeSum(-1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> index.rangeSum(2, 1));
    }

    @Test void outlierDetectionHandlesSignsConstantsAndInclusiveThreshold() {
        List<Double> values = new ArrayList<>(java.util.Collections.nCopies(20, 0.0)); values.set(3, 10.0); values.set(15, -10.0);
        List<Outlier> outliers = SensorToolkit.detectZScoreOutliers(values, 3.0);
        assertEquals(List.of(3, 15), outliers.stream().map(Outlier::index).toList());
        assertTrue(outliers.getFirst().zScore() > 0); assertTrue(outliers.getLast().zScore() < 0);
        assertEquals(List.of(), SensorToolkit.detectZScoreOutliers(java.util.Collections.nCopies(10, 2.0), 3));
        assertEquals(List.of(0, 1), SensorToolkit.detectZScoreOutliers(List.of(-1.0, 1.0), 1).stream().map(Outlier::index).toList());
        assertThrows(IllegalArgumentException.class, () -> SensorToolkit.detectZScoreOutliers(List.of(1.0), 0));
    }

    @Test void generatedAnomaliesAreDetected() {
        SensorDataset dataset = GenerateSensorData.generateSensorData(10_000, 42, null);
        List<Integer> detected = SensorToolkit.detectZScoreOutliers(dataset.readings(), 4).stream().map(Outlier::index).toList();
        assertTrue(detected.containsAll(dataset.anomalyIndices()));
    }

    @Test void analysisCombinesImmutableSnapshots() {
        List<Double> values = new ArrayList<>(java.util.Collections.nCopies(20, 0.0)); values.set(10, 10.0);
        SensorToolkit.AnalysisResult result = SensorToolkit.analyzeSensorReadings(values, 4,
                List.of(new Range(0, 4), new Range(9, 12)), 3);
        assertEquals(17, result.movingAverages().size()); assertEquals(List.of(0.0, 10.0), result.rangeSums());
        assertEquals(List.of(10), result.outliers().stream().map(Outlier::index).toList());
        assertThrows(UnsupportedOperationException.class, () -> result.rangeSums().clear());
    }

    @Test void benchmarkCoversOperationsAndWritesCsv(@TempDir Path directory) throws IOException {
        List<SensorBenchmark.BenchmarkResult> results = SensorBenchmark.runBenchmark(
                GenerateSensorData.generateSensorData(2_000, 7, null), 16, 300, 4, 1, 8);
        assertEquals(List.of("moving_average", "prefix_build", "range_queries", "outlier_detection"),
                results.stream().map(SensorBenchmark.BenchmarkResult::operation).toList());
        assertTrue(results.stream().allMatch(row -> row.toolkitSeconds() > 0 && row.referenceSeconds() > 0 && row.referenceFactor() > 0));
        assertTrue(results.stream().allMatch(row -> row.maximumAbsoluteError() < 1e-8));
        Path destination = directory.resolve("benchmark.csv"); SensorBenchmark.writeBenchmarkCsv(results, destination);
        List<String> rows = Files.readAllLines(destination); assertEquals(5, rows.size());
        assertTrue(rows.getFirst().contains("toolkit_to_reference_factor"));
    }

    @Test void benchmarkRejectsInvalidConfiguration() {
        SensorDataset dataset = GenerateSensorData.generateSensorData(20, 1, null);
        assertThrows(IllegalArgumentException.class, () -> SensorBenchmark.runBenchmark(dataset, 2, 0, 4, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> SensorBenchmark.runBenchmark(dataset, 2, 2, 4, 0, 2));
    }
}
