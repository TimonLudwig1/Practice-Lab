package lab.dsa.module01.basic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntToLongFunction;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RuntimeLabTest {
    @Test
    void curveAIsIndependentOfInputSize() {
        assertEquals(32, RuntimeLab.curveA(2));
        assertEquals(32, RuntimeLab.curveA(10_000));
    }

    @Test
    void curveBCountsHalvingLevels() {
        assertEquals(32, RuntimeLab.curveB(2));
        assertEquals(96, RuntimeLab.curveB(8));
        assertEquals(128, RuntimeLab.curveB(16));
    }

    @Test
    void curveCVisitsEveryInputItem() {
        assertEquals(16, RuntimeLab.curveC(2));
        assertEquals(1_016, RuntimeLab.curveC(127));
    }

    @Test
    void curveDCombinesLinearAndHalvingWork() {
        assertEquals(2, RuntimeLab.curveD(2));
        assertEquals(24, RuntimeLab.curveD(8));
        assertEquals(64, RuntimeLab.curveD(16));
    }

    @Test
    void curveECountsUnorderedPairs() {
        for (int size : List.of(2, 3, 10, 25)) {
            assertEquals((long) size * (size - 1) / 2, RuntimeLab.curveE(size));
        }
    }

    @Test
    void configurationRejectsInvalidValues() {
        List<RuntimeLab.BenchmarkConfig> invalidConfigurations = List.of(
                new RuntimeLab.BenchmarkConfig(List.of(), 5, 0.01, 10),
                new RuntimeLab.BenchmarkConfig(List.of(1, 2), 5, 0.01, 10),
                new RuntimeLab.BenchmarkConfig(List.of(4, 4), 5, 0.01, 10),
                new RuntimeLab.BenchmarkConfig(List.of(8, 4), 5, 0.01, 10),
                new RuntimeLab.BenchmarkConfig(List.of(2, 4), 0, 0.01, 10),
                new RuntimeLab.BenchmarkConfig(List.of(2, 4), 5, 0, 10),
                new RuntimeLab.BenchmarkConfig(List.of(2, 4), 5, 0.01, 0));
        for (RuntimeLab.BenchmarkConfig configuration : invalidConfigurations) {
            assertThrows(IllegalArgumentException.class, configuration::validate);
        }
    }

    @Test
    void logLogSlopeRecoversQuadraticGrowth() {
        List<RuntimeLab.Measurement> points = List.of(2, 4, 8, 16).stream()
                .map(size -> new RuntimeLab.Measurement("X", size, 3.0 * size * size, 1))
                .toList();
        assertEquals(2.0, RuntimeLab.estimateLogLogSlope(points), 1e-12);
    }

    @Test
    void smallBenchmarkReturnsPositiveMeasurements() {
        RuntimeLab.BenchmarkConfig configuration =
                new RuntimeLab.BenchmarkConfig(List.of(8, 16), 2, 0.0001, 1_024);
        Map<String, IntToLongFunction> functions = new LinkedHashMap<>();
        functions.put("A", RuntimeLab::curveA);
        functions.put("C", RuntimeLab::curveC);
        List<RuntimeLab.Measurement> measurements = RuntimeLab.runBenchmarks(functions, configuration);
        assertEquals(4, measurements.size());
        assertTrue(measurements.stream().allMatch(point -> point.secondsPerCall() > 0));
        assertTrue(measurements.stream().allMatch(point -> point.iterationsPerSample() >= 1));
    }

    @Test
    void csvContainsHeaderAndAllMeasurements(@TempDir Path temporaryDirectory) throws IOException {
        List<RuntimeLab.Measurement> points = List.of(
                new RuntimeLab.Measurement("A", 8, 0.001, 16),
                new RuntimeLab.Measurement("B", 8, 0.002, 8));
        Path destination = temporaryDirectory.resolve("measurements.csv");
        RuntimeLab.writeCsv(points, destination);
        List<String> rows = Files.readAllLines(destination);
        assertEquals("curve,input_size,seconds_per_call,iterations_per_sample", rows.getFirst());
        assertEquals(3, rows.size());
        assertTrue(rows.get(1).startsWith("A,8,"));
        assertTrue(rows.get(2).startsWith("B,8,"));
    }
}
