package lab.dsa.module01.medium;

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

class ComplexityDetectiveTest {
    @Test void case01ReturnsFixedPosition() { assertEquals(40, ComplexityDetective.case01(2)); assertEquals(40, ComplexityDetective.case01(10_000)); }
    @Test void case02CountsHalvings() { assertEquals(1, ComplexityDetective.case02(2)); assertEquals(3, ComplexityDetective.case02(8)); assertEquals(4, ComplexityDetective.case02(16)); }
    @Test void case03CombinesTwoPasses() { assertEquals(10, ComplexityDetective.case03(4)); assertEquals(55, ComplexityDetective.case03(10)); }
    @Test void case04SumsShrinkingRanges() { assertEquals(3, ComplexityDetective.case04(2)); assertEquals(15, ComplexityDetective.case04(8)); assertEquals(18, ComplexityDetective.case04(10)); }
    @Test void case05FindsAllSetValues() { assertEquals(2, ComplexityDetective.case05(2)); assertEquals(127, ComplexityDetective.case05(127)); }
    @Test void case06CountsAllListMisses() { assertEquals(2, ComplexityDetective.case06(2)); assertEquals(127, ComplexityDetective.case06(127)); }
    @Test void case07SumsValuesWhileCopying() { assertEquals(1, ComplexityDetective.case07(2)); assertEquals(45, ComplexityDetective.case07(10)); }
    @Test void case08BuildsReversedList() { assertEquals(3, ComplexityDetective.case08(2)); assertEquals(19, ComplexityDetective.case08(10)); }
    @Test void case09ReturnsCombinedLengths() { assertEquals(4, ComplexityDetective.case09(2)); assertEquals(254, ComplexityDetective.case09(127)); }
    @Test void case10IsDeterministic() { assertEquals(1_444_343_149L, ComplexityDetective.case10(2)); assertEquals(4_213_176_899L, ComplexityDetective.case10(32)); assertEquals(4_198_688_481L, ComplexityDetective.case10(64)); }

    @Test
    void configurationRejectsInvalidValues() {
        List<ComplexityDetective.BenchmarkConfig> invalid = List.of(
                new ComplexityDetective.BenchmarkConfig(List.of(), 5, .01, 10),
                new ComplexityDetective.BenchmarkConfig(List.of(2), 5, .01, 10),
                new ComplexityDetective.BenchmarkConfig(List.of(1, 2), 5, .01, 10),
                new ComplexityDetective.BenchmarkConfig(List.of(4, 4), 5, .01, 10),
                new ComplexityDetective.BenchmarkConfig(List.of(8, 4), 5, .01, 10),
                new ComplexityDetective.BenchmarkConfig(List.of(2, 4), 0, .01, 10),
                new ComplexityDetective.BenchmarkConfig(List.of(2, 4), 5, 0, 10),
                new ComplexityDetective.BenchmarkConfig(List.of(2, 4), 5, .01, 0));
        invalid.forEach(config -> assertThrows(IllegalArgumentException.class, config::validate));
    }

    @Test void selectingCasesRejectsUnknownIdentifier() {
        assertThrows(IllegalArgumentException.class, () -> ComplexityDetective.selectCases(List.of("01", "99")));
    }

    @Test void logLogSlopeRecoversLinearGrowth() {
        List<ComplexityDetective.Measurement> points = List.of(2, 4, 8, 16).stream()
                .map(size -> new ComplexityDetective.Measurement("X", size, .25 * size, 1)).toList();
        assertEquals(1.0, ComplexityDetective.estimateLogLogSlope(points), 1e-12);
    }

    @Test void smallBenchmarkReturnsAllPoints() {
        ComplexityDetective.BenchmarkConfig config =
                new ComplexityDetective.BenchmarkConfig(List.of(8, 16), 2, .0001, 1_024);
        Map<String, IntToLongFunction> cases = new LinkedHashMap<>();
        cases.put("01", ComplexityDetective::case01);
        cases.put("03", ComplexityDetective::case03);
        List<ComplexityDetective.Measurement> measurements = ComplexityDetective.runBenchmarks(cases, config);
        assertEquals(4, measurements.size());
        assertTrue(measurements.stream().allMatch(point -> point.secondsPerCall() > 0));
        assertTrue(measurements.stream().allMatch(point -> point.iterationsPerSample() >= 1));
    }

    @Test void csvContainsHeaderAndAllRows(@TempDir Path directory) throws IOException {
        Path destination = directory.resolve("measurements.csv");
        ComplexityDetective.writeCsv(List.of(
                new ComplexityDetective.Measurement("01", 8, .001, 16),
                new ComplexityDetective.Measurement("02", 8, .002, 8)), destination);
        List<String> rows = Files.readAllLines(destination);
        assertEquals("case,input_size,seconds_per_call,iterations_per_sample", rows.getFirst());
        assertEquals(3, rows.size());
        assertTrue(rows.get(1).startsWith("01,8,"));
        assertTrue(rows.get(2).startsWith("02,8,"));
    }
}
