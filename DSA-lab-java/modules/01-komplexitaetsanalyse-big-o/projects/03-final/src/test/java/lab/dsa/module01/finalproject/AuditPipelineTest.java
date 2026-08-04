package lab.dsa.module01.finalproject;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import lab.dsa.module01.finalproject.AuditPipeline.CustomerSummary;
import lab.dsa.module01.finalproject.AuditPipeline.Event;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AuditPipelineTest {
    private static List<Event> sampleEvents() {
        return List.of(
                new Event("E1", "C2", "books", 2_000, 200, "completed", "2025-01-02T10:00"),
                new Event("E2", "C1", "sports", 4_000, 500, "completed", "2025-01-01T10:00"),
                new Event("E3", "C1", "books", 2_500, 0, "completed", "2025-01-03T10:00"),
                new Event("E4", "C2", "toys", 9_000, 0, "cancelled", "2025-01-04T10:00"));
    }

    @Test
    void sameSeedProducesIdenticalFiles(@TempDir Path directory) throws IOException {
        Path first = directory.resolve("first.csv");
        Path second = directory.resolve("second.csv");
        GenerateData.generateEvents(first, 50, 123);
        GenerateData.generateEvents(second, 50, 123);
        assertArrayEquals(Files.readAllBytes(first), Files.readAllBytes(second));
    }

    @Test
    void generatorWritesRequestedNumberOfLoadableRows(@TempDir Path directory) throws IOException {
        Path destination = directory.resolve("events.csv");
        GenerateData.generateEvents(destination, 37, 456);
        List<Event> events = AuditPipeline.loadEvents(destination);
        assertEquals(37, events.size());
        assertTrue(events.stream().allMatch(event -> event.eventId().startsWith("EVT-")));
    }

    @Test
    void generatorRejectsNonpositiveRows(@TempDir Path directory) {
        assertThrows(IllegalArgumentException.class,
                () -> GenerateData.generateEvents(directory.resolve("events.csv"), 0, 1));
    }

    @Test
    void pipelinesMatchOnConstructedEvents() {
        assertEquals(AuditPipeline.inefficientPipeline(sampleEvents()),
                AuditPipeline.optimizedPipeline(sampleEvents()));
    }

    @Test
    void expectedSummaryValuesAreProduced() {
        assertEquals(List.of(
                new CustomerSummary("C1", 2, 6_000, 3_000, 2, "2025-01-03T10:00"),
                new CustomerSummary("C2", 1, 1_800, 1_800, 1, "2025-01-02T10:00")),
                AuditPipeline.optimizedPipeline(sampleEvents()));
    }

    @Test
    void noncompletedEventsAreIgnored() {
        List<Event> ignored = List.of(
                new Event("E1", "C1", "books", 100, 0, "pending", "2025-01-01"),
                new Event("E2", "C2", "toys", 100, 0, "cancelled", "2025-01-02"));
        assertEquals(List.of(), AuditPipeline.inefficientPipeline(ignored));
        assertEquals(List.of(), AuditPipeline.optimizedPipeline(ignored));
    }

    @Test
    void pipelinesMatchOnGeneratedData(@TempDir Path directory) throws IOException {
        Path destination = directory.resolve("events.csv");
        GenerateData.generateEvents(destination, 250, 789);
        List<Event> events = AuditPipeline.loadEvents(destination);
        assertEquals(AuditPipeline.inefficientPipeline(events), AuditPipeline.optimizedPipeline(events));
    }

    @Test
    void summaryCsvHasOneRowPerCustomer(@TempDir Path directory) throws IOException {
        Path destination = directory.resolve("summary.csv");
        AuditPipeline.writeSummaries(AuditPipeline.optimizedPipeline(sampleEvents()), destination);
        List<String> rows = Files.readAllLines(destination);
        assertEquals(3, rows.size());
        assertTrue(rows.get(1).startsWith("C1,"));
        assertTrue(rows.get(2).startsWith("C2,"));
    }

    @Test
    void benchmarkValidationRejectsInvalidSettings() {
        List<Event> events = sampleEvents();
        List<List<Integer>> invalidSizes = List.of(
                List.of(), List.of(0, 2), List.of(2, 2), List.of(3, 2), List.of(2, 5));
        invalidSizes.forEach(sizes -> assertThrows(IllegalArgumentException.class,
                () -> RunAudit.validateBenchmark(events, sizes, 1)));
        assertThrows(IllegalArgumentException.class,
                () -> RunAudit.validateBenchmark(events, List.of(2, 3), 0));
    }

    @Test
    void smallBenchmarkVerifiesAndMeasuresBothPipelines() {
        List<RunAudit.AuditMeasurement> measurements =
                RunAudit.benchmarkPipelines(sampleEvents(), List.of(2, 4), 2);
        assertEquals(2, measurements.size());
        assertTrue(measurements.stream().allMatch(point -> point.baselineSeconds() > 0));
        assertTrue(measurements.stream().allMatch(point -> point.optimizedSeconds() > 0));
        assertTrue(measurements.stream().allMatch(point -> point.speedup() > 0));
    }

    @Test
    void measurementCsvAndMarkdownReportAreWritten(@TempDir Path directory) throws IOException {
        List<RunAudit.AuditMeasurement> measurements = List.of(
                new RunAudit.AuditMeasurement(100, .02, .005, 25),
                new RunAudit.AuditMeasurement(200, .08, .01, 50));
        Path csvPath = directory.resolve("audit.csv");
        Path reportPath = directory.resolve("report.md");
        RunAudit.writeMeasurements(measurements, csvPath);
        RunAudit.writeReport(measurements, reportPath, 123, 3);
        List<String> rows = Files.readAllLines(csvPath);
        String report = Files.readString(reportPath);
        assertEquals(3, rows.size());
        assertTrue(rows.getLast().endsWith(",8.000000"));
        assertTrue(report.contains("Seed: 123"));
        assertTrue(report.contains("8.00x"));
    }
}
