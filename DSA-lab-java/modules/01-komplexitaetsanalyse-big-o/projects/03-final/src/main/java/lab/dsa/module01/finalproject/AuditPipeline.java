package lab.dsa.module01.finalproject;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Baseline and optimized customer aggregation pipelines. */
public final class AuditPipeline {
    private AuditPipeline() {
    }

    /** One immutable event loaded from the source CSV. */
    public record Event(String eventId, String customerId, String category,
                        int amountCents, int discountCents, String status,
                        String eventTimestamp) {
        public Event {
            if (amountCents < 0) {
                throw new IllegalArgumentException("Amount must not be negative");
            }
            if (discountCents < 0 || discountCents > amountCents) {
                throw new IllegalArgumentException("Discount must be between zero and amount");
            }
        }

        public int netCents() {
            return amountCents - discountCents;
        }
    }

    /** Aggregated metrics for one customer. */
    public record CustomerSummary(String customerId, int completedEvents,
                                  long netRevenueCents, long averageNetCents,
                                  int uniqueCategories, String latestEventTimestamp) {
    }

    private static final class Accumulator {
        private int completedEvents;
        private long netRevenueCents;
        private final Set<String> categories = new HashSet<>();
        private String latestEventTimestamp = "";

        private void update(Event event) {
            completedEvents++;
            netRevenueCents += event.netCents();
            categories.add(event.category());
            if (event.eventTimestamp().compareTo(latestEventTimestamp) > 0) {
                latestEventTimestamp = event.eventTimestamp();
            }
        }

        private CustomerSummary toSummary(String customerId) {
            return new CustomerSummary(customerId, completedEvents, netRevenueCents,
                    netRevenueCents / completedEvents, categories.size(), latestEventTimestamp);
        }
    }

    public static List<Event> loadEvents(Path source) throws IOException {
        List<String> lines = Files.readAllLines(source, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            return List.of();
        }
        String expectedHeader = "event_id,customer_id,category,amount_cents,discount_cents,status,event_timestamp";
        if (!lines.getFirst().equals(expectedHeader)) {
            throw new IllegalArgumentException("Unexpected event CSV header");
        }
        List<Event> events = new ArrayList<>(lines.size() - 1);
        for (int lineNumber = 1; lineNumber < lines.size(); lineNumber++) {
            String[] fields = lines.get(lineNumber).split(",", -1);
            if (fields.length != 7) {
                throw new IllegalArgumentException("Invalid event row at line " + (lineNumber + 1));
            }
            events.add(new Event(fields[0], fields[1], fields[2], Integer.parseInt(fields[3]),
                    Integer.parseInt(fields[4]), fields[5], fields[6]));
        }
        return events;
    }

    /** Aggregates through list membership checks and repeated full scans. */
    public static List<CustomerSummary> inefficientPipeline(List<Event> events) {
        List<Event> eventList = new ArrayList<>(events);
        List<String> customerIds = new ArrayList<>();
        for (Event event : eventList) {
            if (event.status().equals("completed") && !customerIds.contains(event.customerId())) {
                customerIds.add(event.customerId());
            }
        }

        List<CustomerSummary> summaries = new ArrayList<>();
        for (String customerId : customerIds) {
            List<Event> customerEvents = new ArrayList<>();
            for (Event event : eventList) {
                if (event.status().equals("completed") && event.customerId().equals(customerId)) {
                    customerEvents.add(event);
                }
            }
            long netRevenueCents = 0;
            Set<String> categories = new HashSet<>();
            String latestTimestamp = "";
            for (Event event : customerEvents) {
                netRevenueCents += event.netCents();
                categories.add(event.category());
                if (event.eventTimestamp().compareTo(latestTimestamp) > 0) {
                    latestTimestamp = event.eventTimestamp();
                }
            }
            summaries.add(new CustomerSummary(customerId, customerEvents.size(), netRevenueCents,
                    netRevenueCents / customerEvents.size(), categories.size(), latestTimestamp));
        }
        summaries.sort(Comparator.comparing(CustomerSummary::customerId));
        return summaries;
    }

    /** Aggregates all completed events in one pass with hash lookups. */
    public static List<CustomerSummary> optimizedPipeline(List<Event> events) {
        Map<String, Accumulator> accumulators = new HashMap<>();
        for (Event event : events) {
            if (!event.status().equals("completed")) {
                continue;
            }
            accumulators.computeIfAbsent(event.customerId(), ignored -> new Accumulator()).update(event);
        }
        return accumulators.keySet().stream().sorted()
                .map(customerId -> accumulators.get(customerId).toSummary(customerId)).toList();
    }

    public static void writeSummaries(List<CustomerSummary> summaries, Path destination) throws IOException {
        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("customer_id,completed_events,net_revenue_cents,average_net_cents,unique_categories,latest_event_timestamp");
            writer.newLine();
            for (CustomerSummary summary : summaries) {
                writer.write(String.join(",", summary.customerId(),
                        Integer.toString(summary.completedEvents()), Long.toString(summary.netRevenueCents()),
                        Long.toString(summary.averageNetCents()), Integer.toString(summary.uniqueCategories()),
                        summary.latestEventTimestamp()));
                writer.newLine();
            }
        }
    }
}
