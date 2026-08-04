package lab.dsa.module01.finalproject;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Random;

/** Generates deterministic synthetic e-commerce events for the audit. */
public final class GenerateData {
    public static final long DEFAULT_SEED = 20_260_716L;
    public static final int DEFAULT_ROWS = 8_000;
    public static final Path DEFAULT_OUTPUT = Paths.get("data", "events.csv");
    private static final List<String> CATEGORIES = List.of(
            "books", "electronics", "fashion", "garden", "grocery", "sports", "toys", "travel");
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm");

    private GenerateData() {
    }

    public static void generateEvents(Path destination, int rows, long seed) throws IOException {
        if (rows < 1) {
            throw new IllegalArgumentException("Rows must be positive");
        }
        Random random = new Random(seed);
        int customerCount = Math.max(25, rows / 4);
        LocalDateTime startTime = LocalDateTime.of(2025, 1, 1, 0, 0);
        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("event_id,customer_id,category,amount_cents,discount_cents,status,event_timestamp");
            writer.newLine();
            for (int index = 0; index < rows; index++) {
                double statusDraw = random.nextDouble();
                String status = statusDraw < .72 ? "completed" : statusDraw < .88 ? "pending" : "cancelled";
                int amountCents = 500 + random.nextInt(74_501);
                int maximumDiscount = Math.min(amountCents / 3, 5_000);
                int discountCents = random.nextInt(maximumDiscount + 1);
                LocalDateTime timestamp = startTime.plusMinutes(random.nextInt(365 * 24 * 60));
                writer.write(String.format("EVT-%07d,CUST-%05d,%s,%d,%d,%s,%s",
                        index, random.nextInt(customerCount), CATEGORIES.get(random.nextInt(CATEGORIES.size())),
                        amountCents, discountCents, status, TIMESTAMP_FORMAT.format(timestamp)));
                writer.newLine();
            }
        }
    }

    public static void main(String[] arguments) throws IOException {
        int rows = DEFAULT_ROWS;
        long seed = DEFAULT_SEED;
        Path output = DEFAULT_OUTPUT;
        for (int index = 0; index < arguments.length; index++) {
            switch (arguments[index]) {
                case "--rows" -> rows = Integer.parseInt(arguments[++index]);
                case "--seed" -> seed = Long.parseLong(arguments[++index]);
                case "--output" -> output = Paths.get(arguments[++index]);
                default -> throw new IllegalArgumentException("Unknown option: " + arguments[index]);
            }
        }
        generateEvents(output, rows, seed);
        System.out.printf("Generated %d events at %s with seed %d.%n", rows, output, seed);
    }
}
