package lab.dsa.module02.basic;

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
import java.util.List;
import java.util.Locale;
import javax.imageio.ImageIO;

/** Visualizes capacity growth and amortized append costs. */
public final class GrowthExperiment {
    private GrowthExperiment() {
    }

    public record AppendMeasurement(int appendNumber, int size, int capacity,
                                    int copiedElements, int actualCost,
                                    double cumulativeAverageCost) {
    }

    public static List<AppendMeasurement> runExperiment(int appendCount, int initialCapacity) {
        if (appendCount < 1) throw new IllegalArgumentException("Append count must be positive");
        if (initialCapacity < 1) throw new IllegalArgumentException("Initial capacity must be at least 1");
        DynamicArray<Integer> array = new DynamicArray<>(initialCapacity);
        List<AppendMeasurement> measurements = new ArrayList<>();
        long cumulativeCost = 0;
        for (int value = 0; value < appendCount; value++) {
            int eventCountBefore = array.growthEvents().size();
            array.append(value);
            int copied = array.growthEvents().size() > eventCountBefore
                    ? array.growthEvents().getLast().copiedElements() : 0;
            int cost = 1 + copied;
            cumulativeCost += cost;
            int appendNumber = value + 1;
            measurements.add(new AppendMeasurement(appendNumber, array.size(), array.capacity(),
                    copied, cost, (double) cumulativeCost / appendNumber));
        }
        return measurements;
    }

    public static void writeCsv(List<AppendMeasurement> measurements, Path destination) throws IOException {
        if (destination.getParent() != null) Files.createDirectories(destination.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(destination, StandardCharsets.UTF_8)) {
            writer.write("append_number,length,capacity,copied_elements,actual_cost,cumulative_average_cost");
            writer.newLine();
            for (AppendMeasurement point : measurements) {
                writer.write(String.format(Locale.ROOT, "%d,%d,%d,%d,%d,%.8f", point.appendNumber(),
                        point.size(), point.capacity(), point.copiedElements(), point.actualCost(),
                        point.cumulativeAverageCost()));
                writer.newLine();
            }
        }
    }

    public static void createPlot(List<AppendMeasurement> measurements, Path destination) throws IOException {
        int width = 1_100;
        int height = 800;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setColor(Color.WHITE); graphics.fillRect(0, 0, width, height);
        graphics.setColor(Color.BLACK); graphics.drawString("DynamicArray: capacity and amortized append cost", 400, 25);
        drawSeries(graphics, measurements, 70, 70, 950, 280, true);
        drawSeries(graphics, measurements, 70, 440, 950, 280, false);
        graphics.dispose();
        if (destination.getParent() != null) Files.createDirectories(destination.getParent());
        ImageIO.write(image, "png", destination.toFile());
    }

    private static void drawSeries(Graphics2D graphics, List<AppendMeasurement> points,
                                   int x, int y, int width, int height, boolean capacity) {
        graphics.setColor(Color.BLACK); graphics.drawRect(x, y, width, height);
        graphics.drawString(capacity ? "Geometric buffer growth" : "Append cost and cumulative average", x + 330, y - 12);
        double maximum = capacity
                ? points.stream().mapToDouble(AppendMeasurement::capacity).max().orElseThrow()
                : points.stream().mapToDouble(AppendMeasurement::actualCost).max().orElseThrow();
        int previousX = -1; int previousY = -1;
        graphics.setColor(capacity ? Color.BLUE : Color.RED);
        for (int index = 0; index < points.size(); index++) {
            AppendMeasurement point = points.get(index);
            double value = capacity ? point.capacity() : point.actualCost();
            int currentX = x + index * width / Math.max(1, points.size() - 1);
            int currentY = y + height - (int) (value / maximum * height);
            if (previousX >= 0) graphics.drawLine(previousX, previousY, currentX, currentY);
            previousX = currentX; previousY = currentY;
        }
        if (!capacity) {
            graphics.setColor(new Color(0, 140, 0)); previousX = -1; previousY = -1;
            for (int index = 0; index < points.size(); index++) {
                int currentX = x + index * width / Math.max(1, points.size() - 1);
                int currentY = y + height - (int) (points.get(index).cumulativeAverageCost() / maximum * height);
                if (previousX >= 0) graphics.drawLine(previousX, previousY, currentX, currentY);
                previousX = currentX; previousY = currentY;
            }
        }
    }

    public static void printSummary(List<AppendMeasurement> measurements) {
        System.out.println("Resize events\n" + "=".repeat(58));
        measurements.stream().filter(point -> point.copiedElements() > 0).forEach(point ->
                System.out.printf("append=%4d, capacity=%4d, copied=%4d, actual_cost=%4d%n",
                        point.appendNumber(), point.capacity(), point.copiedElements(), point.actualCost()));
        AppendMeasurement last = measurements.getLast();
        int totalCopies = measurements.stream().mapToInt(AppendMeasurement::copiedElements).sum();
        System.out.printf(Locale.ROOT, "\nSummary%n  appends: %d%n  final capacity: %d%n  total resize copies: %d%n  cumulative average cost: %.4f%n",
                last.appendNumber(), last.capacity(), totalCopies, last.cumulativeAverageCost());
    }

    public static void main(String[] arguments) throws IOException {
        int appendCount = 64; int initialCapacity = 1; Path outputDirectory = Paths.get("results");
        for (int index = 0; index < arguments.length; index++) {
            switch (arguments[index]) {
                case "--appends" -> appendCount = Integer.parseInt(arguments[++index]);
                case "--initial-capacity" -> initialCapacity = Integer.parseInt(arguments[++index]);
                case "--output-dir" -> outputDirectory = Paths.get(arguments[++index]);
                default -> throw new IllegalArgumentException("Unknown option: " + arguments[index]);
            }
        }
        List<AppendMeasurement> measurements = runExperiment(appendCount, initialCapacity);
        Path csvPath = outputDirectory.resolve("growth_log.csv");
        Path plotPath = outputDirectory.resolve("capacity_and_costs.png");
        writeCsv(measurements, csvPath); createPlot(measurements, plotPath); printSummary(measurements);
        System.out.println("\nArtifacts\n  CSV:  " + csvPath + "\n  Plot: " + plotPath);
    }
}
