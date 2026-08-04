package lab.dsa.module02.finalproject;

import java.util.ArrayList;
import java.util.List;

/** Array-based algorithms for analyzing a one-dimensional sensor series. */
public final class SensorToolkit {
    private SensorToolkit() {
    }

    public record Range(int start, int end) {
    }

    public record Outlier(int index, double value, double zScore) {
    }

    public record AnalysisResult(List<Double> movingAverages,
                                 List<Double> rangeSums,
                                 List<Outlier> outliers) {
        public AnalysisResult {
            movingAverages = List.copyOf(movingAverages);
            rangeSums = List.copyOf(rangeSums);
            outliers = List.copyOf(outliers);
        }
    }

    public static List<Double> movingAverage(List<Double> values, int window) {
        validateReadings(values);
        validateWindow(window, values.size());
        double rollingSum = 0;
        for (int index = 0; index < window; index++) rollingSum += values.get(index);
        List<Double> averages = new ArrayList<>(values.size() - window + 1);
        averages.add(rollingSum / window);
        for (int right = window; right < values.size(); right++) {
            rollingSum += values.get(right) - values.get(right - window);
            averages.add(rollingSum / window);
        }
        return averages;
    }

    public static final class PrefixSumIndex {
        private final double[] prefix;

        private PrefixSumIndex(double[] prefix) {
            this.prefix = prefix;
        }

        public static PrefixSumIndex fromReadings(List<Double> values) {
            validateReadings(values);
            double[] prefix = new double[values.size() + 1];
            for (int index = 0; index < values.size(); index++) {
                prefix[index + 1] = prefix[index] + values.get(index);
            }
            return new PrefixSumIndex(prefix);
        }

        public int size() {
            return prefix.length - 1;
        }

        public double[] prefixValues() {
            return prefix.clone();
        }

        public double rangeSum(int start, int end) {
            if (start < 0 || end < start || end > size()) {
                throw new IndexOutOfBoundsException("Range must satisfy 0 <= start <= end <= size");
            }
            return prefix[end] - prefix[start];
        }

        public List<Double> batchRangeSums(Iterable<Range> ranges) {
            List<Double> results = new ArrayList<>();
            for (Range range : ranges) results.add(rangeSum(range.start(), range.end()));
            return results;
        }
    }

    public static List<Outlier> detectZScoreOutliers(List<Double> values, double threshold) {
        validateReadings(values);
        if (!Double.isFinite(threshold) || threshold <= 0) {
            throw new IllegalArgumentException("Threshold must be finite and positive");
        }
        if (values.isEmpty()) return List.of();
        double sum = 0;
        for (double value : values) sum += value;
        double mean = sum / values.size();
        double squaredDifferenceSum = 0;
        for (double value : values) {
            double difference = value - mean;
            squaredDifferenceSum += difference * difference;
        }
        double standardDeviation = Math.sqrt(squaredDifferenceSum / values.size());
        if (standardDeviation == 0) return List.of();
        List<Outlier> outliers = new ArrayList<>();
        for (int index = 0; index < values.size(); index++) {
            double zScore = (values.get(index) - mean) / standardDeviation;
            if (Math.abs(zScore) >= threshold) outliers.add(new Outlier(index, values.get(index), zScore));
        }
        return outliers;
    }

    public static AnalysisResult analyzeSensorReadings(List<Double> values, int window,
                                                       Iterable<Range> ranges, double threshold) {
        PrefixSumIndex prefix = PrefixSumIndex.fromReadings(values);
        return new AnalysisResult(movingAverage(values, window),
                prefix.batchRangeSums(ranges), detectZScoreOutliers(values, threshold));
    }

    private static void validateReadings(List<Double> values) {
        for (Double value : values) {
            if (value == null) throw new IllegalArgumentException("Readings must not contain null");
            if (!Double.isFinite(value)) throw new IllegalArgumentException("Readings must be finite");
        }
    }

    private static void validateWindow(int window, int size) {
        if (window <= 0) throw new IllegalArgumentException("Window must be positive");
        if (window > size) throw new IllegalArgumentException("Window must not exceed the reading count");
    }
}
