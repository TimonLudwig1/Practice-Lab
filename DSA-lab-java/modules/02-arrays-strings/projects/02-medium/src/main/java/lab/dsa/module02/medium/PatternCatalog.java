package lab.dsa.module02.medium;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Classic array and string patterns with explicit complexity trade-offs. */
public final class PatternCatalog {
    private PatternCatalog() {
    }

    public static <T> void rotateRightInPlace(List<T> values, int steps) {
        if (values.size() < 2) return;
        int normalized = Math.floorMod(steps, values.size());
        if (normalized == 0) return;
        reverseRange(values, 0, values.size() - 1);
        reverseRange(values, 0, normalized - 1);
        reverseRange(values, normalized, values.size() - 1);
    }

    private static <T> void reverseRange(List<T> values, int left, int right) {
        while (left < right) Collections.swap(values, left++, right--);
    }

    /** Merges sorted integer inputs into a target containing trailing null buffer slots. */
    public static void mergeSortedInPlace(List<Integer> target, int validCount, List<Integer> other) {
        if (validCount < 0 || validCount > target.size()) throw new IllegalArgumentException("Valid count is outside target");
        if (target.size() != validCount + other.size()) throw new IllegalArgumentException("Target must contain exactly enough buffer slots");
        if (!isNonDecreasing(target.subList(0, validCount))) throw new IllegalArgumentException("Valid target range must be sorted");
        if (!isNonDecreasing(other)) throw new IllegalArgumentException("Other must be sorted");
        int left = validCount - 1; int right = other.size() - 1; int write = target.size() - 1;
        while (right >= 0) {
            if (left >= 0 && target.get(left) > other.get(right)) target.set(write--, target.get(left--));
            else target.set(write--, other.get(right--));
        }
    }

    private static <T extends Comparable<? super T>> boolean isNonDecreasing(List<T> values) {
        for (int index = 1; index < values.size(); index++) if (values.get(index - 1).compareTo(values.get(index)) > 0) return false;
        return true;
    }

    public static final class PrefixSum {
        private final double[] prefix;
        private PrefixSum(double[] prefix) { this.prefix = prefix; }
        public static PrefixSum fromValues(List<? extends Number> values) {
            double[] prefix = new double[values.size() + 1];
            for (int index = 0; index < values.size(); index++) prefix[index + 1] = prefix[index] + values.get(index).doubleValue();
            return new PrefixSum(prefix);
        }
        public int size() { return prefix.length - 1; }
        public double rangeSum(int start, int end) {
            if (start < 0 || end < start || end > size()) throw new IndexOutOfBoundsException("Range must satisfy 0 <= start <= end <= size");
            return prefix[end] - prefix[start];
        }
    }

    public static boolean areAnagrams(String left, String right) {
        int[] leftPoints = left.codePoints().toArray(); int[] rightPoints = right.codePoints().toArray();
        if (leftPoints.length != rightPoints.length) return false;
        Map<Integer, Integer> counts = new HashMap<>();
        for (int point : leftPoints) counts.merge(point, 1, Integer::sum);
        for (int point : rightPoints) {
            int remaining = counts.getOrDefault(point, 0) - 1;
            if (remaining < 0) return false;
            if (remaining == 0) counts.remove(point); else counts.put(point, remaining);
        }
        return counts.isEmpty();
    }

    public static <T extends Comparable<? super T>> int removeDuplicatesSorted(List<T> values) {
        if (!isNonDecreasing(values)) throw new IllegalArgumentException("Values must be sorted");
        if (values.isEmpty()) return 0;
        int write = 1;
        for (int read = 1; read < values.size(); read++) {
            if (!values.get(read).equals(values.get(write - 1))) values.set(write++, values.get(read));
        }
        values.subList(write, values.size()).clear();
        return write;
    }

    public static int moveZerosToEnd(List<Number> values) {
        int write = 0;
        for (int read = 0; read < values.size(); read++) {
            if (values.get(read).doubleValue() != 0.0) Collections.swap(values, write++, read);
        }
        return write;
    }

    public static long[] productExceptSelf(long[] values) {
        long[] result = new long[values.length];
        long prefix = 1;
        for (int index = 0; index < values.length; index++) { result[index] = prefix; prefix *= values[index]; }
        long suffix = 1;
        for (int index = values.length - 1; index >= 0; index--) { result[index] *= suffix; suffix *= values[index]; }
        return result;
    }

    public static String longestUniqueSubstring(String text) {
        int[] points = text.codePoints().toArray();
        Map<Integer, Integer> lastSeen = new HashMap<>();
        int windowStart = 0; int bestStart = 0; int bestLength = 0;
        for (int index = 0; index < points.length; index++) {
            Integer previous = lastSeen.put(points[index], index);
            if (previous != null && previous >= windowStart) windowStart = previous + 1;
            int length = index - windowStart + 1;
            if (length > bestLength) { bestStart = windowStart; bestLength = length; }
        }
        return new String(points, bestStart, bestLength);
    }

    public static <T> List<T> spiralOrder(List<List<T>> matrix) {
        if (matrix.isEmpty()) return List.of();
        int columns = matrix.getFirst().size();
        if (matrix.stream().anyMatch(row -> row.size() != columns)) throw new IllegalArgumentException("Matrix must be rectangular");
        if (columns == 0) return List.of();
        List<T> result = new ArrayList<>();
        int top = 0; int bottom = matrix.size() - 1; int left = 0; int right = columns - 1;
        while (top <= bottom && left <= right) {
            for (int column = left; column <= right; column++) result.add(matrix.get(top).get(column));
            top++;
            for (int row = top; row <= bottom; row++) result.add(matrix.get(row).get(right));
            right--;
            if (top <= bottom) { for (int column = right; column >= left; column--) result.add(matrix.get(bottom).get(column)); bottom--; }
            if (left <= right) { for (int row = bottom; row >= top; row--) result.add(matrix.get(row).get(left)); left++; }
        }
        return result;
    }

    public static int compressRunsInPlace(List<Character> characters) {
        int write = 0; int read = 0;
        while (read < characters.size()) {
            int runStart = read; char current = characters.get(read);
            while (read < characters.size() && characters.get(read) == current) read++;
            characters.set(write++, current);
            int runLength = read - runStart;
            if (runLength > 1) for (char digit : Integer.toString(runLength).toCharArray()) characters.set(write++, digit);
        }
        characters.subList(write, characters.size()).clear();
        return write;
    }
}
