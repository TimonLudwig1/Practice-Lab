package lab.dsa.module02.medium;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Runs one representative example for every catalog pattern. */
public final class CatalogDemo {
    private CatalogDemo() {
    }

    public static void main(String[] arguments) {
        List<Integer> rotated = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6, 7));
        PatternCatalog.rotateRightInPlace(rotated, 3);
        List<Integer> merged = new ArrayList<>(Arrays.asList(1, 4, 7, null, null, null));
        PatternCatalog.mergeSortedInPlace(merged, 3, List.of(2, 3, 8));
        PatternCatalog.PrefixSum prefix = PatternCatalog.PrefixSum.fromValues(List.of(4, -1, 7, 3, 2));
        List<Integer> unique = new ArrayList<>(List.of(1, 1, 2, 2, 2, 5));
        PatternCatalog.removeDuplicatesSorted(unique);
        List<Number> zeros = new ArrayList<>(List.of(0, 3, 0, 1, 0, 8));
        PatternCatalog.moveZerosToEnd(zeros);
        List<Character> compressed = new ArrayList<>();
        for (char value : "aaabcccccccccc".toCharArray()) compressed.add(value);
        PatternCatalog.compressRunsInPlace(compressed);
        System.out.println("Rotation                         " + rotated);
        System.out.println("Merge                            " + merged);
        System.out.println("Prefix sum [1:4]                 " + prefix.rangeSum(1, 4));
        System.out.println("Anagram                          " + PatternCatalog.areAnagrams("listen", "silent"));
        System.out.println("Deduplication                    " + unique);
        System.out.println("Move zeros                       " + zeros);
        System.out.println("Product except self              " + Arrays.toString(PatternCatalog.productExceptSelf(new long[]{1, 2, 3, 4})));
        System.out.println("Longest unique substring         " + PatternCatalog.longestUniqueSubstring("pwwkew"));
        System.out.println("Spiral                           " + PatternCatalog.spiralOrder(List.of(List.of(1, 2, 3), List.of(4, 5, 6), List.of(7, 8, 9))));
        StringBuilder encoded = new StringBuilder(); compressed.forEach(encoded::append);
        System.out.println("Compression                      " + encoded);
    }
}
