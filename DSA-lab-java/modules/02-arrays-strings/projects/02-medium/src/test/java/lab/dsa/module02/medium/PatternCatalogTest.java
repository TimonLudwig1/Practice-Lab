package lab.dsa.module02.medium;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class PatternCatalogTest {
    @Test void rotationHandlesTypicalLargeNegativeAndSmallInputs() {
        List<Integer> values = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6, 7));
        PatternCatalog.rotateRightInPlace(values, 3); assertEquals(List.of(5, 6, 7, 1, 2, 3, 4), values);
        values = new ArrayList<>(List.of(1, 2, 3)); PatternCatalog.rotateRightInPlace(values, 7); assertEquals(List.of(3, 1, 2), values);
        values = new ArrayList<>(List.of(1, 2, 3, 4)); PatternCatalog.rotateRightInPlace(values, -1); assertEquals(List.of(2, 3, 4, 1), values);
        List<Integer> empty = new ArrayList<>(); PatternCatalog.rotateRightInPlace(empty, 100); assertEquals(List.of(), empty);
    }

    @Test void mergeHandlesInterleavingDuplicatesAndEmptySides() {
        List<Integer> target = new ArrayList<>(Arrays.asList(1, 4, 7, null, null, null));
        PatternCatalog.mergeSortedInPlace(target, 3, List.of(2, 3, 8)); assertEquals(List.of(1, 2, 3, 4, 7, 8), target);
        target = new ArrayList<>(Arrays.asList(1, 2, null, null)); PatternCatalog.mergeSortedInPlace(target, 2, List.of(1, 2)); assertEquals(List.of(1, 1, 2, 2), target);
        target = new ArrayList<>(Arrays.asList(null, null)); PatternCatalog.mergeSortedInPlace(target, 0, List.of(1, 2)); assertEquals(List.of(1, 2), target);
    }

    @Test void mergeRejectsWrongBuffersRangesAndUnsortedInputs() {
        assertThrows(IllegalArgumentException.class, () -> PatternCatalog.mergeSortedInPlace(new ArrayList<>(Arrays.asList(1, null, null)), 1, List.of(2)));
        assertThrows(IllegalArgumentException.class, () -> PatternCatalog.mergeSortedInPlace(new ArrayList<>(Arrays.asList(2, 1, null)), 2, List.of(3)));
        assertThrows(IllegalArgumentException.class, () -> PatternCatalog.mergeSortedInPlace(new ArrayList<>(Arrays.asList(1, null, null)), 1, List.of(3, 2)));
        assertThrows(IllegalArgumentException.class, () -> PatternCatalog.mergeSortedInPlace(new ArrayList<>(List.of(1)), -1, List.of(1, 2)));
    }

    @Test void prefixSumAnswersRangesAndIsIndependentOfSource() {
        List<Integer> source = new ArrayList<>(List.of(4, -1, 7, 3, 2));
        PatternCatalog.PrefixSum prefix = PatternCatalog.PrefixSum.fromValues(source); source.set(0, 100);
        assertEquals(15, prefix.rangeSum(0, 5)); assertEquals(9, prefix.rangeSum(1, 4)); assertEquals(0, prefix.rangeSum(1, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> prefix.rangeSum(-1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> prefix.rangeSum(2, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> prefix.rangeSum(0, 6));
    }

    @Test void anagramsHandleCountsSensitivityUnicodeAndEmptyText() {
        assertTrue(PatternCatalog.areAnagrams("listen", "silent")); assertFalse(PatternCatalog.areAnagrams("Listen", "silent"));
        assertTrue(PatternCatalog.areAnagrams("a b", "ab ")); assertTrue(PatternCatalog.areAnagrams("äöä", "ääö"));
        assertTrue(PatternCatalog.areAnagrams("😀a", "a😀")); assertTrue(PatternCatalog.areAnagrams("", ""));
        assertFalse(PatternCatalog.areAnagrams("aab", "abb")); assertFalse(PatternCatalog.areAnagrams("abc", "ab"));
    }

    @Test void sortedDeduplicationHandlesRepeatedUniqueEmptyAndInvalidInputs() {
        List<Integer> values = new ArrayList<>(List.of(1, 1, 2, 2, 2, 5));
        assertEquals(3, PatternCatalog.removeDuplicatesSorted(values)); assertEquals(List.of(1, 2, 5), values);
        List<Integer> empty = new ArrayList<>(); assertEquals(0, PatternCatalog.removeDuplicatesSorted(empty));
        List<Integer> invalid = new ArrayList<>(List.of(1, 3, 2));
        assertThrows(IllegalArgumentException.class, () -> PatternCatalog.removeDuplicatesSorted(invalid)); assertEquals(List.of(1, 3, 2), invalid);
    }

    @Test void movingZerosIsStable() {
        List<Number> values = new ArrayList<>(List.of(0, 3, 0, 1, 0, 8));
        assertEquals(3, PatternCatalog.moveZerosToEnd(values)); assertEquals(List.of(3, 1, 8, 0, 0, 0), values);
        values = new ArrayList<>(List.of(0.0, 2.5, -0.0, -1.0)); PatternCatalog.moveZerosToEnd(values);
        assertEquals(List.of(2.5, -1.0), values.subList(0, 2));
        assertTrue(values.get(2).doubleValue() == 0.0);
        assertTrue(values.get(3).doubleValue() == 0.0);
    }

    @Test void productExceptSelfHandlesZerosNegativesAndSmallInputs() {
        assertArrayEquals(new long[]{24, 12, 8, 6}, PatternCatalog.productExceptSelf(new long[]{1, 2, 3, 4}));
        assertArrayEquals(new long[]{0, 0, 8, 0}, PatternCatalog.productExceptSelf(new long[]{1, 2, 0, 4}));
        assertArrayEquals(new long[]{-6, 3, -2}, PatternCatalog.productExceptSelf(new long[]{-1, 2, -3}));
        assertArrayEquals(new long[]{1}, PatternCatalog.productExceptSelf(new long[]{7}));
    }

    @Test void longestUniqueSubstringHandlesWindowsTiesUnicodeAndEmptyText() {
        assertEquals("abc", PatternCatalog.longestUniqueSubstring("abcabcbb"));
        assertEquals("ab", PatternCatalog.longestUniqueSubstring("abba"));
        assertEquals("wke", PatternCatalog.longestUniqueSubstring("pwwkew"));
        assertEquals("bcaef", PatternCatalog.longestUniqueSubstring("abcaef"));
        assertEquals("😀ab", PatternCatalog.longestUniqueSubstring("😀ab😀"));
        assertEquals("", PatternCatalog.longestUniqueSubstring(""));
    }

    @Test void spiralTraversesDifferentRectanglesAndRejectsRaggedRows() {
        assertEquals(List.of(1, 2, 3, 6, 9, 8, 7, 4, 5), PatternCatalog.spiralOrder(List.of(List.of(1, 2, 3), List.of(4, 5, 6), List.of(7, 8, 9))));
        assertEquals(List.of(1, 2, 3, 4, 8, 7, 6, 5), PatternCatalog.spiralOrder(List.of(List.of(1, 2, 3, 4), List.of(5, 6, 7, 8))));
        assertEquals(List.of(), PatternCatalog.spiralOrder(List.of()));
        assertThrows(IllegalArgumentException.class, () -> PatternCatalog.spiralOrder(List.of(List.of(1, 2), List.of(3))));
    }

    @Test void compressionHandlesRunsMultiDigitCountsSinglesAndEmptyInput() {
        List<Character> characters = characters("aabcccccaaa");
        assertEquals(7, PatternCatalog.compressRunsInPlace(characters)); assertEquals(characters("a2bc5a3"), characters);
        characters = characters("abbbbbbbbbbbb"); PatternCatalog.compressRunsInPlace(characters); assertEquals(characters("ab12"), characters);
        characters = characters("abcd"); assertEquals(4, PatternCatalog.compressRunsInPlace(characters)); assertEquals(characters("abcd"), characters);
        characters = new ArrayList<>(); assertEquals(0, PatternCatalog.compressRunsInPlace(characters));
    }

    private static List<Character> characters(String text) {
        List<Character> result = new ArrayList<>(); for (char value : text.toCharArray()) result.add(value); return result;
    }
}
