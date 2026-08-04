package lab.dsa.module02.basic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DynamicArrayTest {
    @Test void initialStateUsesFixedBuffer() {
        DynamicArray<Integer> array = new DynamicArray<>(3);
        assertEquals(0, array.size()); assertEquals(3, array.capacity()); assertEquals(List.of(), array.toList());
    }

    @Test void invalidInitialCapacityIsRejected() {
        for (int capacity : List.of(0, -1, -10)) {
            assertThrows(IllegalArgumentException.class, () -> new DynamicArray<>(capacity));
        }
    }

    @Test void appendPreservesOrderAndDoublesCapacity() {
        DynamicArray<Integer> array = new DynamicArray<>();
        List<Integer> capacities = new ArrayList<>();
        for (int value = 0; value < 9; value++) { array.append(value); capacities.add(array.capacity()); }
        assertEquals(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8), array.toList());
        assertEquals(List.of(1, 2, 4, 4, 8, 8, 8, 8, 16), capacities);
    }

    @Test void getSetAndNegativeIndicesWork() {
        DynamicArray<String> array = new DynamicArray<>();
        for (String value : List.of("A", "B", "C")) array.append(value);
        assertEquals("A", array.get(0)); assertEquals("C", array.get(-1));
        array.set(1, "X"); array.set(-1, "Z"); assertEquals(List.of("A", "X", "Z"), array.toList());
    }

    @Test void accessBoundsAreEnforced() {
        DynamicArray<Integer> array = new DynamicArray<>(); array.append(10);
        for (int index : List.of(1, -2, 100)) {
            assertThrows(IndexOutOfBoundsException.class, () -> array.get(index));
            assertThrows(IndexOutOfBoundsException.class, () -> array.set(index, 0));
        }
    }

    @Test void insertionHandlesStartMiddleAndEnd() {
        DynamicArray<String> array = new DynamicArray<>(); array.append("B"); array.append("D");
        array.insert(0, "A"); array.insert(2, "C"); array.insert(array.size(), "E");
        assertEquals(List.of("A", "B", "C", "D", "E"), array.toList());
    }

    @Test void insertionCanTriggerResizeAndRejectsInvalidBounds() {
        DynamicArray<Integer> array = new DynamicArray<>(2); array.append(1); array.append(3); array.insert(1, 2);
        assertEquals(List.of(1, 2, 3), array.toList()); assertEquals(4, array.capacity());
        assertEquals(2, array.growthEvents().getLast().copiedElements());
        for (int index : List.of(-1, 4, 99)) assertThrows(IndexOutOfBoundsException.class, () -> array.insert(index, 0));
    }

    @Test void deletionReturnsValueShiftsAndDoesNotShrink() {
        DynamicArray<String> array = new DynamicArray<>();
        for (String value : List.of("A", "B", "C", "D", "E")) array.append(value);
        int capacity = array.capacity(); assertEquals("B", array.delete(1)); assertEquals("E", array.delete(-1));
        assertEquals(List.of("A", "C", "D"), array.toList()); assertEquals(capacity, array.capacity());
    }

    @Test void deletionBoundsAreEnforced() {
        DynamicArray<Integer> array = new DynamicArray<>();
        assertThrows(IndexOutOfBoundsException.class, () -> array.delete(0));
        assertThrows(IndexOutOfBoundsException.class, () -> array.delete(-1));
    }

    @Test void mixedObjectsAreSupported() {
        Object marker = new Object(); DynamicArray<Object> array = new DynamicArray<>();
        array.append(42); array.append("DSA"); array.append(null); array.append(marker);
        assertEquals(42, array.get(0)); assertEquals("DSA", array.get(1)); assertEquals(null, array.get(2)); assertSame(marker, array.get(3));
    }

    @Test void growthEventsAreExactAndImmutable() {
        DynamicArray<Integer> array = new DynamicArray<>(); for (int value = 0; value < 9; value++) array.append(value);
        assertEquals(List.of(new DynamicArray.GrowthEvent(1, 1, 2, 1), new DynamicArray.GrowthEvent(2, 2, 4, 2),
                new DynamicArray.GrowthEvent(4, 4, 8, 4), new DynamicArray.GrowthEvent(8, 8, 16, 8)), array.growthEvents());
        assertThrows(UnsupportedOperationException.class, () -> array.growthEvents().clear());
        assertEquals(15, array.totalCopiedElements());
    }

    @Test void totalResizeCopiesHaveLinearBound() {
        for (int count : List.of(1, 2, 3, 8, 16, 100, 1_000)) {
            DynamicArray<Integer> array = new DynamicArray<>(); for (int value = 0; value < count; value++) array.append(value);
            assertTrue(array.totalCopiedElements() < 2L * count);
        }
    }

    @Test void iterationAndRepresentationWork() {
        DynamicArray<Integer> array = new DynamicArray<>(2); array.append(3); array.append(5);
        List<Integer> iterated = new ArrayList<>(); array.forEach(iterated::add);
        assertEquals(List.of(3, 5), iterated); assertEquals("DynamicArray([3, 5], capacity=2)", array.toString());
    }

    @Test void firstEightCostsShowResizeSpikes() {
        List<GrowthExperiment.AppendMeasurement> points = GrowthExperiment.runExperiment(8, 1);
        assertEquals(List.of(1, 2, 3, 1, 5, 1, 1, 1), points.stream().map(GrowthExperiment.AppendMeasurement::actualCost).toList());
        assertEquals(List.of(1, 2, 4, 4, 8, 8, 8, 8), points.stream().map(GrowthExperiment.AppendMeasurement::capacity).toList());
    }

    @Test void experimentRejectsInvalidConfiguration() {
        assertThrows(IllegalArgumentException.class, () -> GrowthExperiment.runExperiment(0, 1));
        assertThrows(IllegalArgumentException.class, () -> GrowthExperiment.runExperiment(10, 0));
    }

    @Test void csvContainsOneRowPerAppend(@TempDir Path directory) throws IOException {
        Path destination = directory.resolve("growth.csv");
        GrowthExperiment.writeCsv(GrowthExperiment.runExperiment(5, 1), destination);
        List<String> rows = Files.readAllLines(destination); assertEquals(6, rows.size());
        assertTrue(rows.getLast().contains(",8,4,5,"));
    }
}
