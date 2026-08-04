package lab.dsa.module02.basic;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/** Stores elements in a manually resized contiguous reference buffer. */
public final class DynamicArray<T> implements Iterable<T> {
    public record GrowthEvent(int sizeBefore, int oldCapacity, int newCapacity, int copiedElements) {
    }

    private Object[] buffer;
    private int size;
    private int capacity;
    private final List<GrowthEvent> growthEvents = new ArrayList<>();
    private long totalCopiedElements;

    public DynamicArray() {
        this(1);
    }

    public DynamicArray(int initialCapacity) {
        if (initialCapacity < 1) {
            throw new IllegalArgumentException("Initial capacity must be at least 1");
        }
        capacity = initialCapacity;
        buffer = new Object[initialCapacity];
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    public List<GrowthEvent> growthEvents() {
        return List.copyOf(growthEvents);
    }

    public long totalCopiedElements() {
        return totalCopiedElements;
    }

    @SuppressWarnings("unchecked")
    public T get(int index) {
        return (T) buffer[normalizeIndex(index)];
    }

    public void set(int index, T value) {
        buffer[normalizeIndex(index)] = value;
    }

    public void append(T value) {
        ensureCapacity();
        buffer[size++] = value;
    }

    public void insert(int index, T value) {
        if (index < 0 || index > size) {
            throw new IndexOutOfBoundsException("Insert index out of range: " + index);
        }
        ensureCapacity();
        for (int position = size; position > index; position--) {
            buffer[position] = buffer[position - 1];
        }
        buffer[index] = value;
        size++;
    }

    @SuppressWarnings("unchecked")
    public T delete(int index) {
        int normalized = normalizeIndex(index);
        T removed = (T) buffer[normalized];
        for (int position = normalized; position < size - 1; position++) {
            buffer[position] = buffer[position + 1];
        }
        buffer[--size] = null;
        return removed;
    }

    public List<T> toList() {
        List<T> result = new ArrayList<>(size);
        for (T value : this) {
            result.add(value);
        }
        return result;
    }

    private void ensureCapacity() {
        if (size == capacity) {
            resize(capacity * 2);
        }
    }

    private void resize(int newCapacity) {
        if (newCapacity <= capacity) {
            throw new IllegalArgumentException("New capacity must exceed current capacity");
        }
        int oldCapacity = capacity;
        Object[] newBuffer = new Object[newCapacity];
        for (int index = 0; index < size; index++) {
            newBuffer[index] = buffer[index];
        }
        GrowthEvent event = new GrowthEvent(size, oldCapacity, newCapacity, size);
        buffer = newBuffer;
        capacity = newCapacity;
        growthEvents.add(event);
        totalCopiedElements += event.copiedElements();
    }

    private int normalizeIndex(int index) {
        int normalized = index < 0 ? index + size : index;
        if (normalized < 0 || normalized >= size) {
            throw new IndexOutOfBoundsException("Array index out of range: " + index);
        }
        return normalized;
    }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<>() {
            private int index;

            @Override
            public boolean hasNext() {
                return index < size;
            }

            @Override
            public T next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return get(index++);
            }
        };
    }

    @Override
    public String toString() {
        return "DynamicArray(" + toList() + ", capacity=" + capacity + ")";
    }
}
