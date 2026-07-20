"""Show heapify, sift-up, sift-down, and heap sort as visible states."""

from binary_heap import MinHeap, heap_sort


def main() -> None:
    source = [9, 4, 7, 1, -2, 6, 5]
    heap = MinHeap(source)

    print("BOTTOM-UP HEAPIFY")
    print(f"source: {source}")
    print(f"heap:   {heap.to_list()}")
    print(f"valid:  {heap.is_valid()}")

    print("\nPUSH / SIFT-UP")
    for value in (3, -5, 8):
        before = heap.to_list()
        heap.push(value)
        print(f"push {value:>2}: {before} -> {heap.to_list()}")

    print("\nPOP / SIFT-DOWN")
    while heap:
        before = heap.to_list()
        minimum = heap.pop()
        print(f"pop {minimum:>2}: {before} -> {heap.to_list()}")

    print("\nHEAP SORT")
    unsorted = [7, 2, 9, 2, -1, 5]
    print(f"input:  {unsorted}")
    print(f"output: {heap_sort(unsorted)}")
    print(f"input unchanged: {unsorted}")


if __name__ == "__main__":
    main()
