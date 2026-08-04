# DSA Practice Lab — Java Migration Tracker

This file tracks the one-to-one migration from `../DSA-lab` to Java.

## Next action

> **Translate Module 02 completely: English theory with Java examples, all three
> projects, Maven configurations, Java implementations, datasets, and JUnit tests.**

## Migration rules

1. The Python lab is read-only source material.
2. Work in module order.
3. Translate documentation to English only.
4. Translate every Python source and test semantically to Java.
5. Translate every Python Markdown code block to Java.
6. Copy meaningful tracked datasets and result artifacts; exclude runtime caches.
7. Preserve deterministic seeds and expected behavior.
8. Check source-to-target file coverage after every module.
9. Compile and run JUnit tests when Java 21 becomes available.
10. Commit after each completed module.

## Status

| Module | Theory | Project 01 | Project 02 | Project 03 | Coverage | Java tests | Notes |
|---|---|---|---|---|---|---|---|
| 01 Complexity Analysis & Big-O | complete | complete | complete | complete | complete | blocked | Static checks passed; JDK not installed |
| 02 Arrays & Strings | open | open | open | open | open | blocked | JDK not installed |
| 03 Recursion & Divide and Conquer | open | open | open | open | open | blocked | JDK not installed |
| 04 Linked Lists | open | open | open | open | open | blocked | JDK not installed |
| 05 Stacks & Queues | open | open | open | open | open | blocked | JDK not installed |
| 06 Hashing & Hash Maps | open | open | open | open | open | blocked | JDK not installed |
| 07 Sorting Algorithms | open | open | open | open | open | blocked | JDK not installed |
| 08 Binary Search & Search Variants | open | open | open | open | open | blocked | JDK not installed |
| 09 Two Pointers & Sliding Window | open | open | open | open | open | blocked | JDK not installed |
| 10 Trees & Binary Search Trees | open | open | open | open | open | blocked | JDK not installed |
| 11 Heaps & Priority Queues | open | open | open | open | open | blocked | JDK not installed |
| 12 Graphs I: Representation & Traversal | open | open | open | open | open | blocked | JDK not installed |
| 13 Graphs II: Shortest Paths, Union-Find & MST | open | open | open | open | open | blocked | JDK not installed |
| 14 Greedy Algorithms | open | not present | not present | not present | open | blocked | Only theory exists in source lab |

## Log

| Date | Work | Notes |
|---|---|---|
| 2026-08-04 | Java mirror initialized | English-only documentation, Java 21/Maven/JUnit target, source exclusions, and per-module migration tracker defined. |
| 2026-08-04 | Module 01 translated | Theory rewritten in English with Java examples; three projects migrated to Java 21 with Maven, JUnit, CSV output, plots, generators, and English reports. Source-to-target coverage checked. |
