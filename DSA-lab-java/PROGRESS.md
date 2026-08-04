# DSA Practice Lab — Java Migration Tracker

This file tracks the one-to-one migration from `../DSA-lab` to Java.

## Next action

> **Translate Module 03 completely: English theory with Java examples, all three
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
| 01 Complexity Analysis & Big-O | complete | complete | complete | complete | complete | pending | JDK became available after the module commit; full rerun pending |
| 02 Arrays & Strings | complete | complete | complete | complete | complete | complete | 37/37 JUnit tests pass across three Maven projects |
| 03 Recursion & Divide and Conquer | open | open | open | open | open | pending | Not translated yet |
| 04 Linked Lists | open | open | open | open | open | pending | Not translated yet |
| 05 Stacks & Queues | open | open | open | open | open | pending | Not translated yet |
| 06 Hashing & Hash Maps | open | open | open | open | open | pending | Not translated yet |
| 07 Sorting Algorithms | open | open | open | open | open | pending | Not translated yet |
| 08 Binary Search & Search Variants | open | open | open | open | open | pending | Not translated yet |
| 09 Two Pointers & Sliding Window | open | open | open | open | open | pending | Not translated yet |
| 10 Trees & Binary Search Trees | open | open | open | open | open | pending | Not translated yet |
| 11 Heaps & Priority Queues | open | open | open | open | open | pending | Not translated yet |
| 12 Graphs I: Representation & Traversal | open | open | open | open | open | pending | Not translated yet |
| 13 Graphs II: Shortest Paths, Union-Find & MST | open | open | open | open | open | pending | Not translated yet |
| 14 Greedy Algorithms | open | not present | not present | not present | open | pending | Only theory exists in source lab |

## Log

| Date | Work | Notes |
|---|---|---|
| 2026-08-04 | Java mirror initialized | English-only documentation, Java 21/Maven/JUnit target, source exclusions, and per-module migration tracker defined. |
| 2026-08-04 | Module 01 translated | Theory rewritten in English with Java examples; three projects migrated to Java 21 with Maven, JUnit, CSV output, plots, generators, and English reports. Source-to-target coverage checked. |
| 2026-08-04 | Module 02 translated and tested | Arrays, strings, dynamic growth, pattern catalog, and sensor toolkit migrated to Java 21. NumPy comparison replaced with validated Java reference implementations; 37/37 JUnit tests pass. |
