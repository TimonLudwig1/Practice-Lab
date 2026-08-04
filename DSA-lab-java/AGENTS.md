# AGENTS.md — DSA Practice Lab Java Edition

This repository section is a learning workshop for data structures and algorithms
in Java. The Python lab at `../DSA-lab` is the semantic source and must remain
unchanged while the Java mirror is built.

## Source and progress

- `PROGRESS.md` is the source of truth for migration progress.
- `../DSA-lab` supplies the files and expected behavior to mirror.
- Translate only the next item named in `PROGRESS.md` unless the user explicitly
  requests a broader correction.

## Required language and file rules

1. Documentation is English only.
2. All Java identifiers, comments, Javadocs, string literals, errors, output,
   chart labels, and generated reports are English.
3. Python code and Python code fences are not allowed in the finished Java lab.
4. Markdown code examples use Java and must be valid for Java 21.
5. Python tests become equivalent JUnit 5 tests.
6. Preserve algorithms, invariants, edge cases, asymptotic behavior, fixed seeds,
   datasets, and observable results.
7. Prefer standard Java APIs. Add a dependency only when the Python project truly
   relies on an external capability such as plotting or tabular processing.
8. Use the standard Maven layout: `src/main/java` and `src/test/java`.
9. Do not copy caches, compiled files, virtual environments, or OS metadata.
10. A module is complete only after file coverage has been checked against the
    Python source and, when a JDK is available, all Maven tests pass.

## Git rule

Every Codex commit for this lab must include exactly this trailer after a blank
line:

```text
Co-authored-by: Codex <267193182+codex@users.noreply.github.com>
```
