# DSA Practice Lab — Java Edition

This directory is the Java counterpart of `DSA-lab`. It mirrors the currently
available learning material while replacing Python implementations and examples
with idiomatic Java.

## Purpose

The lab prepares you to study data structures and algorithms in Java without
losing the progression, datasets, experiments, or project ideas from the Python
edition.

## Translation contract

- Markdown documentation is English only.
- Every Python source file is represented by corresponding Java source code.
- Python tests become JUnit 5 tests.
- Python examples inside Markdown become Java examples.
- Identifiers, comments, Javadocs, messages, output, and chart labels are English.
- Existing CSV, JSON, text, and image artifacts are retained when they remain
  meaningful for the Java implementation.
- Runtime artifacts such as `__pycache__`, `.pyc`, `.pytest_cache`, and
  `.DS_Store` are not copied.
- Java code may use Java-specific types and project layout while preserving the
  behavior, learning objective, edge cases, and complexity of the source.

## Toolchain

- Java 21
- Maven 3.9 or newer
- JUnit 5

Each project is an independent Maven project. From a project directory, run:

```bash
mvn test
```

Some projects also provide an executable demo class documented in their
`README.md`.

## Current environment note

A newer JDK and Maven are now available in the workspace. The projects still
target Java 21 bytecode through `maven.compiler.release`, and Maven compilation
plus JUnit execution are part of the module validation workflow.

See `PROGRESS.md` for exact migration status.
