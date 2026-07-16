"""Analyze a directory tree recursively or with an explicit stack."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FileRecord:
    """A matched file with normalized path, size, and containing-directory depth."""

    relative_path: str
    size: int
    depth: int


@dataclass(frozen=True)
class DepthStats:
    """Aggregates for one directory depth."""

    depth: int
    directory_count: int
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class TreeAnalysis:
    """Canonical immutable analysis result for a directory tree."""

    total_bytes: int
    file_count: int
    directory_count: int
    max_depth: int
    depth_stats: tuple[DepthStats, ...]
    bytes_by_extension: tuple[tuple[str, int], ...]
    matches: tuple[FileRecord, ...]


@dataclass
class _DepthAccumulator:
    directory_count: int = 0
    file_count: int = 0
    total_bytes: int = 0


@dataclass
class _Accumulator:
    total_bytes: int = 0
    file_count: int = 0
    directory_count: int = 0
    max_depth: int = 0
    depths: dict[int, _DepthAccumulator] = field(default_factory=dict)
    extensions: dict[str, int] = field(default_factory=dict)
    matches: list[FileRecord] = field(default_factory=list)


def _validate_root_and_pattern(root: Path, pattern: str) -> Path:
    """Validate public inputs and return a resolved directory root."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")
    if not pattern:
        raise ValueError("pattern must not be empty")
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return root.resolve()


def _depth_bucket(accumulator: _Accumulator, depth: int) -> _DepthAccumulator:
    """Return the mutable aggregate bucket for one depth."""
    return accumulator.depths.setdefault(depth, _DepthAccumulator())


def _record_directory(accumulator: _Accumulator, depth: int) -> None:
    """Add one visited directory to the aggregates."""
    accumulator.directory_count += 1
    accumulator.max_depth = max(accumulator.max_depth, depth)
    _depth_bucket(accumulator, depth).directory_count += 1


def _record_file(
    accumulator: _Accumulator,
    root: Path,
    file_path: Path,
    depth: int,
    pattern: str,
) -> None:
    """Add one regular file to all size, depth, extension, and match aggregates."""
    size = file_path.stat().st_size
    relative_path = file_path.relative_to(root).as_posix()
    extension = file_path.suffix.lower() or "<none>"

    accumulator.total_bytes += size
    accumulator.file_count += 1
    accumulator.max_depth = max(accumulator.max_depth, depth)
    bucket = _depth_bucket(accumulator, depth)
    bucket.file_count += 1
    bucket.total_bytes += size
    accumulator.extensions[extension] = accumulator.extensions.get(extension, 0) + size

    if fnmatch.fnmatchcase(file_path.name, pattern) or fnmatch.fnmatchcase(
        relative_path, pattern
    ):
        accumulator.matches.append(FileRecord(relative_path, size, depth))


def _finish(accumulator: _Accumulator) -> TreeAnalysis:
    """Convert mutable traversal state into a deterministic immutable result."""
    depth_stats = tuple(
        DepthStats(
            depth=depth,
            directory_count=bucket.directory_count,
            file_count=bucket.file_count,
            total_bytes=bucket.total_bytes,
        )
        for depth, bucket in sorted(accumulator.depths.items())
    )
    return TreeAnalysis(
        total_bytes=accumulator.total_bytes,
        file_count=accumulator.file_count,
        directory_count=accumulator.directory_count,
        max_depth=accumulator.max_depth,
        depth_stats=depth_stats,
        bytes_by_extension=tuple(sorted(accumulator.extensions.items())),
        matches=tuple(sorted(accumulator.matches, key=lambda item: item.relative_path)),
    )


def analyze_recursive(root: Path, pattern: str = "*") -> TreeAnalysis:
    """Analyze a directory tree with one Python call frame per directory level."""
    resolved_root = _validate_root_and_pattern(root, pattern)
    accumulator = _Accumulator()

    def visit(directory: Path, depth: int) -> None:
        _record_directory(accumulator, depth)
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.is_symlink():
                continue
            if entry.is_dir():
                visit(entry, depth + 1)
            elif entry.is_file():
                _record_file(accumulator, resolved_root, entry, depth, pattern)

    visit(resolved_root, 0)
    return _finish(accumulator)


def analyze_iterative(root: Path, pattern: str = "*") -> TreeAnalysis:
    """Analyze a directory tree depth-first using an explicit list stack."""
    resolved_root = _validate_root_and_pattern(root, pattern)
    accumulator = _Accumulator()
    stack: list[tuple[Path, int]] = [(resolved_root, 0)]

    while stack:
        directory, depth = stack.pop()
        _record_directory(accumulator, depth)
        children: list[tuple[Path, int]] = []
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.is_symlink():
                continue
            if entry.is_dir():
                children.append((entry, depth + 1))
            elif entry.is_file():
                _record_file(accumulator, resolved_root, entry, depth, pattern)
        stack.extend(reversed(children))

    return _finish(accumulator)
