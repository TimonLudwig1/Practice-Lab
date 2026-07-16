"""Generate a reproducible nested directory tree with sparse synthetic files."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# A fixed seed makes paths, sizes, tests, and benchmark inputs reproducible.
DEFAULT_SEED: Final = 20_260_716
DIRECTORY_LABELS: Final = ("raw", "processed", "archive", "models", "reports")
FILE_STEMS: Final = ("sensor", "report", "events", "summary", "model", "image")
EXTENSIONS: Final = ("csv", "json", "log", "txt", "bin")


@dataclass(frozen=True)
class GeneratedFile:
    """One generated file relative to the synthetic root."""

    relative_path: str
    size: int


@dataclass(frozen=True)
class GenerationSummary:
    """Canonical summary of one generated directory tree."""

    directory_count: int
    file_count: int
    total_bytes: int
    max_depth: int
    files: tuple[GeneratedFile, ...]


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate a positive non-boolean integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _prepare_root(root: Path, overwrite: bool) -> None:
    """Create an empty root while guarding against dangerous deletion targets."""
    resolved = root.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to use a filesystem root as generated tree")
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(root)
        if any(root.iterdir()):
            if not overwrite:
                raise FileExistsError("generated tree root is not empty")
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def generate_tree(
    root: Path,
    *,
    seed: int = DEFAULT_SEED,
    max_depth: int = 5,
    max_subdirectories: int = 3,
    max_files_per_directory: int = 5,
    overwrite: bool = False,
) -> GenerationSummary:
    """Generate a bounded recursive directory tree and return its manifest."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool):
        raise TypeError("max_depth must be an integer")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    _validate_positive_integer(max_subdirectories, "max_subdirectories")
    _validate_positive_integer(max_files_per_directory, "max_files_per_directory")
    _prepare_root(root, overwrite)

    random_source = random.Random(seed)
    generated_files: list[GeneratedFile] = []
    directory_count = 1
    next_directory_id = 1
    next_file_id = 1

    def populate(directory: Path, depth: int) -> None:
        nonlocal directory_count, next_directory_id, next_file_id

        file_count = random_source.randint(1, max_files_per_directory)
        for _ in range(file_count):
            stem = random_source.choice(FILE_STEMS)
            extension = random_source.choice(EXTENSIONS)
            file_name = f"{stem}_{next_file_id:05d}.{extension}"
            next_file_id += 1
            size = random_source.randint(64, 8_192)
            file_path = directory / file_name
            with file_path.open("wb") as handle:
                handle.truncate(size)
            generated_files.append(
                GeneratedFile(file_path.relative_to(root).as_posix(), size)
            )

        if depth == max_depth:
            return

        child_count = random_source.randint(1, max_subdirectories)
        for _ in range(child_count):
            label = random_source.choice(DIRECTORY_LABELS)
            child = directory / f"{label}_{next_directory_id:04d}"
            next_directory_id += 1
            child.mkdir()
            directory_count += 1
            populate(child, depth + 1)

    populate(root, 0)
    files = tuple(sorted(generated_files, key=lambda item: item.relative_path))
    return GenerationSummary(
        directory_count=directory_count,
        file_count=len(files),
        total_bytes=sum(item.size for item in files),
        max_depth=max_depth,
        files=files,
    )


def main() -> None:
    """Generate the default learning tree from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/synthetic_tree"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-subdirectories", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Fail instead of replacing an existing generated tree.",
    )
    arguments = parser.parse_args()

    summary = generate_tree(
        arguments.root,
        seed=arguments.seed,
        max_depth=arguments.max_depth,
        max_subdirectories=arguments.max_subdirectories,
        max_files_per_directory=arguments.max_files,
        overwrite=not arguments.keep_existing,
    )
    print(f"Root: {arguments.root.resolve()}")
    print(f"Directories: {summary.directory_count}")
    print(f"Files: {summary.file_count}")
    print(f"Bytes: {summary.total_bytes}")
    print(f"Maximum depth: {summary.max_depth}")


if __name__ == "__main__":
    main()
