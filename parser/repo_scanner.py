from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "external",
        "third_party",
        "bazel-bin",
        "bazel-out",
        "bazel-testlogs",
    }
)

IGNORED_EXTENSIONS: frozenset[str] = frozenset(
    {".pb", ".onnx", ".so", ".dll", ".exe", ".bin"}
)

# Skip files larger than 1 MB
MAX_FILE_SIZE_BYTES: int = 1 * 1024 * 1024


def _is_ignored_directory(path: Path) -> bool:
    name = path.name.lower()
    return name in IGNORED_DIRECTORIES or "pycache" in name


def _is_ignored_file(path: Path) -> bool:
    """Return True if a file should be skipped based on extension or size."""
    if path.suffix.lower() in IGNORED_EXTENSIONS:
        return True
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            logger.debug("Skipping large file", extra={"file_path": str(path)})
            return True
    except OSError:
        pass
    return False


def scan_repository(repository_path: str | Path) -> list[Path]:
    """Recursively return Python source files under a cloned repository."""
    root = Path(repository_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    python_files: list[Path] = []
    stack: list[Path] = [root]

    logger.info("Scanning repository for Python files", extra={"repository_path": str(root)})

    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name.lower())
        except OSError as exc:
            logger.warning(
                "Skipping unreadable directory",
                extra={"directory": str(current), "error": str(exc)},
            )
            continue

        for entry in entries:
            if entry.is_dir():
                if _is_ignored_directory(entry):
                    logger.debug("Ignoring directory during repository scan", extra={"directory": str(entry)})
                    continue
                stack.append(entry)
                continue

            if entry.is_file() and entry.suffix == ".py":
                if not _is_ignored_file(entry):
                    python_files.append(entry)

    python_files.sort()
    logger.info(
        "Completed repository scan",
        extra={"repository_path": str(root), "python_file_count": len(python_files)},
    )
    return python_files


__all__ = ["IGNORED_DIRECTORIES", "scan_repository"]
