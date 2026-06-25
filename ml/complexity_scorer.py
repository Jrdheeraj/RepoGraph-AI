"""Complexity Scorer for RepoGraph AI.

This module provides the ComplexityScorer class, which calculates code complexity
and maintainability metrics using the radon library.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
except ImportError:
    cc_visit = None
    h_visit = None
    mi_visit = None

logger = logging.getLogger(__name__)

MAX_FILES_PER_REPOSITORY = 1000


class ComplexityScorer:
    """Scorer for code complexity, maintainability, and Halstead metrics."""

    def __init__(self) -> None:
        """Initialize the ComplexityScorer."""
        if not all([cc_visit, h_visit, mi_visit]):
            logger.warning("radon library is not installed. Metrics will return 0.0.")

    def calculate_cyclomatic_complexity(self, source_code: str) -> float:
        """Calculate the total cyclomatic complexity of the given source code."""
        if not cc_visit:
            return 0.0
        try:
            blocks = cc_visit(source_code)
            total_cc = sum(getattr(block, "complexity", 0) for block in blocks)
            return float(total_cc)
        except SyntaxError:
            logger.debug("SyntaxError while calculating cyclomatic complexity – skipping.")
            return 0.0
        except Exception:
            logger.error("Failed to calculate cyclomatic complexity", exc_info=True)
            return 0.0

    def calculate_maintainability_index(self, source_code: str) -> float:
        """Calculate the maintainability index of the given source code."""
        if not mi_visit:
            return 0.0
        try:
            mi = mi_visit(source_code, multi=True)
            return float(mi)
        except SyntaxError:
            logger.debug("SyntaxError while calculating maintainability index – skipping.")
            return 0.0
        except Exception:
            logger.error("Failed to calculate maintainability index", exc_info=True)
            return 0.0

    def calculate_halstead_metrics(self, source_code: str) -> dict[str, float]:
        """Calculate Halstead volume, difficulty, and effort for the source code."""
        if not h_visit:
            return {"volume": 0.0, "difficulty": 0.0, "effort": 0.0}
        try:
            h = h_visit(source_code)
            if hasattr(h, "total"):
                metrics = getattr(h, "total")
            else:
                metrics = h

            return {
                "volume": float(getattr(metrics, "volume", 0.0)),
                "difficulty": float(getattr(metrics, "difficulty", 0.0)),
                "effort": float(getattr(metrics, "effort", 0.0)),
            }
        except SyntaxError:
            logger.debug("SyntaxError while calculating Halstead metrics – skipping.")
            return {"volume": 0.0, "difficulty": 0.0, "effort": 0.0}
        except Exception:
            logger.error("Failed to calculate Halstead metrics", exc_info=True)
            return {"volume": 0.0, "difficulty": 0.0, "effort": 0.0}

    def score_file(self, file_path: str | Path) -> dict[str, float]:
        """Score a single Python file for all complexity metrics."""
        path = Path(file_path)
        _zero: dict[str, float] = {
            "cyclomatic_complexity": 0.0,
            "maintainability_index": 0.0,
            "halstead_volume": 0.0,
            "halstead_difficulty": 0.0,
            "halstead_effort": 0.0,
        }

        try:
            if path.stat().st_size > 1 * 1024 * 1024:
                logger.debug("Skipping large file", extra={"file_path": str(path)})
                return _zero
        except OSError:
            return _zero

        try:
            source_code = path.read_text(encoding="utf-8", errors="replace")
        except SyntaxError:
            logger.debug("SyntaxError while reading file – skipping.", extra={"file_path": str(path)})
            return _zero
        except Exception:
            logger.debug("Failed to read file", extra={"file_path": str(path)})
            return _zero

        halstead = self.calculate_halstead_metrics(source_code)

        return {
            "cyclomatic_complexity": self.calculate_cyclomatic_complexity(source_code),
            "maintainability_index": self.calculate_maintainability_index(source_code),
            "halstead_volume": halstead.get("volume", 0.0),
            "halstead_difficulty": halstead.get("difficulty", 0.0),
            "halstead_effort": halstead.get("effort", 0.0),
        }

    def score_repository(self, repo_path: str | Path) -> dict[str, float]:
        """Score an entire repository by averaging metrics across all Python files.

        At most MAX_FILES_PER_REPOSITORY eligible files are analysed to keep
        runtime bounded on large repositories like PyTorch and TensorFlow.
        """
        path = Path(repo_path)
        logger.info("Starting repository complexity analysis...", extra={"repo_path": str(path)})

        # Directories to skip within a repository
        _SKIP_DIRS: frozenset[str] = frozenset(
            {
                "tests", "test", "benchmarks", "examples",
                "third_party", "external", "build", "dist",
            }
        )
        # Filename patterns that indicate generated / protobuf files
        _GENERATED_MARKERS: tuple[str, ...] = ("_pb2.py", "_pb2_grpc.py", ".pb2.py")

        totals: Dict[str, float] = {
            "cyclomatic_complexity": 0.0,
            "maintainability_index": 0.0,
            "halstead_volume": 0.0,
            "halstead_difficulty": 0.0,
            "halstead_effort": 0.0,
        }

        # ------------------------------------------------------------------ #
        # Phase 1 – deterministic file collection (capped at MAX_FILES)       #
        # ------------------------------------------------------------------ #
        eligible: list[Path] = []
        for file_path in sorted(path.rglob("*.py")):          # sorted → deterministic
            if len(eligible) >= MAX_FILES_PER_REPOSITORY:
                break
            try:
                # Skip hidden directories
                if any(part.startswith(".") for part in file_path.parts):
                    continue
                # Skip excluded subdirectories
                if any(part in _SKIP_DIRS for part in file_path.parts):
                    continue
                # Skip generated/protobuf files
                if any(file_path.name.endswith(marker) for marker in _GENERATED_MARKERS):
                    continue
                # Skip files larger than 1 MB
                if file_path.stat().st_size > 1 * 1024 * 1024:
                    logger.debug("Skipping large file", extra={"file_path": str(file_path)})
                    continue
                eligible.append(file_path)
            except Exception:
                logger.debug("Skipping file during collection", extra={"file_path": str(file_path)})
                continue

        sampled = len(eligible) == MAX_FILES_PER_REPOSITORY
        logger.info(
            "Analyzed %d files%s.",
            len(eligible),
            " (sampled)" if sampled else "",
            extra={"repo_path": str(path)},
        )

        # ------------------------------------------------------------------ #
        # Phase 2 – score collected files                                      #
        # ------------------------------------------------------------------ #
        count = 0
        for file_path in eligible:
            try:
                scores = self.score_file(file_path)
                for key in totals:
                    totals[key] += scores[key]
                count += 1
            except Exception:
                logger.debug("Skipping file due to error", extra={"file_path": str(file_path)})
                continue

        logger.info(
            "Completed repository complexity analysis.",
            extra={"repo_path": str(path), "files_scored": count},
        )

        if count == 0:
            return totals

        return {key: value / count for key, value in totals.items()}
