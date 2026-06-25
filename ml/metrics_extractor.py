"""Metrics Extractor for RepoGraph AI.

This module provides the MetricsExtractor class to calculate and extract
repository-level metrics from the dependency graph for ML models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import networkx as nx

from parser.dependency_graph import build_graph

logger = logging.getLogger(__name__)


class MetricsExtractor:
    """Extractor for repository metrics using the dependency graph."""

    def __init__(self, repository_path: str | Path) -> None:
        """Initialize the MetricsExtractor.

        Args:
            repository_path: The root path of the repository to analyze.
        """
        self._repository_path = Path(repository_path)
        self._graph: nx.DiGraph | None = None

    def _get_graph(self) -> nx.DiGraph:
        """Lazily initialize and return the parsed dependency graph."""
        if self._graph is None:
            logger.info("Building dependency graph for MetricsExtractor")
            try:
                self._graph = build_graph(self._repository_path)
            except Exception as e:
                logger.error("Failed to build dependency graph", exc_info=True)
                self._graph = nx.DiGraph()
        return self._graph

    def calculate_loc(self) -> float:
        """Calculate approximate lines of code based on node line ends."""
        try:
            graph = self._get_graph()
            file_max_lines: Dict[str, int] = {}
            for _, data in graph.nodes(data=True):
                path = data.get("path")
                line_end = data.get("line_end")
                if path and isinstance(line_end, int):
                    if path not in file_max_lines or line_end > file_max_lines[path]:
                        file_max_lines[path] = line_end
            
            return float(sum(file_max_lines.values()))
        except Exception:
            logger.error("Error calculating LOC", exc_info=True)
            return 0.0

    def count_functions(self) -> float:
        """Count the total number of function nodes."""
        try:
            graph = self._get_graph()
            count = sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "function")
            return float(count)
        except Exception:
            logger.error("Error counting functions", exc_info=True)
            return 0.0

    def count_classes(self) -> float:
        """Count the total number of class nodes."""
        try:
            graph = self._get_graph()
            count = sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "class")
            return float(count)
        except Exception:
            logger.error("Error counting classes", exc_info=True)
            return 0.0

    def calculate_average_function_length(self) -> float:
        """Calculate the average length of functions in lines."""
        try:
            graph = self._get_graph()
            total_lines = 0
            count = 0
            for _, data in graph.nodes(data=True):
                if data.get("type") == "function":
                    start = data.get("line_start", 0)
                    end = data.get("line_end", 0)
                    total_lines += max(0, end - start)
                    count += 1
            return float(total_lines / count) if count > 0 else 0.0
        except Exception:
            logger.error("Error calculating average function length", exc_info=True)
            return 0.0

    def calculate_dependency_count(self) -> float:
        """Count the total number of depends_on edges."""
        try:
            graph = self._get_graph()
            count = sum(1 for _, _, data in graph.edges(data=True) if data.get("type") == "depends_on")
            return float(count)
        except Exception:
            logger.error("Error calculating dependency count", exc_info=True)
            return 0.0

    def calculate_fan_in(self) -> float:
        """Calculate the maximum fan-in (incoming dependencies) across modules."""
        try:
            graph = self._get_graph()
            in_degrees: Dict[Any, int] = {}
            for u, v, data in graph.edges(data=True):
                if data.get("type") == "depends_on":
                    in_degrees[v] = in_degrees.get(v, 0) + 1
            return float(max(in_degrees.values())) if in_degrees else 0.0
        except Exception:
            logger.error("Error calculating fan-in", exc_info=True)
            return 0.0

    def calculate_fan_out(self) -> float:
        """Calculate the maximum fan-out (outgoing dependencies) across modules."""
        try:
            graph = self._get_graph()
            out_degrees: Dict[Any, int] = {}
            for u, v, data in graph.edges(data=True):
                if data.get("type") == "depends_on":
                    out_degrees[u] = out_degrees.get(u, 0) + 1
            return float(max(out_degrees.values())) if out_degrees else 0.0
        except Exception:
            logger.error("Error calculating fan-out", exc_info=True)
            return 0.0

    def calculate_module_count(self) -> float:
        """Count the total number of file/module nodes."""
        try:
            graph = self._get_graph()
            count = sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "file")
            return float(count)
        except Exception:
            logger.error("Error counting modules", exc_info=True)
            return 0.0

    def extract_metrics(self) -> dict[str, float]:
        """Extract all repository metrics for ML models.

        Returns:
            A dictionary containing all calculated metrics.
        """
        logger.info("Extracting all ML metrics")
        metrics = {
            "loc": self.calculate_loc(),
            "function_count": self.count_functions(),
            "class_count": self.count_classes(),
            "avg_function_length": self.calculate_average_function_length(),
            "dependency_count": self.calculate_dependency_count(),
            "fan_in": self.calculate_fan_in(),
            "fan_out": self.calculate_fan_out(),
            "module_count": self.calculate_module_count(),
        }
        logger.info("Successfully extracted metrics", extra={"metrics": metrics})
        return metrics
