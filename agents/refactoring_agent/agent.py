"""Refactoring Agent for analyzing code quality and suggesting improvements.

This module provides the RefactoringAgent class, which leverages the RAG pipeline
and dependency graph parsing to find code smells, analyze complexity, suggest
refactoring and performance improvements, and detect duplicate code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from parser.dependency_graph import build_graph
from rag.pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)


class RefactoringAgent:
    """Agent responsible for code quality analysis and refactoring suggestions."""

    def __init__(
        self,
        repository_path: str | Path,
        rag_pipeline: Optional[AgenticRAGPipeline] = None,
    ) -> None:
        """Initialize the Refactoring Agent.

        Args:
            repository_path: The root path of the repository to analyze.
            rag_pipeline: An optional pre-configured AgenticRAGPipeline instance.
        """
        self._repository_path = Path(repository_path)
        self._rag_pipeline = rag_pipeline
        self._graph: nx.DiGraph | None = None

    def _get_rag_pipeline(self) -> AgenticRAGPipeline:
        """Lazily initialize and return the RAG pipeline."""
        if self._rag_pipeline is None:
            logger.info("Lazily initializing AgenticRAGPipeline for RefactoringAgent")
            from rag.retrieval.dense_retriever import DenseRetriever
            from rag.retrieval.sparse_retriever import SparseRetriever
            from rag.retrieval.hybrid_retriever import HybridRetriever
            from rag.retrieval.reranker import Reranker

            dense = DenseRetriever()
            sparse = SparseRetriever()
            hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
            reranker = Reranker()

            self._rag_pipeline = AgenticRAGPipeline(
                hybrid_retriever=hybrid, reranker=reranker
            )
        return self._rag_pipeline

    def _get_dependency_graph(self) -> nx.DiGraph:
        """Lazily initialize and return the parsed dependency graph."""
        if self._graph is None:
            logger.info(
                "Building dependency graph for RefactoringAgent",
                extra={"repository_path": str(self._repository_path)}
            )
            self._graph = build_graph(self._repository_path)
        return self._graph

    def _identify_long_functions(self, threshold: int = 50) -> List[Dict[str, Any]]:
        """Identify functions that exceed a certain line length threshold."""
        graph = self._get_dependency_graph()
        long_functions = []
        for node_id, data in graph.nodes(data=True):
            if data.get("type") == "function":
                line_start = data.get("line_start", 0)
                line_end = data.get("line_end", 0)
                length = line_end - line_start
                if length > threshold:
                    long_functions.append({
                        "name": data.get("qualified_name", data.get("label", node_id)),
                        "length": length,
                        "file": data.get("path", "Unknown")
                    })
        return sorted(long_functions, key=lambda x: x["length"], reverse=True)

    def _identify_highly_coupled_modules(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify modules with high in-degree or out-degree of dependencies."""
        graph = self._get_dependency_graph()
        
        # Calculate out-degree for files (dependencies they import)
        out_degree: Dict[str, int] = {}
        for source, target, data in graph.edges(data=True):
            if data.get("type") == "depends_on":
                out_degree[source] = out_degree.get(source, 0) + 1
                
        sorted_modules = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
        
        coupled = []
        for node_id, degree in sorted_modules[:top_n]:
            data = graph.nodes.get(node_id, {})
            coupled.append({
                "module": data.get("label", node_id),
                "dependencies_count": degree,
            })
        return coupled

    def find_code_smells(self) -> str:
        """Analyze the repository for code smells and return a markdown report."""
        logger.info("Finding code smells")
        pipeline = self._get_rag_pipeline()
        
        long_funcs = self._identify_long_functions(threshold=50)
        coupled_mods = self._identify_highly_coupled_modules(top_n=5)
        
        context = (
            f"Long functions identified: {long_funcs[:10]}\n"
            f"Highly coupled modules: {coupled_mods}\n"
        )
        
        query = (
            "Based on the following structural analysis of the codebase, identify "
            "potential code smells. Discuss the implications of these long functions "
            f"and highly coupled modules:\n{context}\n"
            "Provide a comprehensive markdown report."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No code smells could be identified.")
        except Exception as e:
            logger.error("Failed to find code smells", exc_info=True)
            return "Failed to analyze code smells."

    def complexity_analysis(self, target: str) -> str:
        """Analyze the cyclomatic or cognitive complexity of a specific target."""
        logger.info("Analyzing complexity", extra={"target": target})
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Analyze the structural and logical complexity of the following target: {target}. "
            "Identify areas with deep nesting, complex conditionals, or convoluted logic. "
            "Provide suggestions to simplify the code. Output in structured markdown."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No complexity analysis could be generated.")
        except Exception as e:
            logger.error("Failed to analyze complexity", exc_info=True, extra={"target": target})
            return f"Failed to analyze complexity for {target}."

    def refactor_suggestions(self, target: str) -> str:
        """Provide design and refactoring improvements for a target."""
        logger.info("Generating refactoring suggestions", extra={"target": target})
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Provide concrete refactoring and design improvements for: {target}. "
            "Focus on SOLID principles, modularity, and readability. "
            "Provide code snippets demonstrating the 'before' and 'after' states. "
            "Output in structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No refactoring suggestions could be generated.")
        except Exception as e:
            logger.error("Failed to generate refactoring suggestions", exc_info=True, extra={"target": target})
            return f"Failed to suggest refactoring for {target}."

    def performance_improvements(self, target: str) -> str:
        """Suggest performance optimizations for a target."""
        logger.info("Suggesting performance improvements", extra={"target": target})
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Analyze the following target for potential performance bottlenecks: {target}. "
            "Suggest performance optimizations such as better data structures, caching, "
            "algorithmic improvements, or reduced I/O. Output in structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No performance improvements could be suggested.")
        except Exception as e:
            logger.error("Failed to suggest performance improvements", exc_info=True, extra={"target": target})
            return f"Failed to suggest performance improvements for {target}."

    def duplicate_code_detection(self) -> str:
        """Detect duplicate logic across the repository."""
        logger.info("Detecting duplicate code")
        pipeline = self._get_rag_pipeline()
        
        query = (
            "Analyze the codebase to detect areas with duplicate or highly similar logic. "
            "Highlight the specific files and functions involved, and suggest how to "
            "extract the common logic into reusable components. Output in structured markdown."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No duplicate code could be detected.")
        except Exception as e:
            logger.error("Failed to detect duplicate code", exc_info=True)
            return "Failed to detect duplicate code."
