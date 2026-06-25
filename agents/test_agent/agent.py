"""Test Agent for generating repository tests and analyzing test coverage.

This module provides the TestAgent class, which leverages the RAG pipeline
to generate pytest code, suggest edge cases, analyze missing tests, and
create comprehensive test plans for unit and integration testing.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag.pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)


class TestAgent:
    """Agent responsible for test generation and coverage analysis."""

    def __init__(
        self,
        rag_pipeline: Optional[AgenticRAGPipeline] = None,
    ) -> None:
        """Initialize the Test Agent.

        Args:
            rag_pipeline: An optional pre-configured AgenticRAGPipeline instance.
        """
        self._rag_pipeline = rag_pipeline

    def _get_rag_pipeline(self) -> AgenticRAGPipeline:
        """Lazily initialize and return the RAG pipeline."""
        if self._rag_pipeline is None:
            logger.info("Lazily initializing AgenticRAGPipeline for TestAgent")
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

    def generate_pytest(self, target: str, test_type: str = "unit") -> str:
        """Generate pytest code for a specific target.

        Args:
            target: The specific file, module, or class to test.
            test_type: The type of test ('unit' or 'integration').

        Returns:
            A string containing the generated pytest code in markdown format.
        """
        logger.info(
            "Generating pytest code",
            extra={"target": target, "test_type": test_type}
        )
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Generate comprehensive {test_type} tests using pytest for the "
            f"following target: {target}. Include necessary mocks, fixtures, and "
            "assertions. Output only the Python code inside a markdown code block."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No tests could be generated.")
        except Exception as e:
            logger.error("Failed to generate pytest code", exc_info=True, extra={"target": target})
            return f"Failed to generate {test_type} tests for {target}."

    def generate_edge_cases(self, target: str) -> str:
        """Suggest edge cases for testing a specific target.

        Args:
            target: The specific file, module, or class to analyze for edge cases.

        Returns:
            A markdown string containing the suggested edge cases.
        """
        logger.info("Suggesting edge cases", extra={"target": target})
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Identify and suggest edge cases and boundary conditions that should "
            f"be tested for the following target: {target}. Explain why each edge "
            "case is important and how it might break the system. Use structured "
            "markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No edge cases could be identified.")
        except Exception as e:
            logger.error("Failed to suggest edge cases", exc_info=True, extra={"target": target})
            return f"Failed to suggest edge cases for {target}."

    def coverage_analysis(self, target_module: str) -> str:
        """Analyze missing tests and coverage gaps for a module.

        Args:
            target_module: The module or subsystem to analyze for coverage.

        Returns:
            A markdown string detailing missing tests and coverage analysis.
        """
        logger.info("Analyzing test coverage", extra={"target_module": target_module})
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Analyze the test coverage for the module: {target_module}. "
            "Identify what functions or code paths are likely missing tests, "
            "highlight untested complex logic, and suggest what specific tests "
            "should be added to improve coverage. Use structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No coverage analysis could be generated.")
        except Exception as e:
            logger.error("Failed to analyze coverage", exc_info=True, extra={"target_module": target_module})
            return f"Failed to analyze coverage for {target_module}."

    def generate_test_plan(self, target_component: str) -> str:
        """Generate a comprehensive test plan for a component.

        Args:
            target_component: The component or subsystem to plan tests for.

        Returns:
            A markdown string containing the detailed test plan.
        """
        logger.info("Generating test plan", extra={"target_component": target_component})
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Generate a comprehensive test plan for: {target_component}. "
            "Include strategies for both unit and integration tests. Outline the "
            "testing setup, specific features to test, environment requirements, "
            "and execution strategy. Use structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "No test plan could be generated.")
        except Exception as e:
            logger.error("Failed to generate test plan", exc_info=True, extra={"target_component": target_component})
            return f"Failed to generate test plan for {target_component}."
