"""Documentation Agent for generating repository documentation.

This module provides the DocumentationAgent class, which leverages the RAG pipeline
and ArchitectAgent to generate READMEs, API docs, module docs, wikis, and architecture docs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agents.architect_agent.agent import ArchitectAgent
from rag.pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)


class DocumentationAgent:
    """Agent responsible for generating repository documentation."""

    def __init__(
        self,
        repository_path: str | Path,
        rag_pipeline: Optional[AgenticRAGPipeline] = None,
        architect_agent: Optional[ArchitectAgent] = None,
    ) -> None:
        """Initialize the Documentation Agent.

        Args:
            repository_path: The root path of the repository to document.
            rag_pipeline: An optional pre-configured AgenticRAGPipeline instance.
            architect_agent: An optional pre-configured ArchitectAgent instance.
        """
        self._repository_path = Path(repository_path)
        self._rag_pipeline = rag_pipeline
        self._architect_agent = architect_agent

    def _get_rag_pipeline(self) -> AgenticRAGPipeline:
        """Lazily initialize and return the RAG pipeline."""
        if self._rag_pipeline is None:
            logger.info("Lazily initializing AgenticRAGPipeline for DocumentationAgent")
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

    def _get_architect_agent(self) -> ArchitectAgent:
        """Lazily initialize and return the Architect Agent."""
        if self._architect_agent is None:
            logger.info("Lazily initializing ArchitectAgent for DocumentationAgent")
            self._architect_agent = ArchitectAgent(
                repository_path=self._repository_path,
                rag_pipeline=self._get_rag_pipeline()
            )
        return self._architect_agent

    def generate_readme(self) -> str:
        """Generate a comprehensive README.md for the repository.

        Returns:
            A string containing the structured markdown README.
        """
        logger.info("Generating README.md")
        pipeline = self._get_rag_pipeline()
        architect = self._get_architect_agent()

        core_modules_info = architect.identify_core_modules()
        logger.info("Retrieved core modules for README", extra={"core_modules": len(core_modules_info.get("core_modules", []))})
        
        query = (
            "Generate a comprehensive, developer-friendly README.md for this repository. "
            "Include an overview, installation instructions, usage examples, and describe "
            f"the core modules identified as: {core_modules_info}. "
            "Use structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "")
        except Exception as e:
            logger.error("Failed to generate README", exc_info=True)
            return "Failed to generate README documentation."

    def generate_api_docs(self) -> str:
        """Generate API documentation for the repository.

        Returns:
            A string containing the structured markdown API documentation.
        """
        logger.info("Generating API documentation")
        pipeline = self._get_rag_pipeline()
        
        query = (
            "Generate detailed API documentation for the main interfaces and classes "
            "in this repository. Include parameters, return types, and usage examples. "
            "Use structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "")
        except Exception as e:
            logger.error("Failed to generate API docs", exc_info=True)
            return "Failed to generate API documentation."

    def generate_module_docs(self) -> str:
        """Generate detailed module-level documentation.

        Returns:
            A string containing the structured markdown module documentation.
        """
        logger.info("Generating module documentation")
        pipeline = self._get_rag_pipeline()
        architect = self._get_architect_agent()
        
        deps = architect.analyze_dependencies()
        logger.info("Retrieved dependencies for module docs", extra={"total_dependencies": deps.get("total_dependencies", 0)})
        
        query = (
            "Generate comprehensive module documentation. Explain the purpose of each "
            "major module and how they interact. "
            f"Here is some dependency analysis context: {deps}. "
            "Use structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "")
        except Exception as e:
            logger.error("Failed to generate module docs", exc_info=True)
            return "Failed to generate module documentation."

    def generate_wiki(self) -> str:
        """Generate developer wiki documentation.

        Returns:
            A string containing the structured markdown wiki.
        """
        logger.info("Generating developer wiki")
        pipeline = self._get_rag_pipeline()
        
        query = (
            "Generate a developer wiki for this project. Include setup guides, "
            "contribution guidelines, testing practices, and general developer workflows. "
            "Use structured markdown format."
        )
        
        try:
            result = pipeline.query(query)
            return result.get("answer", "")
        except Exception as e:
            logger.error("Failed to generate wiki", exc_info=True)
            return "Failed to generate wiki documentation."

    def generate_architecture_docs(self) -> str:
        """Generate architecture documentation with Mermaid diagrams.

        Returns:
            A string containing the structured markdown architecture documentation.
        """
        logger.info("Generating architecture documentation")
        pipeline = self._get_rag_pipeline()
        architect = self._get_architect_agent()
        
        mermaid_diagram = architect.generate_mermaid_diagram()
        architecture_explanation = architect.explain_architecture(
            "What is the high-level architecture of this system?"
        )
        logger.info("Retrieved architecture explanation and Mermaid diagram")
        
        query = (
            "Generate comprehensive architecture documentation. Incorporate the following "
            f"architectural explanation context: {architecture_explanation}. "
            "Use structured markdown format and include sections on design decisions, "
            "data flow, and component interactions. I will append the Mermaid diagram "
            "after your generation."
        )
        
        try:
            result = pipeline.query(query)
            answer = result.get("answer", "")
            
            docs = (
                f"{answer}\n\n"
                "## Architecture Diagram\n\n"
                "```mermaid\n"
                f"{mermaid_diagram}\n"
                "```\n"
            )
            return docs
        except Exception as e:
            logger.error("Failed to generate architecture docs", exc_info=True)
            return "Failed to generate architecture documentation."
