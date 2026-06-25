"""Architect Agent for analyzing and explaining repository architecture.

This module provides the ArchitectAgent class, which leverages the RAG pipeline
and dependency graph parsing to answer complex architecture questions, identify
core modules, trace execution flow, and generate Mermaid diagrams.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from parser.dependency_graph import build_graph
from rag.pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)


class ArchitectAgent:
    """Agent responsible for understanding repository architecture."""

    def __init__(
        self,
        repository_path: str | Path,
        rag_pipeline: Optional[AgenticRAGPipeline] = None
    ) -> None:
        """Initialize the Architect Agent.

        Args:
            repository_path: The root path of the repository to analyze.
            rag_pipeline: An optional pre-configured AgenticRAGPipeline instance.
                          If None, it will be lazily initialized.
        """
        self._repository_path = Path(repository_path)
        self._rag_pipeline = rag_pipeline
        self._graph: nx.DiGraph | None = None

    def _get_rag_pipeline(self) -> AgenticRAGPipeline:
        """Lazily initialize and return the RAG pipeline."""
        if self._rag_pipeline is None:
            logger.info("Lazily initializing AgenticRAGPipeline for ArchitectAgent")
            from rag.retrieval.dense_retriever import DenseRetriever
            from rag.retrieval.sparse_retriever import SparseRetriever
            from rag.retrieval.hybrid_retriever import HybridRetriever
            from rag.retrieval.reranker import Reranker

            dense = DenseRetriever()
            sparse = SparseRetriever()
            hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
            reranker = Reranker()

            self._rag_pipeline = AgenticRAGPipeline(
                hybrid_retriever=hybrid,
                reranker=reranker
            )
        return self._rag_pipeline

    def _get_dependency_graph(self) -> nx.DiGraph:
        """Lazily initialize and return the parsed dependency graph."""
        if self._graph is None:
            logger.info(
                "Building dependency graph for ArchitectAgent",
                extra={"repository_path": str(self._repository_path)}
            )
            self._graph = build_graph(self._repository_path)
        return self._graph

    def explain_architecture(self, query: str) -> Dict[str, Any]:
        """Explain the architecture using the Agentic RAG pipeline.

        Args:
            query: The architecture-related question from the user.

        Returns:
            A structured dictionary containing the answer and citations.
        """
        logger.info("Explaining architecture", extra={"query": query})
        
        pipeline = self._get_rag_pipeline()
        
        # Frame the query specifically for architecture explanation
        augmented_query = (
            f"As an Architect Agent, explain the repository architecture related to "
            f"the following question: {query}"
        )
        
        try:
            result = pipeline.query(augmented_query)
            return {
                "intent": "architecture_explanation",
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
            }
        except Exception as e:
            logger.error("Failed to explain architecture", exc_info=True)
            return {
                "intent": "architecture_explanation",
                "answer": "An error occurred while analyzing the architecture.",
                "citations": [],
            }

    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram mapping the module dependencies.

        Returns:
            A string containing a formatted Mermaid graph.
        """
        logger.info("Generating Mermaid diagram from dependency graph")
        graph = self._get_dependency_graph()
        
        lines = ["graph TD;"]
        
        for source, target, data in graph.edges(data=True):
            if data.get("type") == "depends_on":
                # Ensure identifiers are Mermaid-safe
                s_id = str(source).replace(":", "_").replace("-", "_").replace(".", "_")
                t_id = str(target).replace(":", "_").replace("-", "_").replace(".", "_")
                
                s_label = graph.nodes.get(source, {}).get("label", s_id)
                t_label = graph.nodes.get(target, {}).get("label", t_id)
                
                lines.append(f'    {s_id}["{s_label}"] -->|depends| {t_id}["{t_label}"]')
                
        diagram = "\n".join(lines)
        logger.info("Successfully generated Mermaid diagram", extra={"lines": len(lines)})
        return diagram

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze and explain module relationships and dependencies.

        Returns:
            A dictionary containing dependency counts and relationship lists.
        """
        logger.info("Analyzing module dependencies")
        graph = self._get_dependency_graph()
        
        relationships = []
        for source, target, data in graph.edges(data=True):
            if data.get("type") == "depends_on":
                s_label = graph.nodes.get(source, {}).get("label", str(source))
                t_label = graph.nodes.get(target, {}).get("label", str(target))
                relationships.append(f"{s_label} depends on {t_label}")

        return {
            "total_modules": graph.number_of_nodes(),
            "total_dependencies": len(relationships),
            "relationships": relationships[:100],  # Truncate for readability
            "summary": "Repository module relationships successfully analyzed."
        }

    def identify_core_modules(self, top_n: int = 10) -> Dict[str, Any]:
        """Identify critical files and services based on in-degree centrality.

        Args:
            top_n: The number of top core modules to return.

        Returns:
            A structured dict of core modules and their centrality scores.
        """
        logger.info("Identifying core modules", extra={"top_n": top_n})
        graph = self._get_dependency_graph()
        
        if graph.number_of_nodes() == 0:
            return {"core_modules": [], "message": "Graph is empty."}
            
        centrality = nx.in_degree_centrality(graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        core_modules = []
        for node_id, score in sorted_nodes:
            if len(core_modules) >= top_n:
                break
                
            node_data = graph.nodes.get(node_id, {})
            # Focus primarily on files or main architectural modules
            if node_data.get("type") in ("file", "module", "class"):
                core_modules.append({
                    "id": node_id,
                    "label": node_data.get("label", str(node_id)),
                    "centrality_score": round(score, 4),
                    "type": node_data.get("type")
                })
                
        return {
            "core_modules": core_modules,
            "metric": "in_degree_centrality",
            "description": "Modules ranked by how frequently they are depended upon."
        }

    def trace_data_flow(self, entry_point: str) -> Dict[str, Any]:
        """Trace the execution and data flow starting from an entry point.

        Uses the RAG pipeline to explain the logical execution flow.

        Args:
            entry_point: The starting point for the data flow trace.

        Returns:
            A structured dictionary explaining the execution flow.
        """
        logger.info("Tracing data flow", extra={"entry_point": entry_point})
        pipeline = self._get_rag_pipeline()
        
        query = (
            f"Explain the execution flow, data flow, and the sequence of function "
            f"calls starting from the entry point: '{entry_point}'. Identify the "
            f"critical files and services involved."
        )
        
        try:
            result = pipeline.query(query)
            return {
                "entry_point": entry_point,
                "execution_flow_explanation": result.get("answer", ""),
                "citations": result.get("citations", [])
            }
        except Exception as e:
            logger.error("Failed to trace data flow", exc_info=True)
            return {
                "entry_point": entry_point,
                "error": "Failed to trace data flow via RAG pipeline.",
                "execution_flow_explanation": "",
                "citations": []
            }
