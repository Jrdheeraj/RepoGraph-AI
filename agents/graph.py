"""LangGraph core workflow builder for RepoGraph AI."""

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from agents.state import RepoGraphState
from agents.router import route_query
from rag.pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)

# Lazily initialized pipeline instance
_rag_pipeline: AgenticRAGPipeline | None = None


def _get_rag_pipeline() -> AgenticRAGPipeline:
    """Lazily initialize and return the Agentic RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        logger.info("Lazily initializing AgenticRAGPipeline dependencies")
        # Imports are isolated to prevent overhead until needed
        from rag.retrieval.dense_retriever import DenseRetriever
        from rag.retrieval.sparse_retriever import SparseRetriever
        from rag.retrieval.hybrid_retriever import HybridRetriever
        from rag.retrieval.reranker import Reranker

        dense = DenseRetriever()
        sparse = SparseRetriever()
        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        reranker = Reranker()

        _rag_pipeline = AgenticRAGPipeline(
            hybrid_retriever=hybrid,
            reranker=reranker
        )
    return _rag_pipeline


def router_node(state: RepoGraphState) -> dict[str, Any]:
    """Node that evaluates the user query to determine its intent."""
    query = state.get("query", "")
    logger.info("Executing router_node")
    intent = route_query(query)
    return {"intent": intent}


def _create_agent_node(agent_name: str):
    """Helper to generate specific agent nodes that tag the state."""
    def node(state: RepoGraphState) -> dict[str, Any]:
        logger.info("Executing specialized agent node", extra={"agent": agent_name})
        return {"agent_used": agent_name}
    return node


# Initialize specialized agent nodes
architect_node = _create_agent_node("Architect Agent")
documentation_node = _create_agent_node("Documentation Agent")
testing_node = _create_agent_node("Test Agent")
refactoring_node = _create_agent_node("Refactoring Agent")


def agentic_rag_node(state: RepoGraphState) -> dict[str, Any]:
    """Node that triggers the Agentic RAG pipeline to generate an answer."""
    logger.info("Executing agentic_rag_node")
    query = state.get("query", "")
    if not query:
        logger.warning("Empty query passed to agentic_rag_node")
        return {}

    pipeline = _get_rag_pipeline()
    
    try:
        result = pipeline.query(query)
        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
        }
    except Exception as e:
        logger.error("Agentic RAG pipeline execution failed", exc_info=True)
        return {
            "answer": "An error occurred while executing the Agentic RAG pipeline.",
            "citations": []
        }


def route_based_on_intent(state: RepoGraphState) -> str:
    """Conditional edge logic for routing post-router_node."""
    intent = state.get("intent", "default")
    logger.info("Routing state based on intent", extra={"intent": intent})
    
    # Map intents to the corresponding workflow branches
    if intent == "architecture":
        return "architect"
    elif intent == "documentation":
        return "documentation"
    elif intent == "testing":
        return "testing"
    elif intent == "refactoring":
        return "refactoring"
    else:
        return "default"


# Lazily compiled LangGraph workflow instance
_compiled_graph = None


def build_graph() -> Any:
    """Build and compile the LangGraph workflow.

    Workflow:
    START
      ↓
    router_node
      ↓ (route_based_on_intent)
    [Architect Agent | Documentation Agent | Test Agent | Refactoring Agent | (default)]
      ↓
    AgenticRAGPipeline
      ↓
    END
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    logger.info("Building LangGraph StateGraph")
    workflow = StateGraph(RepoGraphState)

    # Add core nodes
    workflow.add_node("router", router_node)
    workflow.add_node("architect", architect_node)
    workflow.add_node("documentation", documentation_node)
    workflow.add_node("testing", testing_node)
    workflow.add_node("refactoring", refactoring_node)
    workflow.add_node("agentic_rag", agentic_rag_node)

    # Establish edge from START
    workflow.add_edge(START, "router")

    # Add conditional routing from the router node
    workflow.add_conditional_edges(
        "router",
        route_based_on_intent,
        {
            "architect": "architect",
            "documentation": "documentation",
            "testing": "testing",
            "refactoring": "refactoring",
            "default": "agentic_rag",
        }
    )

    # Merge specialized agents back into the generic RAG pipeline
    workflow.add_edge("architect", "agentic_rag")
    workflow.add_edge("documentation", "agentic_rag")
    workflow.add_edge("testing", "agentic_rag")
    workflow.add_edge("refactoring", "agentic_rag")

    # Final step: edge to END
    workflow.add_edge("agentic_rag", END)

    _compiled_graph = workflow.compile()
    logger.info("LangGraph StateGraph successfully compiled")
    
    return _compiled_graph
