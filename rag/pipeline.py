"""Agentic RAG pipeline for RepoGraph AI.

This module provides the main end-to-end retrieval-augmented generation pipeline,
incorporating query rewriting, hybrid retrieval, cross-encoder reranking, and
answer generation with fallback mechanisms.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from rag.retrieval.hybrid_retriever import HybridRetriever, HybridSearchResult
from rag.retrieval.reranker import Reranker, RerankedResult

logger = logging.getLogger(__name__)


class AgenticRAGPipeline:
    """Agentic RAG pipeline orchestrating retrieval and generation."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        primary_model: str = "deepseek-r1-distill-llama-70b",
        fallback_model: str = "qwen/qwen3-32b",
    ) -> None:
        """Initialize the pipeline.

        Args:
            hybrid_retriever: Pre-configured hybrid retriever instance.
            reranker: Pre-configured cross-encoder reranker instance.
            primary_model: Main Groq model identifier to use.
            fallback_model: Fallback Groq model identifier if primary fails.
        """
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._primary_model = primary_model
        self._fallback_model = fallback_model

        # Lazily loaded LLMs
        self._llm: Runnable | None = None

    def _get_llm(self) -> Runnable:
        """Lazily initialize the LLM chain with fallbacks."""
        if self._llm is None:
            primary = ChatGroq(model=self._primary_model, temperature=0.0)
            fallback = ChatGroq(model=self._fallback_model, temperature=0.0)
            self._llm = primary.with_fallbacks([fallback])
            logger.info(
                "Initialized LLMs",
                extra={
                    "primary": self._primary_model,
                    "fallback": self._fallback_model,
                },
            )
        return self._llm

    def rewrite_query(self, user_query: str) -> str:
        """Rewrite the user query for optimal retrieval.

        Args:
            user_query: Original query from the user.

        Returns:
            A rewritten query optimized for search.
        """
        logger.info("Rewriting query", extra={"original_query": user_query})

        prompt = PromptTemplate.from_template(
            "Rewrite the following query to make it clearer and more specific for a search engine. "
            "Output only the rewritten query text, with no preamble or explanation.\n\n"
            "Query: {query}"
        )

        chain = prompt | self._get_llm() | StrOutputParser()

        try:
            rewritten: str = chain.invoke({"query": user_query})
            rewritten_query = rewritten.strip()
            logger.info(
                "Query successfully rewritten",
                extra={"rewritten_query": rewritten_query},
            )
            return rewritten_query
        except Exception as e:
            logger.error(
                "Failed to rewrite query, falling back to original query",
                exc_info=True,
                extra={"original_query": user_query},
            )
            return user_query

    def retrieve_context(self, query: str, top_k: int = 15) -> list[HybridSearchResult]:
        """Retrieve candidate context using hybrid retrieval.

        Args:
            query: The search query.
            top_k: Number of candidates to retrieve.

        Returns:
            A list of candidate search results.
        """
        logger.info("Retrieving candidate context", extra={"query": query, "top_k": top_k})
        try:
            # We fetch more candidates here to give the reranker headroom
            results = self._retriever.retrieve(query, top_k=top_k)
            logger.info("Retrieved candidates", extra={"count": len(results)})
            return results
        except Exception as e:
            logger.error("Hybrid retrieval failed", exc_info=True)
            return []

    def rerank_context(
        self, query: str, candidates: list[HybridSearchResult], top_k: int = 5
    ) -> list[RerankedResult]:
        """Rerank candidates using the cross-encoder.

        Args:
            query: The search query.
            candidates: The initially retrieved candidate chunks.
            top_k: Final number of documents to keep.

        Returns:
            A list of reranked results.
        """
        logger.info(
            "Reranking candidates",
            extra={"num_candidates": len(candidates), "top_k": top_k},
        )
        if not candidates:
            return []

        try:
            results = self._reranker.rerank(query, candidates, top_k=top_k)
            logger.info("Reranking completed", extra={"count": len(results)})
            return results
        except Exception as e:
            logger.error("Reranking failed, falling back to original ranking", exc_info=True)
            # Wrap original candidates as fallback
            fallback_results = [
                RerankedResult(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    rerank_score=c.score,
                    original_score=c.score,
                    metadata=c.metadata,
                )
                for c in candidates[:top_k]
            ]
            return fallback_results

    def build_prompt(self, query: str, context: list[RerankedResult]) -> str:
        """Construct the prompt combining query and context.

        Args:
            query: The user query.
            context: The reranked context documents.

        Returns:
            The complete prompt string to pass to the LLM.
        """
        logger.info("Building generation prompt", extra={"num_documents": len(context)})

        context_blocks = []
        for i, doc in enumerate(context, start=1):
            source = doc.metadata.get("source", "Unknown Source")
            context_blocks.append(f"[Document {i}] (Source: {source})\n{doc.content}\n")

        context_str = "\n".join(context_blocks)

        prompt = (
            "You are a helpful AI assistant. Use the following retrieved documents "
            "to answer the user's query comprehensively and accurately. "
            "If the provided documents do not contain sufficient information to answer "
            "the query, clearly state that you do not know. "
            "Always cite your sources using the [Document X] notation when referencing "
            "information from the context.\n\n"
            "--- Context ---\n"
            f"{context_str}\n"
            "--- End Context ---\n\n"
            f"User Query: {query}\n\n"
            "Answer:"
        )
        return prompt

    def generate_answer(self, prompt: str) -> str:
        """Generate the final answer using the LLM.

        Args:
            prompt: The fully constructed prompt.

        Returns:
            The generated answer string.
        """
        logger.info("Generating answer via LLM")
        try:
            chain = self._get_llm() | StrOutputParser()
            answer: str = chain.invoke(prompt)
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error("Answer generation failed", exc_info=True)
            return (
                "I apologize, but I encountered an internal error while trying "
                "to generate an answer."
            )

    def query(self, user_query: str) -> dict[str, Any]:
        """Execute the full end-to-end RAG pipeline.

        Args:
            user_query: The raw query from the user.

        Returns:
            A dictionary containing the 'answer' string and a list of 'citations'.
        """
        logger.info("Starting pipeline execution", extra={"user_query": user_query})

        try:
            # 1. Rewrite Query
            rewritten_query = self.rewrite_query(user_query)

            # 2. Retrieve Context (Fetch 15 candidates initially)
            candidates = self.retrieve_context(rewritten_query, top_k=15)

            # 3. Rerank Context (Keep top 5)
            reranked = self.rerank_context(rewritten_query, candidates, top_k=5)

            # 4. Build Prompt
            prompt = self.build_prompt(user_query, reranked)

            # 5. Generate Answer
            answer = self.generate_answer(prompt)

            # 6. Assemble output with citations
            citations = [
                {
                    "chunk_id": doc.chunk_id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": doc.rerank_score,
                }
                for doc in reranked
            ]

            logger.info(
                "Pipeline execution completed successfully",
                extra={"citations_count": len(citations)},
            )
            return {
                "answer": answer,
                "citations": citations,
            }

        except Exception as e:
            logger.critical("Fatal error during pipeline execution", exc_info=True)
            return {
                "answer": "An unexpected error occurred while processing your request.",
                "citations": [],
            }
