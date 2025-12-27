"""Simple RAG pipeline: retrieve once, generate once."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse, observe

from ragbench.index.chromadb_index import ChromaDBIndex
from ragbench.index.bm25_index import BM25Index
from ragbench.clients import EmbeddingClient
from ragbench.retrievers.retrieval import (
    retrieve_dense,
    retrieve_keyword,
    retrieve_hybrid,
    apply_reranker,
    RetrievalResult,
)

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

# Initialize Langfuse client
langfuse = Langfuse()


class SimpleRAGPipeline:
    """Simple RAG: retrieve context -> generate response."""

    def __init__(
        self,
        chroma_index: ChromaDBIndex,
        embedding_client: EmbeddingClient,
        bm25_index: Optional[BM25Index] = None,
        retrieval_mode: str = "dense",
        model: str = "gpt-4o-mini",
        reranker=None,
        top_k: int = 10,
    ):
        self.model = model


        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for LLM")
        self.client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        logger.info("Using OpenRouter API for LLM")
        self.chroma_index = chroma_index
        self.embedding_client = embedding_client
        self.bm25_index = bm25_index
        self.retrieval_mode = retrieval_mode
        self.reranker = reranker
        self.top_k = top_k

    @observe()
    def run(self, query: str) -> dict:
        """Run RAG pipeline: retrieve -> generate."""
        try:
            # Retrieval step with Langfuse tracing
            retrieval_result = self._retrieve_with_tracing(query)

            # Generation step with Langfuse tracing
            response, usage = self._generate_with_tracing(query, retrieval_result)

            # Build conversation messages for tracing
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer based only on the provided context.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{retrieval_result.context}\n\nQuestion: {query}",
                },
                {
                    "role": "assistant",
                    "content": response,
                },
            ]

            return {
                "response": response,
                "context": retrieval_result.context,
                "retrieval_result": retrieval_result.to_dict(),
                "messages": messages,
                "search_steps": 1,  # Simple RAG always does 1 search
            }

        except Exception as e:
            logger.error(f"SimpleRAG pipeline failed: {e}", exc_info=True)
            return {
                "response": None,
                "context": None,
                "retrieval_result": None,
                "messages": [],
                "search_steps": 0,
                "error": str(e),
            }

    @observe()
    def _retrieve_with_tracing(self, query: str) -> RetrievalResult:
        """Retrieve context with Langfuse tracing."""
        retrieval_result = self._retrieve(query)


        langfuse.score_current_trace(
            name="retrieval_chunks",
            value=len(retrieval_result.chunks),
            data_type="NUMERIC",
            comment="Number of retrieved chunks"
        )
        langfuse.score_current_trace(
            name="retrieval_latency",
            value=retrieval_result.latency_ms,
            data_type="NUMERIC",
            comment="Retrieval latency in milliseconds"
        )

        return retrieval_result

    @observe()
    def _generate_with_tracing(self, query: str, retrieval_result: RetrievalResult) -> tuple:
        """Generate response with Langfuse tracing."""
        response_obj = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer based only on the provided context.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{retrieval_result.context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        response = response_obj.choices[0].message.content

        # Log generation metrics to Langfuse
        if response_obj.usage:
            langfuse.score_current_trace(
                name="generation_tokens",
                value=response_obj.usage.total_tokens,
                data_type="NUMERIC",
                comment="Total tokens used in generation"
            )
            langfuse.score_current_trace(
                name="generation_cost",
                value=self._estimate_cost(response_obj.usage),
                data_type="NUMERIC",
                comment="Estimated generation cost in USD"
            )

        return response, response_obj.usage

    def _estimate_cost(self, usage) -> float:
        """Estimate cost based on token usage (approximate for OpenRouter)."""


        cost_per_token = 0.00015 / 1000
        return usage.total_tokens * cost_per_token if usage else 0.0

    def _retrieve(self, query: str) -> RetrievalResult:
        """Retrieve context based on retrieval_mode."""
        if self.retrieval_mode == "dense":
            result = retrieve_dense(query, self.chroma_index, self.embedding_client, self.top_k)

        elif self.retrieval_mode == "keyword":
            if self.bm25_index is None:
                raise ValueError("BM25Index required for keyword retrieval")
            result = retrieve_keyword(query, self.bm25_index, self.top_k)

        elif self.retrieval_mode == "hybrid":
            if self.bm25_index is None:
                raise ValueError("BM25Index required for hybrid retrieval")
            result = retrieve_hybrid(query, self.chroma_index, self.embedding_client, self.bm25_index, self.top_k)

        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

        # Apply reranker if available (orthogonal to retrieval mode)
        if self.reranker is not None:
            result = apply_reranker(result, query, self.reranker, self.top_k)

        return result
