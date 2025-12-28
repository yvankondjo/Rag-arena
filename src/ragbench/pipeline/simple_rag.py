"""Simple RAG pipeline: retrieve once, generate once.

CRITICAL: This pipeline MUST use the same configuration as AgenticRAG:
- Same system prompt (from config_benchmark.py)
- Same temperature (0.0 for reproducibility)
- Same model
"""

import logging
import os
import time
from typing import Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

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
from ragbench.config_benchmark import (
    get_benchmark_config,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    GENERATION_TEMPERATURE,
    MAX_GENERATION_TOKENS,
    RERANKER_OVER_FETCH,
    RANDOM_SEED,
)

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
logger = logging.getLogger(__name__)

# Initialize Langfuse client
langfuse = Langfuse()


@dataclass
class LatencyBreakdown:
    """Detailed latency measurements for fair comparison."""
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class SimpleRAGPipeline:
    """Simple RAG: retrieve context -> generate response.
    
    IMPORTANT: Uses IDENTICAL configuration to AgenticRAG for fair comparison:
    - Same prompt template
    - Same temperature (0.0)
    - Same model
    """

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
        self.benchmark_config = get_benchmark_config()

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

    def run(self, query: str) -> dict:
        """Run RAG pipeline: retrieve -> generate.
        
        Returns dict with:
        - response: Generated answer
        - context: Full context used for generation
        - retrieval_result: Retrieval metadata
        - all_contexts: List of all context chunks (for RAGAS)
        - latency: Detailed latency breakdown
        - messages: Conversation messages
        - search_steps: Always 1 for simple RAG
        """
        total_start = time.perf_counter()
        
        try:
            # Step 1: Retrieval with timing
            retrieval_start = time.perf_counter()
            retrieval_result = self._retrieve(query)
            retrieval_time = (time.perf_counter() - retrieval_start) * 1000
            
            # Step 2: Reranking timing (if applicable)
            reranking_time = 0.0
            if self.reranker is not None:
                rerank_start = time.perf_counter()
                retrieval_result = apply_reranker(
                    retrieval_result, query, self.reranker, self.top_k
                )
                reranking_time = (time.perf_counter() - rerank_start) * 1000
            
            # Step 3: Generation with timing
            generation_start = time.perf_counter()
            response, usage = self._generate(query, retrieval_result)
            generation_time = (time.perf_counter() - generation_start) * 1000
            
            total_time = (time.perf_counter() - total_start) * 1000
            
            # Build latency breakdown
            latency = LatencyBreakdown(
                retrieval_ms=retrieval_time,
                reranking_ms=reranking_time,
                generation_ms=generation_time,
                total_ms=total_time,
            )
            
            # Build conversation messages using UNIFIED prompt template
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        context=retrieval_result.context,
                        question=query
                    ),
                },
                {"role": "assistant", "content": response},
            ]
            
            # CRITICAL: Store ALL context chunks for RAGAS evaluation
            # This ensures RAGAS evaluates on the SAME context used for generation
            all_context_chunks = retrieval_result.documents
            
            # Extract doc_ids for retrieval metrics
            # For simple RAG: first_step = final_step (only 1 retrieval)
            doc_ids = []
            for chunk in retrieval_result.chunks:
                if chunk.document_id and chunk.document_id not in doc_ids:
                    doc_ids.append(chunk.document_id)
            doc_ids = doc_ids[:10]
            
            return {
                "response": response,
                "context": retrieval_result.context,
                "retrieval_result": retrieval_result.to_dict(),
                "messages": messages,
                "search_steps": 1,  # Simple RAG always does 1 search
                # NEW: For fair RAGAS evaluation
                "all_contexts": all_context_chunks,
                "all_retrieval_results": [retrieval_result.to_dict()],
                # NEW: For fair retrieval metrics (first = final for simple RAG)
                "first_step_doc_ids": doc_ids,
                "final_step_doc_ids": doc_ids,
                "latency": {
                    "retrieval_ms": latency.retrieval_ms,
                    "reranking_ms": latency.reranking_ms,
                    "generation_ms": latency.generation_ms,
                    "total_ms": latency.total_ms,
                },
            }

        except Exception as e:
            logger.error(f"SimpleRAG pipeline failed: {e}", exc_info=True)
            return {
                "response": None,
                "context": None,
                "retrieval_result": None,
                "messages": [],
                "search_steps": 0,
                "all_contexts": [],
                "all_retrieval_results": [],
                "first_step_doc_ids": [],
                "final_step_doc_ids": [],
                "latency": {"retrieval_ms": 0, "reranking_ms": 0, "generation_ms": 0, "total_ms": 0},
                "error": str(e),
            }

    def _generate(self, query: str, retrieval_result: RetrievalResult) -> Tuple[str, dict]:
        """Generate response using UNIFIED prompt template and temperature.
        
        CRITICAL: Uses SAME parameters as AgenticRAG:
        - Temperature: 0.0 (from config_benchmark)
        - System prompt: Unified (from config_benchmark)
        - User prompt: Unified template (from config_benchmark)
        """
        response_obj = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        context=retrieval_result.context,
                        question=query
                    ),
                },
            ],
            temperature=GENERATION_TEMPERATURE,  # 0.0 for reproducibility
            max_tokens=MAX_GENERATION_TOKENS,
            seed=RANDOM_SEED,  # Deterministic sampling
        )

        response = response_obj.choices[0].message.content
        return response, response_obj.usage

    def _retrieve(self, query: str) -> RetrievalResult:
        """Retrieve context based on retrieval_mode.
        
        When reranker is active, fetch more candidates (RERANKER_OVER_FETCH * top_k)
        to give the reranker enough documents to reorder effectively.
        
        Note: Reranking is handled separately in run() for accurate timing.
        """
        # Over-fetch when reranker is active to give it more candidates
        fetch_k = self.top_k * RERANKER_OVER_FETCH if self.reranker else self.top_k
        
        if self.retrieval_mode == "dense":
            result = retrieve_dense(
                query, self.chroma_index, self.embedding_client, fetch_k
            )
        elif self.retrieval_mode == "keyword":
            if self.bm25_index is None:
                raise ValueError("BM25Index required for keyword retrieval")
            result = retrieve_keyword(query, self.bm25_index, fetch_k)
        elif self.retrieval_mode == "hybrid":
            if self.bm25_index is None:
                raise ValueError("BM25Index required for hybrid retrieval")
            result = retrieve_hybrid(
                query, self.chroma_index, self.embedding_client, 
                self.bm25_index, fetch_k
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

        return result
