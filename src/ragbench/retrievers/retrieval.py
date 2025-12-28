"""Simple retrieval functions for RAG benchmark."""

import logging
import time
from typing import Optional

from ragbench.index.chromadb_index import ChromaDBIndex
from ragbench.index.bm25_index import BM25Index
from ragbench.clients import EmbeddingClient
from ragbench.retrievers.base import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


def apply_reranker(result: RetrievalResult, query: str, reranker, top_k: int = 10) -> RetrievalResult:
    """Apply reranking to any retrieval result (Cohere or SentenceTransformers).

    Args:
        result: RetrievalResult to rerank
        query: Original query
        reranker: Reranker instance (Cohere ClientV2 or CrossEncoder)
        top_k: Number of results after reranking

    Returns:
        Reranked RetrievalResult with reranking_latency_ms set
    """
    if reranker is None or not result.chunks:
        return result

    try:
        start_time = time.time()
        
        # Cohere ClientV2 detection
        if hasattr(reranker, 'rerank'):
            documents = [chunk.text for chunk in result.chunks]
            
            # Use config model (rerank-v3.5 for reproducibility)
            try:
                from ragbench.config_benchmark import COHERE_RERANK_MODEL
                model = COHERE_RERANK_MODEL
            except ImportError:
                model = "rerank-v3.5"
            
            response = reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model=model,
            )

            reranked = []
            for result_item in response.results:
                original_chunk = result.chunks[result_item.index]
                original_chunk.score = float(result_item.relevance_score)
                reranked.append(original_chunk)

        else:
            # SentenceTransformers CrossEncoder (local, deterministic)
            pairs = [[query, chunk.text] for chunk in result.chunks]
            scores = reranker.predict(pairs, batch_size=32)

            reranked = []
            for chunk, score in sorted(zip(result.chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]:
                chunk.score = float(score)
                reranked.append(chunk)

        reranking_latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Reranking: {len(reranked)} results in {reranking_latency_ms:.1f}ms")

        return RetrievalResult(
            chunks=reranked,
            query=query,
            retriever_type=f"{result.retriever_type}_reranked",
            latency_ms=result.latency_ms,  # Keep original retrieval latency
            reranking_latency_ms=reranking_latency_ms,  # Separate reranking latency
        )
    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        return result


def retrieve_dense(
    query: str,
    chroma_index: ChromaDBIndex,
    embedding_client: EmbeddingClient,
    top_k: int = 10,
) -> RetrievalResult:
    """Retrieve using dense vector similarity (Chroma).
    
    Args:
        query: Query text
        chroma_index: ChromaDB index
        embedding_client: Embedding client
        top_k: Number of results to return
        
    Returns:
        RetrievalResult
    """
    start_time = time.time()
    
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        logger.warning("Empty query provided")
        return RetrievalResult(chunks=[], query=query, retriever_type="dense")
    
    try:
        query_embedding = embedding_client.embed_single(query)
        
        results = chroma_index.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        

        chunks = []
        if results and results.get("documents") and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []
            ids = results.get("ids", [[]])[0] or []
            
            for i, (doc, doc_id) in enumerate(zip(documents, ids)):
                distance = distances[i] if i < len(distances) else 1.0
                similarity = 1.0 - distance
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                chunk = RetrievedChunk(
                    chunk_id=doc_id,
                    text=doc,
                    score=similarity,
                    document_id=metadata.get("document_id", "unknown"),
                    metadata=metadata,
                )
                chunks.append(chunk)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Dense retrieval: {len(chunks)} results in {latency_ms:.1f}ms")
        
        return RetrievalResult(
            chunks=chunks,
            query=query,
            retriever_type="dense",
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Dense retrieval failed: {e}", exc_info=True)
        return RetrievalResult(
            chunks=[],
            query=query,
            retriever_type="dense",
            latency_ms=latency_ms,
        )


def retrieve_keyword(
    query: str,
    bm25_index: BM25Index,
    top_k: int = 10,
) -> RetrievalResult:
    """Retrieve using BM25 keyword search.
    
    Args:
        query: Query text
        bm25_index: BM25Index instance
        top_k: Number of results to return
        
    Returns:
        RetrievalResult
    """
    start_time = time.time()
    
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        logger.warning("Empty query provided")
        return RetrievalResult(chunks=[], query=query, retriever_type="keyword")
    
    if bm25_index.retriever is None:
        logger.error("BM25s index not loaded")
        return RetrievalResult(chunks=[], query=query, retriever_type="keyword")
    
    try:
       
        bm25_result = bm25_index.query(query, top_k=top_k)
        
        chunks = []
        documents = bm25_result.get("documents", [])
        ids = bm25_result.get("ids", [])
        sources = bm25_result.get("sources", [])  
        metadatas = bm25_result.get("metadatas", [])
        scores = bm25_result.get("scores", []) 

        for i, doc_text in enumerate(documents):
            
            document_id = sources[i] if i < len(sources) else "unknown"
            chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
            metadata = metadatas[i] if i < len(metadatas) else {}

          
            if "document_id" not in metadata:
                metadata["document_id"] = document_id

          
            score = scores[i] if i < len(scores) else 1.0 / (i + 1)

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=doc_text,
                score=score,
                document_id=document_id,
                metadata=metadata,
            )
            chunks.append(chunk)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Keyword retrieval: {len(chunks)} results in {latency_ms:.1f}ms")
        
        return RetrievalResult(
            chunks=chunks,
            query=query,
            retriever_type="keyword",
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Keyword retrieval failed: {e}", exc_info=True)
        return RetrievalResult(
            chunks=[],
            query=query,
            retriever_type="keyword",
            latency_ms=latency_ms,
        )


def retrieve_hybrid(
    query: str,
    chroma_index: ChromaDBIndex,
    embedding_client: EmbeddingClient,
    bm25_index: BM25Index,
    top_k: int = 10,
    k_rrf: float = 60.0,
) -> RetrievalResult:
    """Retrieve using hybrid RRF fusion (dense + keyword).
    
    Args:
        query: Query text
        chroma_index: ChromaDB index
        embedding_client: Embedding client
        bm25_index: BM25Index instance
        top_k: Number of results to return
        k_rrf: RRF parameter (typical: 40-80)
        
    Returns:
        RetrievalResult
    """
    start_time = time.time()
    
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        logger.warning("Empty query provided")
        return RetrievalResult(chunks=[], query=query, retriever_type="hybrid")
    
    try:
  
        dense_result = retrieve_dense(query, chroma_index, embedding_client, top_k=top_k * 2)
        keyword_result = retrieve_keyword(query, bm25_index, top_k=top_k * 2)
        
    
        rrf_scores = {}
        
        for rank, chunk in enumerate(dense_result.chunks, 1):
            chunk_id = chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k_rrf + rank)
        
        for rank, chunk in enumerate(keyword_result.chunks, 1):
            chunk_id = chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k_rrf + rank)
        

        chunks_dict = {}
        for chunk in dense_result.chunks:
            if chunk.chunk_id not in chunks_dict:
                chunks_dict[chunk.chunk_id] = chunk
        for chunk in keyword_result.chunks:
            if chunk.chunk_id not in chunks_dict:
                chunks_dict[chunk.chunk_id] = chunk
        

        sorted_chunks = sorted(
            chunks_dict.values(),
            key=lambda c: rrf_scores.get(c.chunk_id, 0),
            reverse=True,
        )[:top_k]
        

        for chunk in sorted_chunks:
            chunk.score = rrf_scores.get(chunk.chunk_id, 0)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Hybrid retrieval: {len(sorted_chunks)} results in {latency_ms:.1f}ms")
        
        return RetrievalResult(
            chunks=sorted_chunks,
            query=query,
            retriever_type="hybrid",
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Hybrid retrieval failed: {e}", exc_info=True)
        return RetrievalResult(
            chunks=[],
            query=query,
            retriever_type="hybrid",
            latency_ms=latency_ms,
        )
