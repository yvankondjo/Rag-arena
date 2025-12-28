"""Run RAGBench experiments."""

import asyncio
import hashlib
import json
import logging
import random
import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import os
from threading import Lock

from ragbench.config import AppConfig, RESULTS_DIR, CONFIG_DIR
from ragbench.index.chromadb_index import ChromaDBIndex
from ragbench.index.bm25_index import BM25Index
from ragbench.pipeline.simple_rag import SimpleRAGPipeline
from ragbench.pipeline.agentic_rag import AgenticRAGGraph
from ragbench.beir_download import load_beir_queries, load_beir_qrels, load_beir_corpus
from ragbench.config_benchmark import (
    get_benchmark_config,
    set_benchmark_config,
    BenchmarkConfig,
    QueryLevelMetrics,
    RERANKER_MODEL,
    RERANKER_BACKEND,
    COHERE_RERANK_MODEL,
    COHERE_TOP_N,
    USE_LOCAL_RERANKER_ONLY,
    RANDOM_SEED,
    MAX_CONCURRENT_QUERIES,
    MAX_CONCURRENT_LLM_CALLS,
)

logger = logging.getLogger(__name__)

# Global lock for index access (thread-safety)
_index_lock = Lock()


def _serialize_messages(messages: List[Any]) -> List[dict]:
    """Convert LangChain messages or plain dicts to JSON-serializable dicts."""
    serialized = []
    for msg in messages:
        if msg is None:
            continue
        if isinstance(msg, dict):
            serialized.append(msg)
            continue
        try:
            from langchain_core.messages import message_to_dict
            serialized.append(message_to_dict(msg))
        except (ImportError, TypeError, AttributeError):
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                serialized.append({
                    "type": getattr(msg, 'type', 'unknown'),
                    "content": str(getattr(msg, 'content', '')),
                })
            elif hasattr(msg, '__dict__'):
                try:
                    serialized.append({k: str(v) for k, v in msg.__dict__.items()})
                except Exception:
                    serialized.append({"content": str(msg)})
            else:
                serialized.append({"content": str(msg)})
    return serialized


@dataclass
class RAGConfig:
    """Configuration for a single RAG run."""
    orchestration_mode: str
    retrieval_mode: str
    use_reranker: bool = False
    dataset: str = "scifact"
    model: str = "gpt-4o-mini"
    top_k: int = 10
    max_agentic_steps: int = 3

    def __hash__(self):
        """Generate deterministic hash."""
        reranker_str = "rerank" if self.use_reranker else "no_rerank"
        s = f"{self.orchestration_mode}_{self.retrieval_mode}_{reranker_str}_{self.dataset}"
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    def to_dict(self):
        return asdict(self)


def get_all_configs() -> List[RAGConfig]:
    """Generate all configurations from axes.yaml and base.yaml."""
    base_config_path = CONFIG_DIR / "base.yaml"
    axes_config_path = CONFIG_DIR / "axes.yaml"

    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    with open(axes_config_path, 'r') as f:
        axes_config = yaml.safe_load(f)

    orch_modes = axes_config["orchestration"]["modes"]
    retrieval_modes = axes_config["retrieval"]["modes"]
    reranker_modes = axes_config["reranker"]["modes"]
    reranker_modes = [mode == "rerank" for mode in reranker_modes]

    configs = []
    for orch in orch_modes:
        for retrieval in retrieval_modes:
            for use_reranker in reranker_modes:
                configs.append(
                    RAGConfig(
                        orchestration_mode=orch,
                        retrieval_mode=retrieval,
                        use_reranker=use_reranker,
                        dataset=base_config.get("dataset_name", "scifact"),
                        model=base_config.get("model", "gpt-4o-mini"),
                        top_k=base_config.get("top_k", 10),
                        max_agentic_steps=base_config.get("max_agentic_steps", 3),
                    )
                )
    return configs


def load_indexes(config: RAGConfig) -> tuple:
    """Load Chroma + BM25 indexes for config (thread-safe)."""
    with _index_lock:
        app_config = AppConfig()
        collection_name = f"{config.dataset}_doclevel"

        chroma_index = ChromaDBIndex(
            collection_name=collection_name,
            persist_directory=str(app_config.persist_directory),
        )

        bm25_dir = app_config.bm25_index_dir / f"{config.dataset}_doclevel"
        bm25_index = BM25Index(str(bm25_dir)) if bm25_dir.exists() else None

        return chroma_index, bm25_index


def create_reranker():
    """Create reranker based on RERANKER_BACKEND config.
    
    Options:
    - "cohere": Use Cohere API (fast, deterministic with rerank-v3.5)
    - "local": Use CrossEncoder (free, reproducible, slower)
    - "auto": Cohere if API key available, else local
    """
    backend = RERANKER_BACKEND.lower()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    use_cohere = False
    if backend == "cohere":
        if cohere_api_key:
            use_cohere = True
        else:
            logger.warning("RERANKER_BACKEND=cohere but no COHERE_API_KEY. Falling back to local.")
    elif backend == "auto":
        use_cohere = bool(cohere_api_key)
    
    if use_cohere:
        try:
            import cohere
            client = cohere.ClientV2(api_key=cohere_api_key)
            logger.info(f"Using Cohere reranker: {COHERE_RERANK_MODEL}")
            return client
        except ImportError:
            logger.warning("cohere package not installed. Falling back to local CrossEncoder.")
        except Exception as e:
            logger.warning(f"Failed to init Cohere client: {e}. Falling back to local.")
    
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Using local CrossEncoder reranker: {RERANKER_MODEL}")
        return reranker
    except Exception as e:
        logger.warning(f"Failed to load CrossEncoder reranker: {e}")
        return None


def validate_document_ids(retrieval_results: List[dict], qrels: Dict[str, Dict[str, int]]) -> bool:
    """Validate document IDs exist in qrels."""
    valid = True
    all_relevant_docs = set()
    for qid, rels in qrels.items():
        all_relevant_docs.update(rels.keys())
    
    for result in retrieval_results:
        metadatas = result.get("metadatas", [])
        for meta in metadatas:
            doc_id = meta.get("document_id") if isinstance(meta, dict) else None
            if doc_id and doc_id not in all_relevant_docs:
                logger.debug(f"Document ID {doc_id} not in qrels (may be irrelevant)")
    
    return valid


def select_queries_unbiased(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    max_queries: int,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Unbiased query sampling with random seed."""
    if not max_queries or len(queries) <= max_queries:
        return queries, qrels
    
    valid_qids = [qid for qid in queries.keys() if qid in qrels and len(qrels[qid]) > 0]
    
    if len(valid_qids) <= max_queries:
        selected_qids = valid_qids
    else:
        rng = random.Random(seed)
        selected_qids = rng.sample(valid_qids, max_queries)
    
    selected_queries = {qid: queries[qid] for qid in selected_qids}
    selected_qrels = {qid: qrels[qid] for qid in selected_qids if qid in qrels}
    
    logger.info(f"Random sampling (seed={seed}): {len(selected_queries)} queries from {len(queries)}")
    return selected_queries, selected_qrels


def compute_query_retrieval_metrics(
    retrieved_doc_ids: List[str],
    relevant_docs: Dict[str, int],
) -> dict:
    """Compute retrieval metrics for a single query using BEIR.
    
    Converts ordered doc_ids to scores (descending) for BEIR compatibility.
    """
    from ragbench.eval.retrieval_metrics import evaluate_single_query_retrieval
    
    # Convert ordered list to {doc_id: score} with descending scores
    # This preserves ranking order for BEIR evaluation
    results = {doc_id: len(retrieved_doc_ids) - i for i, doc_id in enumerate(retrieved_doc_ids)}
    
    return evaluate_single_query_retrieval(results, relevant_docs)


def _run_query_sync(
    pipeline,
    query_text: str,
    orchestration_mode: str,
    max_agentic_steps: int
) -> dict:
    """Synchronous query execution wrapper."""
    try:
        with _index_lock:
            if orchestration_mode == "simple":
                result = pipeline.run(query_text)
            else:
                result = pipeline.run(query_text, max_search_steps=max_agentic_steps)
        return result
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        return {"error": str(e)}


async def run_single_config(
    config: RAGConfig,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    semaphore: asyncio.Semaphore,
    results_dir: Path,
    rate_limiter: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Run a single config on all queries."""
    async with semaphore:
        config_hash = config.__hash__()
        run_dir = results_dir / "runs" / config_hash
        run_dir.mkdir(parents=True, exist_ok=True)
        
        reranker_str = "rerank" if config.use_reranker else "no_rerank"
        logger.info(f"🚀 Starting {config_hash}: {config.orchestration_mode}/{config.retrieval_mode}/{reranker_str}")
        
        try:
            chroma_index, bm25_index = load_indexes(config)
            app_config = AppConfig()
            embedding_client = app_config.create_embedding_client()
            
            reranker = None
            if config.use_reranker:
                reranker = create_reranker()
            
            if config.orchestration_mode == "simple":
                pipeline = SimpleRAGPipeline(
                    chroma_index=chroma_index,
                    embedding_client=embedding_client,
                    bm25_index=bm25_index,
                    retrieval_mode=config.retrieval_mode,
                    model=config.model,
                    reranker=reranker,
                    top_k=config.top_k,
                )
            else:
                pipeline = AgenticRAGGraph(
                    model=config.model,
                    chroma_index=chroma_index,
                    embedding_client=embedding_client,
                    bm25_index=bm25_index,
                    retrieval_mode=config.retrieval_mode,
                    reranker=reranker,
                    max_search_steps=config.max_agentic_steps,
                )
            
            start_time = time.time()
            
            query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
            
            async def process_single_query(query_id: str, query_text: str):
                """Process a single query with rate limiting."""
                async with query_semaphore:
                    async with rate_limiter:
                        try:
                            result = await asyncio.to_thread(
                                _run_query_sync,
                                pipeline,
                                query_text,
                                config.orchestration_mode,
                                config.max_agentic_steps
                            )
                            return query_id, query_text, result, None
                        except Exception as e:
                            logger.error(f"Query {query_id} failed: {e}")
                            return query_id, query_text, None, str(e)
            
            tasks = [
                process_single_query(qid, qtext) 
                for qid, qtext in queries.items()
            ]
            query_results = await asyncio.gather(*tasks)
            
            predictions = []
            traces = []
            query_metrics = []
            
            for query_id, query_text, result, error in query_results:
                if error or result is None:
                    predictions.append({
                        "query_id": query_id,
                        "error": error or "Unknown error",
                    })
                    continue
                
                if "error" in result:
                    predictions.append({
                        "query_id": query_id,
                        "error": result["error"],
                    })
                    continue
                
                all_contexts = result.get("all_contexts", [])
                if not all_contexts:
                    retrieval_result = result.get("retrieval_result", {})
                    all_contexts = retrieval_result.get("documents", [])[:10]
                
                latency = result.get("latency", {})
                
                # Extract first_step and final_step doc_ids
                # For simple RAG: first = final (only 1 step)
                # For agentic RAG: first = first retrieval, final = last retrieval
                first_step_doc_ids = result.get("first_step_doc_ids", [])
                final_step_doc_ids = result.get("final_step_doc_ids", [])
                
                # Fallback: extract from all_retrieval_results if not provided
                all_retrieval_results = result.get("all_retrieval_results", [])
                if not first_step_doc_ids and all_retrieval_results:
                    # Get doc_ids from first retrieval
                    first_result = all_retrieval_results[0]
                    for meta in first_result.get("metadatas", [])[:10]:
                        if isinstance(meta, dict) and "document_id" in meta:
                            first_step_doc_ids.append(meta["document_id"])
                
                if not final_step_doc_ids and all_retrieval_results:
                    # Get doc_ids from last retrieval
                    last_result = all_retrieval_results[-1]
                    for meta in last_result.get("metadatas", [])[:10]:
                        if isinstance(meta, dict) and "document_id" in meta:
                            final_step_doc_ids.append(meta["document_id"])
                
                # Legacy fallback for retrieved_doc_ids (use first_step for fair comparison)
                retrieved_doc_ids = first_step_doc_ids[:10] if first_step_doc_ids else []
                if not retrieved_doc_ids:
                    retrieval_result = result.get("retrieval_result", {})
                    for meta in retrieval_result.get("metadatas", [])[:10]:
                        if isinstance(meta, dict) and "document_id" in meta:
                            retrieved_doc_ids.append(meta["document_id"])
                
                relevant_docs = qrels.get(query_id, {})
                
                # Compute FIRST STEP metrics (for fair comparison)
                first_step_metrics = compute_query_retrieval_metrics(
                    first_step_doc_ids[:10] if first_step_doc_ids else retrieved_doc_ids[:10], 
                    relevant_docs
                )
                
                # Compute FINAL STEP metrics (after query refinement)
                final_step_metrics = compute_query_retrieval_metrics(
                    final_step_doc_ids[:10] if final_step_doc_ids else retrieved_doc_ids[:10],
                    relevant_docs
                )
                
                # Calculate orchestration gain (value of multi-step)
                orchestration_gain_ndcg = final_step_metrics["ndcg_at_10"] - first_step_metrics["ndcg_at_10"]
                orchestration_gain_recall = final_step_metrics["recall_at_5"] - first_step_metrics["recall_at_5"]
                
                qm = QueryLevelMetrics(
                    query_id=query_id,
                    query_text=query_text,
                    config_hash=config_hash,
                    retrieval_latency=latency.get("retrieval_ms", 0) / 1000,
                    reranking_latency=latency.get("reranking_ms", 0) / 1000,
                    generation_latency=latency.get("generation_ms", 0) / 1000,
                    total_latency=latency.get("total_ms", 0) / 1000,
                    # First step metrics (fair comparison)
                    retrieved_doc_ids=first_step_doc_ids[:10] if first_step_doc_ids else retrieved_doc_ids[:10],
                    relevant_doc_ids=list(relevant_docs.keys()),
                    ndcg_at_10=first_step_metrics["ndcg_at_10"],
                    recall_at_5=first_step_metrics["recall_at_5"],
                    mrr_at_10=first_step_metrics["mrr_at_10"],
                    # Final step metrics (after refinement)
                    final_step_doc_ids=final_step_doc_ids[:10] if final_step_doc_ids else retrieved_doc_ids[:10],
                    final_step_ndcg_at_10=final_step_metrics["ndcg_at_10"],
                    final_step_recall_at_5=final_step_metrics["recall_at_5"],
                    final_step_mrr_at_10=final_step_metrics["mrr_at_10"],
                    # Orchestration gain
                    orchestration_gain_ndcg=orchestration_gain_ndcg,
                    orchestration_gain_recall=orchestration_gain_recall,
                    context_chunks=all_contexts[:10],
                    context_text=result.get("context", ""),
                    answer=result.get("response", ""),
                    num_retrieval_steps=result.get("search_steps", 1),
                )
                query_metrics.append(qm)
                
                predictions.append({
                    "query_id": query_id,
                    "query": query_text,
                    "response": result.get("response"),
                    "context": result.get("context"),
                    "retrieval": result.get("retrieval_result", {}),
                    "all_contexts": all_contexts,
                    "latency": latency,
                    "search_steps": result.get("search_steps", 1),
                })
                
                traces.append({
                    "query_id": query_id,
                    "messages": result.get("messages", []),
                    "search_steps": result.get("search_steps", 0),
                })
            
            elapsed = time.time() - start_time
            
            config_file = run_dir / "config.yaml"
            predictions_file = run_dir / "predictions.jsonl"
            traces_file = run_dir / "traces.jsonl"
            metrics_file = run_dir / "metrics.json"
            query_metrics_file = run_dir / "query_metrics.jsonl"
            
            with open(config_file, 'w') as f:
                yaml.dump(config.to_dict(), f)
            
            with open(predictions_file, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
            
            with open(traces_file, 'w') as f:
                for trace in traces:
                    serializable_trace = {
                        "query_id": trace.get("query_id"),
                        "search_steps": trace.get("search_steps", 0),
                        "messages": _serialize_messages(trace.get("messages", [])),
                    }
                    f.write(json.dumps(serializable_trace) + "\n")
            
            with open(query_metrics_file, 'w') as f:
                for qm in query_metrics:
                    f.write(json.dumps(qm.to_dict()) + "\n")
            
            successful_predictions = [p for p in predictions if "error" not in p]
            retrieval_metrics = None
            ragas_metrics = None

            if successful_predictions:
                if query_metrics:
                    retrieval_metrics = {
                        "ndcg_at_10": sum(qm.ndcg_at_10 for qm in query_metrics) / len(query_metrics),
                        "recall_at_5": sum(qm.recall_at_5 for qm in query_metrics) / len(query_metrics),
                        "mrr_at_10": sum(qm.mrr_at_10 for qm in query_metrics) / len(query_metrics),
                    }
                
                try:
                    from ragbench.eval.ragas_metrics import evaluate_rag_with_ragas
                    
                    questions = [p["query"] for p in successful_predictions]
                    answers = [p.get("response", "") for p in successful_predictions]
                    contexts = [p.get("all_contexts", [])[:10] for p in successful_predictions]
                    
                    ragas_metrics = evaluate_rag_with_ragas(
                        questions=questions,
                        answers=answers,
                        contexts=contexts
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute Ragas metrics: {e}")

            latency_breakdown = {
                "retrieval_latency_mean": sum(qm.retrieval_latency for qm in query_metrics) / len(query_metrics) if query_metrics else 0,
                "reranking_latency_mean": sum(qm.reranking_latency for qm in query_metrics) / len(query_metrics) if query_metrics else 0,
                "generation_latency_mean": sum(qm.generation_latency for qm in query_metrics) / len(query_metrics) if query_metrics else 0,
            }

            metrics_data = {
                "config_hash": config_hash,
                "num_queries": len(predictions),
                "successful_queries": len(successful_predictions),
                "error_count": sum(1 for p in predictions if "error" in p),
                "elapsed_seconds": elapsed,
                "avg_time_per_query": elapsed / len(predictions) if predictions else 0,
                **latency_breakdown,
            }

            if retrieval_metrics:
                metrics_data.update({
                    "ndcg_at_10": retrieval_metrics["ndcg_at_10"],
                    "recall_at_5": retrieval_metrics["recall_at_5"],
                    "mrr_at_10": retrieval_metrics["mrr_at_10"],
                })

            # Add final_step and orchestration_gain aggregated metrics
            if query_metrics:
                n = len(query_metrics)
                metrics_data.update({
                    "final_step_ndcg_at_10": sum(qm.final_step_ndcg_at_10 for qm in query_metrics) / n,
                    "final_step_recall_at_5": sum(qm.final_step_recall_at_5 for qm in query_metrics) / n,
                    "final_step_mrr_at_10": sum(qm.final_step_mrr_at_10 for qm in query_metrics) / n,
                    "orchestration_gain_ndcg": sum(qm.orchestration_gain_ndcg for qm in query_metrics) / n,
                    "orchestration_gain_recall": sum(qm.orchestration_gain_recall for qm in query_metrics) / n,
                    "avg_retrieval_steps": sum(qm.num_retrieval_steps for qm in query_metrics) / n,
                })

            if ragas_metrics:
                metrics_data.update({
                    "faithfulness": ragas_metrics.faithfulness,
                    "answer_relevancy": ragas_metrics.answer_relevancy,
                })

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"✓ {config_hash} completed in {elapsed:.1f}s")
            
            return {
                "config_hash": config_hash,
                "status": "success",
                "num_queries": len(predictions),
                "elapsed": elapsed,
            }
        
        except Exception as e:
            logger.error(f"❌ {config_hash} failed: {e}", exc_info=True)
            return {
                "config_hash": config_hash,
                "status": "failed",
                "error": str(e),
            }


async def run_benchmark_parallel(
    configs: List[RAGConfig],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    results_dir: Path,
    max_parallel: int = 3,
    max_concurrent_api_calls: int = 10,
) -> List[Dict]:
    """Run configs in parallel with semaphore and rate limiting."""
    semaphore = asyncio.Semaphore(max_parallel)
    rate_limiter = asyncio.Semaphore(max_concurrent_api_calls)
    
    tasks = [
        run_single_config(cfg, queries, qrels, semaphore, results_dir, rate_limiter)
        for cfg in configs
    ]
    
    results = await asyncio.gather(*tasks)
    return results


def save_selected_queries(results_dir: Path, query_ids: List[str], max_queries: int, seed: int):
    """Save selected query IDs for reproducibility."""
    selection_file = results_dir / "selected_queries.json"
    data = {
        "query_ids": query_ids,
        "max_queries": max_queries,
        "count": len(query_ids),
        "seed": seed,
        "sampling_method": "random",  # Document the method
    }
    with open(selection_file, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(query_ids)} selected query IDs to {selection_file}")


def load_selected_queries(results_dir: Path) -> tuple:
    """Load previously selected query IDs."""
    selection_file = results_dir / "selected_queries.json"
    if not selection_file.exists():
        return None, None
    
    try:
        with open(selection_file, 'r') as f:
            data = json.load(f)
        query_ids = data.get("query_ids", [])
        max_queries = data.get("max_queries")
        logger.info(f"Loaded {len(query_ids)} previously selected query IDs")
        return query_ids, max_queries
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load selected queries: {e}")
        return None, None


def apply_saved_query_selection(
    queries: Dict[str, str], 
    qrels: Dict[str, Dict[str, int]], 
    saved_query_ids: List[str]
) -> tuple:
    """Filter queries and qrels to match saved selection."""
    selected_queries = {qid: queries[qid] for qid in saved_query_ids if qid in queries}
    selected_qrels = {qid: qrels[qid] for qid in saved_query_ids if qid in qrels}
    
    if len(selected_queries) != len(saved_query_ids):
        missing = set(saved_query_ids) - set(selected_queries.keys())
        logger.warning(f"Some saved query IDs not found in dataset: {missing}")
    
    return selected_queries, selected_qrels


def get_completed_configs(results_dir: Path) -> set:
    """Get set of config hashes that have already been completed."""
    completed = set()
    runs_dir = results_dir / "runs"
    
    if not runs_dir.exists():
        return completed
    
    for config_dir in runs_dir.iterdir():
        if not config_dir.is_dir():
            continue
        
        config_hash = config_dir.name
        config_file = config_dir / "config.yaml"
        predictions_file = config_dir / "predictions.jsonl"
        metrics_file = config_dir / "metrics.json"
        
        if config_file.exists() and predictions_file.exists() and metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if metrics.get("successful_queries", 0) > 0:
                        completed.add(config_hash)
            except (json.JSONDecodeError, KeyError):
                pass
    
    return completed


def filter_pending_configs(configs: List[RAGConfig], completed_hashes: set) -> List[RAGConfig]:
    """Filter out configs that have already been completed."""
    pending = []
    for cfg in configs:
        cfg_hash = cfg.__hash__()
        if cfg_hash not in completed_hashes:
            pending.append(cfg)
        else:
            reranker_str = "rerank" if cfg.use_reranker else "no_rerank"
            logger.info(f"⏭️ Skipping {cfg_hash} ({cfg.orchestration_mode}/{cfg.retrieval_mode}/{reranker_str}) - already completed")
    return pending


async def run_benchmark_async(
    dataset: str = "scifact",
    parallel: bool = False,
    max_configs: int = None,
    max_queries: int = None,
    dry_run: bool = False,
    resume: bool = False,
) -> List[Dict]:
    """Run full RAGBench benchmark (async version)."""
    app_config = AppConfig()
    
    benchmark_config = BenchmarkConfig()
    set_benchmark_config(benchmark_config)

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = app_config.raw_data_dir / dataset
    queries = load_beir_queries(dataset_path)
    qrels = load_beir_qrels(dataset_path)

    if resume:
        saved_query_ids, saved_max_queries = load_selected_queries(results_dir)
        
        if saved_query_ids:
            queries, qrels = apply_saved_query_selection(queries, qrels, saved_query_ids)
            print(f"📋 Using {len(queries)} queries from previous run (saved selection)")
            
            if max_queries and max_queries != saved_max_queries:
                print(f"⚠️ Note: --max-queries={max_queries} ignored in resume mode")
        elif max_queries:
            queries, qrels = select_queries_unbiased(queries, qrels, max_queries, RANDOM_SEED)
            save_selected_queries(results_dir, list(queries.keys()), max_queries, RANDOM_SEED)
    elif max_queries:
        queries, qrels = select_queries_unbiased(queries, qrels, max_queries, RANDOM_SEED)
        save_selected_queries(results_dir, list(queries.keys()), max_queries, RANDOM_SEED)

    configs = get_all_configs()
    if max_configs:
        configs = configs[:max_configs]

    skipped_count = 0
    if resume:
        completed_hashes = get_completed_configs(results_dir)
        original_count = len(configs)
        configs = filter_pending_configs(configs, completed_hashes)
        skipped_count = original_count - len(configs)
        
        if skipped_count > 0:
            print(f"\n🔄 RESUME MODE: Skipping {skipped_count} already completed configurations")
            print(f"   Remaining: {len(configs)} configurations to run\n")
        
        if not configs:
            print("✅ All configurations already completed! Nothing to run.")
            return []

    if dry_run:
        print(f"Dry run: {len(configs)} configurations to run")
        print(f"  Using {len(queries)} queries (random sampling, seed={RANDOM_SEED})")
        if resume and skipped_count > 0:
            print(f"  (Skipped {skipped_count} already completed)")
        for cfg in configs:
            reranker_str = "rerank" if cfg.use_reranker else "no_rerank"
            print(f"  {cfg.__hash__()}: {cfg.orchestration_mode}/{cfg.retrieval_mode}/{reranker_str}")
        return []

    logger.info(f"🏃 Running {len(configs)} configurations on {len(queries)} queries")

    if parallel:
        results = await run_benchmark_parallel(
            configs, queries, qrels, results_dir, 
            max_parallel=3, max_concurrent_api_calls=10
        )
    else:
        results = []
        semaphore = asyncio.Semaphore(1)
        rate_limiter = asyncio.Semaphore(1)
        for cfg in configs:
            result = await run_single_config(cfg, queries, qrels, semaphore, results_dir, rate_limiter)
            results.append(result)

    logger.info(f"✓ Benchmark completed: {len(results)} configs")

    success = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    print(f"\n{'='*60}")
    print(f"Results: {success} success, {failed} failed")
    if resume and skipped_count > 0:
        print(f"Skipped (already done): {skipped_count}")
    print(f"Output: {results_dir}/runs/")
    print(f"{'='*60}\n")

    return results


def run_benchmark(
    dataset: str = "scifact",
    parallel: bool = False,
    max_configs: int = None,
    max_queries: int = None,
    dry_run: bool = False,
    resume: bool = False,
) -> List[Dict]:
    """Run full RAGBench benchmark."""
    return asyncio.run(
        run_benchmark_async(dataset, parallel, max_configs, max_queries, dry_run, resume)
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="RAGBench: Run RAG benchmark experiments")
    parser.add_argument("--dataset", default="scifact", help="Dataset name")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--resume", action="store_true", help="Resume from where stopped")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit configs")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries")
    args = parser.parse_args()
    
    run_benchmark(
        dataset=args.dataset,
        parallel=args.parallel,
        max_configs=args.max_configs,
        max_queries=args.max_queries,
        dry_run=args.dry_run,
        resume=args.resume,
    )
