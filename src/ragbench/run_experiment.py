"""Run RAGBench-12x experiments with proper async support."""

import asyncio
import hashlib
import json
import logging
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Any
import time
import os
from threading import Lock

# Cross-platform resource module handling
try:
    import resource  # Unix only
except ImportError:
    resource = None

from ragbench.config import AppConfig, RESULTS_DIR, CONFIG_DIR
from ragbench.index.chromadb_index import ChromaDBIndex
from ragbench.index.bm25_index import BM25Index
from ragbench.pipeline.simple_rag import SimpleRAGPipeline
from ragbench.pipeline.agentic_rag import AgenticRAGGraph
from ragbench.beir_download import load_beir_queries, load_beir_qrels
from pathlib import Path as PathlibPath

logger = logging.getLogger(__name__)

# Global lock for index access (thread-safety)
_index_lock = Lock()


def _serialize_messages(messages: List[Any]) -> List[dict]:
    """Convert LangChain messages or plain dicts to JSON-serializable dicts.
    
    Args:
        messages: List of LangChain message objects, plain dicts, or other objects
        
    Returns:
        List of serializable dictionaries
    """
    serialized = []
    for msg in messages:
        if msg is None:
            continue
        
        # Already a dict (e.g., from SimpleRAGPipeline)
        if isinstance(msg, dict):
            serialized.append(msg)
            continue
        
        # Try to use LangChain's message_to_dict if available
        try:
            from langchain_core.messages import message_to_dict
            serialized.append(message_to_dict(msg))
        except (ImportError, TypeError, AttributeError):
            # Fallback: extract content and type manually
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                serialized.append({
                    "type": getattr(msg, 'type', 'unknown'),
                    "content": str(getattr(msg, 'content', '')),
                })
            elif hasattr(msg, '__dict__'):
                # Try to serialize the object's dict representation
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
    """Load Chroma + BM25 indexes for config (thread-safe).
    
    Note: This creates shared index instances. Access is protected by _index_lock.
    """
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


def _run_query_sync(pipeline, query_text: str, orchestration_mode: str, max_agentic_steps: int) -> dict:
    """Synchronous query execution wrapper for thread pool.
    
    This function isolates ALL blocking I/O (API calls, index access) in one place.
    """
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
    """Run a single config on all queries with proper async execution."""
    async with semaphore:
        config_hash = config.__hash__()
        run_dir = results_dir / "runs" / config_hash
        run_dir.mkdir(parents=True, exist_ok=True)
        
        reranker_str = "rerank" if config.use_reranker else "no_rerank"
        logger.info(f"🚀 Starting {config_hash}: {config.orchestration_mode}/{config.retrieval_mode}/{reranker_str}")
        
        try:
            # Load indexes (thread-safe)
            chroma_index, bm25_index = load_indexes(config)
            app_config = AppConfig()
            embedding_client = app_config.create_embedding_client()
            
            # Create pipeline
            reranker = None
            if config.use_reranker:
                try:
                    # Try Cohere reranker first (requires API key)
                    cohere_api_key = os.getenv("COHERE_API_KEY")
                    if cohere_api_key:
                        from cohere import Client
                        reranker = Client(api_key=cohere_api_key)
                        logger.info("Using Cohere reranker")
                    else:
                        # Fall back to sentence-transformers (no API key needed)
                        from sentence_transformers import CrossEncoder
                        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                        logger.info("Using SentenceTransformers reranker (MiniLM)")
                except Exception as e:
                    logger.warning(f"Reranker not available: {e}")
                    reranker = None
            
            # Disable Langfuse auto-tracing in parallel mode to avoid trace mixing
            # Each config will have isolated traces via config_hash
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
            else:  # agentic
                pipeline = AgenticRAGGraph(
                    model=config.model,
                    chroma_index=chroma_index,
                    embedding_client=embedding_client,
                    bm25_index=bm25_index,
                    retrieval_mode=config.retrieval_mode,
                    reranker=reranker,
                    max_search_steps=config.max_agentic_steps,
                )
            
            # Run queries with proper async execution
            predictions = []
            traces = []
            start_time = time.time()
            
            for query_id, query_text in queries.items():
                try:
                    # Rate limiting for API calls
                    async with rate_limiter:
                        # CRITICAL FIX: Run blocking I/O in thread pool
                        result = await asyncio.to_thread(
                            _run_query_sync,
                            pipeline,
                            query_text,
                            config.orchestration_mode,
                            config.max_agentic_steps
                        )
                    
                    if "error" in result:
                        predictions.append({
                            "query_id": query_id,
                            "error": result["error"],
                        })
                    else:
                        predictions.append({
                            "query_id": query_id,
                            "query": query_text,
                            "response": result.get("response"),
                            "context": result.get("context"),
                            "retrieval": result.get("retrieval_result", {}),
                        })
                        
                        traces.append({
                            "query_id": query_id,
                            "messages": result.get("messages", []),
                            "search_steps": result.get("search_steps", 0),
                        })
                
                except Exception as e:
                    logger.error(f"Query {query_id} failed: {e}")
                    predictions.append({
                        "query_id": query_id,
                        "error": str(e),
                    })
            
            elapsed = time.time() - start_time
            
            # Save results
            config_file = run_dir / "config.yaml"
            predictions_file = run_dir / "predictions.jsonl"
            traces_file = run_dir / "traces.jsonl"
            metrics_file = run_dir / "metrics.json"
            
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(config.to_dict(), f)
            
            with open(predictions_file, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
            
            with open(traces_file, 'w') as f:
                for trace in traces:
                    # Convert LangChain messages to JSON-serializable format
                    serializable_trace = {
                        "query_id": trace.get("query_id"),
                        "search_steps": trace.get("search_steps", 0),
                        "messages": _serialize_messages(trace.get("messages", [])),
                    }
                    f.write(json.dumps(serializable_trace) + "\n")
            
            # Compute metrics
            successful_predictions = [p for p in predictions if "error" not in p]

            retrieval_metrics = None
            ragas_metrics = None

            if successful_predictions:
                answers = []
                contexts = []
                retrieval_results = {}  # {query_id: {doc_id: score}}

                for pred in successful_predictions:
                    query_id = pred["query_id"]
                    answers.append(pred.get("response", ""))

                    # Extract retrieval results for metrics
                    retrieval_result = pred.get("retrieval", {}) or pred.get("retrieval_result", {})
                    if isinstance(retrieval_result, dict):
                        metadatas = retrieval_result.get("metadatas", [])[:10]
                        scores = retrieval_result.get("scores", [])[:10]
                        
                        # Extract document_ids from metadata
                        doc_scores = {}
                        for i, metadata in enumerate(metadatas):
                            doc_id = metadata.get("document_id") if isinstance(metadata, dict) else None
                            if doc_id:
                                score = scores[i] if i < len(scores) else 0.0
                                doc_scores[doc_id] = max(doc_scores.get(doc_id, float("-inf")), score)
                        
                        retrieval_results[query_id] = doc_scores

                        # For Ragas context
                        context_texts = retrieval_result.get("documents", [])[:10]
                        contexts.append(context_texts if context_texts else [])
                    else:
                        retrieval_results[query_id] = {}
                        contexts.append([])

                query_ids = [p["query_id"] for p in successful_predictions]
                if query_ids and all(qid in qrels for qid in query_ids):
                    try:
                        from ragbench.eval.retrieval_metrics import evaluate_retrieval_metrics
                        retrieval_metrics = evaluate_retrieval_metrics(
                            results=retrieval_results,
                            qrels={qid: qrels[qid] for qid in query_ids}
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute retrieval metrics: {e}")

                # Ragas evaluation
                try:
                    from ragbench.eval.ragas_metrics import evaluate_rag_with_ragas
                    ragas_metrics = evaluate_rag_with_ragas(
                        questions=[p["query"] for p in successful_predictions],
                        answers=answers,
                        contexts=contexts
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute Ragas metrics: {e}")

            # Save comprehensive metrics
            metrics_data = {
                "config_hash": config_hash,
                "num_queries": len(predictions),
                "successful_queries": len(successful_predictions),
                "error_count": sum(1 for p in predictions if "error" in p),
                "elapsed_seconds": elapsed,
                "avg_time_per_query": elapsed / len(predictions) if predictions else 0,
            }

            if retrieval_metrics:
                metrics_data.update({
                    "ndcg_at_10": retrieval_metrics.ndcg_at_10,
                    "recall_at_5": retrieval_metrics.recall_at_5,
                    "mrr_at_10": retrieval_metrics.mrr_at_10,
                })

            if ragas_metrics:
                metrics_data.update({
                    "faithfulness": ragas_metrics.faithfulness,
                    "answer_relevancy": ragas_metrics.answer_relevancy,
                    "context_precision": ragas_metrics.context_precision,
                    "context_recall": ragas_metrics.context_recall,
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
    max_concurrent_api_calls: int = 10,  # Rate limiting for API calls
) -> List[Dict]:
    """Run configs in parallel with semaphore and rate limiting.
    
    Args:
        configs: List of RAG configurations
        queries: Query dictionary
        qrels: Relevance judgments
        results_dir: Results directory
        max_parallel: Maximum parallel configurations (default: 3)
        max_concurrent_api_calls: Maximum concurrent API calls across all configs (default: 10)
    
    Returns:
        List of result dictionaries
    """
    semaphore = asyncio.Semaphore(max_parallel)
    rate_limiter = asyncio.Semaphore(max_concurrent_api_calls)
    
    tasks = [
        run_single_config(cfg, queries, qrels, semaphore, results_dir, rate_limiter)
        for cfg in configs
    ]
    
    results = await asyncio.gather(*tasks)
    return results


def select_representative_queries(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], max_queries: int) -> tuple:
    """Select representative queries with good data coverage."""
    if not max_queries or len(queries) <= max_queries:
        return queries, qrels


    query_stats = []
    for qid, _ in queries.items():
        relevant_docs = len(qrels.get(qid, {}))
        has_relevant = relevant_docs > 0

        if has_relevant:  
            score = min(relevant_docs, 5)
            query_stats.append((qid, score, relevant_docs))


    query_stats.sort(key=lambda x: x[1], reverse=True)


    selected_qids = [qid for qid, _, _ in query_stats[:max_queries]]
    selected_queries = {qid: queries[qid] for qid in selected_qids}
    selected_qrels = {qid: qrels[qid] for qid in selected_qids if qid in qrels}

    logger.info(f"Smart selection: {len(selected_queries)} queries (from {len(queries)})")
    return selected_queries, selected_qrels


def save_selected_queries(results_dir: Path, query_ids: List[str], max_queries: int):
    """Save selected query IDs to a file for reproducibility during resume.
    
    Args:
        results_dir: Results directory
        query_ids: List of selected query IDs
        max_queries: The max_queries parameter used for selection
    """
    selection_file = results_dir / "selected_queries.json"
    data = {
        "query_ids": query_ids,
        "max_queries": max_queries,
        "count": len(query_ids),
    }
    with open(selection_file, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(query_ids)} selected query IDs to {selection_file}")


def load_selected_queries(results_dir: Path) -> tuple:
    """Load previously selected query IDs from resume file.
    
    Returns:
        Tuple of (query_ids list, max_queries) or (None, None) if not found
    """
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
    """Filter queries and qrels to match saved selection.
    
    Args:
        queries: Full query dictionary
        qrels: Full qrels dictionary
        saved_query_ids: Previously saved query IDs
        
    Returns:
        Filtered (queries, qrels) matching the saved selection
    """
    selected_queries = {qid: queries[qid] for qid in saved_query_ids if qid in queries}
    selected_qrels = {qid: qrels[qid] for qid in saved_query_ids if qid in qrels}
    
    if len(selected_queries) != len(saved_query_ids):
        missing = set(saved_query_ids) - set(selected_queries.keys())
        logger.warning(f"Some saved query IDs not found in dataset: {missing}")
    
    return selected_queries, selected_qrels


def get_completed_configs(results_dir: Path) -> set:
    """Get set of config hashes that have already been completed.
    
    A config is considered complete if:
    - config.yaml exists
    - predictions.jsonl exists
    - metrics.json exists
    
    Returns:
        Set of completed config hashes
    """
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
                        logger.debug(f"Config {config_hash} already completed")
            except (json.JSONDecodeError, KeyError):

                pass
    
    return completed


def filter_pending_configs(configs: List[RAGConfig], completed_hashes: set) -> List[RAGConfig]:
    """Filter out configs that have already been completed.
    
    Args:
        configs: List of all configurations
        completed_hashes: Set of config hashes already completed
        
    Returns:
        List of configs that still need to be run
    """
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
    """Run full RAGBench-12x benchmark (async version).
    
    Args:
        dataset: Dataset name (default: scifact)
        parallel: Run configs in parallel (default: False)
        max_configs: Limit number of configs to run
        max_queries: Limit number of queries per config
        dry_run: Only show what would be run
        resume: Skip already completed configurations (default: False)
    """
    app_config = AppConfig()


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
                print(f"⚠️ Note: --max-queries={max_queries} ignored in resume mode (using saved: {saved_max_queries})")
        elif max_queries:

            queries, qrels = select_representative_queries(queries, qrels, max_queries)
            save_selected_queries(results_dir, list(queries.keys()), max_queries)
    elif max_queries:

        queries, qrels = select_representative_queries(queries, qrels, max_queries)
        save_selected_queries(results_dir, list(queries.keys()), max_queries)

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
        print(f"  Using {len(queries)} queries")
        if resume and skipped_count > 0:
            print(f"  (Skipped {skipped_count} already completed)")
        for cfg in configs:
            reranker_str = "rerank" if cfg.use_reranker else "no_rerank"
            print(f"  {cfg.__hash__()}: {cfg.orchestration_mode}/{cfg.retrieval_mode}/{reranker_str}")
        return []

    logger.info(f"🏃 Running {len(configs)} configurations on {len(queries)} queries (parallel={parallel}, resume={resume})...")

    if parallel:
        results = await run_benchmark_parallel(
            configs, 
            queries, 
            qrels, 
            results_dir, 
            max_parallel=3,
            max_concurrent_api_calls=10
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
    """Run full RAGBench-12x benchmark.
    
    Args:
        dataset: Dataset name (default: scifact)
        parallel: Run configs in parallel (default: False)
        max_configs: Limit number of configs to run
        max_queries: Limit number of queries per config
        dry_run: Only show what would be run
        resume: Skip already completed configurations and continue from where stopped
    """
    return asyncio.run(
        run_benchmark_async(dataset, parallel, max_configs, max_queries, dry_run, resume)
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    import argparse
    parser = argparse.ArgumentParser(
        description="RAGBench-12x: Run RAG benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 12 configs sequentially
  python -m ragbench.run_experiment
  
  # Run in parallel (max 3 concurrent configs)
  python -m ragbench.run_experiment --parallel
  
  # Resume an interrupted run
  python -m ragbench.run_experiment --resume
  
  # Resume in parallel mode
  python -m ragbench.run_experiment --resume --parallel
  
  # Dry run to see what would be run
  python -m ragbench.run_experiment --resume --dry-run
        """
    )
    parser.add_argument("--dataset", default="scifact", help="Dataset name (default: scifact)")
    parser.add_argument("--parallel", action="store_true", help="Run configs in parallel (max 3)")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from where stopped - skip completed configs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without running")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries per config")
    args = parser.parse_args()
    
    run_benchmark(
        dataset=args.dataset,
        parallel=args.parallel,
        max_configs=args.max_configs,
        max_queries=args.max_queries,
        dry_run=args.dry_run,
        resume=args.resume,
    )
