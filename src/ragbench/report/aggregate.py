"""Aggregate benchmark results from multiple runs."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import pandas as pd
from glob import glob


def load_run_results(run_dir: Path) -> Dict[str, Any]:
    """Load results from a single run directory."""
    config_file = run_dir / "config.yaml"
    predictions_file = run_dir / "predictions.jsonl"
    metrics_file = run_dir / "metrics.json"


    import yaml
    with open(config_file) as f:
        config = yaml.safe_load(f)


    with open(metrics_file) as f:
        metrics = json.load(f)


    success_count = metrics.get("successful_queries", 0)
    error_count = metrics.get("error_count", 0)

    return {
        "config_hash": config["__hash__"] if "__hash__" in config else run_dir.name,
        "orchestration_mode": config.get("orchestration_mode", "unknown"),
        "retrieval_mode": config.get("retrieval_mode", "unknown"),
        "use_reranker": config.get("use_reranker", False),
        "dataset": config.get("dataset", "unknown"),
        "top_k": config.get("top_k", 10),
        "max_agentic_steps": config.get("max_agentic_steps", 3),
        "elapsed_seconds": metrics.get("elapsed_seconds", 0),
        "avg_time_per_query": metrics.get("avg_time_per_query", 0),
        "num_queries": metrics.get("num_queries", 0),
        "success_count": success_count,
        "error_count": error_count,

        "ndcg_at_10": metrics.get("ndcg_at_10"),
        "recall_at_5": metrics.get("recall_at_5"),
        "mrr_at_10": metrics.get("mrr_at_10"),

        "faithfulness": metrics.get("faithfulness"),
        "answer_relevancy": metrics.get("answer_relevancy"),
    }


def aggregate_results(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate results from all runs into summary DataFrames.

    Args:
        results_dir: Directory containing runs/ subdirectory

    Returns:
        Tuple of (results_df, metrics_df) where:
        - results_df: Summary of all runs with config details
        - metrics_df: Aggregated metrics across runs
    """
    runs_dir = results_dir / "runs"

    if not runs_dir.exists():
        raise ValueError(f"Runs directory not found: {runs_dir}")


    run_dirs = [Path(d) for d in glob(str(runs_dir / "*")) if Path(d).is_dir()]
    results = []

    for run_dir in run_dirs:
        try:
            result = load_run_results(run_dir)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {run_dir}: {e}")
            continue

    if not results:
        raise ValueError(f"No valid run results found in {runs_dir}")


    results_df = pd.DataFrame(results)



    metrics_df = pd.DataFrame({
        "metric": ["placeholder"],
        "value": [0.0],
        "description": ["Metrics aggregation not yet implemented"]
    })

    return results_df, metrics_df


def load_query_level_metrics(results_dir: Path) -> pd.DataFrame:
    """Load query-level metrics from all runs for statistical analysis.
    
    FIX #3: Enables proper paired tests on query-level data.
    
    Args:
        results_dir: Directory containing runs/ subdirectory
        
    Returns:
        DataFrame with all query-level metrics
    """
    runs_dir = results_dir / "runs"
    all_metrics = []
    
    import yaml
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        query_metrics_file = run_dir / "query_metrics.jsonl"
        config_file = run_dir / "config.yaml"
        
        if not query_metrics_file.exists():
            continue
        
        # Load config
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
        except Exception:
            continue
        
        # Load query metrics
        with open(query_metrics_file) as f:
            for line in f:
                try:
                    metric = json.loads(line)
                    metric["orchestration_mode"] = config.get("orchestration_mode", "unknown")
                    metric["retrieval_mode"] = config.get("retrieval_mode", "unknown")
                    metric["use_reranker"] = config.get("use_reranker", False)
                    all_metrics.append(metric)
                except json.JSONDecodeError:
                    continue
    
    if not all_metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(all_metrics)


def aggregate_with_latency_breakdown(results_dir: Path) -> pd.DataFrame:
    """Aggregate results with separate latency components.
    
    FIX #4: Returns retrieval, reranking, and generation latencies separately.
    """
    results_df, _ = aggregate_results(results_dir)
    query_df = load_query_level_metrics(results_dir)
    
    if query_df.empty:
        return results_df
    
    # Compute latency breakdown per config
    latency_agg = query_df.groupby("config_hash").agg({
        "retrieval_latency": ["mean", "std"],
        "reranking_latency": ["mean", "std"],
        "generation_latency": ["mean", "std"],
        "total_latency": ["mean", "std"],
    })
    
    # Flatten column names
    latency_agg.columns = [f"{col[0]}_{col[1]}" for col in latency_agg.columns]
    latency_agg = latency_agg.reset_index()
    
    # Merge with results
    if "config_hash" in results_df.columns:
        results_df = results_df.merge(latency_agg, on="config_hash", how="left")
    
    return results_df


def save_aggregated_results(results_df: pd.DataFrame, output_dir: Path):
    """Save aggregated results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_file = output_dir / "results.csv"
    results_df.to_csv(csv_file, index=False)

    # Save as JSON lines for easier processing
    jsonl_file = output_dir / "results.jsonl"
    with open(jsonl_file, 'w') as f:
        for _, row in results_df.iterrows():
            f.write(row.to_json() + "\n")

    print(f"[SUCCESS] Aggregated results saved to {output_dir}")
    print(f"  - CSV: {csv_file}")
    print(f"  - JSONL: {jsonl_file}")
