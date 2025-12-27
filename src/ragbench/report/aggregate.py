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
        "context_precision": metrics.get("context_precision"),
        "context_recall": metrics.get("context_recall"),
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
