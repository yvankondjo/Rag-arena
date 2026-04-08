import json

import yaml

from ragbench.report.aggregate import aggregate_results
from ragbench.report.render_md import render_markdown_report


def test_report_generation_from_fixture(tmp_path):
    """The reporting pipeline should work on a minimal fixture result set."""
    results_dir = tmp_path / "results"
    run_dir = results_dir / "runs" / "abc123"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "orchestration_mode": "simple",
                "retrieval_mode": "dense",
                "use_reranker": False,
                "dataset": "scifact",
                "top_k": 10,
                "max_agentic_steps": 3,
            },
            handle,
            sort_keys=False,
        )

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "successful_queries": 2,
                "error_count": 0,
                "num_queries": 2,
                "elapsed_seconds": 1.5,
                "avg_time_per_query": 0.75,
                "ndcg_at_10": 0.8,
                "recall_at_5": 1.0,
                "mrr_at_10": 0.9,
            },
            handle,
            indent=2,
        )

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"query_id": "q1", "response": "answer"}) + "\n")

    results_df, metrics_df = aggregate_results(results_dir)
    report = render_markdown_report(results_df, metrics_df, results_dir)

    assert len(results_df) == 1
    assert "RAGBench-12x Benchmark Report" in report
    assert "simple_dense_no_rerank" in report
    assert "Successful Runs" in report
