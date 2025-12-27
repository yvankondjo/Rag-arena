"""Render benchmark results to Markdown report."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime


def render_markdown_report(results_df: pd.DataFrame, metrics_df: pd.DataFrame, results_dir: Path) -> str:
    """Generate comprehensive Markdown report from benchmark results.

    Args:
        results_df: DataFrame with run results
        metrics_df: DataFrame with aggregated metrics
        results_dir: Results directory path

    Returns:
        Markdown report as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Summary statistics
    total_runs = len(results_df)
    successful_runs = len(results_df[results_df['error_count'] == 0])
    total_queries = results_df['num_queries'].sum()
    total_errors = results_df['error_count'].sum()
    avg_time_per_run = results_df['elapsed_seconds'].mean()

    # Group by configuration axes
    config_summary = results_df.groupby(['orchestration_mode', 'retrieval_mode', 'use_reranker']).agg({
        'elapsed_seconds': 'mean',
        'num_queries': 'mean',
        'error_count': 'sum',
        'success_count': 'sum'
    }).round(2)

    # Performance by retrieval mode
    retrieval_perf = results_df.groupby('retrieval_mode').agg({
        'elapsed_seconds': 'mean',
        'num_queries': 'mean',
        'error_count': 'sum'
    }).round(2).sort_values('elapsed_seconds')

    # Build markdown report
    report_lines = [
        "# RAGBench-12x Benchmark Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Results Directory:** `{results_dir}`",
        "",
        "## Executive Summary",
        "",
        f"- **Total Configurations:** {total_runs}",
        f"- **Successful Runs:** {successful_runs} ({successful_runs/total_runs*100:.1f}%)",
        f"- **Total Queries Processed:** {total_queries:,}",
        f"- **Total Errors:** {total_errors}",
        f"- **Average Time per Configuration:** {avg_time_per_run:.1f}s",
        "",
        "## Configuration Overview",
        "",
        "### All 12 Configurations",
        "",
        "| Orchestration | Retrieval | Reranker | Avg Time | Queries | Errors |",
        "|---------------|-----------|----------|----------|---------|--------|",
    ]

    # Add configuration table
    for idx, row in config_summary.iterrows():
        orch, retrieval, reranker = idx
        reranker_str = "yes" if reranker else "no"
        report_lines.append(
            f"| {orch} | {retrieval} | {reranker_str} | {row['elapsed_seconds']}s | {row['num_queries']:.0f} | {row['error_count']} |"
        )

    report_lines.extend([
        "",
        "### Performance by Retrieval Mode",
        "",
        "| Retrieval Mode | Avg Time | Queries | Total Errors |",
        "|----------------|----------|---------|--------------|",
    ])

    for idx, row in retrieval_perf.iterrows():
        report_lines.append(
            f"| {idx} | {row['elapsed_seconds']}s | {row['num_queries']:.0f} | {row['error_count']} |"
        )

    report_lines.extend([
        "",
        "## Detailed Results",
        "",
        "### Individual Run Results",
        "",
        "| Config Hash | Config | Time | Queries | Success | Errors |",
        "|-------------|--------|------|---------|---------|--------|",
    ])

    # Add individual results
    for _, row in results_df.iterrows():
        reranker_str = "rerank" if row['use_reranker'] else "no_rerank"
        config = f"{row['orchestration_mode']}_{row['retrieval_mode']}_{reranker_str}"
        report_lines.append(
            f"| {row['config_hash'][:8]} | {config} | {row['elapsed_seconds']:.1f}s | {row['num_queries']} | {row['success_count']} | {row['error_count']} |"
        )

    report_lines.extend([
        "",
        "## Methodology",
        "",
        "### Configuration Axes",
        "",
        "**Orchestration (2 modes):**",
        "- `simple`: Single retrieve -> generate",
        "- `agentic`: LangGraph with query rewriting (max 3 steps)",
        "",
        "**Retrieval (3 modes):**",
        "- `dense`: Dense vector similarity (Chroma only)",
        "- `keyword`: Keyword search (BM25s only)",
        "- `hybrid`: Dense + BM25 with RRF fusion",
        "",
        "**Reranker (2 modes):**",
        "- `no_rerank`: No reranking",
        "- `rerank`: CrossEncoder reranking on top-k results",
        "",
        "### Evaluation Metrics",
        "",
        "**Retrieval Metrics:**",
        "- NDCG@10: Normalized Discounted Cumulative Gain",
        "- Recall@5: Fraction of relevant docs in top-5",
        "- MRR@10: Mean Reciprocal Rank",
        "",
        "**RAG Quality Metrics (Ragas):**",
        "- Faithfulness: Answer faithful to retrieved context",
        "- Answer Relevancy: Answer responds to the question",
        "- Context Precision: Retrieved chunks are useful",
        "- Context Recall: All necessary information retrieved",
        "",
        "**Performance Metrics:**",
        "- Latency P50/P95: Query response time percentiles",
        "- Token Usage: Input/output tokens consumed",
        "- Cost Estimation: Based on OpenAI pricing",
        "",
        "## Recommendations",
        "",
        "### Based on Current Results:",
        "",
        "1. **Fastest Retrieval:** " + f"{retrieval_perf.index[0]} ({retrieval_perf.iloc[0]['elapsed_seconds']}s avg)",
        "2. **Most Reliable:** " + f"{config_summary['error_count'].idxmin()} ({config_summary['error_count'].min()} errors)",
        "3. **Best Performance:** Analysis pending - metrics evaluation not yet implemented",
        "",
        "### Next Steps:",
        "",
        "- Implement comprehensive metrics evaluation (retrieval + RAG quality)",
        "- Add latency and cost tracking",
        "- Generate comparison charts and statistical significance tests",
        "- Profile bottlenecks in slow configurations",
        "",
        "---",
        "",
        "*Report generated by RAGBench-12x*",
    ])

    return "\n".join(report_lines)
